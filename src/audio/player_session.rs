use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::watch;

use crate::audio::{AudioBuffer, AudioFormat};
use crate::generation::{OutputIncarnation, StreamGeneration};

const MAX_STATIC_DELAY_MS: u16 = 5_000;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Playback controls applied independently of media preparation.
pub struct AppliedControls {
    /// Playback volume, clamped to 0..=100 when applied.
    pub volume: u8,
    /// Whether playback is muted.
    pub muted: bool,
    /// Static delay in milliseconds, clamped to at most 5000 when applied.
    pub static_delay_ms: u16,
}

/// A partial playback-control update. Requested fields override pending values
/// for the same field; an unset field keeps its current or queued value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControlUpdate {
    /// Requested playback volume, clamped to 0..=100 when applied.
    pub volume: Option<u8>,
    /// Requested mute state.
    pub muted: Option<bool>,
    /// Requested static delay, clamped to at most 5000 milliseconds when applied.
    pub static_delay_ms: Option<u16>,
}

impl ControlUpdate {
    pub(crate) fn apply(self, controls: &mut AppliedControls) {
        if let Some(value) = self.volume {
            controls.volume = value;
        }
        if let Some(value) = self.muted {
            controls.muted = value;
        }
        if let Some(value) = self.static_delay_ms {
            controls.static_delay_ms = value;
        }
        *controls = controls.clone().normalized();
    }
}

#[cfg(test)]
type ControlPatch = ControlUpdate;

impl AppliedControls {
    pub(crate) fn normalized(self) -> Self {
        Self {
            volume: self.volume.min(100),
            muted: self.muted,
            static_delay_ms: self.static_delay_ms.min(MAX_STATIC_DELAY_MS),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Failure reported by a player output operation.
pub enum OutputError {
    /// A callback, setter, or cleanup operation failed; the session becomes terminal.
    /// The first failure is retained, cleanup is attempted at most once, and cleanup
    /// errors are best-effort rather than replacing the original failure.
    Failed(String),
    /// One media item does not match the output's sample rate or channel count.
    FormatMismatch {
        /// Format accepted by the output.
        expected: AudioFormat,
        /// Format supplied by the item.
        actual: AudioFormat,
    },
}

pub(crate) trait OutputAdapter: Send + 'static {
    fn commit(&mut self, buffer: AudioBuffer) -> Result<(), OutputError>;
    fn clear(&mut self) -> Result<(), OutputError>;
    fn set_volume(&mut self, volume: u8) -> Result<(), OutputError>;
    fn set_mute(&mut self, muted: bool) -> Result<(), OutputError>;
    fn set_static_delay(&mut self, delay_ms: u16) -> Result<(), OutputError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Retained lifecycle state of a role session.
pub enum SessionStatus {
    /// The session accepts controls and current output work.
    Open,
    /// The session was closed and rejects future work.
    Closed,
    /// An output failure closed the session; the first error is retained.
    Failed(OutputError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Why prepared output work was not admitted.
pub enum CommitRejected {
    /// A clear or end boundary, output release/reinstall, or newer receipt superseded the work.
    Obsolete,
    /// The session is closed or failed and admits no further work; terminal receives return
    /// `None` without draining.
    Closed,
    /// The output rejected the item. `FormatMismatch` rejects only that item and leaves
    /// the session open; another output failure is terminal.
    Output(OutputError),
}

struct State<O> {
    floor: StreamGeneration,
    incarnation: OutputIncarnation,
    status: SessionStatus,
    output: Option<O>,
    controls: AppliedControls,
    control_version: u64,
}

pub(crate) struct Shared<O> {
    state: Mutex<State<O>>,
    controls: watch::Sender<AppliedControls>,
    admission: Arc<AdmissionGate>,
}
pub(crate) struct Coordinator<O> {
    shared: Arc<Shared<O>>,
}

impl<O> Coordinator<O> {
    pub(crate) fn from_shared(shared: Arc<Shared<O>>) -> Self {
        Self { shared }
    }
}

pub(crate) struct PreparedAudioWork<O> {
    shared: Arc<Shared<O>>,
    generation: StreamGeneration,
    incarnation: OutputIncarnation,
}

impl<O: OutputAdapter> Coordinator<O> {
    #[cfg(test)]
    pub(crate) fn new(output: O, controls: AppliedControls) -> Self {
        let controls = controls.normalized();
        let (control_tx, _) = watch::channel(controls.clone());
        Self::with_controls_sender(
            output,
            controls,
            control_tx,
            Arc::new(AdmissionGate::new(OutputIncarnation::initial())),
        )
    }
    pub(crate) fn with_controls_sender(
        output: O,
        controls: AppliedControls,
        control_tx: watch::Sender<AppliedControls>,
        admission: Arc<AdmissionGate>,
    ) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    floor: StreamGeneration::initial(),
                    incarnation: OutputIncarnation::initial(),
                    status: SessionStatus::Open,
                    output: Some(output),
                    controls,
                    control_version: 0,
                }),
                controls: control_tx,
                admission,
            }),
        }
    }
    #[cfg(test)]
    pub(crate) fn work(&self, _buffer: AudioBuffer) -> PreparedAudioWork<O> {
        let state = self.shared.state.lock();
        PreparedAudioWork {
            shared: Arc::clone(&self.shared),
            generation: state.floor,
            incarnation: state.incarnation,
        }
    }
    pub(crate) fn work_receipt(
        &self,
        generation: StreamGeneration,
        stamp: ReceiptStamp,
    ) -> PreparedAudioWork<O> {
        PreparedAudioWork {
            shared: Arc::clone(&self.shared),
            generation,
            incarnation: stamp.incarnation,
        }
    }
    pub(crate) fn boundary(&self, generation: StreamGeneration) -> Result<(), OutputError> {
        let mut state = self.shared.state.lock();
        if !matches!(state.status, SessionStatus::Open) || generation <= state.floor {
            return Ok(());
        }
        state.floor = generation;
        let Some(mut output) = state.output.take() else {
            return Ok(());
        };
        if let Err(error) = output.clear() {
            state.status = SessionStatus::Failed(error.clone());
            return Err(error);
        }
        state.output = Some(output);
        Ok(())
    }
    pub(crate) fn close(&self) {
        let mut state = self.shared.state.lock();
        if let Some(mut output) = state.output.take() {
            let _ = output.clear();
        }
        if matches!(state.status, SessionStatus::Open) {
            state.status = SessionStatus::Closed;
        }
        self.shared.admission.close();
    }
    #[cfg(test)]
    fn apply_patch(&self, patch: ControlPatch) {
        let mut state = self.shared.state.lock();
        if !matches!(state.status, SessionStatus::Open) {
            return;
        }
        let before = state.controls.clone();
        let mut next = state.controls.clone();
        patch.apply(&mut next);
        let Some(mut output) = state.output.take() else {
            state.controls = next;
            return;
        };
        let result = (|| {
            if before.volume != next.volume {
                output.set_volume(next.volume)?;
            }
            if before.muted != next.muted {
                output.set_mute(next.muted)?;
            }
            if before.static_delay_ms != next.static_delay_ms {
                output.set_static_delay(next.static_delay_ms)?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                state.controls = next;
                state.output = Some(output)
            }
            Err(error) => terminal_failure(&mut state, output, error, true),
        }
    }
    pub(crate) fn status(&self) -> SessionStatus {
        self.shared.state.lock().status.clone()
    }
    pub(crate) fn shared(&self) -> Arc<Shared<O>> {
        Arc::clone(&self.shared)
    }
    pub(crate) fn current_generation(&self) -> StreamGeneration {
        self.shared.state.lock().floor
    }
    pub(crate) fn apply_initial_controls(
        &self,
        controls: AppliedControls,
    ) -> Result<(), OutputError> {
        let mut state = self.shared.state.lock();
        if !matches!(state.status, SessionStatus::Open) {
            return Ok(());
        }
        let controls = controls.normalized();
        let Some(mut output) = state.output.take() else {
            state.controls = controls;
            return Ok(());
        };
        let result: Result<(), OutputError> = (|| {
            output.set_volume(controls.volume)?;
            output.set_mute(controls.muted)?;
            output.set_static_delay(controls.static_delay_ms)?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                state.controls = controls;
                let _ = self.shared.controls.send_replace(state.controls.clone());
                state.output = Some(output);
                Ok(())
            }
            Err(error) => {
                terminal_failure(&mut state, output, error.clone(), true);
                Err(error)
            }
        }
    }

    pub(crate) fn apply_controls_transaction(
        &self,
        capture: impl FnOnce(u64) -> Option<(ControlUpdate, u64)>,
    ) -> Result<(), OutputError> {
        let mut state = self.shared.state.lock();
        if let SessionStatus::Failed(error) = &state.status {
            return Err(error.clone());
        }
        if matches!(state.status, SessionStatus::Closed) {
            return Err(OutputError::Failed("session is closed".into()));
        }
        let Some((update, version)) = capture(state.control_version) else {
            return Ok(());
        };
        if version <= state.control_version {
            return Ok(());
        }
        let mut controls = state.controls.clone();
        update.apply(&mut controls);
        let before = state.controls.clone();
        let Some(mut output) = state.output.take() else {
            state.controls = controls;
            state.control_version = version;
            let _ = self.shared.controls.send_replace(state.controls.clone());
            return Ok(());
        };
        let result: Result<(), OutputError> = (|| {
            if before.volume != controls.volume {
                output.set_volume(controls.volume)?;
            }
            if before.muted != controls.muted {
                output.set_mute(controls.muted)?;
            }
            if before.static_delay_ms != controls.static_delay_ms {
                output.set_static_delay(controls.static_delay_ms)?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                state.controls = controls;
                state.control_version = version;
                let _ = self.shared.controls.send_replace(state.controls.clone());
                state.output = Some(output);
                Ok(())
            }
            Err(error) => {
                terminal_failure(&mut state, output, error.clone(), true);
                Err(error)
            }
        }
    }
}

impl<O: OutputAdapter> PreparedAudioWork<O> {
    #[cfg(test)]
    pub(crate) fn commit(self, buffer: AudioBuffer) -> Result<(), CommitRejected> {
        let generation = self.generation;
        self.commit_at_with_status(generation, buffer).0
    }
    pub(crate) fn commit_at_with_status(
        self,
        generation: StreamGeneration,
        buffer: AudioBuffer,
    ) -> (Result<(), CommitRejected>, SessionStatus) {
        let coordinator = Coordinator::from_shared(Arc::clone(&self.shared));
        let result = coordinator
            .boundary(generation)
            .map_err(CommitRejected::Output);
        if let Err(error) = result {
            return (Err(error), coordinator.status());
        }
        let mut state = self.shared.state.lock();
        if !matches!(state.status, SessionStatus::Open) {
            return (Err(CommitRejected::Closed), state.status.clone());
        }
        if self.generation != state.floor || self.incarnation != state.incarnation {
            return (Err(CommitRejected::Obsolete), state.status.clone());
        }
        if matches!(self.shared.admission.admit(), Admission::Closed) {
            return (Err(CommitRejected::Closed), state.status.clone());
        }
        let Some(mut output) = state.output.take() else {
            return (Err(CommitRejected::Closed), state.status.clone());
        };
        // Keep the session lock across the short, synchronous callback. This
        // serializes commit with boundary/clear and prevents stale reinsertion.
        let result = output.commit(buffer);
        let closed = {
            let gate = self.shared.admission.state.lock();
            matches!(gate.state, AdmissionState::Closed)
        };
        if closed {
            let _ = output.clear();
            state.status = SessionStatus::Closed;
            return (Err(CommitRejected::Closed), state.status.clone());
        }
        let result = match result {
            Ok(()) => {
                state.output = Some(output);
                Ok(())
            }
            Err(error @ OutputError::FormatMismatch { .. }) => {
                state.output = Some(output);
                Err(CommitRejected::Output(error))
            }
            Err(error) => {
                let rejection = CommitRejected::Output(error.clone());
                terminal_failure(&mut state, output, error, true);
                Err(rejection)
            }
        };
        (result, state.status.clone())
    }
}

fn terminal_failure<O: OutputAdapter>(
    state: &mut State<O>,
    mut output: O,
    error: OutputError,
    cleanup: bool,
) {
    if cleanup {
        let _ = output.clear();
    }
    state.status = SessionStatus::Failed(error);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReceiptStamp {
    pub(crate) incarnation: OutputIncarnation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdmissionState {
    Open(OutputIncarnation),
    Detached,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Admission {
    Open(ReceiptStamp),
    Detached,
    Closed,
}

pub(crate) struct AdmissionGate {
    state: Mutex<AdmissionData>,
}

struct AdmissionData {
    state: AdmissionState,
}

impl AdmissionGate {
    pub(crate) fn new(incarnation: OutputIncarnation) -> Self {
        Self {
            state: Mutex::new(AdmissionData {
                state: AdmissionState::Open(incarnation),
            }),
        }
    }
    pub(crate) fn admit(&self) -> Admission {
        match self.state.lock().state {
            AdmissionState::Open(incarnation) => Admission::Open(ReceiptStamp { incarnation }),
            AdmissionState::Detached => Admission::Detached,
            AdmissionState::Closed => Admission::Closed,
        }
    }
    pub(crate) fn detach(&self) {
        let mut data = self.state.lock();
        if !matches!(data.state, AdmissionState::Closed) {
            data.state = AdmissionState::Detached;
        }
    }
    pub(crate) fn close(&self) {
        self.state.lock().state = AdmissionState::Closed;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LazyState {
    Initial,
    Reserved(u64),
    Installed,
    Detached,
    ReservedDetached(u64),
}

/// Explains why an output-install reservation or installation was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstallRejected {
    /// The session has been explicitly closed.
    Closed,
    /// The session has failed and cannot be recovered by reinstalling output.
    Failed,
    /// An output is already installed.
    AlreadyInstalled,
    /// Another installation reservation is active.
    Reserved,
    /// The reservation belongs to an older output incarnation.
    Superseded,
}

pub(crate) struct LazyShared<O> {
    session: Mutex<State<O>>,
    controls: watch::Sender<AppliedControls>,
    admission: Arc<AdmissionGate>,
    lazy: Mutex<LazyState>,
    next_reservation: Mutex<u64>,
}
pub(crate) struct LazyCoordinator<O> {
    shared: Arc<LazyShared<O>>,
}
pub(crate) struct InstallPermit<O> {
    shared: Arc<LazyShared<O>>,
    incarnation: OutputIncarnation,
    reservation: u64,
    consumed: bool,
}
pub(crate) struct LazyWork<O> {
    shared: Arc<LazyShared<O>>,
    stamp: ReceiptStamp,
    generation: StreamGeneration,
}

impl<O: OutputAdapter> LazyCoordinator<O> {
    pub(crate) fn from_shared(shared: Arc<LazyShared<O>>) -> Self {
        Self { shared }
    }
    pub(crate) fn shared(&self) -> Arc<LazyShared<O>> {
        Arc::clone(&self.shared)
    }
    #[cfg(test)]
    pub(crate) fn new(controls: AppliedControls) -> Self {
        let incarnation = OutputIncarnation::initial();
        Self::with_admission(controls, Arc::new(AdmissionGate::new(incarnation)))
    }
    #[cfg(test)]
    pub(crate) fn with_admission(controls: AppliedControls, admission: Arc<AdmissionGate>) -> Self {
        let controls = controls.normalized();
        let (control_tx, _) = watch::channel(controls.clone());
        Self::with_admission_and_controls(controls, admission, control_tx)
    }
    pub(crate) fn with_admission_and_controls(
        controls: AppliedControls,
        admission: Arc<AdmissionGate>,
        control_tx: watch::Sender<AppliedControls>,
    ) -> Self {
        let incarnation = OutputIncarnation::initial();
        let controls = controls.normalized();
        Self {
            shared: Arc::new(LazyShared {
                session: Mutex::new(State {
                    floor: StreamGeneration::initial(),
                    incarnation,
                    status: SessionStatus::Open,
                    output: None,
                    controls,
                    control_version: 0,
                }),
                controls: control_tx,
                admission,
                lazy: Mutex::new(LazyState::Initial),
                next_reservation: Mutex::new(0),
            }),
        }
    }
    pub(crate) fn admission(&self) -> Arc<AdmissionGate> {
        Arc::clone(&self.shared.admission)
    }
    #[cfg(test)]
    pub(crate) fn receipt(&self) -> Option<LazyWork<O>> {
        let Admission::Open(stamp) = self.shared.admission.admit() else {
            return None;
        };
        let generation = self.shared.session.lock().floor;
        Some(self.receipt_stamped(stamp, generation))
    }
    pub(crate) fn receipt_stamped(
        &self,
        stamp: ReceiptStamp,
        generation: StreamGeneration,
    ) -> LazyWork<O> {
        LazyWork {
            shared: Arc::clone(&self.shared),
            stamp,
            generation,
        }
    }
    pub(crate) fn begin_install(&self) -> Result<InstallPermit<O>, InstallRejected> {
        let state = self.shared.session.lock();
        if !matches!(state.status, SessionStatus::Open) {
            return Err(if matches!(state.status, SessionStatus::Closed) {
                InstallRejected::Closed
            } else {
                InstallRejected::Failed
            });
        }
        let incarnation = state.incarnation;
        let mut lazy = self.shared.lazy.lock();
        let reservation = {
            let mut next = self.shared.next_reservation.lock();
            *next = next.wrapping_add(1);
            *next
        };
        match *lazy {
            LazyState::Initial => *lazy = LazyState::Reserved(reservation),
            LazyState::Detached => *lazy = LazyState::ReservedDetached(reservation),
            LazyState::Reserved(_) | LazyState::ReservedDetached(_) => {
                return Err(InstallRejected::Reserved)
            }
            LazyState::Installed => return Err(InstallRejected::AlreadyInstalled),
        }
        Ok(InstallPermit {
            shared: Arc::clone(&self.shared),
            incarnation,
            reservation,
            consumed: false,
        })
    }
    pub(crate) fn release(&self) -> Result<(), OutputError> {
        let mut state = self.shared.session.lock();
        if !matches!(state.status, SessionStatus::Open) {
            return Err(OutputError::Failed("session is terminal".into()));
        }
        let next = match state.incarnation.checked_advance() {
            Some(next) => next,
            None => {
                let error = OutputError::Failed("output incarnation exhausted".into());
                self.shared.admission.close();
                if let Some(mut output) = state.output.take() {
                    let _ = output.clear();
                }
                state.status = SessionStatus::Failed(error.clone());
                *self.shared.lazy.lock() = LazyState::Detached;
                return Err(error);
            }
        };
        self.shared.admission.detach();
        state.incarnation = next;
        let result = state
            .output
            .take()
            .map_or(Ok(()), |mut output| output.clear());
        if let Err(error) = result {
            state.status = SessionStatus::Failed(error.clone());
            return Err(error);
        }
        *self.shared.lazy.lock() = LazyState::Detached;
        Ok(())
    }
    pub(crate) fn status(&self) -> SessionStatus {
        self.shared.session.lock().status.clone()
    }
    pub(crate) fn current_generation(&self) -> StreamGeneration {
        self.shared.session.lock().floor
    }
    #[cfg(test)]
    pub(crate) fn control_version(&self) -> u64 {
        self.shared.session.lock().control_version
    }
    pub(crate) fn apply_controls_transaction(
        &self,
        capture: impl FnOnce(u64) -> Option<(ControlUpdate, u64)>,
    ) -> Result<(), OutputError> {
        let mut state = self.shared.session.lock();
        if let SessionStatus::Failed(error) = &state.status {
            return Err(error.clone());
        }
        if matches!(state.status, SessionStatus::Closed) {
            return Err(OutputError::Failed("session is closed".into()));
        }
        let Some((update, version)) = capture(state.control_version) else {
            return Ok(());
        };
        if version <= state.control_version {
            return Ok(());
        }
        let mut controls = state.controls.clone();
        update.apply(&mut controls);
        let before = state.controls.clone();
        let Some(mut output) = state.output.take() else {
            state.controls = controls;
            state.control_version = version;
            let _ = self.shared.controls.send_replace(state.controls.clone());
            return Ok(());
        };
        let result: Result<(), OutputError> = (|| {
            if before.volume != controls.volume {
                output.set_volume(controls.volume)?;
            }
            if before.muted != controls.muted {
                output.set_mute(controls.muted)?;
            }
            if before.static_delay_ms != controls.static_delay_ms {
                output.set_static_delay(controls.static_delay_ms)?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => {
                state.controls = controls;
                state.control_version = version;
                let _ = self.shared.controls.send_replace(state.controls.clone());
                state.output = Some(output);
                Ok(())
            }
            Err(error) => {
                terminal_failure(&mut state, output, error.clone(), true);
                Err(error)
            }
        }
    }

    pub(crate) fn close(&self) {
        let mut state = self.shared.session.lock();
        if let Some(mut output) = state.output.take() {
            let _ = output.clear();
        }
        if matches!(state.status, SessionStatus::Open) {
            state.status = SessionStatus::Closed;
        }
        self.shared.admission.close();
    }
    pub(crate) fn boundary(&self, generation: StreamGeneration) -> Result<(), OutputError> {
        let mut state = self.shared.session.lock();
        if !matches!(state.status, SessionStatus::Open) || generation <= state.floor {
            return Ok(());
        }
        state.floor = generation;
        if let Some(mut output) = state.output.take() {
            if let Err(error) = output.clear() {
                state.status = SessionStatus::Failed(error.clone());
                return Err(error);
            }
            state.output = Some(output);
        }
        Ok(())
    }
}

impl<O: OutputAdapter> InstallPermit<O> {
    #[cfg(test)]
    pub(crate) fn install(self, output: O) -> Result<(), InstallRejected> {
        let controls = self.shared.session.lock().controls.clone();
        self.install_with_controls(output, 0, controls, || None)
    }

    pub(crate) fn install_with_controls(
        mut self,
        mut output: O,
        version: u64,
        controls: AppliedControls,
        current_boundary: impl FnOnce() -> Option<StreamGeneration>,
    ) -> Result<(), InstallRejected> {
        let mut state = self.shared.session.lock();
        if !matches!(state.status, SessionStatus::Open) {
            return Err(if matches!(state.status, SessionStatus::Closed) {
                InstallRejected::Closed
            } else {
                InstallRejected::Failed
            });
        }
        if state.incarnation != self.incarnation {
            return Err(InstallRejected::Superseded);
        }
        let mut lazy = self.shared.lazy.lock();
        let reservation_matches = matches!(
            *lazy,
            LazyState::Reserved(id) | LazyState::ReservedDetached(id) if id == self.reservation
        );
        if !reservation_matches {
            return Err(InstallRejected::AlreadyInstalled);
        }
        let (controls, version) = if version <= state.control_version {
            (state.controls.clone(), state.control_version)
        } else {
            (controls.normalized(), version)
        };
        let applied = output
            .set_volume(controls.volume)
            .and_then(|_| output.set_mute(controls.muted))
            .and_then(|_| output.set_static_delay(controls.static_delay_ms));
        if let Err(error) = applied {
            terminal_failure(&mut state, output, error, true);
            *lazy = LazyState::Detached;
            return Err(InstallRejected::Failed);
        }
        let retained_boundary = current_boundary();
        if let Some(boundary) = retained_boundary {
            if boundary > state.floor {
                state.floor = boundary;
            }
            if let Err(error) = output.clear() {
                terminal_failure(&mut state, output, error, false);
                *lazy = LazyState::Detached;
                return Err(InstallRejected::Failed);
            }
        }
        let mut gate = self.shared.admission.state.lock();
        if matches!(gate.state, AdmissionState::Closed) {
            drop(gate);
            let _ = output.clear();
            state.status = SessionStatus::Closed;
            *lazy = LazyState::Detached;
            return Err(InstallRejected::Closed);
        }
        state.controls = controls;
        state.control_version = version;
        let _ = self.shared.controls.send_replace(state.controls.clone());
        state.output = Some(output);
        gate.state = AdmissionState::Open(state.incarnation);
        drop(gate);
        *lazy = LazyState::Installed;
        self.consumed = true;
        Ok(())
    }
}

impl<O> Drop for InstallPermit<O> {
    fn drop(&mut self) {
        if self.consumed {
            return;
        }
        let mut lazy = self.shared.lazy.lock();
        match *lazy {
            LazyState::Reserved(id) if id == self.reservation => *lazy = LazyState::Initial,
            LazyState::ReservedDetached(id) if id == self.reservation => {
                *lazy = LazyState::Detached
            }
            _ => {}
        }
    }
}

impl<O: OutputAdapter> LazyWork<O> {
    #[cfg(test)]
    pub(crate) fn commit(self, buffer: AudioBuffer) -> Result<(), CommitRejected> {
        let generation = self.generation;
        self.commit_at_with_status(generation, buffer).0
    }
    pub(crate) fn commit_at_with_status(
        self,
        generation: StreamGeneration,
        buffer: AudioBuffer,
    ) -> (Result<(), CommitRejected>, SessionStatus) {
        let coordinator = LazyCoordinator::from_shared(Arc::clone(&self.shared));
        if let Err(error) = coordinator
            .boundary(generation)
            .map_err(CommitRejected::Output)
        {
            return (Err(error), coordinator.status());
        }
        let mut state = self.shared.session.lock();
        if !matches!(state.status, SessionStatus::Open) {
            return (Err(CommitRejected::Closed), state.status.clone());
        }
        if self.stamp.incarnation != state.incarnation || self.generation != state.floor {
            return (Err(CommitRejected::Obsolete), state.status.clone());
        }
        if matches!(self.shared.admission.admit(), Admission::Closed) {
            return (Err(CommitRejected::Closed), state.status.clone());
        }
        let Some(mut output) = state.output.take() else {
            return (Err(CommitRejected::Obsolete), state.status.clone());
        };
        // Keep the session lock across the short, synchronous callback. This
        // serializes commit with boundary/clear and prevents stale reinsertion.
        let result = output.commit(buffer);
        let closed = {
            let gate = self.shared.admission.state.lock();
            matches!(gate.state, AdmissionState::Closed)
        };
        if closed {
            let _ = output.clear();
            state.status = SessionStatus::Closed;
            return (Err(CommitRejected::Closed), state.status.clone());
        }
        let result = match result {
            Ok(()) => {
                state.output = Some(output);
                Ok(())
            }
            Err(error @ OutputError::FormatMismatch { .. }) => {
                state.output = Some(output);
                Err(CommitRejected::Output(error))
            }
            Err(error) => {
                let rejection = CommitRejected::Output(error.clone());
                terminal_failure(&mut state, output, error, true);
                Err(rejection)
            }
        };
        (result, state.status.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioBuffer, Codec};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;

    struct FakeOutput {
        format: AudioFormat,
        commits: usize,
        clears: Arc<AtomicUsize>,
        fail_commit: Option<OutputError>,
        fail_clear: Option<OutputError>,
        controls: AppliedControls,
        commit_gate: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
    }
    impl OutputAdapter for FakeOutput {
        fn commit(&mut self, buffer: AudioBuffer) -> Result<(), OutputError> {
            self.commits += 1;
            if let Some((ready, release)) = self.commit_gate.take() {
                ready.send(()).unwrap();
                release.recv().unwrap();
            }
            if let Some(error) = self.fail_commit.clone() {
                return Err(error);
            }
            if self.format.sample_rate != buffer.format.sample_rate
                || self.format.channels != buffer.format.channels
            {
                return Err(OutputError::FormatMismatch {
                    expected: self.format.clone(),
                    actual: buffer.format,
                });
            }
            Ok(())
        }
        fn clear(&mut self) -> Result<(), OutputError> {
            self.clears.fetch_add(1, Ordering::SeqCst);
            self.fail_clear.clone().map_or(Ok(()), Err)
        }
        fn set_volume(&mut self, volume: u8) -> Result<(), OutputError> {
            self.controls.volume = volume;
            Ok(())
        }
        fn set_mute(&mut self, muted: bool) -> Result<(), OutputError> {
            self.controls.muted = muted;
            Ok(())
        }
        fn set_static_delay(&mut self, delay_ms: u16) -> Result<(), OutputError> {
            self.controls.static_delay_ms = delay_ms;
            Ok(())
        }
    }
    fn format(rate: u32, channels: u8) -> AudioFormat {
        AudioFormat {
            codec: Codec::Pcm,
            sample_rate: rate,
            channels,
            bit_depth: 16,
            codec_header: None,
        }
    }
    fn buffer(rate: u32, channels: u8) -> AudioBuffer {
        AudioBuffer {
            timestamp: 0,
            samples: Arc::from(vec![0; channels as usize]),
            format: format(rate, channels),
        }
    }
    fn controls() -> AppliedControls {
        AppliedControls {
            volume: 255,
            muted: true,
            static_delay_ms: 9999,
        }
    }
    fn fake(clears: Arc<AtomicUsize>) -> FakeOutput {
        FakeOutput {
            format: format(48_000, 2),
            commits: 0,
            clears,
            fail_commit: None,
            fail_clear: None,
            controls: AppliedControls {
                volume: 0,
                muted: false,
                static_delay_ms: 0,
            },
            commit_gate: None,
        }
    }
    fn coordinator(clears: Arc<AtomicUsize>) -> Coordinator<FakeOutput> {
        Coordinator::new(fake(clears), controls())
    }

    #[test]
    fn commit_holds_session_lock_until_clear_then_new_generation_is_accepted() {
        let clears = Arc::new(AtomicUsize::new(0));
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut output = fake(Arc::clone(&clears));
        output.commit_gate = Some((ready_tx, release_rx));
        let c = Arc::new(Coordinator::new(output, controls()));
        let old = c.work(buffer(48_000, 2));
        let stale = c.work(buffer(48_000, 2));
        let commit = std::thread::spawn(move || old.commit(buffer(48_000, 2)));
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        let (clear_tx, clear_rx) = mpsc::channel();
        let clear_c = Arc::clone(&c);
        let clear = std::thread::spawn(move || {
            let result = clear_c.boundary(StreamGeneration::initial().checked_advance().unwrap());
            clear_tx.send(result).unwrap();
        });
        assert!(clear_rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err());
        release_tx.send(()).unwrap();
        assert_eq!(commit.join().unwrap(), Ok(()));
        assert_eq!(
            clear_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap(),
            Ok(())
        );
        clear.join().unwrap();
        assert_eq!(
            stale.commit(buffer(48_000, 2)),
            Err(CommitRejected::Obsolete)
        );
        assert_eq!(
            c.current_generation(),
            StreamGeneration::initial().checked_advance().unwrap()
        );
        let next = c.work(buffer(48_000, 2));
        assert_eq!(next.commit(buffer(48_000, 2)), Ok(()));
        assert_eq!(clears.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn stale_commit_and_late_boundary_are_no_ops() {
        let clears = Arc::new(AtomicUsize::new(0));
        let c = coordinator(Arc::clone(&clears));
        let stale = c.work(buffer(48_000, 2));
        let _ = c.boundary(StreamGeneration::initial().checked_advance().unwrap());
        assert_eq!(
            stale.commit(buffer(48_000, 2)),
            Err(CommitRejected::Obsolete)
        );
        let current = c.work(buffer(48_000, 2));
        let _ = c.boundary(StreamGeneration::initial());
        assert_eq!(current.commit(buffer(48_000, 2)), Ok(()));
        assert_eq!(clears.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn terminal_rejects_equal_generation_work() {
        let c = coordinator(Arc::new(AtomicUsize::new(0)));
        let work = c.work(buffer(48_000, 2));
        c.close();
        assert_eq!(work.commit(buffer(48_000, 2)), Err(CommitRejected::Closed));
        assert_eq!(c.status(), SessionStatus::Closed);
    }
    #[test]
    fn closed_control_update_rejects_before_capture() {
        let c = coordinator(Arc::new(AtomicUsize::new(0)));
        c.close();
        let captured = Arc::new(AtomicBool::new(false));
        let result = c.apply_controls_transaction({
            let captured = Arc::clone(&captured);
            move |_| {
                captured.store(true, Ordering::SeqCst);
                Some((
                    ControlUpdate {
                        volume: Some(25),
                        ..ControlUpdate::default()
                    },
                    1,
                ))
            }
        });
        assert_eq!(result, Err(OutputError::Failed("session is closed".into())));
        assert!(!captured.load(Ordering::SeqCst));
    }

    #[test]
    fn failed_control_update_returns_original_failure_before_capture() {
        let c = coordinator(Arc::new(AtomicUsize::new(0)));
        c.shared.state.lock().status =
            SessionStatus::Failed(OutputError::Failed("original".into()));
        let first = c.apply_controls_transaction(|_| {
            Some((
                ControlUpdate {
                    volume: Some(25),
                    ..ControlUpdate::default()
                },
                1,
            ))
        });
        assert_eq!(first, Err(OutputError::Failed("original".into())));
        let captured = Arc::new(AtomicBool::new(false));
        let second = c.apply_controls_transaction({
            let captured = Arc::clone(&captured);
            move |_| {
                captured.store(true, Ordering::SeqCst);
                Some((
                    ControlUpdate {
                        volume: Some(30),
                        ..ControlUpdate::default()
                    },
                    2,
                ))
            }
        });
        assert_eq!(second, Err(OutputError::Failed("original".into())));
        assert!(!captured.load(Ordering::SeqCst));
    }

    #[test]
    fn lazy_closed_control_update_rejects_before_capture() {
        let c = LazyCoordinator::<FakeOutput>::new(controls());
        c.close();
        let captured = Arc::new(AtomicBool::new(false));
        let result = c.apply_controls_transaction({
            let captured = Arc::clone(&captured);
            move |_| {
                captured.store(true, Ordering::SeqCst);
                Some((
                    ControlUpdate {
                        volume: Some(25),
                        ..ControlUpdate::default()
                    },
                    1,
                ))
            }
        });
        assert_eq!(result, Err(OutputError::Failed("session is closed".into())));
        assert!(!captured.load(Ordering::SeqCst));
    }

    #[test]
    fn callback_failure_is_terminal_and_cleanup_runs_once() {
        let clears = Arc::new(AtomicUsize::new(0));
        let c = coordinator(Arc::clone(&clears));
        c.shared.state.lock().output.as_mut().unwrap().fail_commit =
            Some(OutputError::Failed("callback".into()));
        assert_eq!(
            c.work(buffer(48_000, 2)).commit(buffer(48_000, 2)),
            Err(CommitRejected::Output(OutputError::Failed(
                "callback".into()
            )))
        );
        assert_eq!(clears.load(Ordering::SeqCst), 1);
        assert!(matches!(
            c.status(),
            SessionStatus::Failed(OutputError::Failed(_))
        ));
    }
    #[test]
    fn format_mismatch_is_nonterminal() {
        let c = coordinator(Arc::new(AtomicUsize::new(0)));
        assert!(matches!(
            c.work(buffer(44_100, 2)).commit(buffer(44_100, 2)),
            Err(CommitRejected::Output(OutputError::FormatMismatch { .. }))
        ));
        assert_eq!(c.status(), SessionStatus::Open);
        assert_eq!(c.work(buffer(48_000, 2)).commit(buffer(48_000, 2)), Ok(()));
    }
    #[test]
    fn controls_are_clamped_and_merged() {
        let c = coordinator(Arc::new(AtomicUsize::new(0)));
        c.apply_patch(ControlPatch {
            volume: Some(255),
            ..ControlPatch::default()
        });
        c.apply_patch(ControlPatch {
            static_delay_ms: Some(9000),
            ..ControlPatch::default()
        });
        let state = c.shared.state.lock();
        assert_eq!(state.controls.volume, 100);
        assert_eq!(state.controls.static_delay_ms, 5000);
        assert!(state.controls.muted);
    }
    #[test]
    fn coordinator_has_no_arc_self_cycle() {
        let c = coordinator(Arc::new(AtomicUsize::new(0)));
        let weak = Arc::downgrade(&c.shared);
        c.close();
        drop(c);
        assert!(weak.upgrade().is_none());
    }

    fn lazy() -> LazyCoordinator<FakeOutput> {
        LazyCoordinator::new(controls())
    }

    #[test]
    fn incarnation_exhaustion_is_terminal_and_rejects_work_install_and_admission() {
        let c = lazy();
        c.shared.session.lock().incarnation = OutputIncarnation::max_for_test();
        let work = c.receipt().unwrap();
        let permit = c.begin_install().unwrap();
        let error = c.release().unwrap_err();
        assert_eq!(
            error,
            OutputError::Failed("output incarnation exhausted".into())
        );
        assert_eq!(work.commit(buffer(48_000, 2)), Err(CommitRejected::Closed));
        assert!(matches!(c.begin_install(), Err(InstallRejected::Failed)));
        assert!(c.receipt().is_none());
        drop(permit);
    }
    fn installable() -> FakeOutput {
        fake(Arc::new(AtomicUsize::new(0)))
    }
    #[test]
    fn abandoned_install_permit_allows_retry_and_duplicate_is_rejected() {
        let c = lazy();
        let permit = c.begin_install().unwrap();
        drop(permit);
        let permit = c.begin_install().unwrap();
        permit.install(installable()).unwrap();
        assert!(matches!(
            c.begin_install(),
            Err(InstallRejected::AlreadyInstalled)
        ));
    }
    #[test]
    fn install_keeps_retained_controls_for_equal_snapshot_version() {
        let c = lazy();
        c.apply_controls_transaction(|_| {
            Some((
                ControlUpdate {
                    volume: Some(37),
                    muted: Some(true),
                    ..ControlUpdate::default()
                },
                1,
            ))
        })
        .unwrap();
        let stale = AppliedControls {
            volume: 3,
            muted: false,
            static_delay_ms: 1,
        };
        let permit = c.begin_install().unwrap();
        permit
            .install_with_controls(installable(), 1, stale, || None)
            .unwrap();
        let state = c.shared.session.lock();
        assert_eq!(state.control_version, 1);
        assert_eq!(state.controls.volume, 37);
        assert!(state.controls.muted);
        assert_eq!(state.output.as_ref().unwrap().controls.volume, 37);
        assert!(state.output.as_ref().unwrap().controls.muted);
    }
    #[test]
    fn initial_install_accepts_initial_controls_at_version_zero() {
        let c = lazy();
        let permit = c.begin_install().unwrap();
        permit.install(installable()).unwrap();
        let state = c.shared.session.lock();
        assert_eq!(state.control_version, 0);
        assert_eq!(state.controls, controls().normalized());
        assert_eq!(
            state.output.as_ref().unwrap().controls,
            controls().normalized()
        );
    }
    #[test]
    fn release_rejects_old_receipt_and_reinstall_replays_controls() {
        let c = lazy();
        let old = c.receipt().unwrap();
        let permit = c.begin_install().unwrap();
        permit.install(installable()).unwrap();
        c.release().unwrap();
        assert_eq!(old.commit(buffer(48_000, 2)), Err(CommitRejected::Obsolete));
        let permit = c.begin_install().unwrap();
        permit.install(installable()).unwrap();
        assert_eq!(c.receipt().unwrap().commit(buffer(48_000, 2)), Ok(()));
    }
    #[test]
    fn detached_receipt_is_dropped_and_old_permit_cannot_install() {
        let c = lazy();
        let permit = c.begin_install().unwrap();
        c.release().unwrap();
        assert_eq!(
            permit.install(installable()),
            Err(InstallRejected::Superseded)
        );
        assert!(c.receipt().is_none());
    }
    #[test]
    fn clear_spans_install_and_terminal_rejects_install() {
        let c = lazy();
        let _ = c.boundary(StreamGeneration::initial().checked_advance().unwrap());
        let permit = c.begin_install().unwrap();
        permit.install(installable()).unwrap();
        assert!(c.shared.session.lock().floor > StreamGeneration::initial());
        c.shared.session.lock().status = SessionStatus::Closed;
        assert!(matches!(c.begin_install(), Err(InstallRejected::Closed)));
    }
}
