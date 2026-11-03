use std::sync::Arc;

use parking_lot::Mutex;
use thiserror::Error;
use tokio::sync::watch;

pub(crate) struct Lifecycle {
    pub(crate) status: watch::Sender<SessionStatus>,
    pub(crate) current: Arc<Mutex<SessionStatus>>,
    pub(crate) transition: Mutex<()>,
}

impl Lifecycle {
    pub(crate) fn new(
        status: watch::Sender<SessionStatus>,
        current: Arc<Mutex<SessionStatus>>,
    ) -> Self {
        Self {
            status,
            current,
            transition: Mutex::new(()),
        }
    }
}

use super::diagnostics::DiagnosticScope;
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

#[derive(Clone, Debug, Error, PartialEq, Eq)]
/// Failure reported by an output operation.
pub enum OutputError {
    /// A player output operation failed; the player session detaches the output and can recover
    /// through a fresh installation. Custom role sessions retain this as a terminal failure.
    /// The first failure is retained and cleanup errors are best-effort.
    #[error("output failed: {0}")]
    Failed(String),
    /// One media item does not match the output's sample rate or channel count.
    #[error("output format mismatch: expected {expected:?}, got {actual:?}")]
    FormatMismatch {
        /// Format accepted by the output.
        expected: AudioFormat,
        /// Format supplied by the item.
        actual: AudioFormat,
    },
    /// A buffer does not contain a whole number of interleaved frames.
    /// This rejects only the item and leaves the output session open.
    #[error("invalid output buffer: {samples} samples for {channels} channels")]
    InvalidBuffer {
        /// Number of interleaved samples supplied.
        samples: usize,
        /// Number of channels required to form a frame.
        channels: u8,
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
    /// The output is detached after a recoverable device failure; the first error is retained.
    OutputLost(OutputError),
    /// A terminal failure rejects future work.
    Failed(OutputError),
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
/// Why prepared output work was not admitted.
pub enum CommitRejected {
    /// A clear or end boundary or an output release superseded the work, or no output is installed.
    #[error("output work is obsolete")]
    Obsolete,
    /// The session is closed or failed and admits no further work; terminal receives return
    /// `None` without draining.
    #[error("output session is closed")]
    Closed,
    /// The output rejected the item. `FormatMismatch` and `InvalidBuffer` reject only that
    /// item and leave the session open; another player output failure detaches the output.
    #[error("output rejected: {0}")]
    Output(#[source] OutputError),
}

struct State<O> {
    floor: StreamGeneration,
    incarnation: OutputIncarnation,
    status: SessionStatus,
    output: Option<O>,
    /// Controls known to have been applied to the installed output.
    controls: AppliedControls,
    /// Latest requested controls, replayed when an output is installed.
    desired_controls: AppliedControls,
    control_version: u64,
}

fn output_lost<O: OutputAdapter>(state: &mut State<O>, mut output: Option<O>, error: OutputError) {
    if let Some(mut output) = output.take() {
        let _ = output.clear();
    }
    state.output = None;
    if !matches!(state.status, SessionStatus::OutputLost(_)) {
        state.status = SessionStatus::OutputLost(error);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReceiptStamp {
    pub(crate) incarnation: OutputIncarnation,
}

#[derive(Default)]
pub(crate) struct AdmissionGate {
    closed: Mutex<bool>,
}

impl AdmissionGate {
    pub(crate) fn is_closed(&self) -> bool {
        *self.closed.lock()
    }
    pub(crate) fn close(&self) {
        *self.closed.lock() = true;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InstallState {
    Initial,
    Reserved(u64),
    Installed,
    Detached,
    ReservedDetached(u64),
}

/// Explains why an output-install reservation or installation was rejected.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum InstallRejected {
    /// The session has been explicitly closed.
    #[error("session is closed")]
    Closed,
    /// This installation attempt failed while applying controls or clearing a retained boundary.
    /// A player session may retry with a fresh permit after output loss.
    #[error("output installation failed")]
    Failed,
    /// An output is already installed.
    #[error("output is already installed")]
    AlreadyInstalled,
    /// Another installation reservation is active.
    #[error("output installation is already reserved")]
    Reserved,
    /// The reservation belongs to an older output incarnation.
    #[error("output installation was superseded")]
    Superseded,
}

pub(crate) struct Shared<O> {
    session: Mutex<State<O>>,
    controls: watch::Sender<AppliedControls>,
    lifecycle: Arc<Lifecycle>,
    admission: Arc<AdmissionGate>,
    install_state: Mutex<InstallState>,
    next_reservation: Mutex<u64>,
}
pub(crate) struct Coordinator<O> {
    shared: Arc<Shared<O>>,
}
pub(crate) struct InstallPermit<O> {
    shared: Arc<Shared<O>>,
    incarnation: OutputIncarnation,
    reservation: u64,
    consumed: bool,
}
pub(crate) struct PreparedAudioWork<O> {
    shared: Arc<Shared<O>>,
    stamp: ReceiptStamp,
    generation: StreamGeneration,
}

impl<O: OutputAdapter> Coordinator<O> {
    pub(crate) fn from_shared(shared: Arc<Shared<O>>) -> Self {
        Self { shared }
    }
    pub(crate) fn shared(&self) -> Arc<Shared<O>> {
        Arc::clone(&self.shared)
    }
    #[cfg(test)]
    pub(crate) fn new(controls: AppliedControls) -> Self {
        Self::with_admission(controls, Arc::new(AdmissionGate::default()))
    }
    #[cfg(test)]
    pub(crate) fn with_output(output: O, controls: AppliedControls) -> Self {
        let coordinator = Self::new(controls);
        coordinator
            .begin_install()
            .unwrap()
            .install(output)
            .unwrap();
        coordinator
    }
    #[cfg(test)]
    pub(crate) fn work(&self, _buffer: AudioBuffer) -> PreparedAudioWork<O> {
        self.receipt().expect("test output is installed")
    }
    #[cfg(test)]
    fn apply_patch(&self, patch: ControlPatch) {
        let _ = self.apply_controls_transaction(|version| Some((patch, version.wrapping_add(1))));
    }
    #[cfg(test)]
    pub(crate) fn with_admission(controls: AppliedControls, admission: Arc<AdmissionGate>) -> Self {
        let controls = controls.normalized();
        let (control_tx, _) = watch::channel(controls.clone());
        let (status, _) = watch::channel(SessionStatus::Open);
        let current = Arc::new(Mutex::new(SessionStatus::Open));
        Self::with_admission_and_controls(
            controls,
            admission,
            control_tx,
            Arc::new(Lifecycle::new(status, current)),
        )
    }
    pub(crate) fn with_admission_and_controls(
        controls: AppliedControls,
        admission: Arc<AdmissionGate>,
        control_tx: watch::Sender<AppliedControls>,
        lifecycle: Arc<Lifecycle>,
    ) -> Self {
        let incarnation = OutputIncarnation::initial();
        let controls = controls.normalized();
        Self {
            shared: Arc::new(Shared {
                session: Mutex::new(State {
                    floor: StreamGeneration::initial(),
                    incarnation,
                    status: SessionStatus::Open,
                    output: None,
                    controls: controls.clone(),
                    desired_controls: controls,
                    control_version: 0,
                }),
                controls: control_tx,
                lifecycle,
                admission,
                install_state: Mutex::new(InstallState::Initial),
                next_reservation: Mutex::new(0),
            }),
        }
    }
    pub(crate) fn admission(&self) -> Arc<AdmissionGate> {
        Arc::clone(&self.shared.admission)
    }
    #[cfg(test)]
    pub(crate) fn incarnation(&self) -> OutputIncarnation {
        self.shared.session.lock().incarnation
    }
    #[cfg(test)]
    pub(crate) fn receipt(&self) -> Option<PreparedAudioWork<O>> {
        if self.shared.admission.is_closed() {
            return None;
        }
        let generation = self.shared.session.lock().floor;
        Some(self.receipt_stamped(self.stamp(), generation))
    }
    pub(crate) fn stamp(&self) -> ReceiptStamp {
        ReceiptStamp {
            incarnation: self.shared.session.lock().incarnation,
        }
    }
    pub(crate) fn receipt_stamped(
        &self,
        stamp: ReceiptStamp,
        generation: StreamGeneration,
    ) -> PreparedAudioWork<O> {
        PreparedAudioWork {
            shared: Arc::clone(&self.shared),
            stamp,
            generation,
        }
    }
    pub(crate) fn begin_install(&self) -> Result<InstallPermit<O>, InstallRejected> {
        let state = self.shared.session.lock();
        if matches!(
            state.status,
            SessionStatus::Closed | SessionStatus::Failed(_)
        ) {
            return Err(if matches!(state.status, SessionStatus::Closed) {
                InstallRejected::Closed
            } else {
                InstallRejected::Failed
            });
        }
        let incarnation = state.incarnation;
        let mut install_state = self.shared.install_state.lock();
        let reservation = {
            let mut next = self.shared.next_reservation.lock();
            *next = next.wrapping_add(1);
            *next
        };
        match *install_state {
            InstallState::Initial => *install_state = InstallState::Reserved(reservation),
            InstallState::Detached => *install_state = InstallState::ReservedDetached(reservation),
            InstallState::Reserved(_) | InstallState::ReservedDetached(_) => {
                return Err(InstallRejected::Reserved)
            }
            InstallState::Installed => return Err(InstallRejected::AlreadyInstalled),
        }
        Ok(InstallPermit {
            shared: Arc::clone(&self.shared),
            incarnation,
            reservation,
            consumed: false,
        })
    }
    pub(crate) fn release(&self) -> Result<(), OutputError> {
        let _diagnostics = DiagnosticScope::new();
        let mut state = self.shared.session.lock();
        if matches!(
            state.status,
            SessionStatus::Closed | SessionStatus::Failed(_)
        ) {
            return Err(OutputError::Failed("session is terminal".into()));
        }
        state.incarnation = state.incarnation.advance();
        let mut output = state.output.take();
        *self.shared.install_state.lock() = InstallState::Detached;
        drop(state);
        let result = output.as_mut().map_or(Ok(()), |output| output.clear());
        if let Err(error) = result {
            let mut state = self.shared.session.lock();
            self.lose_output_locked(&mut state, None, error.clone(), true);
            self.publish_status_locked(&mut state);
            return Err(error);
        }
        Ok(())
    }
    pub(crate) fn status(&self) -> SessionStatus {
        self.shared.session.lock().status.clone()
    }
    fn publish_status_locked(&self, state: &mut State<O>) {
        let _transition = self.shared.lifecycle.transition.lock();
        let mut current = self.shared.lifecycle.current.lock();
        // A terminal lifecycle transition wins over any worker that admitted
        // work before the transition and only publishes after a callback or
        // cleanup completes. Never let that stale worker reopen the watch.
        if matches!(*current, SessionStatus::Closed | SessionStatus::Failed(_)) {
            state.status = current.clone();
        }
        *current = state.status.clone();
        let _ = self
            .shared
            .lifecycle
            .status
            .send_replace(state.status.clone());
    }

    fn lose_output_locked(
        &self,
        state: &mut State<O>,
        output: Option<O>,
        error: OutputError,
        already_advanced: bool,
    ) {
        if matches!(
            state.status,
            SessionStatus::Closed | SessionStatus::Failed(_)
        ) {
            return;
        }
        if self.shared.admission.is_closed() {
            if let Some(mut output) = output {
                let _ = output.clear();
            }
            state.output = None;
            state.status = SessionStatus::Closed;
            self.publish_status_locked(state);
            return;
        }
        if !already_advanced {
            state.incarnation = state.incarnation.advance();
        }
        output_lost(state, output, error);
        self.publish_status_locked(state);
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
        let _diagnostics = DiagnosticScope::new();
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
        let mut controls = state.desired_controls.clone();
        update.apply(&mut controls);
        let before = state.controls.clone();
        state.desired_controls = controls.clone();
        let Some(mut output) = state.output.take() else {
            state.control_version = version;
            // Keep failed-output requests private until installation succeeds.
            // An open session without output still exposes requested controls.
            if matches!(state.status, SessionStatus::Open) {
                let _ = self
                    .shared
                    .controls
                    .send_replace(state.desired_controls.clone());
            }
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
                if self.shared.admission.is_closed() {
                    let _ = output.clear();
                    state.output = None;
                    state.status = SessionStatus::Closed;
                    self.publish_status_locked(&mut state);
                    *self.shared.install_state.lock() = InstallState::Detached;
                    return Err(OutputError::Failed("session is closed".into()));
                }
                state.controls = controls;
                state.control_version = version;
                let _ = self.shared.controls.send_replace(state.controls.clone());
                state.output = Some(output);
                Ok(())
            }
            Err(error) => {
                self.lose_output_locked(&mut state, None, error.clone(), false);
                *self.shared.install_state.lock() = InstallState::Detached;
                self.publish_status_locked(&mut state);
                drop(state);
                let _ = output.clear();
                Err(error)
            }
        }
    }

    pub(crate) fn close(&self) {
        let _diagnostics = DiagnosticScope::new();
        let mut state = self.shared.session.lock();
        if let Some(mut output) = state.output.take() {
            let _ = output.clear();
        }
        if matches!(
            state.status,
            SessionStatus::Open | SessionStatus::OutputLost(_)
        ) {
            state.status = SessionStatus::Closed;
            self.publish_status_locked(&mut state);
        }
        self.shared.admission.close();
    }
    pub(crate) fn boundary(&self, generation: StreamGeneration) -> Result<(), OutputError> {
        let _diagnostics = DiagnosticScope::new();
        let mut state = self.shared.session.lock();
        if matches!(
            state.status,
            SessionStatus::Closed | SessionStatus::Failed(_)
        ) || generation <= state.floor
        {
            return Ok(());
        }
        state.floor = generation;
        if let Some(mut output) = state.output.take() {
            if let Err(error) = output.clear() {
                self.lose_output_locked(&mut state, Some(output), error.clone(), false);
                *self.shared.install_state.lock() = InstallState::Detached;
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
        self.install_with_controls(output, 0, controls, None, || None)
    }

    pub(crate) fn install_with_controls(
        mut self,
        mut output: O,
        version: u64,
        controls: AppliedControls,
        patch: Option<ControlUpdate>,
        current_boundary: impl FnOnce() -> Option<StreamGeneration>,
    ) -> Result<(), InstallRejected> {
        let _diagnostics = DiagnosticScope::new();
        let mut state = self.shared.session.lock();
        if matches!(
            state.status,
            SessionStatus::Closed | SessionStatus::Failed(_)
        ) {
            return Err(if matches!(state.status, SessionStatus::Closed) {
                InstallRejected::Closed
            } else {
                InstallRejected::Failed
            });
        }
        if state.incarnation != self.incarnation {
            return Err(InstallRejected::Superseded);
        }
        let mut install_state = self.shared.install_state.lock();
        let reservation_matches = matches!(
            *install_state,
            InstallState::Reserved(id) | InstallState::ReservedDetached(id) if id == self.reservation
        );
        if !reservation_matches {
            return Err(InstallRejected::AlreadyInstalled);
        }
        let (controls, version) = if version <= state.control_version {
            (state.desired_controls.clone(), state.control_version)
        } else {
            let mut desired = state.desired_controls.clone();
            if let Some(patch) = patch {
                patch.apply(&mut desired);
                (desired.normalized(), version)
            } else {
                (controls.normalized(), version)
            }
        };
        let desired_controls = controls.clone();
        let applied = output
            .set_volume(controls.volume)
            .and_then(|_| output.set_mute(controls.muted))
            .and_then(|_| output.set_static_delay(controls.static_delay_ms));
        if let Err(error) = applied {
            Coordinator::from_shared(Arc::clone(&self.shared)).lose_output_locked(
                &mut state,
                Some(output),
                error,
                false,
            );
            *install_state = InstallState::Detached;
            return Err(InstallRejected::Failed);
        }
        let retained_boundary = current_boundary();
        if let Some(boundary) = retained_boundary {
            if boundary > state.floor {
                state.floor = boundary;
            }
            if let Err(error) = output.clear() {
                Coordinator::from_shared(Arc::clone(&self.shared)).lose_output_locked(
                    &mut state,
                    Some(output),
                    error,
                    false,
                );
                *install_state = InstallState::Detached;
                return Err(InstallRejected::Failed);
            }
        }
        let closed = self.shared.admission.closed.lock();
        if *closed {
            drop(closed);
            let _ = output.clear();
            state.status = SessionStatus::Closed;
            *install_state = InstallState::Detached;
            return Err(InstallRejected::Closed);
        }
        state.controls = controls.clone();
        state.desired_controls = desired_controls;
        state.control_version = version;
        let _ = self.shared.controls.send_replace(state.controls.clone());
        state.output = Some(output);
        state.status = SessionStatus::Open;
        Coordinator::from_shared(Arc::clone(&self.shared)).publish_status_locked(&mut state);
        drop(closed);
        *install_state = InstallState::Installed;
        self.consumed = true;
        Ok(())
    }
}

impl<O> Drop for InstallPermit<O> {
    fn drop(&mut self) {
        if self.consumed {
            return;
        }
        let mut install_state = self.shared.install_state.lock();
        match *install_state {
            InstallState::Reserved(id) if id == self.reservation => {
                *install_state = InstallState::Initial
            }
            InstallState::ReservedDetached(id) if id == self.reservation => {
                *install_state = InstallState::Detached
            }
            _ => {}
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
        let _diagnostics = DiagnosticScope::new();
        let coordinator = Coordinator::from_shared(Arc::clone(&self.shared));
        if let Err(error) = coordinator
            .boundary(generation)
            .map_err(CommitRejected::Output)
        {
            return (Err(error), coordinator.status());
        }
        let mut state = self.shared.session.lock();
        if matches!(
            state.status,
            SessionStatus::Closed | SessionStatus::Failed(_)
        ) {
            return (Err(CommitRejected::Closed), state.status.clone());
        }
        if matches!(state.status, SessionStatus::OutputLost(_)) {
            return (Err(CommitRejected::Obsolete), state.status.clone());
        }
        if self.stamp.incarnation != state.incarnation || self.generation != state.floor {
            return (Err(CommitRejected::Obsolete), state.status.clone());
        }
        if self.shared.admission.is_closed() {
            return (Err(CommitRejected::Closed), state.status.clone());
        }
        let Some(mut output) = state.output.take() else {
            return (Err(CommitRejected::Obsolete), state.status.clone());
        };
        // Keep the session lock across the short, synchronous callback. This
        // serializes commit with boundary/clear and prevents stale reinsertion.
        let result = output.commit(buffer);
        if self.shared.admission.is_closed() {
            let _ = output.clear();
            state.status = SessionStatus::Closed;
            return (Err(CommitRejected::Closed), state.status.clone());
        }
        let result = match result {
            Ok(()) => {
                state.output = Some(output);
                Ok(())
            }
            Err(
                error @ (OutputError::FormatMismatch { .. } | OutputError::InvalidBuffer { .. }),
            ) => {
                state.output = Some(output);
                Err(CommitRejected::Output(error))
            }
            Err(error) => {
                let rejection = CommitRejected::Output(error.clone());
                Coordinator::from_shared(Arc::clone(&self.shared)).lose_output_locked(
                    &mut state,
                    Some(output),
                    error,
                    false,
                );
                *self.shared.install_state.lock() = InstallState::Detached;
                Err(rejection)
            }
        };
        (result, state.status.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::diagnostics::{
        clear_diagnostic_observer, install_diagnostic_observer, PlaybackDiagnostic,
    };
    use crate::audio::synced_player::{PlaybackControl, SyncedPlayer};
    use crate::audio::{AudioBuffer, Codec};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;

    struct FakeOutput {
        format: AudioFormat,
        commits: usize,
        clears: Arc<AtomicUsize>,
        fail_commit: Option<OutputError>,
        fail_clear: Option<OutputError>,
        fail_set_volume: Option<OutputError>,
        controls: AppliedControls,
        commit_gate: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
        setter_gate: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
        clear_gate: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
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
            if let Some((ready, release)) = self.clear_gate.take() {
                ready.send(()).unwrap();
                release.recv().unwrap();
            }
            self.fail_clear.clone().map_or(Ok(()), Err)
        }
        fn set_volume(&mut self, volume: u8) -> Result<(), OutputError> {
            if let Some((ready, release)) = self.setter_gate.take() {
                ready.send(()).unwrap();
                release.recv().unwrap();
            }
            if let Some(error) = self.fail_set_volume.clone() {
                return Err(error);
            }
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
            fail_set_volume: None,
            controls: AppliedControls {
                volume: 0,
                muted: false,
                static_delay_ms: 0,
            },
            commit_gate: None,
            setter_gate: None,
            clear_gate: None,
        }
    }
    fn coordinator(clears: Arc<AtomicUsize>) -> Coordinator<FakeOutput> {
        Coordinator::with_output(fake(clears), controls())
    }

    struct ControlOutput(PlaybackControl);

    impl OutputAdapter for ControlOutput {
        fn commit(&mut self, buffer: AudioBuffer) -> Result<(), OutputError> {
            self.0.commit(buffer)
        }
        fn clear(&mut self) -> Result<(), OutputError> {
            self.0.clear()
        }
        fn set_volume(&mut self, volume: u8) -> Result<(), OutputError> {
            self.0.set_volume(volume)
        }
        fn set_mute(&mut self, muted: bool) -> Result<(), OutputError> {
            self.0.set_mute(muted)
        }
        fn set_static_delay(&mut self, delay_ms: u16) -> Result<(), OutputError> {
            self.0.set_static_delay(delay_ms)
        }
    }

    #[test]
    fn invalid_buffer_rejection_keeps_session_open_for_next_commit() {
        let control = SyncedPlayer::control_for_test(format(48_000, 2));
        let coordinator = Coordinator::with_output(ControlOutput(control), controls());
        let work = coordinator.work(buffer(48_000, 2));
        let mut partial = buffer(48_000, 2);
        partial.samples = Arc::from(vec![0; 3].into_boxed_slice());
        assert_eq!(
            work.commit(partial),
            Err(CommitRejected::Output(OutputError::InvalidBuffer {
                samples: 3,
                channels: 2,
            }))
        );
        assert_eq!(coordinator.status(), SessionStatus::Open);
        let next = coordinator.work(buffer(48_000, 2));
        assert_eq!(next.commit(buffer(48_000, 2)), Ok(()));
    }

    #[test]
    fn setter_failure_emits_clear_after_session_lock_is_released() {
        let control = SyncedPlayer::control_for_test(format(48_000, 2));
        let coordinator = Coordinator::with_output(ControlOutput(control.clone()), controls());
        control.set_error_for_test("callback failed");
        let shared = coordinator.shared();
        let observations = Arc::new(Mutex::new(Vec::new()));
        let observer_observations = Arc::clone(&observations);
        install_diagnostic_observer(move |diagnostic| {
            if let PlaybackDiagnostic::Cleared {
                buffers, queued_us, ..
            } = diagnostic
            {
                assert!(shared.session.try_lock().is_some());
                observer_observations.lock().push((queued_us, buffers));
            }
        });

        let result = coordinator.apply_controls_transaction(|version| {
            Some((
                ControlUpdate {
                    volume: Some(25),
                    ..ControlUpdate::default()
                },
                version + 1,
            ))
        });

        assert_eq!(result, Err(OutputError::Failed("callback failed".into())));
        let observations = observations.lock();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0], (0, 0));
        clear_diagnostic_observer();
    }

    #[test]
    fn commit_holds_session_lock_until_clear_then_new_generation_is_accepted() {
        let clears = Arc::new(AtomicUsize::new(0));
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let mut output = fake(Arc::clone(&clears));
        output.commit_gate = Some((ready_tx, release_rx));
        let c = Arc::new(Coordinator::with_output(output, controls()));
        let old = c.work(buffer(48_000, 2));
        let stale = c.work(buffer(48_000, 2));
        let commit = std::thread::spawn(move || old.commit(buffer(48_000, 2)));
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        let (clear_tx, clear_rx) = mpsc::channel();
        let clear_c = Arc::clone(&c);
        let clear = std::thread::spawn(move || {
            let result = clear_c.boundary(StreamGeneration::initial().advance());
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
            StreamGeneration::initial().advance()
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
        let _ = c.boundary(StreamGeneration::initial().advance());
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
        c.shared.session.lock().status =
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
        let c = Coordinator::<FakeOutput>::new(controls());
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
    fn closed_admission_during_blocked_setter_publishes_closed_and_joins_worker() {
        let c = Arc::new(coordinator(Arc::new(AtomicUsize::new(0))));
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        c.shared.session.lock().output.as_mut().unwrap().setter_gate = Some((ready_tx, release_rx));
        let status = c.shared.lifecycle.status.subscribe();
        let worker = {
            let c = Arc::clone(&c);
            std::thread::spawn(move || {
                c.apply_patch(ControlPatch {
                    volume: Some(30),
                    ..ControlPatch::default()
                })
            })
        };
        ready_rx.recv().unwrap();
        c.shared.admission.close();
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        assert_eq!(c.status(), SessionStatus::Closed);
        assert!(!matches!(*status.borrow(), SessionStatus::OutputLost(_)));
        assert!(status.has_changed().unwrap());
    }

    #[test]
    fn output_loss_retains_episode_error_across_failed_recovery_then_reopens() {
        let c = coordinator(Arc::new(AtomicUsize::new(0)));
        c.shared.session.lock().output.as_mut().unwrap().fail_commit =
            Some(OutputError::Failed("A".into()));
        assert!(c.work(buffer(48_000, 2)).commit(buffer(48_000, 2)).is_err());
        let first = c.status();
        let permit = c.begin_install().unwrap();
        let mut failed = installable();
        failed.fail_set_volume = Some(OutputError::Failed("B".into()));
        assert_eq!(permit.install(failed), Err(InstallRejected::Failed));
        assert_eq!(c.status(), first);
        c.begin_install().unwrap().install(installable()).unwrap();
        assert_eq!(c.status(), SessionStatus::Open);
        c.shared.session.lock().output.as_mut().unwrap().fail_commit =
            Some(OutputError::Failed("C".into()));
        assert_eq!(
            c.work(buffer(48_000, 2)).commit(buffer(48_000, 2)),
            Err(CommitRejected::Output(OutputError::Failed("C".into())))
        );
        assert_eq!(
            c.status(),
            SessionStatus::OutputLost(OutputError::Failed("C".into()))
        );
    }

    #[test]
    fn callback_failure_is_output_loss_and_cleanup_runs_once() {
        let clears = Arc::new(AtomicUsize::new(0));
        let c = coordinator(Arc::clone(&clears));
        c.shared.session.lock().output.as_mut().unwrap().fail_commit =
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
            SessionStatus::OutputLost(OutputError::Failed(_))
        ));
    }
    #[test]
    fn commit_failure_cleanup_race_reconciles_closed_lifecycle() {
        let clears = Arc::new(AtomicUsize::new(0));
        let (clear_entered_tx, clear_entered_rx) = mpsc::channel();
        let (clear_release_tx, clear_release_rx) = mpsc::channel();
        let mut output = fake(Arc::clone(&clears));
        output.fail_commit = Some(OutputError::Failed("commit".into()));
        output.clear_gate = Some((clear_entered_tx, clear_release_rx));
        let c = Arc::new(Coordinator::with_output(output, controls()));
        let mut status = c.shared.lifecycle.status.subscribe();
        let work = c.work(buffer(48_000, 2));
        let commit = {
            let c = Arc::clone(&c);
            std::thread::spawn(move || c.work(buffer(48_000, 2)).commit(buffer(48_000, 2)))
        };
        clear_entered_rx.recv().unwrap();
        c.shared.admission.close();
        clear_release_tx.send(()).unwrap();
        assert_eq!(
            commit.join().unwrap(),
            Err(CommitRejected::Output(OutputError::Failed("commit".into())))
        );
        c.close();
        assert_eq!(work.commit(buffer(48_000, 2)), Err(CommitRejected::Closed));
        assert_eq!(c.status(), SessionStatus::Closed);
        assert_eq!(*c.shared.lifecycle.current.lock(), SessionStatus::Closed);
        assert_eq!(*status.borrow(), SessionStatus::Closed);
        while status.has_changed().unwrap() {
            assert_ne!(
                *status.borrow_and_update(),
                SessionStatus::OutputLost(OutputError::Failed("commit".into()))
            );
        }
        assert_eq!(clears.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn failed_setter_cleanup_does_not_erase_concurrent_controls_or_install() {
        let clears = Arc::new(AtomicUsize::new(0));
        let (clear_entered_tx, clear_entered_rx) = mpsc::channel();
        let (clear_release_tx, clear_release_rx) = mpsc::channel();
        let output = fake(Arc::clone(&clears));
        let c = Arc::new(Coordinator::with_output(output, controls()));
        {
            let mut state = c.shared.session.lock();
            let output = state.output.as_mut().unwrap();
            output.fail_set_volume = Some(OutputError::Failed("setter A".into()));
            output.clear_gate = Some((clear_entered_tx, clear_release_rx));
        }
        let failed = {
            let c = Arc::clone(&c);
            std::thread::spawn(move || {
                c.apply_controls_transaction(|version| {
                    Some((
                        ControlUpdate {
                            volume: Some(11),
                            ..ControlUpdate::default()
                        },
                        version + 1,
                    ))
                })
            })
        };
        clear_entered_rx.recv().unwrap();
        c.apply_patch(ControlPatch {
            volume: Some(77),
            ..ControlPatch::default()
        });
        let replacement = fake(Arc::clone(&clears));
        let permit = c.begin_install().unwrap();
        assert_eq!(permit.install(replacement), Ok(()));
        clear_release_tx.send(()).unwrap();
        assert_eq!(
            failed.join().unwrap(),
            Err(OutputError::Failed("setter A".into()))
        );
        let state = c.shared.session.lock();
        assert_eq!(state.status, SessionStatus::Open);
        assert_eq!(state.controls.volume, 77);
        assert_eq!(state.desired_controls.volume, 77);
        assert!(matches!(
            *c.shared.install_state.lock(),
            InstallState::Installed
        ));
        assert_eq!(state.output.as_ref().unwrap().controls.volume, 77);
        drop(state);
        assert_eq!(c.work(buffer(48_000, 2)).commit(buffer(48_000, 2)), Ok(()));
        assert_eq!(clears.load(Ordering::SeqCst), 1);
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
        let state = c.shared.session.lock();
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

    fn lazy() -> Coordinator<FakeOutput> {
        Coordinator::new(controls())
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
            .install_with_controls(installable(), 1, stale, None, || None)
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
    fn old_permit_cannot_install_after_release() {
        let c = lazy();
        let permit = c.begin_install().unwrap();
        c.release().unwrap();
        assert_eq!(
            permit.install(installable()),
            Err(InstallRejected::Superseded)
        );
    }
    #[test]
    fn clear_spans_install_and_terminal_rejects_install() {
        let c = lazy();
        let _ = c.boundary(StreamGeneration::initial().advance());
        let permit = c.begin_install().unwrap();
        permit.install(installable()).unwrap();
        assert!(c.shared.session.lock().floor > StreamGeneration::initial());
        c.shared.session.lock().status = SessionStatus::Closed;
        assert!(matches!(c.begin_install(), Err(InstallRejected::Closed)));
    }
}
