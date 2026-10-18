use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Barrier;

use parking_lot::Mutex;
use tokio::sync::{mpsc, watch, Notify};

use super::player_session::{
    Admission, AppliedControls, CommitRejected, Coordinator, InstallRejected, LazyCoordinator,
    OutputAdapter, OutputError, SessionStatus,
};
use super::synced_player::PlaybackControl;
use super::AudioBuffer;
use crate::generation::StreamGeneration;
use crate::protocol::client::AudioChunk;
pub use crate::protocol::messages::StreamPlayerConfig;
use crate::role_core::ProducerError;

type LazyOutput = Box<dyn OutputAdapter>;
type CommitFn = Box<dyn FnOnce(StreamGeneration, AudioBuffer) -> Result<(), CommitRejected> + Send>;
#[cfg(test)]
type SetterGate = Arc<(
    std::sync::mpsc::Sender<()>,
    parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
)>;

/// A synchronous device-independent player output.
///
/// Calls are serialized, short, nonblocking, and must not reenter the session.
pub trait PlayerOutput: Clone + Send + Sync + 'static {
    /// Commits one decoded buffer.
    fn commit(&self, buffer: AudioBuffer) -> Result<(), OutputError>;
    /// Clears queued output.
    fn clear(&self) -> Result<(), OutputError>;
    /// Applies normalized volume.
    fn set_volume(&self, volume: u8) -> Result<(), OutputError>;
    /// Applies mute state.
    fn set_mute(&self, muted: bool) -> Result<(), OutputError>;
    /// Applies normalized static delay.
    fn set_static_delay(&self, delay_ms: u16) -> Result<(), OutputError>;
}

#[derive(Clone)]
struct OutputAdapterHandle<O: PlayerOutput>(O);
impl<O: PlayerOutput> OutputAdapter for OutputAdapterHandle<O> {
    fn commit(&mut self, buffer: AudioBuffer) -> Result<(), OutputError> {
        self.0.commit(buffer)
    }
    fn clear(&mut self) -> Result<(), OutputError> {
        self.0.clear()
    }
    fn set_volume(&mut self, value: u8) -> Result<(), OutputError> {
        self.0.set_volume(value)
    }
    fn set_mute(&mut self, value: bool) -> Result<(), OutputError> {
        self.0.set_mute(value)
    }
    fn set_static_delay(&mut self, value: u16) -> Result<(), OutputError> {
        self.0.set_static_delay(value)
    }
}

impl OutputAdapter for Box<dyn OutputAdapter> {
    fn commit(&mut self, buffer: AudioBuffer) -> Result<(), OutputError> {
        self.as_mut().commit(buffer)
    }
    fn clear(&mut self) -> Result<(), OutputError> {
        self.as_mut().clear()
    }
    fn set_volume(&mut self, value: u8) -> Result<(), OutputError> {
        self.as_mut().set_volume(value)
    }
    fn set_mute(&mut self, value: bool) -> Result<(), OutputError> {
        self.as_mut().set_mute(value)
    }
    fn set_static_delay(&mut self, value: u16) -> Result<(), OutputError> {
        self.as_mut().set_static_delay(value)
    }
}

enum RawEvent {
    Start(StreamPlayerConfig),
    Audio {
        generation: StreamGeneration,
        chunk: AudioChunk,
        receipt: super::player_session::ReceiptStamp,
    },
    Clear(StreamGeneration),
    End(StreamGeneration),
}

#[derive(Clone, Copy, Default)]
struct CommandPatch {
    volume: Option<u8>,
    muted: Option<bool>,
    static_delay_ms: Option<u16>,
}
impl CommandPatch {
    fn apply_to(self, mut controls: AppliedControls) -> AppliedControls {
        self.apply(&mut controls);
        controls
    }
    fn apply(self, controls: &mut AppliedControls) {
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
#[derive(Clone)]
struct UrgentState {
    version: u64,
    boundary: Option<StreamGeneration>,
    commands: CommandPatch,
    closed: bool,
}

struct SessionFence {
    status: watch::Sender<SessionStatus>,
    status_value: Mutex<SessionStatus>,
    controls: watch::Sender<AppliedControls>,
    urgent: watch::Sender<UrgentState>,
    boundary: Mutex<StreamGeneration>,
    admission: Arc<super::player_session::AdmissionGate>,
    #[cfg(test)]
    driver_start_gate: Mutex<Option<Arc<Notify>>>,
    #[cfg(test)]
    commit_admission_gate: Mutex<Option<(Arc<Barrier>, Arc<Barrier>)>>,
    cancel: Arc<Notify>,
    cancelled: AtomicBool,
}
impl SessionFence {
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.cancel.notify_waiters();
    }
    fn fail(&self, error: OutputError) {
        {
            let mut status = self.status_value.lock();
            if !matches!(*status, SessionStatus::Open) {
                return;
            }
            *status = SessionStatus::Failed(error.clone());
            let _ = self.status.send_replace(SessionStatus::Failed(error));
            self.admission.close();
        }
        self.cancel();
    }
    fn close(&self) {
        self.cancel.notify_waiters();
        self.admission.close();
        {
            let mut status = self.status_value.lock();
            if !matches!(*status, SessionStatus::Open) {
                return;
            }
            *status = SessionStatus::Closed;
            let _ = self.status.send_replace(SessionStatus::Closed);
            self.urgent.send_modify(|urgent| urgent.closed = true);
        }
        self.cancel();
    }
}

/// Opaque audio work bound to the session that issued it.
///
/// Preparation is application-owned; `commit` is synchronous and never waits for decoding.
pub struct AudioWork {
    commit: Option<CommitFn>,
    generation: StreamGeneration,
    chunk: AudioChunk,
    fence: Arc<SessionFence>,
}

impl AudioWork {
    fn new<W: Send + 'static>(
        work: W,
        generation: StreamGeneration,
        chunk: AudioChunk,
        fence: Arc<SessionFence>,
        commit: impl FnOnce(W, StreamGeneration, AudioBuffer) -> Result<(), CommitRejected>
            + Send
            + 'static,
    ) -> Self {
        let commit_fence = Arc::clone(&fence);
        Self {
            commit: Some(Box::new(move |generation, buffer| {
                #[cfg(test)]
                if let Some((reached, release)) = commit_fence.commit_admission_gate.lock().take() {
                    reached.wait();
                    release.wait();
                }
                let result = commit(work, generation, buffer);
                if let Err(CommitRejected::Output(error)) = &result {
                    if matches!(error, OutputError::Failed(_)) {
                        commit_fence.fail(error.clone());
                    }
                }
                result
            })),
            generation,
            chunk,
            fence,
        }
    }

    /// Returns the server-time timestamp in microseconds.
    pub fn timestamp(&self) -> i64 {
        self.chunk.timestamp
    }
    /// Returns the encoded payload.
    pub fn data(&self) -> &[u8] {
        &self.chunk.data
    }
    /// Returns the encoded chunk, including its payload.
    pub fn chunk(&self) -> &AudioChunk {
        &self.chunk
    }

    /// Commits prepared audio.
    ///
    /// For the player, `FormatMismatch` is a per-item `CommitRejected::Output` and
    /// leaves the session open. Callback failures are retained as terminal status
    /// and reject later work. Clear/end/release, missing installed output, and
    /// terminal closure make older work `Obsolete` or `Closed`. Each work item is
    /// one-shot; dropping it has no session effect.
    pub fn commit(mut self, prepared: AudioBuffer) -> Result<(), CommitRejected> {
        if !matches!(*self.fence.status_value.lock(), SessionStatus::Open) {
            return Err(CommitRejected::Closed);
        }
        let receipt_generation = self.generation;
        let boundary = self.fence.urgent.borrow().boundary;
        if boundary.is_some_and(|boundary| boundary > receipt_generation) {
            return Err(CommitRejected::Obsolete);
        }
        let generation = boundary.unwrap_or(receipt_generation);
        self.commit.take().expect("audio work commits once")(generation, prepared)
    }
}

impl fmt::Debug for AudioWork {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AudioWork")
            .field("timestamp", &self.timestamp())
            .field("data_len", &self.data().len())
            .finish()
    }
}

/// Ordered player event.
pub enum PlayerEvent {
    /// Announces a format/configuration change.
    Start(StreamPlayerConfig),
    /// Provides encoded audio work for application preparation.
    Audio(AudioWork),
    /// Clears work issued before this boundary.
    Clear,
    /// Ends the current stream while allowing a later start.
    End,
}

impl fmt::Debug for PlayerEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start(config) => formatter.debug_tuple("Start").field(config).finish(),
            Self::Audio(work) => formatter.debug_tuple("Audio").field(work).finish(),
            Self::Clear => formatter.write_str("Clear"),
            Self::End => formatter.write_str("End"),
        }
    }
}

pub(crate) struct PlayerProducer {
    tx: mpsc::UnboundedSender<RawEvent>,
    fence: Arc<SessionFence>,
    generation: Arc<Mutex<StreamGeneration>>,
}
impl PlayerProducer {
    pub(crate) fn start(&self, config: StreamPlayerConfig) -> Result<(), ProducerError> {
        self.tx
            .send(RawEvent::Start(config))
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    #[cfg(test)]
    pub(crate) fn set_generation_max_for_test(&self) {
        *self.generation.lock() = StreamGeneration::max_for_test();
    }
    #[cfg(test)]
    pub(crate) fn set_controls(&self, controls: AppliedControls) {
        self.fence.urgent.send_modify(|urgent| {
            urgent.version = urgent.version.wrapping_add(1);
            urgent.commands = CommandPatch {
                volume: Some(controls.volume),
                muted: Some(controls.muted),
                static_delay_ms: Some(controls.static_delay_ms),
            };
        });
    }
    pub(crate) fn audio(&self, chunk: AudioChunk) -> Result<(), ProducerError> {
        let generation = *self.generation.lock();
        let receipt = match self.fence.admission.admit() {
            Admission::Open(receipt) => receipt,
            Admission::Detached => return Ok(()),
            Admission::Closed => return Err(ProducerError::ReceiverClosed),
        };
        self.tx
            .send(RawEvent::Audio {
                generation,
                chunk,
                receipt,
            })
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    pub(crate) fn set_volume(&self, value: u8) {
        self.fence.urgent.send_modify(|urgent| {
            urgent.version = urgent.version.wrapping_add(1);
            urgent.commands.volume = Some(value);
        });
    }
    pub(crate) fn set_mute(&self, value: bool) {
        self.fence.urgent.send_modify(|urgent| {
            urgent.version = urgent.version.wrapping_add(1);
            urgent.commands.muted = Some(value);
        });
    }
    pub(crate) fn set_static_delay(&self, value: u16) {
        self.fence.urgent.send_modify(|urgent| {
            urgent.version = urgent.version.wrapping_add(1);
            urgent.commands.static_delay_ms = Some(value);
        });
    }
    pub(crate) fn clear(&self) -> Result<(), ProducerError> {
        let generation = {
            let mut current = self.generation.lock();
            let next = match current.checked_advance() {
                Some(next) => next,
                None => {
                    self.fence.close();
                    return Err(ProducerError::GenerationExhausted);
                }
            };
            *current = next;
            next
        };
        *self.fence.boundary.lock() = generation;
        self.fence.urgent.send_modify(|urgent| {
            urgent.boundary = Some(generation);
        });
        self.tx
            .send(RawEvent::Clear(generation))
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    pub(crate) fn end(&self) -> Result<(), ProducerError> {
        let generation = {
            let mut current = self.generation.lock();
            let next = match current.checked_advance() {
                Some(next) => next,
                None => {
                    self.fence.close();
                    return Err(ProducerError::GenerationExhausted);
                }
            };
            *current = next;
            next
        };
        *self.fence.boundary.lock() = generation;
        self.fence.urgent.send_modify(|urgent| {
            urgent.boundary = Some(generation);
        });
        self.tx
            .send(RawEvent::End(generation))
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    pub(crate) fn close(&self) {
        self.fence.close();
    }
}
impl Drop for PlayerProducer {
    fn drop(&mut self) {
        self.fence.close();
    }
}

/// Receiver for one player role stream.
pub struct PlayerReceiver {
    rx: mpsc::UnboundedReceiver<RawEvent>,
    fence: Arc<SessionFence>,
}
impl PlayerReceiver {
    pub(crate) fn channel(initial: AppliedControls) -> (Self, PlayerProducer) {
        let (tx, rx) = mpsc::unbounded_channel();
        let controls = initial.normalized();
        let (status, _) = watch::channel(SessionStatus::Open);
        let (control_tx, _) = watch::channel(controls.clone());
        let (urgent, _) = watch::channel(UrgentState {
            version: 0,
            boundary: None,
            commands: CommandPatch::default(),
            closed: false,
        });
        let admission = Arc::new(super::player_session::AdmissionGate::new(
            crate::generation::OutputIncarnation::initial(),
        ));
        let fence = Arc::new(SessionFence {
            status,
            status_value: Mutex::new(SessionStatus::Open),
            controls: control_tx,
            urgent,
            boundary: Mutex::new(StreamGeneration::initial()),
            admission: Arc::clone(&admission),
            #[cfg(test)]
            driver_start_gate: Mutex::new(None),
            #[cfg(test)]
            commit_admission_gate: Mutex::new(None),
            cancel: Arc::new(Notify::new()),
            cancelled: AtomicBool::new(false),
        });
        let generation = Arc::new(Mutex::new(StreamGeneration::initial()));
        let producer = PlayerProducer {
            tx,
            fence: Arc::clone(&fence),
            generation: Arc::clone(&generation),
        };
        let receiver = Self { rx, fence };
        (receiver, producer)
    }
    /// Attaches an output and starts the urgent-control driver on the current Tokio runtime.
    ///
    /// `initial_controls` is normalized (`volume` ≤ 100 and `static_delay_ms` ≤ 5000),
    /// then overlaid with any pre-attachment command for each named field; the
    /// resulting values are applied before the session is returned. Output callbacks
    /// are synchronous, short, nonblocking, serialized, and must not reenter this session.
    ///
    /// ```rust,no_run
    /// use sendspin::audio::{AppliedControls, PlayerOutput, PlayerReceiver, PlayerSession};
    ///
    /// fn attach_player<O: PlayerOutput>(
    ///     receiver: PlayerReceiver,
    ///     output: O,
    /// ) -> PlayerSession<O> {
    ///     receiver.attach(
    ///         output,
    ///         AppliedControls {
    ///             volume: 100,
    ///             muted: false,
    ///             static_delay_ms: 0,
    ///         },
    ///     )
    /// }
    /// ```
    pub fn attach<O: PlayerOutput>(
        self,
        output: O,
        initial_controls: AppliedControls,
    ) -> PlayerSession<O> {
        let mut controls = initial_controls.normalized();
        self.fence.urgent.borrow().commands.apply(&mut controls);
        let coordinator = Coordinator::with_controls_sender(
            OutputAdapterHandle(output.clone()),
            controls.clone(),
            self.fence.controls.clone(),
            Arc::clone(&self.fence.admission),
        );
        let shared = coordinator.shared();
        let initial = self.fence.urgent.borrow().clone();
        if let Err(error) =
            Coordinator::from_shared(Arc::clone(&shared)).apply_initial_controls(controls.clone())
        {
            self.fence.fail(error);
        } else {
            let _ = self.fence.controls.send_replace(controls);
        }
        let coordinator = Coordinator::from_shared(Arc::clone(&shared));
        if let Some(boundary) = initial.boundary {
            if let Err(error) = coordinator.boundary(boundary) {
                self.fence.fail(error);
            }
        }
        if initial.closed || !matches!(*self.fence.status_value.lock(), SessionStatus::Open) {
            coordinator.close();
        }
        let driver = Driver::start(Arc::clone(&self.fence), shared);
        PlayerSession {
            rx: self.rx,
            coordinator,
            fence: self.fence,
            driver: Some(driver),
        }
    }
    /// Creates a lazy session whose output can be installed later on the current Tokio runtime.
    ///
    /// `initial_controls` is normalized and overlaid with pre-attachment commands by
    /// named field. Controls are published before an output exists and replayed before
    /// a successful installation is admitted; an installation that fails while applying
    /// those controls or clearing a retained boundary is terminal; reservation errors
    /// remain retryable.
    pub fn attach_lazy(self, initial_controls: AppliedControls) -> LazyPlayerSession {
        self.attach_lazy_impl(initial_controls)
    }

    #[cfg(test)]
    fn attach_with_driver_gate<O: PlayerOutput>(
        self,
        output: O,
        initial_controls: AppliedControls,
        gate: Arc<Notify>,
    ) -> PlayerSession<O> {
        *self.fence.driver_start_gate.lock() = Some(gate);
        self.attach(output, initial_controls)
    }

    #[cfg(test)]
    fn attach_lazy_for_test<O: PlayerOutput>(
        self,
        initial_controls: AppliedControls,
    ) -> LazyPlayerSession {
        let _ = std::marker::PhantomData::<O>;
        self.attach_lazy_impl(initial_controls)
    }

    #[cfg(test)]
    fn attach_lazy_with_driver_gate(
        self,
        initial_controls: AppliedControls,
        gate: Arc<Notify>,
    ) -> LazyPlayerSession {
        *self.fence.driver_start_gate.lock() = Some(gate);
        self.attach_lazy_impl(initial_controls)
    }

    fn attach_lazy_impl(self, initial_controls: AppliedControls) -> LazyPlayerSession {
        let mut controls = initial_controls.normalized();
        self.fence.urgent.borrow().commands.apply(&mut controls);
        let fence = Arc::clone(&self.fence);
        let coordinator = LazyCoordinator::<LazyOutput>::with_admission_and_controls(
            controls.clone(),
            Arc::clone(&fence.admission),
            fence.controls.clone(),
        );
        let shared = coordinator.shared();
        let initial = fence.urgent.borrow().clone();
        let coordinator = super::player_session::LazyCoordinator::from_shared(Arc::clone(&shared));
        let _ = fence.controls.send_replace(controls.clone());
        if let Some(boundary) = initial.boundary {
            if let Err(error) = coordinator.boundary(boundary) {
                fence.fail(error);
            }
        }
        if initial.closed || !matches!(*fence.status_value.lock(), SessionStatus::Open) {
            coordinator.close();
        }
        let driver = Driver::start_lazy(Arc::clone(&fence), shared);
        LazyPlayerSession {
            receiver: self,
            coordinator,
            driver: Some(driver),
        }
    }
}

struct Driver {
    task: Option<tokio::task::JoinHandle<()>>,
}
impl Driver {
    fn start<O: OutputAdapter + 'static>(
        fence: Arc<SessionFence>,
        shared: Arc<super::player_session::Shared<O>>,
    ) -> Self {
        let mut urgent = fence.urgent.subscribe();
        let mut status = fence.status.subscribe();
        let cancel = Arc::clone(&fence.cancel);
        let task = tokio::spawn(async move {
            loop {
                if fence.cancelled.load(Ordering::Acquire) {
                    break;
                }
                let snapshot = urgent.borrow().clone();
                #[cfg(test)]
                let driver_gate = { fence.driver_start_gate.lock().take() };
                #[cfg(test)]
                if let Some(gate) = driver_gate {
                    gate.notified().await;
                }
                let coordinator = Coordinator::from_shared(Arc::clone(&shared));
                if snapshot.closed {
                    coordinator.close();
                    break;
                }
                if let Some(boundary) = snapshot.boundary {
                    if let Err(error) = coordinator.boundary(boundary) {
                        fence.fail(error);
                        break;
                    }
                }
                let mut controls = coordinator.controls();
                snapshot.commands.apply(&mut controls);
                if let Err(error) = coordinator.apply_controls_versioned(controls, snapshot.version)
                {
                    fence.fail(error);
                    break;
                }
                tokio::select! {
                    _ = cancel.notified() => break,
                    changed = status.changed() => {
                        if changed.is_err() || !matches!(*status.borrow(), SessionStatus::Open) { break; }
                    }
                    changed = urgent.changed() => {
                        if changed.is_err() { break; }
                    }
                }
            }
            Coordinator::from_shared(Arc::clone(&shared)).close();
        });
        Self { task: Some(task) }
    }
    async fn join(mut self) {
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
    fn start_lazy<O: OutputAdapter + 'static>(
        fence: Arc<SessionFence>,
        coordinator: Arc<super::player_session::LazyShared<O>>,
    ) -> Self {
        let mut urgent = fence.urgent.subscribe();
        let mut status = fence.status.subscribe();
        let cancel = Arc::clone(&fence.cancel);
        let task = tokio::spawn(async move {
            loop {
                if fence.cancelled.load(Ordering::Acquire) {
                    break;
                }
                let snapshot = urgent.borrow().clone();
                #[cfg(test)]
                let driver_gate = { fence.driver_start_gate.lock().take() };
                #[cfg(test)]
                if let Some(gate) = driver_gate {
                    gate.notified().await;
                }
                let coordinator =
                    super::player_session::LazyCoordinator::from_shared(Arc::clone(&coordinator));
                if snapshot.closed {
                    coordinator.close();
                    break;
                }
                if let Some(boundary) = snapshot.boundary {
                    if let Err(error) = coordinator.boundary(boundary) {
                        fence.fail(error);
                        break;
                    }
                }
                let mut controls = coordinator.controls();
                snapshot.commands.apply(&mut controls);
                if let Err(error) = coordinator.apply_controls_versioned(controls, snapshot.version)
                {
                    fence.fail(error);
                    break;
                }
                tokio::select! {
                    _ = cancel.notified() => break,
                    changed = status.changed() => {
                        if changed.is_err() || !matches!(*status.borrow(), SessionStatus::Open) { break; }
                    }
                    changed = urgent.changed() => {
                        if changed.is_err() { break; }
                    }
                }
            }
            super::player_session::LazyCoordinator::from_shared(Arc::clone(&coordinator)).close();
        });
        Self { task: Some(task) }
    }
}
impl Drop for Driver {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

/// Session receiving ordered player events for an attached output.
pub struct PlayerSession<O: PlayerOutput> {
    rx: mpsc::UnboundedReceiver<RawEvent>,
    coordinator: Coordinator<OutputAdapterHandle<O>>,
    fence: Arc<SessionFence>,
    driver: Option<Driver>,
}
impl<O: PlayerOutput> PlayerSession<O> {
    /// Receives the next ordered event; returns `None` after closure or failure.
    ///
    /// Waiting is cancellation-safe: cancelling this future does not consume the next
    /// event. After a terminal status, subsequent calls return `None` without draining
    /// queued events.
    pub async fn recv(&mut self) -> Option<PlayerEvent> {
        loop {
            let mut status = self.fence.status.subscribe();
            if !matches!(*status.borrow(), SessionStatus::Open) {
                return None;
            }
            let urgent = self.fence.urgent.borrow().clone();
            if let Some(boundary) = urgent.boundary {
                if let Err(error) = self.coordinator.boundary(boundary) {
                    self.fence.fail(error);
                    return None;
                }
            }
            let event = tokio::select! {
                biased;
                changed = status.changed() => {
                    if changed.is_err() || !matches!(*status.borrow(), SessionStatus::Open) { return None; }
                    return None;
                },
                event = self.rx.recv() => event?,
            };
            return Some(match event {
                RawEvent::Start(config) => PlayerEvent::Start(config),
                RawEvent::Audio {
                    generation,
                    chunk,
                    receipt,
                } => {
                    if generation < self.coordinator.current_generation() {
                        continue;
                    }
                    PlayerEvent::Audio(AudioWork::new(
                        self.coordinator.work_receipt(generation, receipt),
                        generation,
                        chunk,
                        Arc::clone(&self.fence),
                        |work, generation, buffer| {
                            let (result, status) = work.commit_at_with_status(generation, buffer);
                            if let SessionStatus::Failed(error) = status {
                                return Err(CommitRejected::Output(error));
                            }
                            result
                        },
                    ))
                }
                RawEvent::Clear(generation) => {
                    if let Err(error) = self.coordinator.boundary(generation) {
                        self.fence.fail(error);
                        return None;
                    }
                    PlayerEvent::Clear
                }
                RawEvent::End(generation) => {
                    if let Err(error) = self.coordinator.boundary(generation) {
                        self.fence.fail(error);
                        return None;
                    }
                    PlayerEvent::End
                }
            });
        }
    }
    /// Subscribes to retained lifecycle status without polling `recv`.
    pub fn status(&self) -> watch::Receiver<SessionStatus> {
        self.fence.status.subscribe()
    }
    /// Subscribes to the latest normalized controls.
    pub fn controls(&self) -> watch::Receiver<AppliedControls> {
        self.fence.controls.subscribe()
    }
    /// Closes the session, then joins its control driver.
    pub async fn shutdown(mut self) {
        self.fence.close();
        self.rx.close();
        while self.rx.recv().await.is_some() {}
        if let Some(driver) = self.driver.take() {
            driver.join().await;
        }
    }
}
impl<O: PlayerOutput> Drop for PlayerSession<O> {
    fn drop(&mut self) {
        self.fence.close();
        self.rx.close();
        self.driver.take();
        self.coordinator.close();
    }
}

/// One-shot reservation for installing lazy output.
pub struct OutputInstallPermit {
    permit: Option<super::player_session::InstallPermit<LazyOutput>>,
    shared: Arc<super::player_session::LazyShared<LazyOutput>>,
    fence: Arc<SessionFence>,
}
impl OutputInstallPermit {
    /// Installs the reserved output and replays current normalized controls.
    ///
    /// The permit is one-shot. Dropping it abandons the reservation so a later
    /// reservation can be made. Only an installation that fails while applying
    /// controls or clearing a retained boundary is terminal; reservation, closure,
    /// and already-installed errors are not.
    pub fn install(self, output: PlaybackControl) -> Result<(), InstallRejected> {
        self.install_adapter(Box::new(OutputAdapterHandle(output)))
    }

    #[cfg(test)]
    fn install_for_test<O: PlayerOutput>(self, output: O) -> Result<(), InstallRejected> {
        self.install_adapter(Box::new(OutputAdapterHandle(output)))
    }

    fn install_adapter(mut self, output: LazyOutput) -> Result<(), InstallRejected> {
        if !matches!(*self.fence.status_value.lock(), SessionStatus::Open) {
            return Err(match *self.fence.status_value.lock() {
                SessionStatus::Closed => InstallRejected::Closed,
                SessionStatus::Failed(_) => InstallRejected::Failed,
                SessionStatus::Open => unreachable!(),
            });
        }
        let urgent = self.fence.urgent.borrow().clone();
        let current_controls = self.fence.controls.borrow().clone();
        let controls = urgent.commands.apply_to(current_controls);
        let result = self
            .permit
            .take()
            .ok_or(InstallRejected::Reserved)?
            .install_with_controls(output, urgent.version, controls, || {
                self.fence.urgent.borrow().boundary
            });
        if matches!(result, Err(InstallRejected::Failed)) {
            if let SessionStatus::Failed(error) =
                LazyCoordinator::from_shared(Arc::clone(&self.shared)).status()
            {
                self.fence.fail(error);
            }
        }
        result
    }
}

/// Session that accepts player events before and after output installation.
pub struct LazyPlayerSession {
    receiver: PlayerReceiver,
    coordinator: LazyCoordinator<LazyOutput>,
    driver: Option<Driver>,
}
impl LazyPlayerSession {
    /// Reserves the single output-install slot.
    ///
    /// Only one reservation may exist at a time. Clear and end advance the invalidation
    /// floor but do not consume the slot; release fences old work and permits by output
    /// incarnation. Terminal closure or failure rejects installation permanently.
    pub fn begin_output_install(&self) -> Result<OutputInstallPermit, InstallRejected> {
        if !matches!(
            *self.receiver.fence.status_value.lock(),
            SessionStatus::Open
        ) {
            return Err(match *self.receiver.fence.status_value.lock() {
                SessionStatus::Closed => InstallRejected::Closed,
                SessionStatus::Failed(_) => InstallRejected::Failed,
                SessionStatus::Open => unreachable!(),
            });
        }
        let coordinator = &self.coordinator;
        let boundary = *self.receiver.fence.boundary.lock();
        if let Err(error) = coordinator.boundary(boundary) {
            self.receiver.fence.fail(error);
        }
        let permit = coordinator.begin_install().map_err(|error| match error {
            InstallRejected::Closed => InstallRejected::Closed,
            InstallRejected::Failed => InstallRejected::Failed,
            InstallRejected::AlreadyInstalled => InstallRejected::AlreadyInstalled,
            InstallRejected::Reserved => InstallRejected::Reserved,
            InstallRejected::Superseded => InstallRejected::Superseded,
        })?;
        Ok(OutputInstallPermit {
            shared: Arc::clone(&coordinator.shared()),
            permit: Some(permit),
            fence: Arc::clone(&self.receiver.fence),
        })
    }

    /// Installs output using a newly reserved slot.
    pub fn install(&mut self, output: PlaybackControl) -> Result<(), InstallRejected> {
        self.install_adapter(Box::new(OutputAdapterHandle(output)))
    }

    #[cfg(test)]
    fn install_for_test<O: PlayerOutput>(&mut self, output: O) -> Result<(), InstallRejected> {
        self.install_adapter(Box::new(OutputAdapterHandle(output)))
    }

    fn install_adapter(&mut self, output: LazyOutput) -> Result<(), InstallRejected> {
        if !matches!(
            *self.receiver.fence.status_value.lock(),
            SessionStatus::Open
        ) {
            return Err(match *self.receiver.fence.status_value.lock() {
                SessionStatus::Closed => InstallRejected::Closed,
                SessionStatus::Failed(_) => InstallRejected::Failed,
                SessionStatus::Open => unreachable!(),
            });
        }
        self.begin_output_install()?.install_adapter(output)
    }
    /// Releases the installed output while keeping the session open for reinstall.
    ///
    /// Receipt-time admission is detached first, so queued and outstanding old work is
    /// rejected or skipped. The output is cleared before the result is returned; a clear
    /// failure is the retained terminal failure and prevents reinstall.
    ///
    /// The output-owning worker must acknowledge dropping its device before the application
    /// advertises an external source. The session remains open and retains controls and the
    /// current boundary for a later installation; the worker retains the latest start/configuration.
    ///
    /// ```rust,no_run
    /// use sendspin::audio::{LazyPlayerSession, PlaybackControl};
    /// use sendspin::{protocol::messages::PlayerState, WsSender};
    ///
    /// async fn handoff(
    ///     session: &mut LazyPlayerSession,
    ///     sender: &WsSender,
    ///     worker: &mut OutputWorker,
    /// ) -> Result<(), Box<dyn std::error::Error>> {
    ///     session.release_output().map_err(|error| format!("{error:?}"))?;
    ///     worker.drop_device_and_ack().await;
    ///     sender.enter_external_source().await?;
    ///
    ///     // The caller supplies the latest protocol state and reads any OS/device state
    ///     // through its platform APIs before constructing this value.
    ///     let initial_state = worker.read_player_state().await?;
    ///     let controls = session.controls().borrow().clone();
    ///     sender.exit_external_source(Some(with_session_controls(initial_state, controls))).await?;
    ///     let permit = session
    ///         .begin_output_install()
    ///         .map_err(|error| format!("{error:?}"))?;
    ///     let output = worker.create_output().await?;
    ///     permit.install(output).map_err(|error| format!("{error:?}"))?;
    ///     Ok(())
    /// }
    ///
    /// fn with_session_controls(
    ///     initial_state: PlayerState,
    ///     controls: sendspin::audio::AppliedControls,
    /// ) -> PlayerState {
    ///     PlayerState {
    ///         volume: Some(controls.volume),
    ///         muted: Some(controls.muted),
    ///         static_delay_ms: Some(controls.static_delay_ms),
    ///         ..initial_state
    ///     }
    /// }
    ///
    /// struct OutputWorker;
    /// impl OutputWorker {
    ///     async fn drop_device_and_ack(&mut self) {}
    ///     async fn read_player_state(&mut self) -> Result<PlayerState, Box<dyn std::error::Error>> {
    ///         Ok(PlayerState::default())
    ///     }
    ///     async fn create_output(&mut self) -> Result<PlaybackControl, Box<dyn std::error::Error>> {
    ///         todo!()
    ///     }
    /// }
    ///
    /// ```
    pub fn release_output(&mut self) -> Result<(), OutputError> {
        let coordinator = &self.coordinator;
        match coordinator.release() {
            Ok(()) => Ok(()),
            Err(error) => {
                self.receiver.fence.fail(error.clone());
                Err(error)
            }
        }
    }
    /// Receives the next event, skipping audio from detached or superseded output
    /// incarnations, including audio received before reinstallation.
    ///
    /// Waiting is cancellation-safe and does not lose the next event. Terminal status
    /// returns `None` immediately without draining queued events; subsequent calls also
    /// return `None`.
    pub async fn recv(&mut self) -> Option<PlayerEvent> {
        loop {
            let receiver = &mut self.receiver;
            let mut status = receiver.fence.status.subscribe();
            if !matches!(*status.borrow(), SessionStatus::Open) {
                return None;
            }
            let coordinator = &self.coordinator;
            let boundary = receiver.fence.urgent.borrow().boundary;
            if let Some(boundary) = boundary {
                if let Err(error) = coordinator.boundary(boundary) {
                    receiver.fence.fail(error);
                    return None;
                }
            }
            let event = tokio::select! {
                biased;
                changed = status.changed() => {
                    if changed.is_err() || !matches!(*status.borrow(), SessionStatus::Open) { return None; }
                    return None;
                },
                event = receiver.rx.recv() => event?,
            };
            match event {
                RawEvent::Audio {
                    generation,
                    chunk,
                    receipt,
                } => {
                    let coordinator = &self.coordinator;
                    let admitted = match coordinator.admission().admit() {
                        Admission::Open(current) => current == receipt,
                        Admission::Detached | Admission::Closed => false,
                    };
                    if !admitted || generation < coordinator.current_generation() {
                        continue;
                    }
                    let work = coordinator.receipt_stamped(receipt, generation);
                    return Some(PlayerEvent::Audio(AudioWork::new(
                        work,
                        generation,
                        chunk,
                        Arc::clone(&receiver.fence),
                        |work, generation, buffer| {
                            let (result, status) = work.commit_at_with_status(generation, buffer);
                            if let SessionStatus::Failed(error) = status {
                                return Err(CommitRejected::Output(error));
                            }
                            result
                        },
                    )));
                }
                RawEvent::Start(config) => return Some(PlayerEvent::Start(config)),
                RawEvent::Clear(generation) => {
                    let coordinator = &self.coordinator;
                    if let Err(error) = coordinator.boundary(generation) {
                        receiver.fence.fail(error);
                        return None;
                    }
                    return Some(PlayerEvent::Clear);
                }
                RawEvent::End(generation) => {
                    let coordinator = &self.coordinator;
                    if let Err(error) = coordinator.boundary(generation) {
                        receiver.fence.fail(error);
                        return None;
                    }
                    return Some(PlayerEvent::End);
                }
            }
        }
    }
    /// Subscribes to retained lifecycle status without polling `recv`.
    pub fn status(&self) -> watch::Receiver<SessionStatus> {
        self.receiver.fence.status.subscribe()
    }
    /// Subscribes to the latest normalized controls.
    pub fn controls(&self) -> watch::Receiver<AppliedControls> {
        self.receiver.fence.controls.subscribe()
    }
    /// Closes the lazy session and joins its control driver.
    pub async fn shutdown(mut self) {
        self.receiver.fence.close();
        self.receiver.rx.close();
        self.coordinator.close();
        if let Some(driver) = self.driver.take() {
            driver.join().await;
        }
    }
}
impl Drop for LazyPlayerSession {
    fn drop(&mut self) {
        self.receiver.fence.close();
        self.receiver.rx.close();
        self.driver.take();
        self.coordinator.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioFormat, Codec};
    use futures_util::FutureExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::time::timeout;

    #[derive(Clone)]
    struct FakeOutput {
        commits: Arc<AtomicUsize>,
        clears: Arc<AtomicUsize>,
        clear_notify: Arc<Notify>,
        clear_gate: Arc<AtomicBool>,
        fail_clear: bool,
        clear_format_mismatch: bool,
        fail_volume: bool,
        fail_commit: bool,
        mismatch_commit: bool,
        setter_gate: Option<SetterGate>,
        controls: Arc<parking_lot::Mutex<AppliedControls>>,
        calls: Arc<parking_lot::Mutex<Vec<String>>>,
    }

    impl PlayerOutput for FakeOutput {
        fn commit(&self, buffer: AudioBuffer) -> Result<(), OutputError> {
            self.commits.fetch_add(1, Ordering::SeqCst);
            if self.mismatch_commit && self.commits.load(Ordering::SeqCst) == 1 {
                return Err(OutputError::FormatMismatch {
                    expected: buffer.format.clone(),
                    actual: buffer.format,
                });
            }
            if self.fail_commit {
                Err(OutputError::Failed("commit".into()))
            } else {
                Ok(())
            }
        }
        fn clear(&self) -> Result<(), OutputError> {
            while self.clear_gate.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            self.clears.fetch_add(1, Ordering::SeqCst);
            self.clear_notify.notify_waiters();
            if self.clear_format_mismatch {
                Err(OutputError::FormatMismatch {
                    expected: AudioFormat {
                        codec: Codec::Pcm,
                        sample_rate: 48_000,
                        channels: 2,
                        bit_depth: 16,
                        codec_header: None,
                    },
                    actual: AudioFormat {
                        codec: Codec::Pcm,
                        sample_rate: 44_100,
                        channels: 2,
                        bit_depth: 16,
                        codec_header: None,
                    },
                })
            } else if self.fail_clear {
                Err(OutputError::Failed("clear".into()))
            } else {
                Ok(())
            }
        }
        fn set_volume(&self, volume: u8) -> Result<(), OutputError> {
            #[cfg(test)]
            if let Some(gate) = &self.setter_gate {
                gate.0.send(()).unwrap();
                gate.1.lock().recv().unwrap();
            }
            self.controls.lock().volume = volume;
            self.calls.lock().push(format!("volume:{volume}"));
            if self.fail_volume {
                Err(OutputError::Failed("volume".into()))
            } else {
                Ok(())
            }
        }
        fn set_mute(&self, muted: bool) -> Result<(), OutputError> {
            self.controls.lock().muted = muted;
            Ok(())
        }
        fn set_static_delay(&self, delay_ms: u16) -> Result<(), OutputError> {
            self.controls.lock().static_delay_ms = delay_ms;
            Ok(())
        }
    }

    #[derive(Clone)]
    struct AlternateOutput(FakeOutput);

    impl PlayerOutput for AlternateOutput {
        fn commit(&self, buffer: AudioBuffer) -> Result<(), OutputError> {
            self.0.commit(buffer)
        }
        fn clear(&self) -> Result<(), OutputError> {
            self.0.clear()
        }
        fn set_volume(&self, volume: u8) -> Result<(), OutputError> {
            self.0.set_volume(volume)
        }
        fn set_mute(&self, muted: bool) -> Result<(), OutputError> {
            self.0.set_mute(muted)
        }
        fn set_static_delay(&self, delay_ms: u16) -> Result<(), OutputError> {
            self.0.set_static_delay(delay_ms)
        }
    }

    fn output() -> FakeOutput {
        FakeOutput {
            commits: Arc::new(AtomicUsize::new(0)),
            clears: Arc::new(AtomicUsize::new(0)),
            clear_notify: Arc::new(Notify::new()),
            clear_gate: Arc::new(AtomicBool::new(false)),
            fail_clear: false,
            clear_format_mismatch: false,
            fail_volume: false,
            fail_commit: false,
            mismatch_commit: false,
            setter_gate: None,
            controls: Arc::new(parking_lot::Mutex::new(AppliedControls {
                volume: 0,
                muted: false,
                static_delay_ms: 0,
            })),
            calls: Arc::new(parking_lot::Mutex::new(Vec::new())),
        }
    }

    #[tokio::test]
    async fn player_generation_exhaustion_closes_held_work() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach(output(), controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        producer.set_generation_max_for_test();
        assert_eq!(
            producer.clear(),
            Err(crate::role_core::ProducerError::GenerationExhausted)
        );
        assert_eq!(work.commit(buffer()), Err(CommitRejected::Closed));
        assert!(session.recv().await.is_none());
    }

    #[tokio::test]
    async fn receiver_can_attach_different_outputs_from_identical_type() {
        let (receiver_a, _) = PlayerReceiver::channel(controls());
        let (receiver_b, _) = PlayerReceiver::channel(controls());
        let _session_a: PlayerSession<FakeOutput> = receiver_a.attach(output(), controls());
        let _session_b: PlayerSession<AlternateOutput> =
            receiver_b.attach(AlternateOutput(output()), controls());
    }

    fn chunk() -> AudioChunk {
        AudioChunk {
            timestamp: 123,
            data: Arc::from([0x01, 0x02, 0x03].as_slice()),
        }
    }

    fn buffer() -> AudioBuffer {
        AudioBuffer {
            timestamp: 0,
            samples: Arc::from(vec![0, 0]),
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            },
        }
    }

    fn controls() -> AppliedControls {
        AppliedControls {
            volume: 80,
            muted: false,
            static_delay_ms: 0,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn public_commit_close_between_wrapper_check_and_admission_is_closed_without_callback() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let installed = output();
        let commits = Arc::clone(&installed.commits);
        session.install_for_test(installed).unwrap();
        producer.audio(chunk()).unwrap();
        let work = match session.recv().await {
            Some(PlayerEvent::Audio(work)) => work,
            _ => panic!("audio event"),
        };
        let reached = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        *session.receiver.fence.commit_admission_gate.lock() =
            Some((Arc::clone(&reached), Arc::clone(&release)));
        let commit = std::thread::spawn(move || work.commit(buffer()));
        reached.wait();
        producer.close();
        release.wait();
        assert_eq!(commit.join().unwrap(), Err(CommitRejected::Closed));
        assert_eq!(commits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn install_setter_blocked_then_producer_close_rejects_without_publication() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        let installed = output();
        let clears = Arc::clone(&installed.clears);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mut gated = installed.clone();
        gated.setter_gate = Some(Arc::new((entered_tx, parking_lot::Mutex::new(release_rx))));
        let install = std::thread::spawn(move || permit.install_for_test(gated));
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        let (closed_tx, closed_rx) = std::sync::mpsc::channel();
        let close = std::thread::spawn(move || {
            producer.close();
            closed_tx.send(()).unwrap();
        });
        closed_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        close.join().unwrap();
        release_tx.send(()).unwrap();
        assert_eq!(install.join().unwrap(), Err(InstallRejected::Closed));
        assert_eq!(clears.load(Ordering::SeqCst), 1);
        assert!(matches!(
            session.coordinator.admission().admit(),
            Admission::Closed
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn handoff_waits_for_device_drop() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let installed = output();
        session.install_for_test(installed).unwrap();

        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(old_work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        session.release_output().unwrap();
        assert_eq!(old_work.commit(buffer()), Err(CommitRejected::Obsolete));

        let (drop_ack_tx, drop_ack_rx) = tokio::sync::oneshot::channel();
        let advertised = Arc::new(AtomicUsize::new(0));
        let created = Arc::new(AtomicUsize::new(0));
        let advertised_by_handoff = Arc::clone(&advertised);
        let mut handoff = Box::pin(async move {
            drop_ack_rx.await.expect("device-drop acknowledgement");
            advertised_by_handoff.fetch_add(1, Ordering::SeqCst);
        });
        assert!(handoff.as_mut().now_or_never().is_none());
        assert_eq!(advertised.load(Ordering::SeqCst), 0);
        assert_eq!(created.load(Ordering::SeqCst), 0);

        drop_ack_tx.send(()).unwrap();
        handoff.await;
        let replacement = output();
        created.fetch_add(1, Ordering::SeqCst);
        let replacement_controls = Arc::clone(&replacement.controls);
        let permit = session.begin_output_install().unwrap();
        permit.install_for_test(replacement).unwrap();

        assert_eq!(advertised.load(Ordering::SeqCst), 1);
        assert_eq!(created.load(Ordering::SeqCst), 1);
        assert_eq!(*replacement_controls.lock(), controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(new_work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        new_work.commit(buffer()).unwrap();
    }

    #[tokio::test]
    async fn blocked_output_creation_does_not_block_controls_or_end() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        producer.audio(chunk()).unwrap();
        let old_work = match session.recv().await {
            Some(PlayerEvent::Audio(work)) => work,
            other => panic!("expected audio work, got {other:?}"),
        };
        let permit = session.begin_output_install().unwrap();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let created = Arc::new(AtomicUsize::new(0));
        let created_by_worker = Arc::clone(&created);
        let install = std::thread::spawn(move || {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            created_by_worker.fetch_add(1, Ordering::SeqCst);
            permit.install_for_test(output())
        });
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("output factory did not block before creation");
        assert_eq!(created.load(Ordering::SeqCst), 0);

        let mut control_watch = session.controls();
        producer.set_volume(42);
        timeout(Duration::from_secs(5), async {
            loop {
                if control_watch.borrow().volume == 42 {
                    break;
                }
                control_watch.changed().await.unwrap();
            }
        })
        .await
        .expect("controls were not published without an output");
        producer.end().unwrap();
        assert_eq!(old_work.commit(buffer()), Err(CommitRejected::Obsolete));
        assert_eq!(created.load(Ordering::SeqCst), 0);

        release_tx.send(()).unwrap();
        assert_eq!(install.join().unwrap(), Ok(()));
        assert_eq!(created.load(Ordering::SeqCst), 1);
        session.shutdown().await;
    }

    #[tokio::test]
    async fn lazy_public_session_replays_preinstall_work_and_rejects_old_work_after_release() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        assert_eq!(work.timestamp(), 123);
        assert_eq!(work.data(), &[0x01, 0x02, 0x03]);
        assert_eq!(work.chunk().data.as_ref(), &[0x01, 0x02, 0x03]);
        let first = output();
        session.install_for_test(first.clone()).unwrap();
        work.commit(buffer()).unwrap();
        assert_eq!(first.commits.load(Ordering::SeqCst), 1);
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(old_work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        session.release_output().unwrap();
        assert_eq!(old_work.commit(buffer()), Err(CommitRejected::Obsolete));
        session.install_for_test(output()).unwrap();
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(new_work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        new_work.commit(buffer()).unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_audio_before_reinstall_does_not_hide_start() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        producer.audio(chunk()).unwrap();
        session.release_output().unwrap();
        producer
            .start(crate::protocol::messages::StreamPlayerConfig {
                codec: "pcm".into(),
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            })
            .unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Start(_))));
        assert!(session.recv().now_or_never().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn command_between_attach_snapshot_and_driver_start_updates_output_and_controls() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let applied = Arc::clone(&output.controls);
        let gate = Arc::new(Notify::new());
        let session = receiver.attach_with_driver_gate(output, controls(), Arc::clone(&gate));
        let mut controls_rx = session.controls();
        producer.set_volume(42);
        gate.notify_one();
        controls_rx.changed().await.unwrap();
        assert_eq!(applied.lock().volume, 42);
        assert_eq!(controls_rx.borrow().volume, 42);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_install_does_not_regress_controls_published_without_output() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        producer.set_volume(20);
        let session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let mut control_watch = session.controls();
        loop {
            let volume = {
                let controls = control_watch.borrow();
                controls.volume
            };
            if volume == 20 {
                break;
            }
            control_watch.changed().await.unwrap();
        }
        producer.set_volume(50);
        loop {
            let volume = {
                let controls = control_watch.borrow();
                controls.volume
            };
            if volume == 50 {
                break;
            }
            control_watch.changed().await.unwrap();
        }
        let installed = output();
        let calls = Arc::clone(&installed.calls);
        let permit = session.begin_output_install().unwrap();
        permit.install_for_test(installed).unwrap();
        tokio::task::yield_now().await;
        assert_eq!(control_watch.borrow().volume, 50);
        assert_eq!(session.coordinator.control_version(), 2);
        assert_eq!(&*calls.lock(), &["volume:50"]);
        session.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_recv_reconciles_urgent_boundary_before_dequeue() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_clear = true;
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        session.install_for_test(failing).unwrap();
        producer.audio(chunk()).unwrap();
        producer.clear().unwrap();
        assert!(session.recv().await.is_none());
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::Failed(OutputError::Failed("clear".into()))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_recv_skips_detached_audio_and_preserves_following_start() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        session.release_output().unwrap();
        producer.audio(chunk()).unwrap();
        producer
            .start(crate::protocol::messages::StreamPlayerConfig {
                codec: "pcm".into(),
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            })
            .unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Start(_))));
        assert!(session.recv().now_or_never().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_driver_rejects_stale_snapshot_after_newer_install_version() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        producer.set_volume(20);
        let gate = Arc::new(Notify::new());
        let session = receiver.attach_lazy_with_driver_gate(controls(), Arc::clone(&gate));
        let permit = session.begin_output_install().unwrap();
        producer.set_volume(50);
        let installed = output();
        let calls = Arc::clone(&installed.calls);
        permit.install_for_test(installed).unwrap();
        gate.notify_one();
        let mut control_watch = session.controls();
        while control_watch.borrow().volume != 50 {
            control_watch.changed().await.unwrap();
        }
        assert_eq!(control_watch.borrow().volume, 50);
        assert_eq!(session.coordinator.control_version(), 2);
        assert_eq!(&*calls.lock(), &["volume:50"]);
        session.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_install_publishes_controls_across_driver_yields() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        producer.set_volume(20);
        let installed = output();
        let applied = Arc::clone(&installed.controls);
        permit.install_for_test(installed).unwrap();
        producer.set_volume(50);
        let mut controls_watch = session.controls();
        loop {
            let volume = {
                let controls = controls_watch.borrow();
                controls.volume
            };
            if volume == 50 {
                break;
            }
            controls_watch.changed().await.unwrap();
        }
        assert_eq!(applied.lock().volume, 50);
        assert_eq!(controls_watch.borrow().volume, 50);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eager_recv_reports_clear_failure_before_driver_poll() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_clear = true;
        let mut session = receiver.attach(failing, controls());
        producer.clear().unwrap();
        assert!(session.recv().await.is_none());
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::Failed(OutputError::Failed("clear".into()))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eager_recv_bridges_format_mismatch_clear_to_terminal_status() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.clear_format_mismatch = true;
        let mut session = receiver.attach(failing, controls());
        producer.clear().unwrap();
        assert!(session.recv().await.is_none());
        assert!(matches!(
            *session.status().borrow(),
            SessionStatus::Failed(OutputError::FormatMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn public_eager_format_mismatch_keeps_session_open_and_next_commit_succeeds() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut mismatch = output();
        mismatch.mismatch_commit = true;
        let mut session = receiver.attach(mismatch, controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!()
        };
        assert!(matches!(
            work.commit(buffer()),
            Err(CommitRejected::Output(OutputError::FormatMismatch { .. }))
        ));
        assert_eq!(*session.status().borrow(), SessionStatus::Open);
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(work.commit(buffer()), Ok(()));
    }

    #[tokio::test]
    async fn public_lazy_format_mismatch_keeps_session_open_and_next_commit_succeeds() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let mut mismatch = output();
        mismatch.mismatch_commit = true;
        session.install_for_test(mismatch).unwrap();
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!()
        };
        assert!(matches!(
            work.commit(buffer()),
            Err(CommitRejected::Output(OutputError::FormatMismatch { .. }))
        ));
        assert_eq!(*session.status().borrow(), SessionStatus::Open);
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(work.commit(buffer()), Ok(()));
    }

    #[tokio::test]
    async fn predriver_combined_controls_apply_all_three_fields() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let session = receiver.attach(output(), controls());
        let mut controls_rx = session.controls();
        producer.set_controls(AppliedControls {
            volume: 31,
            muted: true,
            static_delay_ms: 47,
        });
        controls_rx.changed().await.unwrap();
        assert_eq!(
            *controls_rx.borrow(),
            AppliedControls {
                volume: 31,
                muted: true,
                static_delay_ms: 47,
            }
        );
    }

    #[tokio::test]
    async fn failed_setter_does_not_publish_unapplied_controls() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_volume = true;
        let session = receiver.attach(failing, controls());
        let controls_rx = session.controls();
        let mut status = session.status();
        producer.set_volume(12);
        while matches!(*status.borrow(), SessionStatus::Open) {
            status.changed().await.unwrap();
        }
        assert!(matches!(*status.borrow(), SessionStatus::Failed(_)));
        assert_eq!(controls_rx.borrow().volume, 80);
    }

    #[tokio::test]
    async fn eager_attach_publishes_all_normalized_initial_controls() {
        let (receiver, _) = PlayerReceiver::channel(controls());
        let output = output();
        let state = Arc::clone(&output.controls);
        let session = receiver.attach(
            output,
            AppliedControls {
                volume: 255,
                muted: true,
                static_delay_ms: 9_999,
            },
        );
        assert_eq!(
            *session.controls().borrow(),
            AppliedControls {
                volume: 100,
                muted: true,
                static_delay_ms: 5_000,
            }
        );
        assert_eq!(
            *state.lock(),
            AppliedControls {
                volume: 100,
                muted: true,
                static_delay_ms: 5_000
            }
        );
    }

    #[tokio::test]
    async fn preattach_boundary_is_applied_before_postboundary_data() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        producer.clear().unwrap();
        let output = output();
        let mut session = receiver.attach(output.clone(), controls());
        producer.audio(chunk()).unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Clear)));
        assert!(matches!(session.recv().await, Some(PlayerEvent::Audio(_))));
        assert_eq!(output.clears.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn public_eager_clear_invalidates_issued_work_while_recv_is_idle() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let mut session = receiver.attach(output.clone(), controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("audio")
        };
        producer.clear().unwrap();
        assert_eq!(work.commit(buffer()), Err(CommitRejected::Obsolete));
        assert!(matches!(session.recv().await, Some(PlayerEvent::Clear)));
    }

    #[tokio::test]
    async fn producer_command_applies_after_attach_without_recv() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let session = receiver.attach(output(), controls());
        let mut controls_rx = session.controls();
        producer.set_volume(33);
        tokio::time::timeout(std::time::Duration::from_secs(1), controls_rx.changed())
            .await
            .expect("timed out waiting for controls")
            .unwrap();
        assert_eq!(controls_rx.borrow().volume, 33);
    }

    #[tokio::test]
    async fn preattach_command_overlays_only_named_initial_field() {
        let (receiver, producer) = PlayerReceiver::channel(AppliedControls {
            volume: 11,
            muted: true,
            static_delay_ms: 222,
        });
        producer.set_volume(33);
        let session = receiver.attach(
            output(),
            AppliedControls {
                volume: 80,
                muted: false,
                static_delay_ms: 7,
            },
        );
        let controls = session.controls();
        assert_eq!(controls.borrow().volume, 33);
        assert!(!controls.borrow().muted);
        assert_eq!(controls.borrow().static_delay_ms, 7);
    }

    #[tokio::test]
    async fn public_lazy_permit_drop_retries_and_preinstall_commands_replay() {
        let (receiver, producer) = PlayerReceiver::channel(AppliedControls {
            volume: 255,
            muted: false,
            static_delay_ms: 9999,
        });
        let session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let mut controls_rx = session.controls();
        producer.set_controls(AppliedControls {
            volume: 33,
            muted: true,
            static_delay_ms: 42,
        });
        controls_rx.changed().await.unwrap();
        let permit = session.begin_output_install().unwrap();
        let installed = output();
        let state = Arc::clone(&installed.controls);
        drop(permit);
        let permit = session.begin_output_install().unwrap();
        permit.install_for_test(installed).unwrap();
        assert_eq!(
            *state.lock(),
            AppliedControls {
                volume: 33,
                muted: true,
                static_delay_ms: 42
            }
        );
        assert_eq!(session.controls().borrow().volume, 33);
        assert_eq!(session.controls().borrow().static_delay_ms, 42);
    }

    #[tokio::test]
    async fn public_producer_close_rejects_held_work_without_recv_and_cancel_keeps_next_event() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach(output(), controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("audio")
        };
        producer.close();
        assert_eq!(work.commit(buffer()), Err(CommitRejected::Closed));
        assert!(session.recv().await.is_none());

        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach(output(), controls());
        producer.audio(chunk()).unwrap();
        let pending = session.recv();
        drop(pending);
        let event = session.recv().await;
        assert!(matches!(event, Some(PlayerEvent::Audio(_))));
        assert!(session.recv().now_or_never().is_none());
    }

    #[tokio::test]
    async fn public_clear_failure_and_setter_failure_are_terminal() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing_clear = output();
        failing_clear.fail_clear = true;
        let session = receiver.attach(failing_clear, controls());
        let mut status = session.status();
        producer.clear().unwrap();
        while matches!(*status.borrow(), SessionStatus::Open) {
            status.changed().await.unwrap();
        }
        assert!(matches!(*status.borrow(), SessionStatus::Failed(_)));

        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing_setter = output();
        failing_setter.fail_volume = true;
        let session = receiver.attach(failing_setter, controls());
        let mut status = session.status();
        producer.set_controls(AppliedControls {
            volume: 12,
            ..controls()
        });
        while matches!(*status.borrow(), SessionStatus::Open) {
            status.changed().await.unwrap();
        }
        assert!(matches!(*status.borrow(), SessionStatus::Failed(_)));
    }

    #[tokio::test]
    async fn lazy_queued_audio_before_release_is_not_replayed_after_reinstall() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("audio")
        };
        let permit = session.begin_output_install().unwrap();
        permit.install_for_test(output()).unwrap();
        session.release_output().unwrap();
        assert_eq!(work.commit(buffer()), Err(CommitRejected::Obsolete));
        assert!(session.recv().now_or_never().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_work_keeps_generation_across_synchronous_clear() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(old_work) = session.recv().await.unwrap() else {
            panic!("audio")
        };
        producer.clear().unwrap();
        assert_eq!(old_work.commit(buffer()), Err(CommitRejected::Obsolete));
        assert!(matches!(session.recv().await, Some(PlayerEvent::Clear)));
        session.install_for_test(output()).unwrap();
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(new_work) = session.recv().await.unwrap() else {
            panic!("post-clear audio")
        };
        assert_eq!(new_work.commit(buffer()), Ok(()));
    }

    #[tokio::test]
    async fn detached_audio_is_skipped_but_fresh_audio_is_delivered() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        producer.audio(chunk()).unwrap();
        session.release_output().unwrap();
        assert_eq!(producer.audio(chunk()), Ok(()));
        session.install_for_test(output()).unwrap();
        producer.audio(chunk()).unwrap();
        let event = session.recv().await;
        assert!(matches!(event, Some(PlayerEvent::Audio(_))));
        assert!(session.recv().now_or_never().is_none());
    }

    #[tokio::test]
    async fn reservation_identity_preserves_new_install_after_old_permit_drop_or_failure() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let old = session.begin_output_install().unwrap();
        session.release_output().unwrap();
        let new = session.begin_output_install().unwrap();
        drop(old);
        new.install_for_test(output()).unwrap();

        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let old = session.begin_output_install().unwrap();
        session.release_output().unwrap();
        let new = session.begin_output_install().unwrap();
        assert_eq!(
            old.install_for_test(output()),
            Err(InstallRejected::Superseded)
        );
        new.install_for_test(output()).unwrap();
    }

    #[test]
    fn concurrent_begin_and_install_complete_without_lock_cycle() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let coordinator = Arc::new(crate::audio::player_session::LazyCoordinator::<
            OutputAdapterHandle<FakeOutput>,
        >::new(controls()));
        let permit = coordinator.begin_install().unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let (done_tx, done_rx) = mpsc::channel();
        let begin_coordinator = Arc::clone(&coordinator);
        let begin_barrier = Arc::clone(&barrier);
        let begin_thread = thread::spawn(move || {
            begin_barrier.wait();
            let result = begin_coordinator.begin_install();
            done_tx.send(result).unwrap();
        });
        barrier.wait();
        assert_eq!(permit.install(OutputAdapterHandle(output())), Ok(()));
        assert!(matches!(
            done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            Err(crate::audio::player_session::InstallRejected::AlreadyInstalled)
                | Err(crate::audio::player_session::InstallRejected::Reserved)
        ));
        begin_thread.join().unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_rejects_outstanding_install_before_driver_poll() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        producer.close();
        assert_eq!(
            permit.install_for_test(output()),
            Err(InstallRejected::Closed)
        );
        assert!(session.recv().await.is_none());
    }

    #[tokio::test]
    async fn producer_drop_fences_held_work_and_pending_recv_and_clears_once() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let mut session = receiver.attach(output.clone(), controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!()
        };
        let clear_done = output.clear_notify.notified();
        let pending = session.recv();
        drop(producer);
        assert_eq!(work.commit(buffer()), Err(CommitRejected::Closed));
        tokio::time::timeout(std::time::Duration::from_secs(1), clear_done)
            .await
            .expect("driver cleanup must complete");
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), pending)
                .await
                .expect("pending recv must wake after producer drop")
                .is_none()
        );
        assert_eq!(output.clears.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn failed_commit_late_status_is_retained_and_shutdown_wakes_recv() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_commit = true;
        let mut session = receiver.attach(failing, controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(
            work.commit(buffer()),
            Err(CommitRejected::Output(OutputError::Failed("commit".into())))
        );
        let status = session.status();
        assert_eq!(
            *status.borrow(),
            SessionStatus::Failed(OutputError::Failed("commit".into()))
        );
        let pending = session.recv();
        assert!(pending.await.is_none());
        session.shutdown().await;
        assert_eq!(
            *status.borrow(),
            SessionStatus::Failed(OutputError::Failed("commit".into()))
        );
    }

    #[tokio::test]
    async fn eager_shutdown_completes_with_live_producer() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let session = receiver.attach(output(), controls());
        tokio::time::timeout(std::time::Duration::from_secs(1), session.shutdown())
            .await
            .expect("eager shutdown must not wait for producer drop");
    }

    #[tokio::test]
    async fn lazy_shutdown_completes_with_live_producer() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        tokio::time::timeout(std::time::Duration::from_secs(1), session.shutdown())
            .await
            .expect("lazy shutdown must not wait for producer drop");
    }

    #[tokio::test]
    async fn session_drop_clears_output_with_held_work() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let weak = Arc::downgrade(&output.clears);
        let mut session = receiver.attach(output.clone(), controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!()
        };
        drop(session);
        drop(output);
        assert_eq!(work.commit(buffer()), Err(CommitRejected::Closed));
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn lazy_shutdown_awaits_driver_and_releases_shared_state() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let shared = session.coordinator.shared();
        let weak = Arc::downgrade(&shared);
        drop(producer);
        session.shutdown().await;
        drop(shared);
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn controls_mirror_publishes_install_and_retains_close() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        producer.set_volume(37);
        let session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let mut controls_watch = session.controls();
        while controls_watch.borrow().volume != 37 {
            controls_watch.changed().await.unwrap();
        }
        let permit = session.begin_output_install().unwrap();
        permit.install_for_test(output()).unwrap();
        assert_eq!(controls_watch.borrow().volume, 37);
        let mut status = session.status();
        producer.close();
        status.changed().await.unwrap();
        assert_eq!(controls_watch.borrow().volume, 37);
        session.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn producer_close_fences_before_gated_driver_cleanup() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let clear_done = Arc::clone(&output.clear_notify);
        let gate = Arc::clone(&output.clear_gate);
        gate.store(true, Ordering::Release);
        let driver_gate = Arc::new(Notify::new());
        let mut session =
            receiver.attach_with_driver_gate(output.clone(), controls(), Arc::clone(&driver_gate));
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("audio")
        };
        producer.close();
        assert_eq!(work.commit(buffer()), Err(CommitRejected::Closed));
        assert_eq!(output.clears.load(Ordering::SeqCst), 0);
        assert_eq!(output.clears.load(Ordering::SeqCst), 0);
        driver_gate.notify_one();
        gate.store(false, Ordering::Release);
        tokio::time::timeout(std::time::Duration::from_secs(1), clear_done.notified())
            .await
            .expect("driver cleanup must complete");
        assert_eq!(output.clears.load(Ordering::SeqCst), 1);
        session.shutdown().await;
    }

    #[tokio::test]
    async fn session_drop_cleans_output_once_after_driver_abort() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let clears = Arc::clone(&output.clears);
        let session = receiver.attach(output, controls());
        drop(producer);
        drop(session);
        assert_eq!(clears.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn lazy_public_session_install_retry_and_closed_rejection() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        assert!(session.begin_output_install().is_ok());
        drop(session);

        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        producer.close();
        assert!(session.recv().await.is_none());
        drop(session);
    }

    #[tokio::test]
    async fn producer_close_cleans_eager_output_before_driver_poll() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let weak = Arc::downgrade(&output.clears);
        let session = receiver.attach(output.clone(), controls());
        let status = session.status();
        let clear_done = output.clear_notify.notified();
        producer.close();
        tokio::time::timeout(std::time::Duration::from_secs(1), clear_done)
            .await
            .expect("driver cleanup must complete");
        assert_eq!(output.clears.load(Ordering::SeqCst), 1);
        assert_eq!(*status.borrow(), SessionStatus::Closed);
        drop(session);
        drop(output);
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn lazy_clear_failure_before_recv_publishes_original_failure() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_clear = true;
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        session.install_for_test(failing).unwrap();
        producer.clear().unwrap();
        assert!(session.recv().await.is_none());
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::Failed(OutputError::Failed("clear".into()))
        );
    }

    #[tokio::test]
    async fn lazy_install_setter_failure_publishes_public_failure() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let mut failing = output();
        failing.fail_volume = true;
        assert_eq!(
            session.install_for_test(failing),
            Err(InstallRejected::Failed)
        );
        assert!(matches!(
            *session.status().borrow(),
            SessionStatus::Failed(OutputError::Failed(ref error)) if error == "volume"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_install_clears_queued_audio_after_end_before_permit() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        producer.audio(chunk()).unwrap();
        producer.end().unwrap();
        let installed = output();
        let clears = Arc::clone(&installed.clears);
        permit.install_for_test(installed).unwrap();
        assert_eq!(clears.load(Ordering::SeqCst), 1);
        assert!(matches!(session.recv().await, Some(PlayerEvent::End)));
        assert!(session.recv().now_or_never().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_install_reconciles_end_while_setter_is_blocked() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        producer.audio(chunk()).unwrap();
        let installed = output();
        let clears = Arc::clone(&installed.clears);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mut gated = installed.clone();
        gated.setter_gate = Some(Arc::new((entered_tx, parking_lot::Mutex::new(release_rx))));
        let install = std::thread::spawn(move || permit.install_for_test(gated));
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        producer.end().unwrap();
        release_tx.send(()).unwrap();
        assert_eq!(install.join().unwrap(), Ok(()));
        assert_eq!(clears.load(Ordering::SeqCst), 1);
        assert!(matches!(session.recv().await, Some(PlayerEvent::End)));
        assert!(session.recv().now_or_never().is_none());
    }

    #[tokio::test]
    async fn lazy_release_clear_failure_retains_public_failure() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_lazy_for_test::<FakeOutput>(controls());
        let mut failing = output();
        failing.fail_clear = true;
        session.install_for_test(failing).unwrap();
        assert_eq!(
            session.release_output(),
            Err(OutputError::Failed("clear".into()))
        );
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::Failed(OutputError::Failed("clear".into()))
        );
    }
}
