use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(test)]
use std::sync::Barrier;

use parking_lot::Mutex;
use tokio::sync::{mpsc, watch, Notify};

use super::player_session::{
    AppliedControls, CommitRejected, ControlUpdate, Coordinator, InstallRejected, Lifecycle,
    OutputAdapter, OutputError, SessionStatus,
};
use super::synced_player::PlaybackControl;
use super::AudioBuffer;
use crate::generation::StreamGeneration;
use crate::protocol::client::AudioChunk;
pub use crate::protocol::messages::StreamPlayerConfig;
use crate::role_core::ProducerError;

type CommitFn = Box<dyn FnOnce(StreamGeneration, AudioBuffer) -> Result<(), CommitRejected> + Send>;
#[cfg(test)]
type SetterGate = Arc<(
    std::sync::mpsc::SyncSender<()>,
    parking_lot::Mutex<std::sync::mpsc::Receiver<()>>,
)>;

/// A synchronous device-independent player output.
///
/// Calls are serialized, short, nonblocking, and must not reenter the session.
pub trait PlayerOutput: Send + 'static {
    /// Commits one decoded buffer.
    fn commit(&self, buffer: AudioBuffer) -> Result<(), OutputError>;
    /// Clears queued output.
    fn clear(&self) -> Result<(), OutputError>;
    /// Applies clamped volume.
    fn set_volume(&self, volume: u8) -> Result<(), OutputError>;
    /// Applies mute state.
    fn set_mute(&self, muted: bool) -> Result<(), OutputError>;
    /// Applies clamped static delay.
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
    },
    Clear(StreamGeneration),
    End(StreamGeneration),
}

#[derive(Clone)]
struct UrgentState {
    version: u64,
    boundary: Option<StreamGeneration>,
    commands: ControlUpdate,
    closed: bool,
}

struct SessionFence {
    status: watch::Sender<SessionStatus>,
    status_value: Arc<Mutex<SessionStatus>>,
    lifecycle: Arc<Lifecycle>,
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
    fn publish_terminal(&self, proposed: SessionStatus) {
        let _transition = self.lifecycle.transition.lock();
        let mut status = self.status_value.lock();
        if matches!(*status, SessionStatus::Closed | SessionStatus::Failed(_)) {
            return;
        }
        *status = proposed.clone();
        let _ = self.status.send_replace(proposed);
    }
    fn close(&self) {
        self.cancel.notify_waiters();
        self.admission.close();
        self.publish_terminal(SessionStatus::Closed);
        self.urgent.send_modify(|urgent| urgent.closed = true);
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
        #[cfg(test)]
        let commit_fence = Arc::clone(&fence);
        Self {
            commit: Some(Box::new(move |generation, buffer| {
                #[cfg(test)]
                if let Some((reached, release)) = commit_fence.commit_admission_gate.lock().take() {
                    reached.wait();
                    release.wait();
                }
                commit(work, generation, buffer)
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

    /// Commits once; dropping work has no session effect. `FormatMismatch` leaves the session open.
    /// Output failure detaches as `OutputLost`, rejecting admitted work; recovery needs new output/work.
    /// Clear/end/release or closure reject older work as `Obsolete` or `Closed`.
    pub fn commit(mut self, prepared: AudioBuffer) -> Result<(), CommitRejected> {
        if matches!(
            *self.fence.status_value.lock(),
            SessionStatus::Closed | SessionStatus::Failed(_)
        ) {
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
    pub(crate) fn set_controls(&self, controls: AppliedControls) {
        self.fence.urgent.send_modify(|urgent| {
            urgent.version = urgent.version.wrapping_add(1);
            urgent.commands = ControlUpdate {
                volume: Some(controls.volume),
                muted: Some(controls.muted),
                static_delay_ms: Some(controls.static_delay_ms),
            };
        });
    }
    pub(crate) fn audio(&self, chunk: AudioChunk) -> Result<(), ProducerError> {
        let generation = *self.generation.lock();
        if self.fence.admission.is_closed() {
            return Err(ProducerError::ReceiverClosed);
        }
        self.tx
            .send(RawEvent::Audio { generation, chunk })
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
            *current = current.advance();
            *current
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
            *current = current.advance();
            *current
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
    pub(crate) fn controls(&self) -> watch::Receiver<AppliedControls> {
        self.fence.controls.subscribe()
    }

    pub(crate) fn channel(initial: AppliedControls) -> (Self, PlayerProducer) {
        let (tx, rx) = mpsc::unbounded_channel();
        let controls = initial.normalized();
        let (status, _) = watch::channel(SessionStatus::Open);
        let status_value = Arc::new(Mutex::new(SessionStatus::Open));
        let lifecycle = Arc::new(Lifecycle::new(status.clone(), Arc::clone(&status_value)));
        let (control_tx, _) = watch::channel(controls.clone());
        let (urgent, _) = watch::channel(UrgentState {
            version: 0,
            boundary: None,
            commands: ControlUpdate::default(),
            closed: false,
        });
        let admission = Arc::new(super::player_session::AdmissionGate::default());
        let fence = Arc::new(SessionFence {
            status,
            status_value,
            lifecycle,
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
    /// `initial_controls` is applied as the starting snapshot, then overlaid with
    /// any pre-attachment command for each named field; requested values are clamped
    /// when applied. Output callbacks are synchronous, short, nonblocking, serialized,
    /// and must not reenter this session. If the output rejects the initial controls,
    /// the session starts without an output and reports `OutputLost`.
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
        let coordinator = Coordinator::<OutputAdapterHandle<O>>::with_admission_and_controls(
            controls.clone(),
            Arc::clone(&self.fence.admission),
            self.fence.controls.clone(),
            Arc::clone(&self.fence.lifecycle),
        );
        let shared = coordinator.shared();
        let initial = self.fence.urgent.borrow().clone();
        let latest = initial.clone();
        if let Ok(permit) = coordinator.begin_install() {
            let mut latest_controls = controls.clone();
            latest.commands.apply(&mut latest_controls);
            let result = permit.install_with_controls(
                OutputAdapterHandle(output),
                latest.version,
                latest_controls.clone(),
                Some(latest.commands),
                || latest.boundary,
            );
            if result.is_err() {
                let _ = self.fence.controls.send_replace(controls.clone());
            }
        } else {
            let _ = self.fence.controls.send_replace(controls.clone());
        }
        if initial.closed
            || !matches!(
                *self.fence.status_value.lock(),
                SessionStatus::Open | SessionStatus::OutputLost(_)
            )
        {
            coordinator.close();
        }
        let driver = Driver::start(Arc::clone(&self.fence), shared);
        PlayerSession {
            receiver: self,
            coordinator,
            driver: Some(driver),
        }
    }
    /// Creates a session with no output on the current Tokio runtime.
    /// Initial controls are overlaid by pre-attachment commands and remain updatable.
    /// Installation replays retained controls before admitting work.
    pub fn attach_empty<O: PlayerOutput>(
        self,
        initial_controls: AppliedControls,
    ) -> PlayerSession<O> {
        self.attach_empty_impl(initial_controls)
    }
    /// Creates a `PlaybackControl` session whose output can be installed later.
    pub fn attach_lazy(self, initial_controls: AppliedControls) -> PlayerSession<PlaybackControl> {
        self.attach_empty(initial_controls)
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
    fn attach_empty_with_driver_gate<O: PlayerOutput>(
        self,
        initial_controls: AppliedControls,
        gate: Arc<Notify>,
    ) -> PlayerSession<O> {
        *self.fence.driver_start_gate.lock() = Some(gate);
        self.attach_empty_impl::<O>(initial_controls)
    }

    fn attach_empty_impl<O: PlayerOutput>(
        self,
        initial_controls: AppliedControls,
    ) -> PlayerSession<O> {
        let mut controls = initial_controls.normalized();
        self.fence.urgent.borrow().commands.apply(&mut controls);
        let fence = Arc::clone(&self.fence);
        let coordinator = Coordinator::<OutputAdapterHandle<O>>::with_admission_and_controls(
            controls.clone(),
            Arc::clone(&fence.admission),
            fence.controls.clone(),
            Arc::clone(&fence.lifecycle),
        );
        let shared = coordinator.shared();
        let initial = fence.urgent.borrow().clone();
        let coordinator = super::player_session::Coordinator::from_shared(Arc::clone(&shared));
        let _ = fence.controls.send_replace(controls.clone());
        if let Some(boundary) = initial.boundary {
            let _ = coordinator.boundary(boundary);
        }
        if initial.closed
            || !matches!(
                *fence.status_value.lock(),
                SessionStatus::Open | SessionStatus::OutputLost(_)
            )
        {
            coordinator.close();
        }
        let driver = Driver::start(Arc::clone(&fence), shared);
        PlayerSession {
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
                    let cancelled = cancel.notified();
                    tokio::pin!(cancelled);
                    cancelled.as_mut().enable();
                    if fence.cancelled.load(Ordering::Acquire) {
                        break;
                    }
                    tokio::select! {
                        _ = cancelled => break,
                        _ = gate.notified() => {}
                    }
                }
                let coordinator = Coordinator::from_shared(Arc::clone(&shared));
                if snapshot.closed {
                    coordinator.close();
                    break;
                }
                if let Some(boundary) = snapshot.boundary {
                    let _ = coordinator.boundary(boundary);
                }
                let result = coordinator.apply_controls_transaction(|control_version| {
                    let mut captured = None;
                    fence.urgent.send_if_modified(|latest| {
                        if latest.version > control_version {
                            captured = Some((latest.commands, latest.version));
                            latest.commands = ControlUpdate::default();
                            true
                        } else {
                            false
                        }
                    });
                    captured
                });
                let _ = result;
                tokio::select! {
                    _ = cancel.notified() => break,
                    changed = status.changed() => {
                        if changed.is_err()
                            || matches!(
                                *status.borrow(),
                                SessionStatus::Closed | SessionStatus::Failed(_)
                            )
                        {
                            break;
                        }
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
}
impl Drop for Driver {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

/// One-shot reservation for installing an output.
pub struct OutputInstallPermit<O: PlayerOutput> {
    permit: Option<super::player_session::InstallPermit<OutputAdapterHandle<O>>>,
    shared: Arc<super::player_session::Shared<OutputAdapterHandle<O>>>,
    fence: Arc<SessionFence>,
}
impl<O: PlayerOutput> OutputInstallPermit<O> {
    /// Installs the output and replays pending controls. The permit is consumed by the attempt.
    /// Dropping it releases the reservation; an install failure reports `OutputLost` and requires
    /// a fresh reservation. Only `Closed` is terminal for the player session.
    pub fn install(mut self, output: O) -> Result<(), InstallRejected> {
        let status = self.fence.status_value.lock().clone();
        if !matches!(status, SessionStatus::Open | SessionStatus::OutputLost(_)) {
            return Err(match status {
                SessionStatus::Closed => InstallRejected::Closed,
                SessionStatus::Failed(_) => InstallRejected::Failed,
                SessionStatus::Open | SessionStatus::OutputLost(_) => unreachable!(),
            });
        }
        let urgent = self.fence.urgent.borrow().clone();
        let current_controls = self.fence.controls.borrow().clone();
        let patch = urgent.commands;
        let mut controls = current_controls;
        patch.apply(&mut controls);
        let mut result = self
            .permit
            .take()
            .ok_or(InstallRejected::Reserved)?
            .install_with_controls(
                OutputAdapterHandle(output),
                urgent.version,
                controls,
                Some(patch),
                || self.fence.urgent.borrow().boundary,
            );
        if result.is_ok() {
            let status = Coordinator::from_shared(Arc::clone(&self.shared)).status();
            if matches!(status, SessionStatus::Open)
                && matches!(*self.fence.status_value.lock(), SessionStatus::Closed)
            {
                Coordinator::from_shared(Arc::clone(&self.shared)).close();
                result = Err(InstallRejected::Closed);
            }
        }
        result
    }
}

/// Session that accepts player events before and after output installation.
pub struct PlayerSession<O: PlayerOutput = PlaybackControl> {
    receiver: PlayerReceiver,
    coordinator: Coordinator<OutputAdapterHandle<O>>,
    driver: Option<Driver>,
}
impl<O: PlayerOutput> PlayerSession<O> {
    /// Reserves the single output-install slot.
    ///
    /// Only one reservation may exist at a time. Clear and end advance the invalidation
    /// floor but do not consume the slot; release fences old work and permits by output
    /// incarnation. `Closed` rejects installation permanently; an installation failure
    /// consumes that attempt and recovery requires a fresh reservation.
    pub fn begin_output_install(&self) -> Result<OutputInstallPermit<O>, InstallRejected> {
        let status = self.receiver.fence.status_value.lock().clone();
        if !matches!(status, SessionStatus::Open | SessionStatus::OutputLost(_)) {
            return Err(match status {
                SessionStatus::Closed => InstallRejected::Closed,
                SessionStatus::Failed(_) => InstallRejected::Failed,
                SessionStatus::Open | SessionStatus::OutputLost(_) => unreachable!(),
            });
        }
        let coordinator = &self.coordinator;
        let boundary = *self.receiver.fence.boundary.lock();
        let _ = coordinator.boundary(boundary);
        let permit = coordinator.begin_install()?;
        Ok(OutputInstallPermit::<O> {
            shared: Arc::clone(&coordinator.shared()),
            permit: Some(permit),
            fence: Arc::clone(&self.receiver.fence),
        })
    }

    /// Installs output using a newly reserved slot.
    pub fn install(&mut self, output: O) -> Result<(), InstallRejected> {
        self.begin_output_install()?.install(output)
    }
    /// Releases the installed output while keeping the session open for reinstall.
    ///
    /// Audio work already returned by [`recv`](Self::recv) becomes obsolete, while audio
    /// still queued is kept for the next installation. The output is cleared before the
    /// result is returned; a clear failure is retained as `OutputLost` and a fresh
    /// installation may recover the session.
    ///
    /// The output-owning worker must acknowledge dropping its device before the application
    /// advertises an external source. The session remains open and retains controls and the
    /// current boundary for a later installation; the worker retains the latest start/configuration.
    ///
    /// ```rust,no_run
    /// use sendspin::audio::{PlaybackControl, PlayerSession};
    /// use sendspin::{protocol::messages::PlayerState, WsSender};
    ///
    /// async fn handoff(
    ///     session: &mut PlayerSession<PlaybackControl>,
    ///     sender: &WsSender,
    ///     worker: &mut OutputWorker,
    /// ) -> Result<(), Box<dyn std::error::Error>> {
    ///     session.release_output()?;
    ///     worker.drop_device_and_ack().await;
    ///     sender.enter_external_source().await?;
    ///
    ///     // The caller supplies the latest protocol state and reads any OS/device state
    ///     // through its platform APIs before constructing this value.
    ///     let initial_state = worker.read_player_state().await?;
    ///     let controls = session.controls().borrow().clone();
    ///     sender.exit_external_source(Some(with_session_controls(initial_state, controls))).await?;
    ///
    ///     let permit = session.begin_output_install()?;
    ///     let output = worker.create_output().await?;
    ///     permit.install(output)?;
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
        self.coordinator.release()
    }
    /// Receives the next event, skipping audio superseded by a clear or end.
    ///
    /// Work returned while no output is installed can be committed after the next
    /// installation, and committing it earlier returns `Obsolete`.
    ///
    /// Waiting is cancellation-safe and does not lose the next event. Terminal status
    /// returns `None` immediately without draining queued events; subsequent calls also
    /// return `None`.
    pub async fn recv(&mut self) -> Option<PlayerEvent> {
        loop {
            let receiver = &mut self.receiver;
            let mut status = receiver.fence.status.subscribe();
            if matches!(
                *status.borrow(),
                SessionStatus::Closed | SessionStatus::Failed(_)
            ) {
                return None;
            }
            let coordinator = &self.coordinator;
            let boundary = receiver.fence.urgent.borrow().boundary;
            if let Some(boundary) = boundary {
                let _ = coordinator.boundary(boundary);
            }
            let event = tokio::select! {
                biased;
                changed = status.changed() => {
                    if changed.is_err()
                        || matches!(
                            *status.borrow(),
                            SessionStatus::Closed | SessionStatus::Failed(_)
                        )
                    {
                        return None;
                    }
                    continue;
                },
                event = receiver.rx.recv() => event?,
            };
            match event {
                RawEvent::Audio { generation, chunk } => {
                    let coordinator = &self.coordinator;
                    if coordinator.admission().is_closed()
                        || generation < coordinator.current_generation()
                    {
                        continue;
                    }
                    let work = coordinator.receipt_stamped(coordinator.stamp(), generation);
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
                    let _ = coordinator.boundary(generation);
                    return Some(PlayerEvent::Clear);
                }
                RawEvent::End(generation) => {
                    let coordinator = &self.coordinator;
                    let _ = coordinator.boundary(generation);
                    return Some(PlayerEvent::End);
                }
            }
        }
    }
    /// Subscribes to retained lifecycle status without polling `recv`.
    pub fn status(&self) -> watch::Receiver<SessionStatus> {
        self.receiver.fence.status.subscribe()
    }
    /// Subscribes to the latest applied control state.
    pub fn controls(&self) -> watch::Receiver<AppliedControls> {
        self.receiver.fence.controls.subscribe()
    }
    /// Applies the requested fields while output is installed or detached.
    pub fn update_controls(&self, update: ControlUpdate) -> Result<(), OutputError> {
        self.coordinator.apply_controls_transaction(|_| {
            let mut captured = None;
            self.receiver.fence.urgent.send_modify(|urgent| {
                let pending = urgent.commands;
                urgent.version = urgent.version.wrapping_add(1);
                captured = Some((
                    ControlUpdate {
                        volume: update.volume.or(pending.volume),
                        muted: update.muted.or(pending.muted),
                        static_delay_ms: update.static_delay_ms.or(pending.static_delay_ms),
                    },
                    urgent.version,
                ));
                urgent.commands = ControlUpdate::default();
            });
            captured
        })
    }
    /// Applies a complete control snapshot while output is installed or detached.
    pub fn set_controls(&self, controls: AppliedControls) -> Result<(), OutputError> {
        self.update_controls(ControlUpdate {
            volume: Some(controls.volume),
            muted: Some(controls.muted),
            static_delay_ms: Some(controls.static_delay_ms),
        })
    }
    /// Closes the session and joins its control driver.
    pub async fn shutdown(mut self) {
        self.receiver.fence.close();
        self.receiver.rx.close();
        self.coordinator.close();
        if let Some(driver) = self.driver.take() {
            driver.join().await;
        }
    }
}
impl<O: PlayerOutput> Drop for PlayerSession<O> {
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
        fail_volume_after_attach: Arc<AtomicBool>,
        fail_commit: bool,
        mismatch_commit: bool,
        setter_gate: Option<SetterGate>,
        setter_gate_enabled: Arc<AtomicBool>,
        record_static_delay: bool,
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
            let gate = self.setter_gate.clone();
            if self.setter_gate_enabled.load(Ordering::Acquire) {
                if let Some(gate) = gate {
                    gate.0.send(()).unwrap();
                    gate.1.lock().recv().unwrap();
                }
            }
            self.controls.lock().volume = volume;
            self.calls.lock().push(format!("volume:{volume}"));
            if self.fail_volume || self.fail_volume_after_attach.load(Ordering::Acquire) {
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
            if self.record_static_delay {
                self.calls.lock().push(format!("static_delay:{delay_ms}"));
            }
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
            fail_volume_after_attach: Arc::new(AtomicBool::new(false)),
            fail_commit: false,
            mismatch_commit: false,
            setter_gate: None,
            setter_gate_enabled: Arc::new(AtomicBool::new(true)),
            record_static_delay: false,
            controls: Arc::new(parking_lot::Mutex::new(AppliedControls {
                volume: 0,
                muted: false,
                static_delay_ms: 0,
            })),
            calls: Arc::new(parking_lot::Mutex::new(Vec::new())),
        }
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

    #[tokio::test]
    async fn public_set_controls_reapplies_equal_server_value_after_local_change() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let calls = Arc::clone(&output.calls);
        let session = receiver.attach(output, controls());
        session
            .set_controls(AppliedControls {
                volume: 25,
                ..controls()
            })
            .unwrap();
        producer.set_volume(60);
        tokio::task::yield_now().await;
        assert_eq!(session.controls().borrow().volume, 60);
        assert!(
            calls
                .lock()
                .iter()
                .filter(|call| *call == "volume:60")
                .count()
                >= 1
        );
    }

    #[tokio::test]
    async fn public_update_controls_then_server_applies() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let calls = Arc::clone(&output.calls);
        let session = receiver.attach(output, controls());
        calls.lock().clear();
        session
            .update_controls(ControlUpdate {
                volume: Some(25),
                ..ControlUpdate::default()
            })
            .unwrap();
        assert_eq!(calls.lock().as_slice(), ["volume:25"]);
        producer.set_volume(60);
        tokio::task::yield_now().await;
        assert_eq!(session.controls().borrow().volume, 60);
        assert_eq!(calls.lock().as_slice(), ["volume:25", "volume:60"]);
    }

    #[tokio::test]
    async fn public_update_controls_preserves_local_volume_for_mute_and_delay() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let calls = Arc::clone(&output.calls);
        let session = receiver.attach(output, controls());
        calls.lock().clear();
        session
            .update_controls(ControlUpdate {
                volume: Some(25),
                ..ControlUpdate::default()
            })
            .unwrap();
        producer.set_mute(true);
        producer.set_static_delay(60);
        tokio::task::yield_now().await;
        let state = session.controls().borrow().clone();
        assert_eq!(
            (state.volume, state.muted, state.static_delay_ms),
            (25, true, 60)
        );
        assert!(calls.lock().iter().all(|call| call != "volume:60"));
    }

    #[tokio::test]
    async fn public_update_controls_does_not_lose_gated_server_command() {
        let driver_gate = Arc::new(Notify::new());
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let session =
            receiver.attach_with_driver_gate(output(), controls(), Arc::clone(&driver_gate));
        producer.set_volume(60);
        session
            .update_controls(ControlUpdate {
                volume: Some(25),
                ..ControlUpdate::default()
            })
            .unwrap();
        driver_gate.notify_one();
        tokio::task::yield_now().await;
        assert_eq!(session.controls().borrow().volume, 25);
    }

    #[tokio::test]
    async fn public_update_replays_pending_mute_and_delay_without_losing_local_volume() {
        let driver_gate = Arc::new(Notify::new());
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let output_state = Arc::clone(&output.controls);
        let session =
            receiver.attach_with_driver_gate(output, controls(), Arc::clone(&driver_gate));
        let mut controls_rx = session.controls();
        producer.set_mute(true);
        producer.set_static_delay(37);
        session
            .update_controls(ControlUpdate {
                volume: Some(25),
                ..ControlUpdate::default()
            })
            .unwrap();
        assert_eq!(session.controls().borrow().volume, 25);
        driver_gate.notify_one();
        tokio::time::timeout(Duration::from_secs(1), controls_rx.changed())
            .await
            .expect("timed out waiting for gated producer controls")
            .unwrap();
        assert_eq!(
            *controls_rx.borrow(),
            AppliedControls {
                volume: 25,
                muted: true,
                static_delay_ms: 37,
            }
        );
        assert_eq!(*output_state.lock(), *controls_rx.borrow());
    }

    #[tokio::test]
    async fn concurrent_local_updates_serialize_admission_and_preserve_disjoint_fields() {
        let driver_gate = Arc::new(Notify::new());
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let (first_entered_tx, first_entered_rx) = std::sync::mpsc::sync_channel(1);
        let (first_release_tx, first_release_rx) = std::sync::mpsc::channel();
        let mut output = output();
        output.setter_gate = Some(Arc::new((
            first_entered_tx,
            parking_lot::Mutex::new(first_release_rx),
        )));
        let setter_gate_enabled = Arc::clone(&output.setter_gate_enabled);
        output.record_static_delay = true;
        let calls = Arc::clone(&output.calls);
        let actual = Arc::clone(&output.controls);
        output.setter_gate_enabled.store(false, Ordering::Release);
        let session = Arc::new(receiver.attach_with_driver_gate(
            output,
            controls(),
            Arc::clone(&driver_gate),
        ));
        assert_eq!(*session.status().borrow(), SessionStatus::Open);
        calls.lock().clear();
        setter_gate_enabled.store(true, Ordering::Release);

        let first_session = Arc::clone(&session);
        let (first_result_tx, first_result_rx) = std::sync::mpsc::sync_channel(1);
        let first = std::thread::spawn(move || {
            let result = first_session.update_controls(ControlUpdate {
                volume: Some(25),
                ..ControlUpdate::default()
            });
            first_result_tx.send(result).unwrap();
        });
        first_entered_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap();

        let second_session = Arc::clone(&session);
        let (second_started_tx, second_started_rx) = std::sync::mpsc::sync_channel(1);
        let (second_result_tx, second_result_rx) = std::sync::mpsc::sync_channel(1);
        let second = std::thread::spawn(move || {
            second_started_tx.send(()).unwrap();
            let result = second_session.update_controls(ControlUpdate {
                static_delay_ms: Some(37),
                ..ControlUpdate::default()
            });
            second_result_tx.send(result).unwrap();
        });
        second_started_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        assert!(second_result_rx.try_recv().is_err());

        first_release_tx.send(()).unwrap();
        assert!(first_result_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .is_ok());
        assert!(second_result_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .is_ok());
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(*session.status().borrow(), SessionStatus::Open);

        driver_gate.notify_one();
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut controls_rx = session.controls();
            while controls_rx.borrow().volume != 25 || controls_rx.borrow().static_delay_ms != 37 {
                controls_rx.changed().await.unwrap();
            }
        })
        .await
        .expect("timed out waiting for concurrent control updates");
        assert_eq!(
            *actual.lock(),
            AppliedControls {
                volume: 25,
                muted: false,
                static_delay_ms: 37,
            }
        );
        assert_eq!(calls.lock().as_slice(), ["volume:25", "static_delay:37"]);
        drop(producer);
    }

    #[tokio::test]
    async fn public_update_failure_is_output_loss_and_keeps_session_alive() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_commit = true;
        let mut session = receiver.attach(failing, controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        assert_eq!(
            work.commit(buffer()),
            Err(CommitRejected::Output(OutputError::Failed("commit".into())))
        );
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::OutputLost(OutputError::Failed("commit".into()))
        );
        session
            .update_controls(ControlUpdate {
                volume: Some(25),
                ..ControlUpdate::default()
            })
            .unwrap();
        producer
            .start(StreamPlayerConfig {
                codec: "pcm".into(),
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            })
            .unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Start(_))));
        session.install(output()).unwrap();
        assert_eq!(*session.status().borrow(), SessionStatus::Open);
    }

    #[tokio::test]
    async fn blocked_output_creation_does_not_block_controls_or_end() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
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
            permit.install(output())
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
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        assert_eq!(work.timestamp(), 123);
        assert_eq!(work.data(), &[0x01, 0x02, 0x03]);
        assert_eq!(work.chunk().data.as_ref(), &[0x01, 0x02, 0x03]);
        let first = output();
        session.install(first.clone()).unwrap();
        work.commit(buffer()).unwrap();
        assert_eq!(first.commits.load(Ordering::SeqCst), 1);
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(old_work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        session.release_output().unwrap();
        assert_eq!(old_work.commit(buffer()), Err(CommitRejected::Obsolete));
        session.install(output()).unwrap();
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(new_work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        new_work.commit(buffer()).unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn command_between_attach_snapshot_and_driver_start_updates_output_and_controls() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let output = output();
        let applied = Arc::clone(&output.controls);
        let session = receiver.attach(output, controls());
        producer.set_volume(42);
        for _ in 0..100 {
            if applied.lock().volume == 42 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(applied.lock().volume, 42);
        assert_eq!(session.controls().borrow().volume, 42);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eager_attach_preserves_controls_published_during_initial_setters() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mut output = output();
        let actual = Arc::clone(&output.controls);
        let setter_gate_enabled = Arc::clone(&output.setter_gate_enabled);
        output.setter_gate = Some(Arc::new((entered_tx, parking_lot::Mutex::new(release_rx))));
        let handle = tokio::runtime::Handle::current();
        let attach = std::thread::spawn(move || {
            let _entered = handle.enter();
            receiver.attach(output, controls())
        });
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("initial setter must block attach");
        producer.set_mute(true);
        producer.set_volume(0);
        setter_gate_enabled.store(false, Ordering::Release);
        release_tx.send(()).unwrap();
        let mut session = attach.join().unwrap();
        tokio::task::yield_now().await;
        assert_eq!(
            *session.controls().borrow(),
            AppliedControls {
                volume: 0,
                muted: true,
                static_delay_ms: 0,
            }
        );
        assert_eq!(*actual.lock(), *session.controls().borrow());
        producer
            .start(StreamPlayerConfig {
                codec: "pcm".into(),
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            })
            .unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Start(_))));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_install_does_not_regress_controls_published_without_output() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        producer.set_volume(20);
        let session = receiver.attach_empty::<FakeOutput>(controls());
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
        permit.install(installed).unwrap();
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
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        session.install(failing).unwrap();
        producer.audio(chunk()).unwrap();
        producer.clear().unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Clear)));
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::OutputLost(OutputError::Failed("clear".into()))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_driver_rejects_stale_snapshot_after_newer_install_version() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        producer.set_volume(20);
        let gate = Arc::new(Notify::new());
        let session = receiver.attach_empty_with_driver_gate(controls(), Arc::clone(&gate));
        let permit = session.begin_output_install().unwrap();
        producer.set_volume(50);
        let installed = output();
        let calls = Arc::clone(&installed.calls);
        permit.install(installed).unwrap();
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
        let session = receiver.attach_empty::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        producer.set_volume(20);
        let installed = output();
        let applied = Arc::clone(&installed.controls);
        permit.install(installed).unwrap();
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
        assert!(matches!(session.recv().await, Some(PlayerEvent::Clear)));
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::OutputLost(OutputError::Failed("clear".into()))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eager_recv_bridges_format_mismatch_clear_to_output_loss() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.clear_format_mismatch = true;
        let mut session = receiver.attach(failing, controls());
        producer.clear().unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Clear)));
        assert!(matches!(
            *session.status().borrow(),
            SessionStatus::OutputLost(OutputError::FormatMismatch { .. })
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
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let mut mismatch = output();
        mismatch.mismatch_commit = true;
        session.install(mismatch).unwrap();
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
        assert!(matches!(*status.borrow(), SessionStatus::OutputLost(_)));
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
        let session = receiver.attach_empty::<FakeOutput>(controls());
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
        permit.install(installed).unwrap();
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
    async fn public_clear_failure_and_setter_failure_are_output_loss() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing_clear = output();
        failing_clear.fail_clear = true;
        let session = receiver.attach(failing_clear, controls());
        let mut status = session.status();
        producer.clear().unwrap();
        while matches!(*status.borrow(), SessionStatus::Open) {
            status.changed().await.unwrap();
        }
        assert!(matches!(*status.borrow(), SessionStatus::OutputLost(_)));

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
        assert!(matches!(*status.borrow(), SessionStatus::OutputLost(_)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_work_keeps_generation_across_synchronous_clear() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(old_work) = session.recv().await.unwrap() else {
            panic!("audio")
        };
        producer.clear().unwrap();
        assert_eq!(old_work.commit(buffer()), Err(CommitRejected::Obsolete));
        assert!(matches!(session.recv().await, Some(PlayerEvent::Clear)));
        session.install(output()).unwrap();
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(new_work) = session.recv().await.unwrap() else {
            panic!("post-clear audio")
        };
        assert_eq!(new_work.commit(buffer()), Ok(()));
    }

    #[tokio::test]
    async fn reservation_identity_preserves_new_install_after_old_permit_drop_or_failure() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let old = session.begin_output_install().unwrap();
        session.release_output().unwrap();
        let new = session.begin_output_install().unwrap();
        drop(old);
        new.install(output()).unwrap();

        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let old = session.begin_output_install().unwrap();
        session.release_output().unwrap();
        let new = session.begin_output_install().unwrap();
        assert_eq!(old.install(output()), Err(InstallRejected::Superseded));
        new.install(output()).unwrap();
    }

    #[test]
    fn concurrent_begin_and_install_complete_without_lock_cycle() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let coordinator = Arc::new(crate::audio::player_session::Coordinator::<
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
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        producer.close();
        assert_eq!(permit.install(output()), Err(InstallRejected::Closed));
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
    async fn failed_commit_late_output_loss_is_retained_and_shutdown_wakes_recv() {
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
            SessionStatus::OutputLost(OutputError::Failed("commit".into()))
        );
        producer
            .start(StreamPlayerConfig {
                codec: "pcm".into(),
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            })
            .unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Start(_))));
        session.install(output()).unwrap();
        assert_eq!(*session.status().borrow(), SessionStatus::Open);
        session.shutdown().await;
        assert_eq!(*status.borrow(), SessionStatus::Closed);
    }

    #[tokio::test]
    async fn output_loss_keeps_session_open_and_reinstalls_fresh_output() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_commit = true;
        let mut session = receiver.attach(failing, controls());
        producer.audio(chunk()).unwrap();
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("audio event")
        };
        let PlayerEvent::Audio(stale) = session.recv().await.unwrap() else {
            panic!("stale audio event")
        };
        assert_eq!(
            work.commit(buffer()),
            Err(CommitRejected::Output(OutputError::Failed("commit".into())))
        );
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::OutputLost(OutputError::Failed("commit".into()))
        );
        assert_eq!(stale.commit(buffer()), Err(CommitRejected::Obsolete));
        producer
            .start(StreamPlayerConfig {
                codec: "pcm".into(),
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            })
            .unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Start(_))));
        session.install(output()).unwrap();
        assert_eq!(*session.status().borrow(), SessionStatus::Open);
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("fresh audio event")
        };
        assert_eq!(work.commit(buffer()), Ok(()));
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
        let session = receiver.attach_empty::<FakeOutput>(controls());
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
        let session = receiver.attach_empty::<FakeOutput>(controls());
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
        let session = receiver.attach_empty::<FakeOutput>(controls());
        let mut controls_watch = session.controls();
        while controls_watch.borrow().volume != 37 {
            controls_watch.changed().await.unwrap();
        }
        let permit = session.begin_output_install().unwrap();
        permit.install(output()).unwrap();
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
        let session = receiver.attach_empty::<FakeOutput>(controls());
        assert!(session.begin_output_install().is_ok());
        drop(session);

        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
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
    async fn lazy_clear_failure_before_recv_publishes_output_loss() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_clear = true;
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        session.install(failing).unwrap();
        producer.clear().unwrap();
        assert!(matches!(session.recv().await, Some(PlayerEvent::Clear)));
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::OutputLost(OutputError::Failed("clear".into()))
        );
    }

    #[tokio::test]
    async fn lazy_install_setter_failure_publishes_output_loss() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let mut failing = output();
        failing.fail_volume = true;
        assert_eq!(session.install(failing), Err(InstallRejected::Failed));
        assert!(matches!(
            *session.status().borrow(),
            SessionStatus::OutputLost(OutputError::Failed(ref error)) if error == "volume"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_install_clears_queued_audio_after_end_before_permit() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        producer.audio(chunk()).unwrap();
        producer.end().unwrap();
        let installed = output();
        let clears = Arc::clone(&installed.clears);
        permit.install(installed).unwrap();
        assert_eq!(clears.load(Ordering::SeqCst), 1);
        assert!(matches!(session.recv().await, Some(PlayerEvent::End)));
        assert!(session.recv().now_or_never().is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lazy_install_reconciles_end_while_setter_is_blocked() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        producer.audio(chunk()).unwrap();
        let installed = output();
        let clears = Arc::clone(&installed.clears);
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mut gated = installed.clone();
        gated.setter_gate = Some(Arc::new((entered_tx, parking_lot::Mutex::new(release_rx))));
        let install = std::thread::spawn(move || permit.install(gated));
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
    async fn lazy_release_clear_failure_retains_output_loss() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let mut failing = output();
        failing.fail_clear = true;
        session.install(failing).unwrap();
        assert_eq!(
            session.release_output(),
            Err(OutputError::Failed("clear".into()))
        );
        assert_eq!(
            *session.status().borrow(),
            SessionStatus::OutputLost(OutputError::Failed("clear".into()))
        );
    }

    #[tokio::test]
    async fn public_loss_invalidates_two_held_works_before_reinstall() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_commit = true;
        let mut session = receiver.attach(failing, controls());
        producer.audio(chunk()).unwrap();
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(first) = session.recv().await.unwrap() else {
            panic!()
        };
        let PlayerEvent::Audio(second) = session.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(
            first.commit(buffer()),
            Err(CommitRejected::Output(OutputError::Failed("commit".into())))
        );
        session.install(output()).unwrap();
        assert_eq!(second.commit(buffer()), Err(CommitRejected::Obsolete));
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(fresh) = session.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(fresh.commit(buffer()), Ok(()));
    }

    #[tokio::test]
    async fn public_pending_recv_gets_start_after_commit_loss() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_commit = true;
        let mut session = receiver.attach(failing, controls());
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!()
        };
        let pending = tokio::spawn(async move { session.recv().await });
        assert_eq!(
            work.commit(buffer()),
            Err(CommitRejected::Output(OutputError::Failed("commit".into())))
        );
        producer
            .start(StreamPlayerConfig {
                codec: "pcm".into(),
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            })
            .unwrap();
        let event = timeout(Duration::from_secs(5), pending)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(event, Some(PlayerEvent::Start(_))));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn public_loss_shutdown_before_driver_poll_closes_and_joins() {
        let (receiver, _producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let mut failing = output();
        failing.fail_volume = true;
        assert_eq!(session.install(failing), Err(InstallRejected::Failed));
        assert!(matches!(
            *session.status().borrow(),
            SessionStatus::OutputLost(_)
        ));
        timeout(Duration::from_secs(5), session.shutdown())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn public_failed_setter_replays_requested_controls_after_reinstall() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let failing = output();
        failing
            .fail_volume_after_attach
            .store(true, Ordering::Release);
        let mut session = receiver.attach(
            failing,
            AppliedControls {
                volume: 50,
                muted: false,
                static_delay_ms: 0,
            },
        );
        let controls_watch = session.controls();
        producer.set_volume(70);
        timeout(Duration::from_secs(5), async {
            while !matches!(*session.status().borrow(), SessionStatus::OutputLost(_)) {
                session.status().changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert_eq!(controls_watch.borrow().volume, 50);
        assert!(!controls_watch.borrow().muted);
        producer.set_mute(true);
        assert_eq!(controls_watch.borrow().volume, 50);
        assert!(!controls_watch.borrow().muted);
        let replacement = output();
        let replacement_state = replacement.controls.clone();
        session.install(replacement).unwrap();
        assert_eq!(replacement_state.lock().volume, 70);
        assert!(replacement_state.lock().muted);
        assert_eq!(controls_watch.borrow().volume, 70);
        assert!(controls_watch.borrow().muted);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pending_volume_patch_overrides_failed_desired_volume_on_install() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let failing = output();
        failing
            .fail_volume_after_attach
            .store(true, Ordering::Release);
        let mut session = receiver.attach(
            failing,
            AppliedControls {
                volume: 50,
                muted: false,
                static_delay_ms: 0,
            },
        );
        producer.set_volume(70);
        timeout(Duration::from_secs(5), async {
            while !matches!(*session.status().borrow(), SessionStatus::OutputLost(_)) {
                session.status().changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        producer.set_volume(50);
        let replacement = output();
        let replacement_state = replacement.controls.clone();
        session.install(replacement).unwrap();
        assert_eq!(replacement_state.lock().volume, 50);
        assert_eq!(session.controls().borrow().volume, 50);
    }

    #[tokio::test]
    async fn public_pending_boundary_loss_keeps_install_recovery_available() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_clear = true;
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        session.install(failing).unwrap();
        producer.clear().unwrap();
        let permit = session.begin_output_install().unwrap();
        assert!(matches!(
            *session.status().borrow(),
            SessionStatus::OutputLost(_)
        ));
        permit.install(output()).unwrap();
        assert_eq!(*session.status().borrow(), SessionStatus::Open);
    }

    #[tokio::test]
    async fn core_install_and_close_publication_are_monotonic() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut session = receiver.attach_empty::<FakeOutput>(controls());
        let status = session.status();
        session.install(output()).unwrap();
        assert_eq!(*status.borrow(), SessionStatus::Open);
        producer.close();
        session.shutdown().await;
        assert_eq!(*status.borrow(), SessionStatus::Closed);
    }

    #[tokio::test]
    async fn release_clear_failure_advances_incarnation_once_and_retains_loss() {
        let c = Coordinator::<OutputAdapterHandle<FakeOutput>>::new(controls());
        let held = c.receipt().unwrap();
        let mut failing = output();
        failing.fail_clear = true;
        c.begin_install()
            .unwrap()
            .install(OutputAdapterHandle(failing))
            .unwrap();
        let initial = c.incarnation();
        let expected = initial.advance();
        let error = c.release().unwrap_err();
        assert_eq!(error, OutputError::Failed("clear".into()));
        assert_eq!(c.incarnation(), expected);
        assert_eq!(c.status(), SessionStatus::OutputLost(error.clone()));
        assert_eq!(held.commit(buffer()), Err(CommitRejected::Obsolete));
        c.begin_install()
            .unwrap()
            .install(OutputAdapterHandle(output()))
            .unwrap();
        assert_eq!(c.status(), SessionStatus::Open);
        assert_eq!(c.incarnation(), expected);
    }

    #[tokio::test]
    async fn pending_recv_wakes_with_start_after_loss_and_reinstall() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let mut failing = output();
        failing.fail_commit = true;
        let mut session = receiver.attach(failing, controls());
        let mut pending = Box::pin(session.recv());
        assert!(matches!(
            futures_util::poll!(&mut pending),
            std::task::Poll::Pending
        ));
        drop(pending);
        producer.audio(chunk()).unwrap();
        let PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!("expected audio work")
        };
        assert_eq!(
            work.commit(buffer()),
            Err(CommitRejected::Output(OutputError::Failed("commit".into())))
        );
        assert!(matches!(
            *session.status().borrow(),
            SessionStatus::OutputLost(_)
        ));
        producer
            .start(StreamPlayerConfig {
                codec: "pcm".into(),
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            })
            .unwrap();
        session.install(output()).unwrap();
        let event = timeout(Duration::from_secs(5), session.recv())
            .await
            .expect("recovered recv must not remain pending")
            .expect("recovered recv must yield queued start");
        assert!(matches!(event, PlayerEvent::Start(_)));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn public_install_cannot_reopen_after_concurrent_close() {
        let (receiver, producer) = PlayerReceiver::channel(controls());
        let session = receiver.attach_empty::<FakeOutput>(controls());
        let permit = session.begin_output_install().unwrap();
        let (entered_tx, entered_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let mut gated = output();
        gated.setter_gate = Some(Arc::new((entered_tx, parking_lot::Mutex::new(release_rx))));
        let install = std::thread::spawn(move || permit.install(gated));
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        producer.close();
        release_tx.send(()).unwrap();
        assert_eq!(install.join().unwrap(), Err(InstallRejected::Closed));
        assert_eq!(*session.status().borrow(), SessionStatus::Closed);
    }
}
