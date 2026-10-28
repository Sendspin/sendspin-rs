use crate::audio::{CommitRejected, OutputError};
use crate::generation::StreamGeneration;
use crate::protocol::client::{ArtworkChunk, VisualizerChunk};
use crate::protocol::messages::{StreamArtworkConfig, StreamVisualizerConfig};
use crate::role_core::{self, BoundaryAction, Coordinator, ProducerError, RoleAdapter};
use parking_lot::Mutex;
use std::fmt;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Receives encoded artwork frames and stream boundary notifications.
pub trait ArtworkOutput: Clone + Send + Sync + 'static {
    /// Prepared representation committed for an artwork frame.
    type Prepared: Send + 'static;
    /// Commits a frame for the specified artwork channel.
    ///
    /// Calls are serialized, short, nonblocking, and must not reenter the session.
    fn commit(&self, channel: u8, prepared: Self::Prepared) -> Result<(), OutputError>;
    /// Ends the current artwork stream. Cleanup on close or failure is attempted once
    /// and is best-effort; the first failure remains terminal.
    ///
    /// Calls are serialized, short, nonblocking, and must not reenter the session.
    fn end(&self) -> Result<(), OutputError>;
}
/// Receives encoded visualizer frames and stream boundary notifications.
pub trait VisualizerOutput: Clone + Send + Sync + 'static {
    /// Prepared representation committed for a visualizer frame.
    type Prepared: Send + 'static;
    /// Commits a prepared visualizer frame.
    ///
    /// Calls are serialized, short, nonblocking, and must not reenter the session.
    fn commit(&self, prepared: Self::Prepared) -> Result<(), OutputError>;
    /// Clears the current visualizer output.
    ///
    /// Calls are serialized, short, nonblocking, and must not reenter the session.
    fn clear(&self) -> Result<(), OutputError>;
    /// Ends the current visualizer stream. Cleanup on close or failure is attempted once
    /// and is best-effort; the first failure remains terminal.
    ///
    /// Calls are serialized, short, nonblocking, and must not reenter the session.
    fn end(&self) -> Result<(), OutputError>;
}
struct ArtAdapter<O: ArtworkOutput>(O);
impl<O: ArtworkOutput> RoleAdapter for ArtAdapter<O> {
    type Prepared = (u8, O::Prepared);
    fn commit(&mut self, p: (u8, O::Prepared)) -> Result<(), OutputError> {
        self.0.commit(p.0, p.1)
    }
    fn boundary(&mut self, a: BoundaryAction) -> Result<(), OutputError> {
        if a == BoundaryAction::End {
            self.0.end()
        } else {
            Ok(())
        }
    }
    fn cleanup(&mut self) -> Result<(), OutputError> {
        self.0.end()
    }
}
struct VizAdapter<O: VisualizerOutput>(O);
impl<O: VisualizerOutput> RoleAdapter for VizAdapter<O> {
    type Prepared = O::Prepared;
    fn commit(&mut self, p: O::Prepared) -> Result<(), OutputError> {
        self.0.commit(p)
    }
    fn boundary(&mut self, a: BoundaryAction) -> Result<(), OutputError> {
        match a {
            BoundaryAction::Clear => self.0.clear(),
            BoundaryAction::End => self.0.end(),
        }
    }
    fn cleanup(&mut self) -> Result<(), OutputError> {
        self.0.end()
    }
}

enum ArtRaw {
    Start(StreamArtworkConfig),
    Data(ArtworkChunk, StreamGeneration),
    End(StreamGeneration),
}
enum VizRaw {
    Start(StreamVisualizerConfig),
    Data(VisualizerChunk, StreamGeneration),
    Clear(StreamGeneration),
    End(StreamGeneration),
}

/// Receives artwork stream data and attaches an [`ArtworkOutput`].
pub struct ArtworkReceiver {
    rx: mpsc::UnboundedReceiver<ArtRaw>,
    urgent: Arc<tokio::sync::watch::Sender<role_core::Urgent>>,
    terminated: Arc<std::sync::atomic::AtomicBool>,
}
/// Produces artwork stream events for an attached receiver.
pub(crate) struct ArtworkProducer {
    tx: mpsc::UnboundedSender<ArtRaw>,
    generation: Arc<Mutex<StreamGeneration>>,
    urgent: Arc<tokio::sync::watch::Sender<role_core::Urgent>>,
    terminated: Arc<std::sync::atomic::AtomicBool>,
}
/// Consumes artwork events and commits prepared frames.
pub struct ArtworkSession<O: ArtworkOutput> {
    rx: mpsc::UnboundedReceiver<ArtRaw>,
    coordinator: Coordinator<ArtAdapter<O>>,
    driver: Option<tokio::task::JoinHandle<()>>,
}
type CommitCapability<P> = Box<dyn FnOnce(P) -> Result<(), CommitRejected> + Send>;

/// Artwork frame data awaiting a single commit.
pub struct ArtworkWork<P> {
    commit: Option<CommitCapability<P>>,
    chunk: ArtworkChunk,
}
impl<P> fmt::Debug for ArtworkWork<P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArtworkWork")
            .field("channel", &self.channel())
            .field("timestamp", &self.timestamp())
            .field("data_len", &self.data().len())
            .field("is_clear", &self.is_clear())
            .finish()
    }
}

/// Event received from an artwork session.
pub enum ArtworkEvent<P: Send + 'static> {
    /// Starts an artwork stream with the supplied configuration.
    Start(StreamArtworkConfig),
    /// Provides one encoded artwork frame.
    Data(ArtworkWork<P>),
    /// Ends the current artwork stream.
    End,
}

impl ArtworkReceiver {
    /// Creates a receiver and producer pair for an artwork stream.
    pub(crate) fn channel() -> (Self, ArtworkProducer) {
        let (tx, rx) = role_core::channel();
        let (g, _) = tokio::sync::watch::channel(role_core::Urgent::default());
        let terminated = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let generation = Arc::new(Mutex::new(StreamGeneration::initial()));
        let p = ArtworkProducer {
            tx: tx.clone(),
            generation: Arc::clone(&generation),
            urgent: Arc::new(g),
            terminated: Arc::clone(&terminated),
        };
        (
            Self {
                rx,
                urgent: Arc::clone(&p.urgent),
                terminated,
            },
            p,
        )
    }
    /// Attaches the output and starts the independent urgent-control driver on the
    /// current Tokio runtime. Output calls are serialized, short, nonblocking, and
    /// must not reenter the session.
    pub fn attach<O: ArtworkOutput>(self, output: O) -> ArtworkSession<O> {
        let c = Coordinator::with_urgent(
            ArtAdapter(output),
            (*self.urgent).clone(),
            Arc::clone(&self.terminated),
        );
        if let Err(error) = c.reconcile_urgent() {
            c.fail(error);
        }
        if self.urgent.borrow().closed || self.terminated.load(std::sync::atomic::Ordering::Acquire)
        {
            c.close();
        }
        let d = driver(Arc::clone(&c.shared), false);
        ArtworkSession {
            rx: self.rx,
            coordinator: c,
            driver: Some(d),
        }
    }
}
impl ArtworkProducer {
    #[cfg(test)]
    pub(crate) fn set_generation_max_for_test(&self) {
        *self.generation.lock() = StreamGeneration::max_for_test();
    }
    fn is_terminated(&self) -> bool {
        self.terminated.load(std::sync::atomic::Ordering::Acquire)
    }
    /// Sends a stream start event.
    pub(crate) fn start(&self, c: StreamArtworkConfig) -> Result<(), ProducerError> {
        if self.is_terminated() {
            return Err(ProducerError::ReceiverClosed);
        }
        self.tx
            .send(ArtRaw::Start(c))
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    /// Sends one encoded artwork frame.
    pub(crate) fn data(&self, c: ArtworkChunk) -> Result<(), ProducerError> {
        if self.is_terminated() {
            return Err(ProducerError::ReceiverClosed);
        }
        let generation = self.current_generation();
        self.tx
            .send(ArtRaw::Data(c, generation))
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    pub(crate) fn end(&self) -> Result<(), ProducerError> {
        if self.is_terminated() {
            return Err(ProducerError::ReceiverClosed);
        }
        let generation = match role_core::advance(&self.generation) {
            Ok(generation) => generation,
            Err(error) => {
                self.close();
                return Err(error);
            }
        };
        role_core::publish_boundary(&self.urgent, BoundaryAction::End, generation);
        self.tx
            .send(ArtRaw::End(generation))
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    fn current_generation(&self) -> StreamGeneration {
        *self.generation.lock()
    }
    pub(crate) fn close(&self) {
        self.terminated
            .store(true, std::sync::atomic::Ordering::Release);
        self.urgent.send_modify(|urgent| urgent.closed = true);
    }
}
impl Drop for ArtworkProducer {
    fn drop(&mut self) {
        self.close();
    }
}
impl<O: ArtworkOutput> ArtworkSession<O> {
    /// Subscribes to retained lifecycle status without polling `recv`.
    pub fn status(&self) -> tokio::sync::watch::Receiver<crate::audio::SessionStatus> {
        self.coordinator.status_watch()
    }
    /// Waits for the next artwork event; returns `None` after shutdown or output failure.
    ///
    /// The wait is cancellation-safe: cancellation does not consume the next event.
    /// Terminal status returns `None` without draining queued events; subsequent calls
    /// continue to return `None`.
    pub async fn recv(&mut self) -> Option<ArtworkEvent<O::Prepared>> {
        loop {
            let mut status = self.coordinator.status_watch();
            if self.coordinator.terminal() {
                return None;
            }
            if let Err(error) = self.coordinator.reconcile_urgent() {
                self.coordinator.fail(error);
                return None;
            }
            let r = tokio::select! {
                biased;
                changed = status.changed() => {
                    if changed.is_err() || !matches!(*status.borrow(), crate::audio::SessionStatus::Open) { return None; }
                    return None;
                },
                r = self.rx.recv() => r?,
            };
            if self.coordinator.terminal() {
                return None;
            }
            return match r {
                ArtRaw::Start(c) => Some(ArtworkEvent::Start(c)),
                ArtRaw::Data(c, generation) => {
                    if generation < self.coordinator.current_generation() {
                        continue;
                    }
                    let channel = c.channel;
                    let work = self.coordinator.work(generation);
                    Some(ArtworkEvent::Data(ArtworkWork {
                        commit: Some(Box::new(move |prepared| work.commit((channel, prepared)))),
                        chunk: c,
                    }))
                }
                ArtRaw::End(generation) => {
                    if let Err(error) = self.coordinator.boundary(generation, BoundaryAction::End) {
                        self.coordinator.fail(error);
                        return None;
                    }
                    Some(ArtworkEvent::End)
                }
            };
        }
    }
    /// Stops the session, closes pending input, and releases the attached output.
    pub async fn shutdown(mut self) {
        self.coordinator.close();
        self.rx.close();
        if let Some(driver) = self.driver.take() {
            let _ = driver.await;
        }
    }
}
impl<O: ArtworkOutput> Drop for ArtworkSession<O> {
    fn drop(&mut self) {
        self.coordinator.close();
        let _ = self.driver.take();
    }
}
impl<P> ArtworkWork<P> {
    /// Returns the artwork channel identified by this frame.
    pub fn channel(&self) -> u8 {
        self.chunk.channel
    }
    /// Returns the server-time timestamp in microseconds.
    pub fn timestamp(&self) -> i64 {
        self.chunk.timestamp
    }
    /// Returns whether this frame clears the channel.
    pub fn is_clear(&self) -> bool {
        self.chunk.is_clear()
    }
    /// Returns the encoded artwork data.
    pub fn data(&self) -> &[u8] {
        &self.chunk.data
    }
    /// Commits the frame, or rejects it if obsolete, output-rejected, or terminal.
    ///
    /// The work is one-shot; dropping it is a no-op for session state.
    pub fn commit(mut self, prepared: P) -> Result<(), CommitRejected> {
        self.commit.take().expect("work commits once")(prepared)
    }
}

impl<P: Send + 'static> fmt::Debug for ArtworkEvent<P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start(config) => formatter.debug_tuple("Start").field(config).finish(),
            Self::Data(work) => formatter.debug_tuple("Data").field(work).finish(),
            Self::End => formatter.write_str("End"),
        }
    }
}

/// Receives visualizer stream data and attaches a [`VisualizerOutput`].
pub struct VisualizerReceiver {
    rx: mpsc::UnboundedReceiver<VizRaw>,
    urgent: Arc<tokio::sync::watch::Sender<role_core::Urgent>>,
    terminated: Arc<std::sync::atomic::AtomicBool>,
}
/// Produces visualizer stream events for an attached receiver.
pub(crate) struct VisualizerProducer {
    tx: mpsc::UnboundedSender<VizRaw>,
    generation: Arc<Mutex<StreamGeneration>>,
    urgent: Arc<tokio::sync::watch::Sender<role_core::Urgent>>,
    terminated: Arc<std::sync::atomic::AtomicBool>,
}
/// Consumes visualizer events and commits prepared frames.
pub struct VisualizerSession<O: VisualizerOutput> {
    rx: mpsc::UnboundedReceiver<VizRaw>,
    coordinator: Coordinator<VizAdapter<O>>,
    driver: Option<tokio::task::JoinHandle<()>>,
}
/// Visualizer frame data awaiting a single commit.
pub struct VisualizerWork<P> {
    commit: Option<CommitCapability<P>>,
    chunk: VisualizerChunk,
}
impl<P> fmt::Debug for VisualizerWork<P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VisualizerWork")
            .field("type_id", &self.type_id())
            .field("timestamp", &self.timestamp())
            .field("data_len", &self.data().len())
            .finish()
    }
}

/// Event received from a visualizer session.
pub enum VisualizerEvent<P: Send + 'static> {
    /// Starts a visualizer stream with the supplied configuration.
    Start(StreamVisualizerConfig),
    /// Provides one encoded visualizer frame.
    Data(VisualizerWork<P>),
    /// Clears the current visualizer output.
    Clear,
    /// Ends the current visualizer stream.
    End,
}
impl VisualizerReceiver {
    /// Creates a receiver and producer pair for a visualizer stream.
    pub(crate) fn channel() -> (Self, VisualizerProducer) {
        let (tx, rx) = role_core::channel();
        let (g, _) = tokio::sync::watch::channel(role_core::Urgent::default());
        let generation = Arc::new(Mutex::new(StreamGeneration::initial()));
        let terminated = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let p = VisualizerProducer {
            tx: tx.clone(),
            generation: Arc::clone(&generation),
            urgent: Arc::new(g),
            terminated: Arc::clone(&terminated),
        };
        (
            Self {
                rx,
                urgent: Arc::clone(&p.urgent),
                terminated,
            },
            p,
        )
    }
    /// Attaches the output and starts the independent urgent-control driver on the
    /// current Tokio runtime. Output calls are serialized, short, nonblocking, and
    /// must not reenter the session.
    pub fn attach<O: VisualizerOutput>(self, output: O) -> VisualizerSession<O> {
        let c = Coordinator::with_urgent(
            VizAdapter(output),
            (*self.urgent).clone(),
            Arc::clone(&self.terminated),
        );
        if let Err(error) = c.reconcile_urgent() {
            c.fail(error);
        }
        if self.urgent.borrow().closed || self.terminated.load(std::sync::atomic::Ordering::Acquire)
        {
            c.close();
        }
        let d = driver(Arc::clone(&c.shared), true);
        VisualizerSession {
            rx: self.rx,
            coordinator: c,
            driver: Some(d),
        }
    }
}
impl VisualizerProducer {
    #[cfg(test)]
    pub(crate) fn set_generation_max_for_test(&self) {
        *self.generation.lock() = StreamGeneration::max_for_test();
    }
    fn is_terminated(&self) -> bool {
        self.terminated.load(std::sync::atomic::Ordering::Acquire)
    }
    fn boundary(&self, a: BoundaryAction) -> Result<(), ProducerError> {
        if self.is_terminated() {
            return Err(ProducerError::ReceiverClosed);
        }
        let g = match role_core::advance(&self.generation) {
            Ok(generation) => generation,
            Err(error) => {
                self.close();
                return Err(error);
            }
        };
        role_core::publish_boundary(&self.urgent, a, g);
        self.tx
            .send(match a {
                BoundaryAction::Clear => VizRaw::Clear(g),
                BoundaryAction::End => VizRaw::End(g),
            })
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    /// Sends a stream start event.
    pub(crate) fn start(&self, c: StreamVisualizerConfig) -> Result<(), ProducerError> {
        if self.is_terminated() {
            return Err(ProducerError::ReceiverClosed);
        }
        self.tx
            .send(VizRaw::Start(c))
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    /// Sends one encoded visualizer frame.
    pub(crate) fn data(&self, c: VisualizerChunk) -> Result<(), ProducerError> {
        if self.is_terminated() {
            return Err(ProducerError::ReceiverClosed);
        }
        let generation = *self.generation.lock();
        self.tx
            .send(VizRaw::Data(c, generation))
            .map_err(|_| ProducerError::ReceiverClosed)
    }
    /// Clears the current visualizer output.
    pub(crate) fn clear(&self) -> Result<(), ProducerError> {
        self.boundary(BoundaryAction::Clear)
    }
    /// Ends the current visualizer stream.
    pub(crate) fn end(&self) -> Result<(), ProducerError> {
        self.boundary(BoundaryAction::End)
    }
    pub(crate) fn close(&self) {
        self.terminated
            .store(true, std::sync::atomic::Ordering::Release);
        self.urgent.send_modify(|urgent| urgent.closed = true)
    }
}
impl Drop for VisualizerProducer {
    fn drop(&mut self) {
        self.close();
    }
}
impl<O: VisualizerOutput> VisualizerSession<O> {
    /// Subscribes to retained lifecycle status without polling `recv`.
    pub fn status(&self) -> tokio::sync::watch::Receiver<crate::audio::SessionStatus> {
        self.coordinator.status_watch()
    }
    /// Waits for the next visualizer event; returns `None` after shutdown or output failure.
    ///
    /// The wait is cancellation-safe: cancellation does not consume the next event.
    /// Terminal status returns `None` without draining queued events; subsequent calls
    /// continue to return `None`.
    pub async fn recv(&mut self) -> Option<VisualizerEvent<O::Prepared>> {
        loop {
            let mut status = self.coordinator.status_watch();
            if self.coordinator.terminal() {
                return None;
            }
            if let Err(error) = self.coordinator.reconcile_urgent() {
                self.coordinator.fail(error);
                return None;
            }
            let r = tokio::select! {
                biased;
                changed = status.changed() => {
                    if changed.is_err() || !matches!(*status.borrow(), crate::audio::SessionStatus::Open) { return None; }
                    return None;
                },
                r = self.rx.recv() => r?,
            };
            if self.coordinator.terminal() {
                return None;
            }
            return Some(match r {
                VizRaw::Start(c) => VisualizerEvent::Start(c),
                VizRaw::Data(c, generation) => {
                    if generation < self.coordinator.current_generation() {
                        continue;
                    }
                    let work = self.coordinator.work(generation);
                    VisualizerEvent::Data(VisualizerWork {
                        commit: Some(Box::new(move |prepared| work.commit(prepared))),
                        chunk: c,
                    })
                }
                VizRaw::Clear(generation) => {
                    if let Err(error) = self.coordinator.boundary(generation, BoundaryAction::Clear)
                    {
                        self.coordinator.fail(error);
                        return None;
                    }
                    VisualizerEvent::Clear
                }
                VizRaw::End(generation) => {
                    if let Err(error) = self.coordinator.boundary(generation, BoundaryAction::End) {
                        self.coordinator.fail(error);
                        return None;
                    }
                    VisualizerEvent::End
                }
            });
        }
    }
    /// Stops the session, closes pending input, and releases the attached output.
    pub async fn shutdown(mut self) {
        self.coordinator.close();
        self.rx.close();
        if let Some(driver) = self.driver.take() {
            let _ = driver.await;
        }
    }
}
impl<O: VisualizerOutput> Drop for VisualizerSession<O> {
    fn drop(&mut self) {
        self.coordinator.close();
        let _ = self.driver.take();
    }
}
impl<P> VisualizerWork<P> {
    /// Returns the visualizer data type identifier.
    pub fn type_id(&self) -> u8 {
        self.chunk.type_id
    }
    /// Returns the server-time timestamp in microseconds.
    pub fn timestamp(&self) -> i64 {
        self.chunk.timestamp
    }
    /// Returns the encoded visualizer data.
    pub fn data(&self) -> &[u8] {
        &self.chunk.data
    }
    /// Commits the frame, or rejects it if obsolete, output-rejected, or terminal.
    ///
    /// The work is one-shot; dropping it is a no-op for session state.
    pub fn commit(mut self, prepared: P) -> Result<(), CommitRejected> {
        self.commit.take().expect("work commits once")(prepared)
    }
}
impl<P: Send + 'static> fmt::Debug for VisualizerEvent<P> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start(config) => formatter.debug_tuple("Start").field(config).finish(),
            Self::Data(work) => formatter.debug_tuple("Data").field(work).finish(),
            Self::Clear => formatter.write_str("Clear"),
            Self::End => formatter.write_str("End"),
        }
    }
}

fn driver<A: RoleAdapter>(
    shared: Arc<role_core::Shared<A>>,
    visualizer: bool,
) -> tokio::task::JoinHandle<()> {
    let mut u = shared.urgent.subscribe();
    let mut status = shared.status.subscribe();
    tokio::spawn(async move {
        let mut initial = true;
        loop {
            if !initial {
                tokio::select! {
                    changed = u.changed() => {
                        if changed.is_err() { break; }
                    }
                    changed = status.changed() => {
                        if changed.is_err() || !matches!(*status.borrow(), crate::audio::SessionStatus::Open) {
                            break;
                        }
                    }
                }
            }
            initial = false;
            let s = u.borrow().clone();
            let c = Coordinator {
                shared: Arc::clone(&shared),
            };
            if s.closed {
                c.close();
                break;
            }
            let mut boundaries = Vec::new();
            if visualizer {
                if let Some(g) = s.clear {
                    boundaries.push((g, BoundaryAction::Clear));
                }
            }
            if let Some(g) = s.end {
                boundaries.push((g, BoundaryAction::End));
            }
            boundaries.sort_by_key(|(generation, _)| *generation);
            for (generation, action) in boundaries {
                if let Err(e) = c.boundary(generation, action) {
                    c.fail(e);
                    return;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::messages::{
        ArtworkSource, ImageFormat, StreamArtworkChannelConfig, VisualizerDataType,
    };
    use std::sync::Arc;
    use tokio::sync::Notify;

    struct Prepared(Vec<u8>);
    #[derive(Clone)]
    struct ArtworkFake {
        events: Arc<Mutex<Vec<String>>>,
        boundary: Arc<Notify>,
    }
    impl ArtworkOutput for ArtworkFake {
        type Prepared = Prepared;
        fn commit(&self, channel: u8, p: Prepared) -> Result<(), OutputError> {
            self.events.lock().push(format!("{channel}:{}", p.0.len()));
            Ok(())
        }
        fn end(&self) -> Result<(), OutputError> {
            self.events.lock().push("end".into());
            self.boundary.notify_waiters();
            Ok(())
        }
    }
    #[derive(Clone)]
    struct VisualFake {
        events: Arc<Mutex<Vec<&'static str>>>,
        clear: Arc<Notify>,
    }
    impl VisualizerOutput for VisualFake {
        type Prepared = Prepared;
        fn commit(&self, _: Prepared) -> Result<(), OutputError> {
            self.events.lock().push("data");
            Ok(())
        }
        fn clear(&self) -> Result<(), OutputError> {
            self.events.lock().push("clear");
            self.clear.notify_waiters();
            Ok(())
        }
        fn end(&self) -> Result<(), OutputError> {
            self.events.lock().push("end");
            self.clear.notify_waiters();
            Ok(())
        }
    }
    fn art(data: &[u8]) -> ArtworkChunk {
        ArtworkChunk {
            channel: 0,
            timestamp: 1,
            data: Arc::from(data),
        }
    }
    fn viz(data: &[u8]) -> VisualizerChunk {
        VisualizerChunk {
            type_id: 0x10,
            timestamp: 1,
            data: Arc::from(data),
        }
    }
    fn art_config() -> StreamArtworkConfig {
        StreamArtworkConfig {
            channels: vec![StreamArtworkChannelConfig {
                source: ArtworkSource::Album,
                format: ImageFormat::Png,
                width: 1,
                height: 1,
            }],
        }
    }
    #[derive(Clone)]
    struct FailingArtwork {
        events: Arc<Mutex<Vec<&'static str>>>,
        fail_commit: Option<OutputError>,
        fail_end: Option<OutputError>,
    }
    impl ArtworkOutput for FailingArtwork {
        type Prepared = Prepared;
        fn commit(&self, _: u8, _: Prepared) -> Result<(), OutputError> {
            self.events.lock().push("commit");
            self.fail_commit.clone().map_or(Ok(()), Err)
        }
        fn end(&self) -> Result<(), OutputError> {
            self.events.lock().push("end");
            self.fail_end.clone().map_or(Ok(()), Err)
        }
    }
    #[derive(Clone)]
    struct FailingVisualizer {
        events: Arc<Mutex<Vec<&'static str>>>,
        fail_commit: Option<OutputError>,
        fail_clear: Option<OutputError>,
        fail_end: Option<OutputError>,
        boundary_seen: Option<Arc<Notify>>,
    }
    impl VisualizerOutput for FailingVisualizer {
        type Prepared = Prepared;
        fn commit(&self, _: Prepared) -> Result<(), OutputError> {
            self.events.lock().push("commit");
            self.fail_commit.clone().map_or(Ok(()), Err)
        }
        fn clear(&self) -> Result<(), OutputError> {
            self.events.lock().push("clear");
            if let Some(gate) = &self.boundary_seen {
                gate.notify_waiters();
            }
            self.fail_clear.clone().map_or(Ok(()), Err)
        }
        fn end(&self) -> Result<(), OutputError> {
            self.events.lock().push("end");
            if let Some(gate) = &self.boundary_seen {
                gate.notify_waiters();
            }
            self.fail_end.clone().map_or(Ok(()), Err)
        }
    }
    fn viz_config() -> StreamVisualizerConfig {
        StreamVisualizerConfig {
            types: vec![VisualizerDataType::Loudness],
            rate_max: 1,
            tracks_downbeats: None,
            spectrum: None,
        }
    }

    #[tokio::test]
    async fn artwork_generation_exhaustion_closes_all_role_sinks() {
        let (receiver, producer) = ArtworkReceiver::channel();
        let (other_receiver, other_producer) = VisualizerReceiver::channel();
        let mut artwork = receiver.attach(ArtworkFake {
            events: Arc::new(Mutex::new(Vec::new())),
            boundary: Arc::new(Notify::new()),
        });
        let mut visualizer = other_receiver.attach(VisualFake {
            events: Arc::new(Mutex::new(Vec::new())),
            clear: Arc::new(Notify::new()),
        });
        producer.set_generation_max_for_test();
        assert_eq!(
            producer.end(),
            Err(crate::role_core::ProducerError::GenerationExhausted)
        );
        other_producer.close();
        assert!(artwork.recv().await.is_none());
        assert!(visualizer.recv().await.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn artwork_recv_boundary_failure_returns_none_and_retains_failure_before_driver_poll() {
        let error = OutputError::Failed("end-failed".into());
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(FailingArtwork {
            events: Arc::new(Mutex::new(Vec::new())),
            fail_commit: None,
            fail_end: Some(error.clone()),
        });
        let pending = session.recv();
        producer.end().unwrap();
        assert!(pending.await.is_none());
        assert_eq!(
            *session.status().borrow(),
            crate::audio::SessionStatus::Failed(error.clone())
        );
        assert_eq!(
            producer.start(art_config()),
            Err(crate::role_core::ProducerError::ReceiverClosed)
        );
        assert_eq!(
            producer.data(art(&[2])),
            Err(crate::role_core::ProducerError::ReceiverClosed)
        );
        assert_eq!(
            producer.end(),
            Err(crate::role_core::ProducerError::ReceiverClosed)
        );
        assert_eq!(
            *session.status().borrow(),
            crate::audio::SessionStatus::Failed(error)
        );
    }

    #[tokio::test]
    async fn artwork_recv_skips_queued_old_data_across_boundary_and_start() {
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(ArtworkFake {
            events: Arc::new(Mutex::new(Vec::new())),
            boundary: Arc::new(Notify::new()),
        });
        producer.data(art(&[1])).unwrap();
        producer.end().unwrap();
        producer.start(art_config()).unwrap();
        producer.data(art(&[2])).unwrap();
        assert!(matches!(session.recv().await, Some(ArtworkEvent::End)));
        assert!(matches!(session.recv().await, Some(ArtworkEvent::Start(_))));
        let ArtworkEvent::Data(work) = session.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(work.data(), &[2]);
        work.commit(Prepared(vec![2])).unwrap();
        session.shutdown().await;
    }

    #[tokio::test]
    async fn visualizer_recv_skips_queued_old_data_after_urgent_clear() {
        let (receiver, producer) = VisualizerReceiver::channel();
        let mut session = receiver.attach(VisualFake {
            events: Arc::new(Mutex::new(Vec::new())),
            clear: Arc::new(Notify::new()),
        });
        producer.data(viz(&[1])).unwrap();
        producer.clear().unwrap();
        producer.start(viz_config()).unwrap();
        producer.data(viz(&[2])).unwrap();
        assert!(matches!(session.recv().await, Some(VisualizerEvent::Clear)));
        assert!(matches!(
            session.recv().await,
            Some(VisualizerEvent::Start(_))
        ));
        let VisualizerEvent::Data(work) = session.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(work.data(), &[2]);
        work.commit(Prepared(vec![2])).unwrap();
        session.shutdown().await;
    }

    #[tokio::test]
    async fn artwork_fifo_accepts_nonclone_prepared_and_empty_payload() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(ArtworkFake {
            events: Arc::clone(&events),
            boundary: Arc::new(Notify::new()),
        });
        producer.start(art_config()).unwrap();
        producer.data(art(&[1])).unwrap();
        producer.data(art(&[])).unwrap();
        assert!(matches!(session.recv().await, Some(ArtworkEvent::Start(_))));
        let first = match session.recv().await.unwrap() {
            ArtworkEvent::Data(w) => w,
            _ => panic!(),
        };
        assert_eq!(first.data(), &[1]);
        first.commit(Prepared(vec![1])).unwrap();
        let second = match session.recv().await.unwrap() {
            ArtworkEvent::Data(w) => w,
            _ => panic!(),
        };
        assert!(second.is_clear());
        second.commit(Prepared(Vec::new())).unwrap();
        assert_eq!(&*events.lock(), &["0:1", "0:0"]);
        session.shutdown().await;
    }

    #[tokio::test]
    async fn artwork_end_invalidates_held_work_without_driver_poll() {
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(ArtworkFake {
            events: Arc::new(Mutex::new(Vec::new())),
            boundary: Arc::new(Notify::new()),
        });
        producer.data(art(&[1])).unwrap();
        let ArtworkEvent::Data(work) = session.recv().await.unwrap() else {
            panic!()
        };
        producer.end().unwrap();
        assert_eq!(
            work.commit(Prepared(vec![1])),
            Err(CommitRejected::Obsolete)
        );
        producer.start(art_config()).unwrap();
        assert!(matches!(session.recv().await, Some(ArtworkEvent::End)));
        assert!(matches!(session.recv().await, Some(ArtworkEvent::Start(_))));
        session.shutdown().await;
    }

    #[tokio::test]
    async fn visualizer_generation_exhaustion_closes_held_work() {
        let (receiver, producer) = VisualizerReceiver::channel();
        let mut session = receiver.attach(VisualFake {
            events: Arc::new(Mutex::new(Vec::new())),
            clear: Arc::new(Notify::new()),
        });
        producer.data(viz(&[1])).unwrap();
        let work = match session.recv().await.unwrap() {
            VisualizerEvent::Data(work) => work,
            _ => panic!(),
        };
        producer.set_generation_max_for_test();
        assert_eq!(
            producer.clear(),
            Err(crate::role_core::ProducerError::GenerationExhausted)
        );
        assert_eq!(work.commit(Prepared(vec![1])), Err(CommitRejected::Closed));
        assert!(session.recv().await.is_none());
    }

    #[tokio::test]
    async fn visualizer_clear_invalidates_issued_work_before_recv() {
        let clear = Arc::new(Notify::new());
        let (receiver, producer) = VisualizerReceiver::channel();
        let mut session = receiver.attach(VisualFake {
            events: Arc::new(Mutex::new(Vec::new())),
            clear: Arc::clone(&clear),
        });
        producer.data(viz(&[1])).unwrap();
        let work = match session.recv().await.unwrap() {
            VisualizerEvent::Data(w) => w,
            _ => panic!(),
        };
        producer.clear().unwrap();
        assert_eq!(
            work.commit(Prepared(vec![1])),
            Err(CommitRejected::Obsolete)
        );
        assert!(matches!(session.recv().await, Some(VisualizerEvent::Clear)));
        session.shutdown().await;
    }

    #[tokio::test]
    async fn visualizer_end_then_start_does_not_repeat_end() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let boundary = Arc::new(Notify::new());
        let (receiver, producer) = VisualizerReceiver::channel();
        let mut session = receiver.attach(VisualFake {
            events: Arc::clone(&events),
            clear: Arc::clone(&boundary),
        });
        producer.end().unwrap();
        producer.start(viz_config()).unwrap();
        producer.data(viz(&[2])).unwrap();
        assert!(matches!(session.recv().await, Some(VisualizerEvent::End)));
        assert!(matches!(
            session.recv().await,
            Some(VisualizerEvent::Start(_))
        ));
        let work = match session.recv().await.unwrap() {
            VisualizerEvent::Data(w) => w,
            _ => panic!(),
        };
        work.commit(Prepared(vec![2])).unwrap();
        assert_eq!(&*events.lock(), &["end", "data"]);
        session.shutdown().await;
    }

    #[tokio::test]
    async fn artwork_commit_failure_is_terminal_and_preserves_first_error() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let first = OutputError::Failed("commit-first".into());
        let cleanup = OutputError::Failed("cleanup-second".into());
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(FailingArtwork {
            events: Arc::clone(&events),
            fail_commit: Some(first.clone()),
            fail_end: Some(cleanup),
        });
        producer.data(art(&[1])).unwrap();
        let work = match session.recv().await.unwrap() {
            ArtworkEvent::Data(w) => w,
            _ => panic!(),
        };
        assert_eq!(
            work.commit(Prepared(vec![1])),
            Err(CommitRejected::Output(first.clone()))
        );
        assert_eq!(
            *session.status().borrow(),
            crate::audio::SessionStatus::Failed(first)
        );
        assert!(session.recv().await.is_none());
        assert_eq!(&*events.lock(), &["commit", "end"]);
        session.shutdown().await;
    }

    #[tokio::test]
    async fn artwork_failure_late_status_watch_retains_failure_after_shutdown() {
        let first = OutputError::Failed("commit-first".into());
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(FailingArtwork {
            events: Arc::new(Mutex::new(Vec::new())),
            fail_commit: Some(first.clone()),
            fail_end: Some(OutputError::Failed("cleanup".into())),
        });
        producer.data(art(&[1])).unwrap();
        let ArtworkEvent::Data(work) = session.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(
            work.commit(Prepared(vec![1])),
            Err(CommitRejected::Output(first.clone()))
        );
        let status = session.status();
        assert_eq!(*status.borrow(), crate::audio::SessionStatus::Failed(first));
        session.shutdown().await;
        assert_eq!(
            *status.borrow(),
            crate::audio::SessionStatus::Failed(OutputError::Failed("commit-first".into()))
        );
    }

    #[tokio::test]
    async fn visualizer_commit_failure_is_terminal_and_cleanup_runs_once() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let error = OutputError::Failed("visual-commit".into());
        let (receiver, producer) = VisualizerReceiver::channel();
        let mut session = receiver.attach(FailingVisualizer {
            events: Arc::clone(&events),
            fail_commit: Some(error.clone()),
            fail_clear: None,
            fail_end: Some(OutputError::Failed("cleanup".into())),
            boundary_seen: None,
        });
        producer.data(viz(&[1])).unwrap();
        let work = match session.recv().await.unwrap() {
            VisualizerEvent::Data(w) => w,
            _ => panic!(),
        };
        assert_eq!(
            work.commit(Prepared(vec![1])),
            Err(CommitRejected::Output(error.clone()))
        );
        assert_eq!(
            *session.status().borrow(),
            crate::audio::SessionStatus::Failed(error)
        );
        assert!(session.recv().await.is_none());
        assert_eq!(&*events.lock(), &["commit", "end"]);
        session.shutdown().await;
    }

    #[tokio::test]
    async fn visualizer_boundary_failure_fences_held_work_and_reports_status() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let error = OutputError::Failed("clear-failed".into());
        let boundary_seen = Arc::new(Notify::new());
        let (receiver, producer) = VisualizerReceiver::channel();
        let mut session = receiver.attach(FailingVisualizer {
            events: Arc::clone(&events),
            fail_commit: None,
            fail_clear: Some(error.clone()),
            fail_end: None,
            boundary_seen: Some(Arc::clone(&boundary_seen)),
        });
        producer.data(viz(&[1])).unwrap();
        let work = match session.recv().await.unwrap() {
            VisualizerEvent::Data(w) => w,
            _ => panic!(),
        };
        let boundary_ready = boundary_seen.notified();
        producer.clear().unwrap();
        boundary_ready.await;
        assert_eq!(work.commit(Prepared(vec![1])), Err(CommitRejected::Closed));
        assert_eq!(
            producer.start(viz_config()),
            Err(crate::role_core::ProducerError::ReceiverClosed)
        );
        assert_eq!(
            producer.data(viz(&[2])),
            Err(crate::role_core::ProducerError::ReceiverClosed)
        );
        assert_eq!(
            producer.clear(),
            Err(crate::role_core::ProducerError::ReceiverClosed)
        );
        assert_eq!(
            producer.end(),
            Err(crate::role_core::ProducerError::ReceiverClosed)
        );
        assert_eq!(
            *session.status().borrow(),
            crate::audio::SessionStatus::Failed(error)
        );
        assert!(session.recv().await.is_none());
        assert_eq!(&*events.lock(), &["clear", "end"]);
        session.shutdown().await;
    }

    #[tokio::test]
    async fn failed_artwork_driver_exits_while_session_and_producer_remain_alive() {
        let error = OutputError::Failed("driver-end".into());
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(FailingArtwork {
            events: Arc::new(Mutex::new(Vec::new())),
            fail_commit: None,
            fail_end: Some(error.clone()),
        });
        let mut status = session.status();
        let producer = producer;
        producer.end().unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), status.changed())
            .await
            .expect("driver did not publish failure")
            .unwrap();
        assert_eq!(*status.borrow(), crate::audio::SessionStatus::Failed(error));
        assert!(session
            .driver
            .as_ref()
            .expect("driver handle")
            .is_finished());
        let _producer = producer;
        let _session = &mut session;
    }

    #[tokio::test]
    async fn cancelled_recv_does_not_consume_next_config_or_data() {
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(ArtworkFake {
            events: Arc::new(Mutex::new(Vec::new())),
            boundary: Arc::new(Notify::new()),
        });
        let pending = session.recv();
        drop(pending);
        producer.start(art_config()).unwrap();
        producer.data(art(&[2])).unwrap();
        assert!(matches!(session.recv().await, Some(ArtworkEvent::Start(_))));
        assert!(matches!(session.recv().await, Some(ArtworkEvent::Data(_))));
        session.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_releases_shared_state_and_output_after_driver_finishes() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let (receiver, producer) = ArtworkReceiver::channel();
        let session = receiver.attach(ArtworkFake {
            events,
            boundary: Arc::new(Notify::new()),
        });
        let weak = Arc::downgrade(&session.coordinator.shared);
        drop(producer);
        session.shutdown().await;
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn producer_drop_rejects_held_work_and_wakes_pending_recv() {
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(ArtworkFake {
            events: Arc::new(Mutex::new(Vec::new())),
            boundary: Arc::new(Notify::new()),
        });
        producer.data(art(&[1])).unwrap();
        let work = match session.recv().await.unwrap() {
            ArtworkEvent::Data(w) => w,
            _ => panic!(),
        };
        let pending = session.recv();
        drop(producer);
        assert_eq!(work.commit(Prepared(vec![1])), Err(CommitRejected::Closed));
        assert!(pending.await.is_none());
    }

    #[tokio::test]
    async fn producer_drop_without_recv_ends_output_and_releases_driver() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let (receiver, producer) = ArtworkReceiver::channel();
        let session = receiver.attach(ArtworkFake {
            events: Arc::clone(&events),
            boundary: Arc::new(Notify::new()),
        });
        let weak = Arc::downgrade(&session.coordinator.shared);
        let mut status = session.status();
        drop(producer);
        tokio::time::timeout(std::time::Duration::from_secs(1), status.changed())
            .await
            .expect("status watch must observe producer drop")
            .unwrap();
        assert_eq!(*status.borrow(), crate::audio::SessionStatus::Closed);
        assert_eq!(&*events.lock(), &["end"]);
        drop(session);
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn producer_close_rejects_held_work_without_recv() {
        let (receiver, producer) = ArtworkReceiver::channel();
        let mut session = receiver.attach(ArtworkFake {
            events: Arc::new(Mutex::new(Vec::new())),
            boundary: Arc::new(Notify::new()),
        });
        producer.data(art(&[1])).unwrap();
        let work = match session.recv().await.unwrap() {
            ArtworkEvent::Data(w) => w,
            _ => panic!(),
        };
        producer.close();
        assert_eq!(work.commit(Prepared(vec![1])), Err(CommitRejected::Closed));
        assert!(session.recv().await.is_none());
    }
}
