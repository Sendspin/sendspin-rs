// ABOUTME: WebSocket client implementation for Sendspin protocol
// ABOUTME: Handles connection, message routing, and protocol state machine

use crate::error::Error;
use crate::protocol::messages::{
    ClientCommand, ClientGoodbye, ClientHello, ClientState, ClientTime, ControllerCommand,
    ControllerCommandType, GoodbyeReason, Message, RepeatMode, ServerHello,
};
use crate::sync::raw_clock::Clock;
use crate::sync::ClockSync;
use futures_util::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};

/// `Goodbye` is one variant (not `Send` + `Close`) so the writer processes it
/// atomically: once dequeued it flushes goodbye + close and exits, so nothing
/// *enqueued after it* reaches the wire.
enum WriteCommand {
    Send {
        msg: WsMessage,
        ack: tokio::sync::oneshot::Sender<Result<(), Error>>,
    },
    Goodbye {
        reason: GoodbyeReason,
        ack: tokio::sync::oneshot::Sender<Result<(), Error>>,
    },
}

async fn writer_task<S>(
    mut sink: SplitSink<WebSocketStream<S>, WsMessage>,
    mut rx: UnboundedReceiver<WriteCommand>,
) where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    while let Some(cmd) = rx.recv().await {
        match cmd {
            WriteCommand::Send { msg, ack } => {
                let result = sink
                    .send(msg)
                    .await
                    .map_err(|e| Error::WebSocket(e.to_string()));
                let failed = result.is_err();
                // Ignore SendError: the caller may have dropped its receiver.
                let _ = ack.send(result);
                if failed {
                    break;
                }
            }
            WriteCommand::Goodbye { reason, ack } => {
                let _ = ack.send(perform_goodbye(&mut sink, reason).await);
                break;
            }
        }
    }
    // On exit `rx` drops, dropping the ack sender of any still-queued command;
    // callers awaiting those acks see the cancellation and treat it as a closed
    // connection (see `WsSender::send_message`).
}

async fn perform_goodbye<S>(
    sink: &mut SplitSink<WebSocketStream<S>, WsMessage>,
    reason: GoodbyeReason,
) -> Result<(), Error>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let goodbye = Message::ClientGoodbye(ClientGoodbye { reason });
    let json = serde_json::to_string(&goodbye).map_err(|e| Error::Protocol(e.to_string()))?;
    sink.send(WsMessage::Text(json))
        .await
        .map_err(|e| Error::WebSocket(e.to_string()))?;
    sink.close()
        .await
        .map_err(|e| Error::WebSocket(e.to_string()))
}

/// Connection components returned by [`ProtocolClient::split()`].
/// Use the fields you need; ignore the rest.
pub struct Connection {
    /// Protocol messages from the server
    pub messages: UnboundedReceiver<Message>,
    /// Audio chunks from the server
    pub audio: UnboundedReceiver<AudioChunk>,
    /// Artwork chunks from the server
    pub artwork: UnboundedReceiver<ArtworkChunk>,
    /// Visualizer chunks from the server
    pub visualizer: UnboundedReceiver<VisualizerChunk>,
    /// Clock synchronization state
    pub clock_sync: Arc<Mutex<ClockSync>>,
    /// Sender for writing messages to the server
    pub sender: WsSender,
    /// Controller handle, if the server granted the `controller@v1` role
    pub controller: Option<Controller>,
    /// The `server/hello` received during handshake. Carries `server_id`,
    /// `connection_reason`, and `active_roles` — required for the
    /// multi-server arbitration policy described on [`ProtocolListener`].
    ///
    /// [`ProtocolListener`]: crate::protocol::listener::ProtocolListener
    pub server_hello: ServerHello,
    /// Must be held alive; dropping aborts background tasks
    pub guard: ConnectionGuard,
}

/// Cheap to clone. `send_message` returns once the writer has reported the
/// underlying `sink.send` result, so the `Result` reflects the wire-write
/// outcome rather than queue insertion.
#[derive(Debug, Clone)]
pub struct WsSender {
    tx: UnboundedSender<WriteCommand>,
}

impl WsSender {
    /// Send a message to the server.
    pub async fn send_message(&self, msg: Message) -> Result<(), Error> {
        let json = serde_json::to_string(&msg).map_err(|e| Error::Protocol(e.to_string()))?;
        log::debug!("Sending message: {}", json);

        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(WriteCommand::Send {
                msg: WsMessage::Text(json),
                ack: ack_tx,
            })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;

        // A cancelled ack means the writer dropped the command unsent — the
        // connection is gone either way.
        ack_rx
            .await
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?
    }

    fn send_goodbye(
        &self,
        reason: GoodbyeReason,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<(), Error>>, Error> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(WriteCommand::Goodbye {
                reason,
                ack: ack_tx,
            })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
        Ok(ack_rx)
    }
}

/// Controller handle for sending playback commands to the server.
///
/// Only available when the server grants the `controller@v1` role.
/// Obtained via [`ProtocolClient::split()`].
#[derive(Debug, Clone)]
pub struct Controller {
    sender: WsSender,
}

impl Controller {
    async fn send_controller_command(&self, cmd: ControllerCommand) -> Result<(), Error> {
        let msg = Message::ClientCommand(ClientCommand {
            controller: Some(cmd),
        });
        self.sender.send_message(msg).await
    }

    async fn send_simple_command(&self, command: ControllerCommandType) -> Result<(), Error> {
        self.send_controller_command(ControllerCommand {
            command,
            volume: None,
            mute: None,
        })
        .await
    }

    /// Resume playback
    pub async fn play(&self) -> Result<(), Error> {
        self.send_simple_command(ControllerCommandType::Play).await
    }

    /// Pause playback
    pub async fn pause(&self) -> Result<(), Error> {
        self.send_simple_command(ControllerCommandType::Pause).await
    }

    /// Stop playback
    pub async fn stop(&self) -> Result<(), Error> {
        self.send_simple_command(ControllerCommandType::Stop).await
    }

    /// Skip to next track
    pub async fn next(&self) -> Result<(), Error> {
        self.send_simple_command(ControllerCommandType::Next).await
    }

    /// Skip to previous track
    pub async fn previous(&self) -> Result<(), Error> {
        self.send_simple_command(ControllerCommandType::Previous)
            .await
    }

    /// Set group volume (0-100). Values above 100 are clamped.
    pub async fn set_volume(&self, volume: u8) -> Result<(), Error> {
        self.send_controller_command(ControllerCommand {
            command: ControllerCommandType::Volume,
            volume: Some(volume.clamp(0, 100)),
            mute: None,
        })
        .await
    }

    /// Set group mute state
    pub async fn set_mute(&self, muted: bool) -> Result<(), Error> {
        self.send_controller_command(ControllerCommand {
            command: ControllerCommandType::Mute,
            volume: None,
            mute: Some(muted),
        })
        .await
    }

    /// Set repeat mode
    pub async fn repeat(&self, mode: RepeatMode) -> Result<(), Error> {
        let command = match mode {
            RepeatMode::Off => ControllerCommandType::RepeatOff,
            RepeatMode::One => ControllerCommandType::RepeatOne,
            RepeatMode::All => ControllerCommandType::RepeatAll,
        };
        self.send_simple_command(command).await
    }

    /// Enable or disable shuffle
    pub async fn shuffle(&self, enabled: bool) -> Result<(), Error> {
        let command = if enabled {
            ControllerCommandType::Shuffle
        } else {
            ControllerCommandType::Unshuffle
        };
        self.send_simple_command(command).await
    }

    /// Switch to next group
    pub async fn switch(&self) -> Result<(), Error> {
        self.send_simple_command(ControllerCommandType::Switch)
            .await
    }
}

/// Binary message type IDs per Sendspin spec
pub mod binary_types {
    /// Player audio chunk (types 4-7, we use 4)
    pub const PLAYER_AUDIO: u8 = 0x04;
    /// Artwork channel 0 (type 8)
    pub const ARTWORK_CHANNEL_0: u8 = 0x08;
    /// Artwork channel 1 (type 9)
    pub const ARTWORK_CHANNEL_1: u8 = 0x09;
    /// Artwork channel 2 (type 10)
    pub const ARTWORK_CHANNEL_2: u8 = 0x0A;
    /// Artwork channel 3 (type 11)
    pub const ARTWORK_CHANNEL_3: u8 = 0x0B;
    /// Visualizer data (type 16)
    pub const VISUALIZER: u8 = 0x10;

    /// Check if a binary type ID is for artwork (8-11)
    pub fn is_artwork(type_id: u8) -> bool {
        (ARTWORK_CHANNEL_0..=ARTWORK_CHANNEL_3).contains(&type_id)
    }

    /// Get artwork channel number from type ID (0-3)
    pub fn artwork_channel(type_id: u8) -> Option<u8> {
        if is_artwork(type_id) {
            Some(type_id - ARTWORK_CHANNEL_0)
        } else {
            None
        }
    }
}

/// Audio chunk from server (binary type 4)
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Server timestamp in microseconds
    pub timestamp: i64,
    /// Raw audio data bytes
    pub data: Arc<[u8]>,
}

impl AudioChunk {
    /// Parse from WebSocket binary frame (type 4 = player audio)
    pub fn from_bytes(frame: &[u8]) -> Result<Self, Error> {
        if frame.len() < 9 {
            return Err(Error::Protocol(format!(
                "Audio chunk too short: got {} bytes, need at least 9",
                frame.len()
            )));
        }

        // Per spec: player audio uses binary type 4
        if frame[0] != binary_types::PLAYER_AUDIO {
            return Err(Error::Protocol(format!(
                "Invalid audio chunk type: expected {}, got {}",
                binary_types::PLAYER_AUDIO,
                frame[0]
            )));
        }

        let timestamp = i64::from_be_bytes([
            frame[1], frame[2], frame[3], frame[4], frame[5], frame[6], frame[7], frame[8],
        ]);

        let data = Arc::from(&frame[9..]);

        Ok(Self { timestamp, data })
    }
}

/// Artwork chunk from server (binary types 8-11)
#[derive(Debug, Clone)]
pub struct ArtworkChunk {
    /// Artwork channel (0-3)
    pub channel: u8,
    /// Server timestamp in microseconds
    pub timestamp: i64,
    /// Image data bytes (JPEG, PNG, or BMP)
    /// Empty payload means clear the artwork
    pub data: Arc<[u8]>,
}

impl ArtworkChunk {
    /// Parse from WebSocket binary frame (types 8-11 = artwork channels 0-3)
    pub fn from_bytes(frame: &[u8]) -> Result<Self, Error> {
        if frame.len() < 9 {
            return Err(Error::Protocol(format!(
                "Artwork chunk too short: got {} bytes, need at least 9",
                frame.len()
            )));
        }

        let type_id = frame[0];
        let channel = binary_types::artwork_channel(type_id)
            .ok_or_else(|| Error::Protocol(format!("Invalid artwork chunk type: {}", type_id)))?;

        let timestamp = i64::from_be_bytes([
            frame[1], frame[2], frame[3], frame[4], frame[5], frame[6], frame[7], frame[8],
        ]);

        let data = Arc::from(&frame[9..]);

        Ok(Self {
            channel,
            timestamp,
            data,
        })
    }

    /// Check if this is a clear command (empty payload)
    pub fn is_clear(&self) -> bool {
        self.data.is_empty()
    }
}

/// Visualizer chunk from server (binary type 16)
#[derive(Debug, Clone)]
pub struct VisualizerChunk {
    /// Server timestamp in microseconds
    pub timestamp: i64,
    /// FFT/visualization data bytes
    pub data: Arc<[u8]>,
}

impl VisualizerChunk {
    /// Parse from WebSocket binary frame (type 16 = visualizer)
    pub fn from_bytes(frame: &[u8]) -> Result<Self, Error> {
        if frame.len() < 9 {
            return Err(Error::Protocol(format!(
                "Visualizer chunk too short: got {} bytes, need at least 9",
                frame.len()
            )));
        }

        if frame[0] != binary_types::VISUALIZER {
            return Err(Error::Protocol(format!(
                "Invalid visualizer chunk type: expected {}, got {}",
                binary_types::VISUALIZER,
                frame[0]
            )));
        }

        let timestamp = i64::from_be_bytes([
            frame[1], frame[2], frame[3], frame[4], frame[5], frame[6], frame[7], frame[8],
        ]);

        let data = Arc::from(&frame[9..]);

        Ok(Self { timestamp, data })
    }
}

/// Binary frame from server (any type)
#[derive(Debug, Clone)]
pub enum BinaryFrame {
    /// Player audio (type 4)
    Audio(AudioChunk),
    /// Artwork image (types 8-11)
    Artwork(ArtworkChunk),
    /// Visualizer data (type 16)
    Visualizer(VisualizerChunk),
    /// Unknown binary type
    Unknown {
        /// The unknown type ID
        type_id: u8,
        /// Raw data after the type byte
        data: Arc<[u8]>,
    },
}

impl BinaryFrame {
    /// Parse any binary frame from WebSocket
    pub fn from_bytes(frame: &[u8]) -> Result<Self, Error> {
        if frame.is_empty() {
            return Err(Error::Protocol("Empty binary frame".to_string()));
        }

        let type_id = frame[0];

        match type_id {
            binary_types::PLAYER_AUDIO => Ok(BinaryFrame::Audio(AudioChunk::from_bytes(frame)?)),
            t if binary_types::is_artwork(t) => {
                Ok(BinaryFrame::Artwork(ArtworkChunk::from_bytes(frame)?))
            }
            binary_types::VISUALIZER => {
                Ok(BinaryFrame::Visualizer(VisualizerChunk::from_bytes(frame)?))
            }
            _ => {
                log::debug!("Unknown binary type: {}", type_id);
                Ok(BinaryFrame::Unknown {
                    type_id,
                    data: Arc::from(&frame[1..]),
                })
            }
        }
    }
}

/// WebSocket client for Sendspin protocol
pub struct ProtocolClient {
    out_tx: UnboundedSender<WriteCommand>,
    audio_rx: UnboundedReceiver<AudioChunk>,
    artwork_rx: UnboundedReceiver<ArtworkChunk>,
    visualizer_rx: UnboundedReceiver<VisualizerChunk>,
    message_rx: UnboundedReceiver<Message>,
    clock_sync: Arc<Mutex<ClockSync>>,
    server_hello: ServerHello,
    /// Background task guard, aborts tasks on drop
    guard: ConnectionGuard,
}

/// Aborts background tasks on drop. Hold this alive for the lifetime of the
/// connection.
pub struct ConnectionGuard {
    sender: WsSender,
    router_handle: Option<tokio::task::JoinHandle<()>>,
    sync_handle: Option<tokio::task::JoinHandle<()>>,
    writer_handle: Option<tokio::task::JoinHandle<()>>,
}

impl ConnectionGuard {
    /// Gracefully disconnect: enqueue `client/goodbye`, await the writer's
    /// ack so the goodbye + close frames are known to have flushed (or
    /// surface the wire error if they didn't), then reap the writer.
    pub async fn disconnect(mut self, reason: GoodbyeReason) -> Result<(), Error> {
        // Stop clock-sync first so it can't enqueue time samples behind the
        // goodbye. The reader stays up until the goodbye/close has flushed
        // (below) so the socket isn't half-closed while we're still writing.
        if let Some(h) = self.sync_handle.take() {
            h.abort();
        }

        let ack_rx = self.sender.send_goodbye(reason)?;
        let goodbye_result = ack_rx
            .await
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;

        // Reap the writer separately from awaiting its ack — the ack arrives
        // just before the task returns, so this only joins the trailing
        // teardown.
        if let Some(h) = self.writer_handle.take() {
            let _ = h.await;
        }

        // Goodbye + close are flushed; tear the reader down now.
        if let Some(h) = self.router_handle.take() {
            h.abort();
        }

        goodbye_result
    }
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        if let Some(h) = self.router_handle.take() {
            h.abort();
        }
        if let Some(h) = self.sync_handle.take() {
            h.abort();
        }
        if let Some(h) = self.writer_handle.take() {
            h.abort();
        }
    }
}

impl ProtocolClient {
    /// Connect to Sendspin server
    pub(crate) async fn connect<R>(
        request: R,
        hello: ClientHello,
        initial_state: ClientState,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, Error>
    where
        R: IntoClientRequest + Unpin,
    {
        let (ws_stream, _) = connect_async(request)
            .await
            .map_err(|e| Error::Connection(e.to_string()))?;

        Self::drive(ws_stream, hello, initial_state, clock).await
    }

    /// Drive the protocol-client state machine over an already-handshaked
    /// WebSocket stream. Shared between outbound `connect()` and inbound
    /// acceptor paths.
    pub(crate) async fn drive<S>(
        ws_stream: WebSocketStream<S>,
        hello: ClientHello,
        initial_state: ClientState,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, Error>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let (mut write, mut read) = ws_stream.split();

        // The handshake exchange (hello + state) sends directly on the sink
        // rather than through the writer task, so handshake failures are
        // returned synchronously instead of through an ack channel.
        let hello_msg = Message::ClientHello(hello);
        let hello_json =
            serde_json::to_string(&hello_msg).map_err(|e| Error::Protocol(e.to_string()))?;
        log::debug!("Sending client/hello: {}", hello_json);
        write
            .send(WsMessage::Text(hello_json))
            .await
            .map_err(|e| Error::WebSocket(e.to_string()))?;

        log::debug!("Waiting for server/hello...");
        let server_hello = loop {
            let Some(result) = read.next().await else {
                log::error!("Connection closed before receiving server/hello");
                return Err(Error::Connection("No server hello received".to_string()));
            };
            match result {
                Ok(WsMessage::Text(text)) => {
                    log::debug!("Received text message: {}", text);
                    let msg: Message = serde_json::from_str(&text).map_err(|e| {
                        log::error!("Failed to parse server message: {}", e);
                        Error::Protocol(e.to_string())
                    })?;

                    match msg {
                        Message::ServerHello(server_hello) => {
                            log::info!(
                                "Connected to server: {} ({})",
                                server_hello.name,
                                server_hello.server_id
                            );
                            break server_hello;
                        }
                        _ => {
                            log::error!("Expected server/hello, got: {:?}", msg);
                            return Err(Error::Protocol("Expected server/hello".to_string()));
                        }
                    }
                }
                Ok(WsMessage::Ping(_)) | Ok(WsMessage::Pong(_)) => {
                    log::debug!("Received Ping/Pong, continuing to wait for server/hello");
                    continue;
                }
                Ok(WsMessage::Close(_)) => {
                    log::error!("Server closed connection");
                    return Err(Error::Connection("Server closed connection".to_string()));
                }
                Ok(other) => {
                    log::warn!(
                        "Unexpected message type while waiting for hello: {:?}",
                        other
                    );
                    continue;
                }
                Err(e) => {
                    log::error!("WebSocket error: {}", e);
                    return Err(Error::WebSocket(e.to_string()));
                }
            }
        };

        let state_msg = Message::ClientState(initial_state);
        let state_json =
            serde_json::to_string(&state_msg).map_err(|e| Error::Protocol(e.to_string()))?;
        log::debug!("Sending initial client/state: {}", state_json);
        write
            .send(WsMessage::Text(state_json))
            .await
            .map_err(|e| Error::WebSocket(e.to_string()))?;

        let (out_tx, out_rx) = unbounded_channel::<WriteCommand>();
        let (audio_tx, audio_rx) = unbounded_channel();
        let (artwork_tx, artwork_rx) = unbounded_channel();
        let (visualizer_tx, visualizer_rx) = unbounded_channel();
        let (message_tx, message_rx) = unbounded_channel();
        let clock_sync = Arc::new(Mutex::new(ClockSync::new(Arc::clone(&clock))));

        let writer_handle = tokio::spawn(writer_task(write, out_rx));

        let clock_sync_router = Arc::clone(&clock_sync);
        let clock_router = Arc::clone(&clock);
        let router_handle = tokio::spawn(async move {
            Self::message_router(
                read,
                audio_tx,
                artwork_tx,
                visualizer_tx,
                message_tx,
                clock_sync_router,
                clock_router,
            )
            .await;
        });

        // First two samples fire 10ms apart so the Kalman filter converges
        // in ~20ms; the remainder run at 1Hz for ongoing drift correction.
        let sync_sender = WsSender { tx: out_tx.clone() };
        let sync_handle = tokio::spawn(async move {
            let mut sample_count: u32 = 0;
            'sync: loop {
                let t1 = clock.now_micros();
                let msg = Message::ClientTime(ClientTime {
                    client_transmitted: t1,
                });
                match sync_sender.send_message(msg).await {
                    Ok(()) => {
                        sample_count = sample_count.saturating_add(1);
                    }
                    Err(e) => {
                        log::info!("Clock sync task exiting: {}", e);
                        break 'sync;
                    }
                }
                let delay = if sample_count < 2 {
                    tokio::time::Duration::from_millis(10)
                } else {
                    tokio::time::Duration::from_secs(1)
                };
                tokio::time::sleep(delay).await;
            }
        });

        Ok(Self {
            out_tx: out_tx.clone(),
            audio_rx,
            artwork_rx,
            visualizer_rx,
            message_rx,
            clock_sync,
            server_hello,
            guard: ConnectionGuard {
                sender: WsSender { tx: out_tx },
                router_handle: Some(router_handle),
                sync_handle: Some(sync_handle),
                writer_handle: Some(writer_handle),
            },
        })
    }

    async fn message_router<S>(
        mut read: SplitStream<WebSocketStream<S>>,
        audio_tx: UnboundedSender<AudioChunk>,
        artwork_tx: UnboundedSender<ArtworkChunk>,
        visualizer_tx: UnboundedSender<VisualizerChunk>,
        message_tx: UnboundedSender<Message>,
        clock_sync: Arc<Mutex<ClockSync>>,
        clock: Arc<dyn Clock>,
    ) where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let mut audio_closed = false;
        let mut artwork_closed = false;
        let mut visualizer_closed = false;
        let mut message_closed = false;

        while let Some(msg) = read.next().await {
            match msg {
                Ok(WsMessage::Binary(data)) => {
                    log::debug!("Received binary frame ({} bytes)", data.len());
                    match BinaryFrame::from_bytes(&data) {
                        Ok(BinaryFrame::Audio(chunk)) => {
                            log::debug!(
                                "Parsed audio chunk: timestamp={}, data_len={}",
                                chunk.timestamp,
                                chunk.data.len()
                            );
                            if !audio_closed && audio_tx.send(chunk).is_err() {
                                log::error!(
                                    "Audio receiver dropped — audio data will be discarded"
                                );
                                audio_closed = true;
                            }
                        }
                        Ok(BinaryFrame::Artwork(chunk)) => {
                            log::debug!(
                                "Parsed artwork chunk: channel={}, timestamp={}, data_len={}",
                                chunk.channel,
                                chunk.timestamp,
                                chunk.data.len()
                            );
                            if !artwork_closed && artwork_tx.send(chunk).is_err() {
                                log::error!(
                                    "Artwork receiver dropped — artwork data will be discarded"
                                );
                                artwork_closed = true;
                            }
                        }
                        Ok(BinaryFrame::Visualizer(chunk)) => {
                            log::debug!(
                                "Parsed visualizer chunk: timestamp={}, data_len={}",
                                chunk.timestamp,
                                chunk.data.len()
                            );
                            if !visualizer_closed && visualizer_tx.send(chunk).is_err() {
                                log::error!("Visualizer receiver dropped — visualizer data will be discarded");
                                visualizer_closed = true;
                            }
                        }
                        Ok(BinaryFrame::Unknown { type_id, .. }) => {
                            log::warn!("Received unknown binary type: {}", type_id);
                        }
                        Err(e) => {
                            log::warn!("Failed to parse binary frame: {}", e);
                        }
                    }
                }
                Ok(WsMessage::Text(text)) => {
                    // Capture receive time before deserialization so
                    // t4 is as close to the true arrival time as possible.
                    let t4 = clock.now_micros();
                    log::debug!("Received text message: {}", text);
                    match serde_json::from_str::<Message>(&text) {
                        Ok(msg) => {
                            log::debug!("Parsed message: {:?}", msg);
                            // ServerTime is consumed here for clock sync
                            // and intentionally NOT forwarded to message_rx
                            // consumers — it's an internal protocol detail.
                            if let Message::ServerTime(ref st) = msg {
                                clock_sync.lock().update(
                                    st.client_transmitted,
                                    st.server_received,
                                    st.server_transmitted,
                                    t4,
                                );
                            } else if !message_closed && message_tx.send(msg).is_err() {
                                log::error!(
                                    "Message receiver dropped — messages will be discarded"
                                );
                                message_closed = true;
                            }
                        }
                        Err(e) => {
                            log::warn!("Failed to parse message: {}", e);
                        }
                    }
                }
                Ok(WsMessage::Ping(_)) | Ok(WsMessage::Pong(_)) => {}
                Ok(WsMessage::Close(_)) => {
                    log::info!("Server closed connection");
                    break;
                }
                Err(e) => {
                    log::error!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
    }

    /// Gracefully disconnect: sends `client/goodbye`, closes the WebSocket,
    /// and aborts background tasks.
    pub async fn disconnect(self, reason: GoodbyeReason) -> Result<(), Error> {
        self.guard.disconnect(reason).await
    }

    /// Get reference to clock sync
    pub fn clock_sync(&self) -> Arc<Mutex<ClockSync>> {
        Arc::clone(&self.clock_sync)
    }

    /// The `server/hello` received during handshake. Carries `server_id`,
    /// `connection_reason`, and `active_roles` — required for the
    /// multi-server arbitration policy described on [`ProtocolListener`].
    ///
    /// [`ProtocolListener`]: crate::protocol::listener::ProtocolListener
    pub fn server_hello(&self) -> &ServerHello {
        &self.server_hello
    }

    /// Split into separate receivers for concurrent processing.
    ///
    /// This allows using `tokio::select!` to process messages and binary
    /// data concurrently. Use the fields you need; ignore the rest.
    pub fn split(self) -> Connection {
        let sender = WsSender { tx: self.out_tx };
        let controller = self
            .server_hello
            .active_roles
            .iter()
            .any(|r| r == "controller@v1")
            .then(|| Controller {
                sender: sender.clone(),
            });
        Connection {
            messages: self.message_rx,
            audio: self.audio_rx,
            artwork: self.artwork_rx,
            visualizer: self.visualizer_rx,
            clock_sync: self.clock_sync,
            sender,
            controller,
            server_hello: self.server_hello,
            guard: self.guard,
        }
    }
}
