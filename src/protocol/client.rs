// ABOUTME: WebSocket client implementation for Sendspin protocol
// ABOUTME: Handles connection, message routing, and protocol state machine

use crate::error::Error;
use crate::protocol::messages::{
    ArtworkV1Support, AudioFormatSpec, ClientHello, DeviceInfo, Message, PlayerV1Support,
    VisualizerV1Support,
};
use crate::sync::ClockSync;
use futures_util::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};
use typed_builder::TypedBuilder;

type WsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, WsMessage>;
type FullSplit = (
    UnboundedReceiver<Message>,
    UnboundedReceiver<AudioChunk>,
    UnboundedReceiver<ArtworkChunk>,
    UnboundedReceiver<VisualizerChunk>,
    Arc<Mutex<ClockSync>>,
    WsSender,
);

/// WebSocket sender wrapper for sending messages
pub struct WsSender {
    tx: Arc<tokio::sync::Mutex<WsSink>>,
}

impl WsSender {
    /// Send a message to the server
    pub async fn send_message(&self, msg: Message) -> Result<(), Error> {
        let json = serde_json::to_string(&msg).map_err(|e| Error::Protocol(e.to_string()))?;
        log::debug!("Sending message: {}", json);

        let mut tx = self.tx.lock().await;
        tx.send(WsMessage::Text(json))
            .await
            .map_err(|e| Error::WebSocket(e.to_string()))
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

/// Intermediate builder struct before finalization
#[derive(Clone)]
pub struct ProtocolClientBuilderRaw {
    client_id: String,
    name: String,
    product_name: Option<String>,
    manufacturer: Option<String>,
    player_v1_support: Option<PlayerV1Support>,
    artwork_v1_support: Option<ArtworkV1Support>,
    visualizer_v1_support: Option<VisualizerV1Support>,
}

impl From<ProtocolClientBuilderRaw> for ProtocolClientBuilder {
    fn from(raw: ProtocolClientBuilderRaw) -> Self {
        // Build supported_roles based on which supports are configured
        let mut supported_roles = Vec::new();

        // Default player support if not explicitly set
        let player_v1_support = raw.player_v1_support.or_else(|| {
            Some(PlayerV1Support {
                supported_formats: vec![AudioFormatSpec {
                    codec: "pcm".to_string(),
                    channels: 2,
                    sample_rate: 48000,
                    bit_depth: 24,
                }],
                buffer_capacity: 50 * 1024 * 1024,
                supported_commands: vec!["volume".to_string(), "mute".to_string()],
            })
        });

        // Player role is always present (since we set a default)
        supported_roles.push("player@v1".to_string());

        if raw.artwork_v1_support.is_some() {
            supported_roles.push("artwork@v1".to_string());
        }
        if raw.visualizer_v1_support.is_some() {
            supported_roles.push("visualizer@v1".to_string());
        }

        ProtocolClientBuilder {
            client_id: raw.client_id,
            name: raw.name,
            product_name: raw.product_name,
            manufacturer: raw.manufacturer,
            supported_roles,
            player_v1_support,
            artwork_v1_support: raw.artwork_v1_support,
            visualizer_v1_support: raw.visualizer_v1_support,
        }
    }
}

#[derive(TypedBuilder, Clone)]
#[builder(build_method(into = ProtocolClientBuilder))]
/// Builder Class for ProtocolClient
pub struct ProtocolClientBuilderFields {
    client_id: String,
    name: String,
    #[builder(default = None)]
    product_name: Option<String>,
    #[builder(default = None)]
    manufacturer: Option<String>,
    #[builder(default = None, setter(transform = |x: PlayerV1Support| Some(x)))]
    player_v1_support: Option<PlayerV1Support>,
    #[builder(default = None, setter(transform = |x: ArtworkV1Support| Some(x)))]
    artwork_v1_support: Option<ArtworkV1Support>,
    #[builder(default = None, setter(transform = |x: VisualizerV1Support| Some(x)))]
    visualizer_v1_support: Option<VisualizerV1Support>,
}

impl From<ProtocolClientBuilderFields> for ProtocolClientBuilder {
    fn from(fields: ProtocolClientBuilderFields) -> Self {
        let raw = ProtocolClientBuilderRaw {
            client_id: fields.client_id,
            name: fields.name,
            product_name: fields.product_name,
            manufacturer: fields.manufacturer,
            player_v1_support: fields.player_v1_support,
            artwork_v1_support: fields.artwork_v1_support,
            visualizer_v1_support: fields.visualizer_v1_support,
        };
        raw.into()
    }
}

/// Builder Class for ProtocolClient
#[derive(Clone)]
pub struct ProtocolClientBuilder {
    client_id: String,
    name: String,
    product_name: Option<String>,
    software_version: Option<String>,
    manufacturer: Option<String>,
    supported_roles: Vec<String>,
    player_v1_support: Option<PlayerV1Support>,
    artwork_v1_support: Option<ArtworkV1Support>,
    visualizer_v1_support: Option<VisualizerV1Support>,
}

impl ProtocolClientBuilder {
    /// Create a new builder
    pub fn builder() -> ProtocolClientBuilderFieldsBuilder {
        ProtocolClientBuilderFields::builder()
    }

    /// Get the supported roles that will be sent in the client hello
    pub fn supported_roles(&self) -> &[String] {
        &self.supported_roles
    }

    /// Get the player v1 support configuration
    pub fn player_v1_support(&self) -> Option<&PlayerV1Support> {
        self.player_v1_support.as_ref()
    }

    /// Connect to Sendspin server
    pub async fn connect(self, url: &str) -> Result<ProtocolClient, Error> {
        let hello = ClientHello {
            client_id: self.client_id.clone(),
            name: self.name.clone(),
            version: 1,
            supported_roles: self.supported_roles.clone(),
            device_info: Some(DeviceInfo {
                product_name: self.product_name.clone(),
                manufacturer: Some(self.manufacturer.unwrap_or("Sendspin".to_string())),
                software_version: self.software_version.clone(),
            }),
            player_v1_support: self.player_v1_support.clone(),
            artwork_v1_support: self.artwork_v1_support.clone(),
            visualizer_v1_support: self.visualizer_v1_support.clone(),
        };
        ProtocolClient::connect(url, hello).await
    }
}

/// WebSocket client for Sendspin protocol
pub struct ProtocolClient {
    ws_tx: Arc<tokio::sync::Mutex<WsSink>>,
    audio_rx: UnboundedReceiver<AudioChunk>,
    artwork_rx: UnboundedReceiver<ArtworkChunk>,
    visualizer_rx: UnboundedReceiver<VisualizerChunk>,
    message_rx: UnboundedReceiver<Message>,
    clock_sync: Arc<Mutex<ClockSync>>,
}

impl ProtocolClient {
    /// Connect to Sendspin server
    pub async fn connect<R>(request: R, hello: ClientHello) -> Result<Self, Error>
    where
        R: IntoClientRequest + Unpin,
    {
        // Connect WebSocket
        let (ws_stream, _) = connect_async(request)
            .await
            .map_err(|e| Error::Connection(e.to_string()))?;

        let (mut write, read) = ws_stream.split();

        // Send client hello
        let hello_msg = Message::ClientHello(hello);
        let hello_json =
            serde_json::to_string(&hello_msg).map_err(|e| Error::Protocol(e.to_string()))?;

        log::debug!("Sending client/hello: {}", hello_json);

        write
            .send(WsMessage::Text(hello_json))
            .await
            .map_err(|e| Error::WebSocket(e.to_string()))?;

        // Wait for server hello (handle Ping/Pong first)
        let mut read_temp = read;
        log::debug!("Waiting for server/hello...");

        loop {
            if let Some(result) = read_temp.next().await {
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
                                break; // Exit loop, we got the server/hello
                            }
                            _ => {
                                log::error!("Expected server/hello, got: {:?}", msg);
                                return Err(Error::Protocol("Expected server/hello".to_string()));
                            }
                        }
                    }
                    Ok(WsMessage::Ping(_)) | Ok(WsMessage::Pong(_)) => {
                        // Ping/Pong are handled automatically by tokio-tungstenite
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
            } else {
                log::error!("Connection closed before receiving server/hello");
                return Err(Error::Connection("No server hello received".to_string()));
            }
        }

        // Create channels for message routing
        let (audio_tx, audio_rx) = unbounded_channel();
        let (artwork_tx, artwork_rx) = unbounded_channel();
        let (visualizer_tx, visualizer_rx) = unbounded_channel();
        let (message_tx, message_rx) = unbounded_channel();

        let clock_sync = Arc::new(Mutex::new(ClockSync::new()));

        // Spawn message router task
        let clock_sync_clone = Arc::clone(&clock_sync);
        tokio::spawn(async move {
            Self::message_router(
                read_temp,
                audio_tx,
                artwork_tx,
                visualizer_tx,
                message_tx,
                clock_sync_clone,
            )
            .await;
        });

        Ok(Self {
            ws_tx: Arc::new(tokio::sync::Mutex::new(write)),
            audio_rx,
            artwork_rx,
            visualizer_rx,
            message_rx,
            clock_sync,
        })
    }

    async fn message_router(
        mut read: SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>,
        audio_tx: UnboundedSender<AudioChunk>,
        artwork_tx: UnboundedSender<ArtworkChunk>,
        visualizer_tx: UnboundedSender<VisualizerChunk>,
        message_tx: UnboundedSender<Message>,
        _clock_sync: Arc<Mutex<ClockSync>>,
    ) {
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
                            let _ = audio_tx.send(chunk);
                        }
                        Ok(BinaryFrame::Artwork(chunk)) => {
                            log::debug!(
                                "Parsed artwork chunk: channel={}, timestamp={}, data_len={}",
                                chunk.channel,
                                chunk.timestamp,
                                chunk.data.len()
                            );
                            let _ = artwork_tx.send(chunk);
                        }
                        Ok(BinaryFrame::Visualizer(chunk)) => {
                            log::debug!(
                                "Parsed visualizer chunk: timestamp={}, data_len={}",
                                chunk.timestamp,
                                chunk.data.len()
                            );
                            let _ = visualizer_tx.send(chunk);
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
                    log::debug!("Received text message: {}", text);
                    match serde_json::from_str::<Message>(&text) {
                        Ok(msg) => {
                            log::debug!("Parsed message: {:?}", msg);
                            let _ = message_tx.send(msg);
                        }
                        Err(e) => {
                            log::warn!("Failed to parse message: {}", e);
                        }
                    }
                }
                Ok(WsMessage::Ping(_)) | Ok(WsMessage::Pong(_)) => {
                    // Handled automatically by tokio-tungstenite
                }
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

    /// Receive next audio chunk
    pub async fn recv_audio_chunk(&mut self) -> Option<AudioChunk> {
        self.audio_rx.recv().await
    }

    /// Receive next artwork chunk
    pub async fn recv_artwork_chunk(&mut self) -> Option<ArtworkChunk> {
        self.artwork_rx.recv().await
    }

    /// Receive next visualizer chunk
    pub async fn recv_visualizer_chunk(&mut self) -> Option<VisualizerChunk> {
        self.visualizer_rx.recv().await
    }

    /// Receive next protocol message
    pub async fn recv_message(&mut self) -> Option<Message> {
        self.message_rx.recv().await
    }

    /// Send a message to the server
    pub async fn send_message(&self, msg: &Message) -> Result<(), Error> {
        let json = serde_json::to_string(msg).map_err(|e| Error::Protocol(e.to_string()))?;
        log::debug!("Sending message: {}", json);

        let mut tx = self.ws_tx.lock().await;
        tx.send(WsMessage::Text(json))
            .await
            .map_err(|e| Error::WebSocket(e.to_string()))
    }

    /// Get reference to clock sync
    pub fn clock_sync(&self) -> Arc<Mutex<ClockSync>> {
        Arc::clone(&self.clock_sync)
    }

    /// Split into separate receivers for concurrent processing
    ///
    /// This allows using tokio::select! to process messages and binary data concurrently
    /// without borrow checker issues
    pub fn split(
        self,
    ) -> (
        UnboundedReceiver<Message>,
        UnboundedReceiver<AudioChunk>,
        Arc<Mutex<ClockSync>>,
        WsSender,
    ) {
        (
            self.message_rx,
            self.audio_rx,
            self.clock_sync,
            WsSender { tx: self.ws_tx },
        )
    }

    /// Split into all receivers including artwork and visualizer
    ///
    /// Use this when you need to handle all binary frame types
    pub fn split_full(self) -> FullSplit {
        (
            self.message_rx,
            self.audio_rx,
            self.artwork_rx,
            self.visualizer_rx,
            self.clock_sync,
            WsSender { tx: self.ws_tx },
        )
    }
}
