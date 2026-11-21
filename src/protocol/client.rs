// ABOUTME: WebSocket client implementation for Sendspin protocol
// ABOUTME: Handles connection, message routing, and protocol state machine

use crate::audio::PlayerReceiver;
use crate::error::Error;
use crate::log_sampling::should_log_sample;
use crate::protocol::activation::admit;
use crate::protocol::messages::{
    ArtworkFormatRequest, ClientCommand, ClientGoodbye, ClientHello, ClientState, ClientSyncState,
    ClientTime, ControllerCommand, ControllerCommandType, GoodbyeReason, Message,
    PlayerFormatRequest, PlayerState, RepeatMode, ServerActivate, ServerHello, ServerTime,
    StreamEnd, StreamRequestFormat, StreamStart, VisualizerDataType, VisualizerFormatRequest,
};
use crate::protocol::noise::{
    self, Decryptor, Encryptor, Identity, PskCategory, Session, HANDSHAKE_TIMEOUT,
    MESSAGE_TYPE_JSON,
};
use crate::roles::{ArtworkReceiver, VisualizerReceiver};
use crate::sync::clock::max_error;
use crate::sync::raw_clock::Clock;
use crate::sync::ClockSync;
use futures_util::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::{oneshot, watch};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};

/// `Goodbye` is one variant (not `Send` + `Close`) so the writer processes it
/// atomically: once dequeued it flushes goodbye + close and exits, so nothing
/// *enqueued after it* reaches the wire.
pub(crate) enum WriteCommand {
    Send {
        msg: Box<Message>,
        ack: oneshot::Sender<Result<(), Error>>,
    },
    Goodbye {
        reason: GoodbyeReason,
        ack: oneshot::Sender<Result<(), Error>>,
    },
    /// Close the WebSocket without sending a Sendspin message.
    Close,
    ReportArtwork,
}

async fn writer_task<S>(
    mut sink: SplitSink<WebSocketStream<S>, WsMessage>,
    mut rx: UnboundedReceiver<WriteCommand>,
    mut encryptor: Encryptor,
    activation: watch::Receiver<ServerActivate>,
) where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let mut artwork = None;
    while let Some(cmd) = rx.recv().await {
        match cmd {
            WriteCommand::ReportArtwork => {
                if !activation.borrow().is_active(ARTWORK_V1) {
                    continue;
                }
                if let Some(artwork) = artwork.clone() {
                    let msg = Message::ClientState(ClientState {
                        artwork: Some(artwork),
                        ..Default::default()
                    });
                    if noise::send_encrypted(&mut sink, &mut encryptor, &msg)
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
            WriteCommand::Send { msg, ack } => {
                if let Message::ClientState(state) = msg.as_ref() {
                    if let Some(update) = &state.artwork {
                        artwork = Some(update.clone());
                    }
                }
                let gated = strip_inactive_roles(*msg, &activation.borrow());
                let Some(msg) = gated else {
                    let _ = ack.send(Ok(()));
                    continue;
                };
                let result = noise::send_encrypted(&mut sink, &mut encryptor, &msg).await;
                let failed = result.is_err();
                // Ignore SendError: the caller may have dropped its receiver.
                let _ = ack.send(result);
                if failed {
                    break;
                }
            }
            WriteCommand::Goodbye { reason, ack } => {
                let _ = ack.send(perform_goodbye(&mut sink, &mut encryptor, reason).await);
                break;
            }
            WriteCommand::Close => {
                let _ = sink.close().await;
                break;
            }
        }
    }
    log::debug!("Writer task exiting");
    // On exit `rx` drops, dropping the ack sender of any still-queued command;
    // callers awaiting those acks see the cancellation and treat it as a closed
    // connection (see `WsSender::send_message`).
}

/// Remove role objects for inactive roles, returning `None` when nothing is left to send.
fn strip_inactive_roles(msg: Message, activation: &ServerActivate) -> Option<Message> {
    match msg {
        Message::ClientState(mut state) => {
            if !activation.is_active(PLAYER_V1) {
                state.player = None;
            }
            if !activation.is_active(ARTWORK_V1) {
                state.artwork = None;
            }
            (state.state.is_some() || state.player.is_some() || state.artwork.is_some())
                .then_some(Message::ClientState(state))
        }
        other => Some(other),
    }
}

async fn perform_goodbye<S>(
    sink: &mut SplitSink<WebSocketStream<S>, WsMessage>,
    encryptor: &mut Encryptor,
    reason: GoodbyeReason,
) -> Result<(), Error>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let goodbye = Message::ClientGoodbye(ClientGoodbye { reason });
    noise::send_encrypted(sink, encryptor, &goodbye).await?;
    sink.close()
        .await
        .map_err(|e| Error::WebSocket(e.to_string()))
}

/// Connection components returned by [`ProtocolClient::split()`].
/// Drop role receivers the application does not consume to disable delivery and
/// avoid unread buffering; retain `guard` until shutdown.
pub struct Connection {
    /// Protocol messages from the server
    pub messages: UnboundedReceiver<Message>,
    /// Player stream receiver. Attach it with [`PlayerReceiver::attach`] to an output
    /// to consume ordered events.
    pub player: PlayerReceiver,
    /// Artwork stream receiver. Attach it with [`ArtworkReceiver::attach`] to an output
    /// to consume ordered events.
    pub artwork: ArtworkReceiver,
    /// Visualizer stream receiver. Attach it with [`VisualizerReceiver::attach`] to an output
    /// to consume ordered events.
    pub visualizer: VisualizerReceiver,
    /// Clock synchronization state
    pub clock_sync: Arc<Mutex<ClockSync>>,
    /// Sender for writing messages to the server
    pub sender: WsSender,
    /// Controller handle. Its commands fail while `controller@v1` is not active.
    pub controller: Controller,
    /// The server's `server_id` from `server/init`, its static public key.
    pub server_id: String,
    /// The `server/hello` received during the handshake.
    pub server_hello: ServerHello,
    /// The first `server/activate`. Later activations arrive on `messages`.
    pub activation: ServerActivate,
    /// Must be held alive; dropping aborts background tasks
    pub guard: ConnectionGuard,
}

/// Bare role names as they appear in `stream/end` role lists — distinct from
/// the versioned `player@v1` names used during role negotiation.
const ROLE_PLAYER: &str = "player";
const ROLE_ARTWORK: &str = "artwork";
const ROLE_VISUALIZER: &str = "visualizer";

const PLAYER_V1: &str = "player@v1";
const ARTWORK_V1: &str = "artwork@v1";
const VISUALIZER_V1: &str = "visualizer@v1";
const CONTROLLER_V1: &str = "controller@v1";

/// Which role streams are currently active, updated by the message router from
/// `stream/start` and `stream/end`.
#[derive(Debug, Default)]
struct StreamState {
    player_active: AtomicBool,
    artwork_active: AtomicBool,
    visualizer_active: AtomicBool,
}

impl StreamState {
    /// A `stream/start` for one role must not disturb another's stream, so
    /// absent roles are left untouched rather than cleared.
    fn note_stream_start(&self, start: &StreamStart) {
        if start.player.is_some() {
            self.player_active.store(true, Ordering::Release);
        }
        if start.artwork.is_some() {
            self.artwork_active.store(true, Ordering::Release);
        }
        if start.visualizer.is_some() {
            self.visualizer_active.store(true, Ordering::Release);
        }
    }

    /// `stream/end` with no roles ends every stream; otherwise only those listed.
    fn note_stream_end(&self, end: &StreamEnd) {
        if role_ended(end, ROLE_PLAYER) {
            self.player_active.store(false, Ordering::Release);
        }
        if role_ended(end, ROLE_ARTWORK) {
            self.artwork_active.store(false, Ordering::Release);
        }
        if role_ended(end, ROLE_VISUALIZER) {
            self.visualizer_active.store(false, Ordering::Release);
        }
    }

    fn is_player_active(&self) -> bool {
        self.player_active.load(Ordering::Acquire)
    }

    fn is_artwork_active(&self) -> bool {
        self.artwork_active.load(Ordering::Acquire)
    }

    fn is_visualizer_active(&self) -> bool {
        self.visualizer_active.load(Ordering::Acquire)
    }
}

fn role_ended(end: &StreamEnd, role: &str) -> bool {
    end.roles
        .as_ref()
        .is_none_or(|roles| roles.iter().any(|r| r == role))
}

/// Cheap to clone. `send_message` returns once the writer has reported the
/// underlying `sink.send` result, so the `Result` reflects the wire-write
/// outcome rather than queue insertion.
#[derive(Debug, Clone)]
pub struct WsSender {
    tx: UnboundedSender<WriteCommand>,
    reporter: Option<UnboundedSender<crate::protocol::state_reporter::Command>>,
    controls: tokio::sync::watch::Receiver<crate::audio::AppliedControls>,
    /// The router updates this *before* forwarding the triggering `stream/start`
    /// / `stream/end`, so a consumer that reacts to those messages already
    /// observes the settled state.
    stream_state: Arc<StreamState>,
}

impl WsSender {
    /// Send a message to the server.
    pub async fn send_message(&self, msg: Message) -> Result<(), Error> {
        match msg {
            Message::ClientState(state) => {
                let Some(reporter) = &self.reporter else {
                    return self.send_raw(Message::ClientState(state)).await;
                };
                let (ack, result) = tokio::sync::oneshot::channel();
                reporter
                    .send(crate::protocol::state_reporter::Command::State {
                        state,
                        boundary_applied_controls: self.controls.borrow().clone(),
                        ack,
                    })
                    .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
                result
                    .await
                    .map_err(|_| Error::WebSocket("connection closed".to_string()))?
            }
            message => self.send_raw(message).await,
        }
    }

    async fn send_raw(&self, msg: Message) -> Result<(), Error> {
        // Time pings go out in bursts for as long as the connection lives; keep
        // that housekeeping at trace so debug shows only meaningful traffic.
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .send(WriteCommand::Send {
                msg: Box::new(msg),
                ack: ack_tx,
            })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;

        // A cancelled ack means the writer dropped the command unsent — the
        // connection is gone either way.
        ack_rx
            .await
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?
    }

    /// Send a top-level client synchronization state update through the state reporter.
    pub async fn send_sync_state(&self, state: ClientSyncState) -> Result<(), Error> {
        let Some(reporter) = &self.reporter else {
            return self
                .send_message(Message::ClientState(ClientState {
                    artwork: None,
                    state: Some(state),
                    player: None,
                }))
                .await;
        };
        let (ack, result) = tokio::sync::oneshot::channel();
        reporter
            .send(crate::protocol::state_reporter::Command::State {
                state: ClientState {
                    artwork: None,
                    state: Some(state),
                    player: None,
                },
                boundary_applied_controls: self.controls.borrow().clone(),
                ack,
            })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
        result
            .await
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?
    }

    /// Send a partial player state update, merging it into the retained state.
    ///
    /// With automatic reporting, later reports take volume, mute, and static delay
    /// from the session controls, so change those through `PlayerSession::update_controls`.
    pub async fn set_player_state(&self, player: PlayerState) -> Result<(), Error> {
        let Some(reporter) = &self.reporter else {
            return self
                .send_message(Message::ClientState(ClientState {
                    artwork: None,
                    state: None,
                    player: Some(player),
                }))
                .await;
        };
        let (ack, result) = tokio::sync::oneshot::channel();
        reporter
            .send(crate::protocol::state_reporter::Command::State {
                state: ClientState {
                    artwork: None,
                    state: None,
                    player: Some(player),
                },
                boundary_applied_controls: self.controls.borrow().clone(),
                ack,
            })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
        result
            .await
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?
    }

    /// Tell the server this client is temporarily owned by another audio source.
    ///
    /// Release any Sendspin-owned output first so the external source can open
    /// the device without racing this client's audio stream.
    pub async fn enter_external_source(&self) -> Result<(), Error> {
        let Some(reporter) = &self.reporter else {
            return self.send_sync_state(ClientSyncState::ExternalSource).await;
        };
        let (ack, result) = tokio::sync::oneshot::channel();
        reporter
            .send(crate::protocol::state_reporter::Command::Enter { ack })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
        result
            .await
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?
    }

    /// Tell the server this client's clock filter has converged enough to resume
    /// synchronized playback scheduling.
    ///
    /// Apply volume, mute, or static delay that changed while the external source
    /// owned the device through [`PlayerSession::update_controls`](crate::audio::PlayerSession::update_controls)
    /// first, since automatic reports take these fields from the session controls.
    /// Hardware/OS mixer changes must be read through platform APIs; this library
    /// only tracks its own software [`GainControl`](crate::audio::GainControl).
    pub async fn exit_external_source(&self, player: Option<PlayerState>) -> Result<(), Error> {
        let Some(reporter) = &self.reporter else {
            return self
                .send_message(Message::ClientState(ClientState {
                    artwork: None,
                    state: Some(ClientSyncState::Synchronized),
                    player,
                }))
                .await;
        };
        let (ack, result) = tokio::sync::oneshot::channel();
        let controls = self.controls.borrow().clone();
        reporter
            .send(crate::protocol::state_reporter::Command::Exit {
                player,
                controls,
                ack,
            })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
        result
            .await
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?
    }

    /// Request a change to the active stream format.
    ///
    /// Sendspin servers may use this advisory message to switch codecs,
    /// sample rates, artwork dimensions, or other stream properties in
    /// response to changing network, CPU, or display conditions. Fields left
    /// as `None` are unconstrained by the client.
    ///
    /// A requested component is rejected unless that role's stream is currently
    /// active (between its `stream/start` and `stream/end`): there is nothing to
    /// renegotiate for a role the server is not streaming.
    pub async fn request_stream_format(
        &self,
        player: Option<PlayerFormatRequest>,
        artwork: Option<ArtworkFormatRequest>,
    ) -> Result<(), Error> {
        self.request_stream_formats(player, artwork, None).await
    }

    /// Request changes to any combination of active stream formats.
    ///
    /// Each supplied component must have a corresponding active stream. The
    /// existing [`Self::request_stream_format`] method remains available for
    /// player/artwork-only callers.
    pub async fn request_stream_formats(
        &self,
        player: Option<PlayerFormatRequest>,
        artwork: Option<ArtworkFormatRequest>,
        visualizer: Option<VisualizerFormatRequest>,
    ) -> Result<(), Error> {
        if player.is_none() && artwork.is_none() && visualizer.is_none() {
            return Err(Error::Protocol(
                "stream/request-format requires a player, artwork, or visualizer request"
                    .to_string(),
            ));
        }

        if let Some(request) = visualizer.as_ref() {
            request
                .validate()
                .map_err(|message| Error::Protocol(message.to_string()))?;
        }

        if player.is_some() && !self.stream_state.is_player_active() {
            return Err(Error::Protocol(
                "stream/request-format requires an active player stream".to_string(),
            ));
        }

        if artwork.is_some() && !self.stream_state.is_artwork_active() {
            return Err(Error::Protocol(
                "stream/request-format requires an active artwork stream".to_string(),
            ));
        }

        if visualizer.is_some() && !self.stream_state.is_visualizer_active() {
            return Err(Error::Protocol(
                "stream/request-format requires an active visualizer stream".to_string(),
            ));
        }

        self.send_message(Message::StreamRequestFormat(StreamRequestFormat {
            player,
            artwork,
            visualizer,
        }))
        .await
    }

    /// Request a change to the active player/audio stream format.
    pub async fn request_player_format(&self, player: PlayerFormatRequest) -> Result<(), Error> {
        self.request_stream_format(Some(player), None).await
    }

    /// Request a change to an active artwork stream format.
    pub async fn request_artwork_format(&self, artwork: ArtworkFormatRequest) -> Result<(), Error> {
        self.request_stream_format(None, Some(artwork)).await
    }

    /// Request a change to an active visualizer stream format.
    pub async fn request_visualizer_format(
        &self,
        visualizer: VisualizerFormatRequest,
    ) -> Result<(), Error> {
        self.request_stream_formats(None, None, Some(visualizer))
            .await
    }

    fn send_goodbye(
        &self,
        reason: GoodbyeReason,
    ) -> Result<oneshot::Receiver<Result<(), Error>>, Error> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .send(WriteCommand::Goodbye {
                reason,
                ack: ack_tx,
            })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
        Ok(ack_rx)
    }

    fn close(&self) {
        // A failed send means the writer is already gone.
        let _ = self.tx.send(WriteCommand::Close);
    }

    /// Queue a report of the retained player object without waiting for it.
    fn queue_player_report(&self) {
        let Some(reporter) = &self.reporter else {
            return;
        };
        let (ack, _) = oneshot::channel();
        let _ = reporter.send(crate::protocol::state_reporter::Command::State {
            state: ClientState {
                artwork: None,
                state: None,
                player: Some(PlayerState::default()),
            },
            boundary_applied_controls: self.controls.borrow().clone(),
            ack,
        });
    }
}

/// Controller handle for sending playback commands to the server.
///
/// Commands fail while the server has not activated `controller@v1`.
/// Obtained via [`ProtocolClient::split()`].
#[derive(Debug, Clone)]
pub struct Controller {
    sender: WsSender,
    activation: watch::Receiver<ServerActivate>,
}

impl Controller {
    async fn send_controller_command(&self, cmd: ControllerCommand) -> Result<(), Error> {
        if !self.activation.borrow().is_active(CONTROLLER_V1) {
            return Err(Error::Protocol(
                "the controller@v1 role is not active".to_string(),
            ));
        }
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
            position_ms: None,
            offset_ms: None,
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
            position_ms: None,
            offset_ms: None,
        })
        .await
    }

    /// Set group mute state
    pub async fn set_mute(&self, muted: bool) -> Result<(), Error> {
        self.send_controller_command(ControllerCommand {
            command: ControllerCommandType::Mute,
            volume: None,
            mute: Some(muted),
            position_ms: None,
            offset_ms: None,
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

    /// Seek to an absolute playback position in milliseconds.
    ///
    /// Only send this when `seek` is in the server's `supported_commands`.
    /// Per the spec, the server ignores the command if `position_ms` is
    /// outside the range 0 to
    /// [`ControllerState::seek_max_ms`](crate::protocol::messages::ControllerState::seek_max_ms).
    pub async fn seek(&self, position_ms: u64) -> Result<(), Error> {
        self.send_controller_command(ControllerCommand {
            command: ControllerCommandType::Seek,
            volume: None,
            mute: None,
            position_ms: Some(position_ms),
            offset_ms: None,
        })
        .await
    }

    /// Seek by a signed offset in milliseconds from the current position
    /// (positive forward, negative backward).
    ///
    /// Only send this when `seek_relative` is in the server's
    /// `supported_commands`. The server applies the offset on a best-effort
    /// basis and clamps the result to the seekable range.
    pub async fn seek_relative(&self, offset_ms: i64) -> Result<(), Error> {
        self.send_controller_command(ControllerCommand {
            command: ControllerCommandType::SeekRelative,
            volume: None,
            mute: None,
            position_ms: None,
            offset_ms: Some(offset_ms),
        })
        .await
    }
}

/// Binary message type IDs per Sendspin spec.
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
    /// Visualizer loudness data (type 16).
    pub const VISUALIZER_LOUDNESS: u8 = 0x10;
    /// Visualizer beat data (type 17).
    pub const VISUALIZER_BEAT: u8 = 0x11;
    /// Visualizer dominant-frequency data (type 18).
    pub const VISUALIZER_F_PEAK: u8 = 0x12;
    /// Visualizer spectrum data (type 19).
    pub const VISUALIZER_SPECTRUM: u8 = 0x13;
    /// Visualizer energy-onset data (type 20).
    pub const VISUALIZER_PEAK: u8 = 0x14;
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

    /// Check if a binary type ID is for visualizer data (16-20).
    pub fn is_visualizer(type_id: u8) -> bool {
        (VISUALIZER_LOUDNESS..=VISUALIZER_PEAK).contains(&type_id)
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

/// Visualizer chunk from server (binary types 16-20).
#[derive(Debug, Clone)]
pub struct VisualizerChunk {
    /// Visualizer binary message type (16-20).
    pub type_id: u8,
    /// Server timestamp in microseconds.
    pub timestamp: i64,
    /// Raw visualization data bytes, left for the application to decode.
    pub data: Arc<[u8]>,
}

impl VisualizerChunk {
    /// Return the typed visualizer data kind represented by this chunk.
    ///
    /// Returns `None` if a chunk was constructed manually with an invalid
    /// `type_id`; frames parsed by [`Self::from_bytes`] always return `Some`.
    pub fn data_type(&self) -> Option<VisualizerDataType> {
        match self.type_id {
            binary_types::VISUALIZER_LOUDNESS => Some(VisualizerDataType::Loudness),
            binary_types::VISUALIZER_BEAT => Some(VisualizerDataType::Beat),
            binary_types::VISUALIZER_F_PEAK => Some(VisualizerDataType::FPeak),
            binary_types::VISUALIZER_SPECTRUM => Some(VisualizerDataType::Spectrum),
            binary_types::VISUALIZER_PEAK => Some(VisualizerDataType::Peak),
            _ => None,
        }
    }

    /// Parse from a WebSocket binary frame (visualizer types 16-20).
    pub fn from_bytes(frame: &[u8]) -> Result<Self, Error> {
        if frame.len() < 9 {
            return Err(Error::Protocol(format!(
                "Visualizer chunk too short: got {} bytes, need at least 9",
                frame.len()
            )));
        }

        if !binary_types::is_visualizer(frame[0]) {
            return Err(Error::Protocol(format!(
                "Invalid visualizer chunk type: expected 16-20, got {}",
                frame[0]
            )));
        }

        let timestamp = i64::from_be_bytes([
            frame[1], frame[2], frame[3], frame[4], frame[5], frame[6], frame[7], frame[8],
        ]);

        let data = Arc::from(&frame[9..]);

        Ok(Self {
            type_id: frame[0],
            timestamp,
            data,
        })
    }
}

/// Binary frame from server (any type)
#[derive(Debug, Clone)]
pub enum BinaryFrame {
    /// Player audio (type 4)
    Audio(AudioChunk),
    /// Artwork image (types 8-11)
    Artwork(ArtworkChunk),
    /// Visualizer data (types 16-20)
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
            t if binary_types::is_visualizer(t) => {
                Ok(BinaryFrame::Visualizer(VisualizerChunk::from_bytes(frame)?))
            }
            // The router warns when it sees the Unknown variant; parsing
            // itself stays quiet to avoid reporting the same frame twice.
            _ => Ok(BinaryFrame::Unknown {
                type_id,
                data: Arc::from(&frame[1..]),
            }),
        }
    }
}

/// `client/time` messages per burst.
const TIME_BURST_SIZE: usize = 8;

/// How long a burst waits for each `server/time` reply before moving on.
/// Eight timeouts still fit within the 10s the spec allows between burst
/// starts.
const TIME_REPLY_TIMEOUT: Duration = Duration::from_secs(1);

/// Spacing between burst starts while the clock estimate settles.
const TIME_BURST_INTERVAL_SETTLING: Duration = Duration::from_secs(1);

/// Spacing between burst starts once settled, short enough that
/// [`ClockSync::is_stale`] stays false while replies arrive.
const TIME_BURST_INTERVAL: Duration = Duration::from_secs(4);

/// Sends `client/time` in bursts, each message after the previous reply, and
/// feeds the clock filter only the burst's lowest-`max_error` reply above 0.
async fn time_sync(
    sender: WsSender,
    clock: Arc<dyn Clock>,
    clock_sync: Arc<Mutex<ClockSync>>,
    mut replies: UnboundedReceiver<(ServerTime, i64)>,
    mut synced: Option<oneshot::Sender<()>>,
) {
    let reply_max_error = |(st, t4): &(ServerTime, i64)| {
        max_error(
            st.client_transmitted,
            st.server_received,
            st.server_transmitted,
            *t4,
        )
    };
    loop {
        let burst_start = tokio::time::Instant::now();
        let mut best: Option<(ServerTime, i64)> = None;
        for _ in 0..TIME_BURST_SIZE {
            let t1 = clock.now_micros();
            let msg = Message::ClientTime(ClientTime {
                client_transmitted: t1,
            });
            if let Err(e) = sender.send_message(msg).await {
                log::info!("Clock sync task exiting: {}", e);
                return;
            }
            let reply = tokio::time::timeout(TIME_REPLY_TIMEOUT, async {
                // Skip late replies to earlier messages that timed out.
                loop {
                    match replies.recv().await {
                        Some(reply) if reply.0.client_transmitted == t1 => return Some(reply),
                        Some(_) => {}
                        None => return None,
                    }
                }
            })
            .await;
            match reply {
                Ok(Some(reply)) => {
                    let error = reply_max_error(&reply);
                    if error > 0 && best.as_ref().is_none_or(|b| error < reply_max_error(b)) {
                        best = Some(reply);
                    }
                }
                Ok(None) => return,
                Err(_) => {
                    log::debug!("Clock sync: no server/time reply within {TIME_REPLY_TIMEOUT:?}")
                }
            }
        }
        let interval = {
            let mut sync = clock_sync.lock();
            let accepted = best.is_some_and(|(st, t4)| {
                sync.update(
                    st.client_transmitted,
                    st.server_received,
                    st.server_transmitted,
                    t4,
                )
            });
            if accepted || sync.is_synchronized() {
                if let Some(synced) = synced.take() {
                    let _ = synced.send(());
                }
            }
            if accepted && !sync.is_synchronized() {
                Duration::ZERO
            } else if !sync.is_settled() {
                TIME_BURST_INTERVAL_SETTLING
            } else {
                TIME_BURST_INTERVAL
            }
        };
        tokio::time::sleep_until(burst_start + interval).await;
    }
}

/// WebSocket client for Sendspin protocol
pub struct ProtocolClient {
    out_tx: UnboundedSender<WriteCommand>,
    player: PlayerReceiver,
    artwork: ArtworkReceiver,
    visualizer: VisualizerReceiver,
    message_rx: UnboundedReceiver<Message>,
    clock_sync: Arc<Mutex<ClockSync>>,
    server_id: String,
    server_hello: ServerHello,
    first_activation: ServerActivate,
    activation: watch::Receiver<ServerActivate>,
    stream_state: Arc<StreamState>,
    /// Background task guard, aborts tasks on drop
    guard: ConnectionGuard,
}

/// What a connection needs from the builder.
pub(crate) struct ConnectParams {
    pub(crate) identity: Identity,
    pub(crate) hello: ClientHello,
    pub(crate) initial_state: ClientState,
    pub(crate) clock: Arc<dyn Clock>,
    pub(crate) automatic_player_state: bool,
}

/// Aborts background tasks on drop. Hold this alive for the lifetime of the
/// connection.
pub struct ConnectionGuard {
    sender: WsSender,
    router_handle: Option<tokio::task::JoinHandle<()>>,
    sync_handle: Option<tokio::task::JoinHandle<()>>,
    writer_handle: Option<tokio::task::JoinHandle<()>>,
    reporter_handle: Option<tokio::task::JoinHandle<()>>,
}

impl ConnectionGuard {
    /// Gracefully disconnect: enqueue `client/goodbye`, await the writer's
    /// ack so the goodbye + close frames are known to have flushed (or
    /// surface the wire error if they didn't), then reap the writer.
    pub async fn disconnect(mut self, reason: GoodbyeReason) -> Result<(), Error> {
        log::debug!("Disconnecting (reason: {reason:?})");
        // Stop producers before goodbye so no report can be queued after it.
        if let Some(h) = self.reporter_handle.take() {
            h.abort();
            let _ = h.await;
        }
        if let Some(h) = self.sync_handle.take() {
            h.abort();
            let _ = h.await;
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

        log::debug!("Disconnect complete");
        goodbye_result
    }

    /// Resolves once the connection is dead: the router task has exited
    /// (peer close, transport failure, or teardown). Cancel-safe.
    ///
    /// Liveness means the *reader*. A write-side failure alone does not
    /// fire this — on TCP it resets the read side too in short order, and
    /// sends toward a dead writer fail fast rather than hang.
    pub(crate) async fn closed(&mut self) {
        if let Some(h) = &mut self.router_handle {
            let _ = h.await;
        }
    }

    /// Non-blocking [`Self::closed`].
    pub(crate) fn is_closed(&self) -> bool {
        self.router_handle
            .as_ref()
            .is_none_or(tokio::task::JoinHandle::is_finished)
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
        if let Some(h) = self.reporter_handle.take() {
            h.abort();
        }
    }
}

impl Connection {
    /// See [`WsSender::enter_external_source`].
    pub async fn enter_external_source(&self) -> Result<(), Error> {
        self.sender.enter_external_source().await
    }

    /// See [`WsSender::exit_external_source`].
    pub async fn exit_external_source(&self, player: Option<PlayerState>) -> Result<(), Error> {
        self.sender.exit_external_source(player).await
    }
}

impl ProtocolClient {
    /// Connect to Sendspin server
    pub(crate) async fn connect<R>(request: R, params: ConnectParams) -> Result<Self, Error>
    where
        R: IntoClientRequest + Unpin,
    {
        let (ws_stream, _) = connect_async(request)
            .await
            .map_err(|e| Error::Connection(e.to_string()))?;

        Self::drive(ws_stream, params).await
    }

    /// Drive the protocol-client state machine over an already-upgraded
    /// WebSocket stream, through the first `server/activate` and the initial
    /// `client/state`. Shared between outbound `connect()` and inbound
    /// acceptor paths.
    pub(crate) async fn drive<S>(
        ws_stream: WebSocketStream<S>,
        params: ConnectParams,
    ) -> Result<Self, Error>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let ConnectParams {
            identity,
            hello,
            initial_state,
            clock,
            automatic_player_state,
        } = params;
        let unpaired_access = hello.unpaired_access.enabled;
        let (mut write, mut read) = ws_stream.split();

        // Setup messages go directly on the sink rather than through the
        // writer task, so failures are returned synchronously.
        let Session {
            server_id,
            psk_category,
            mut encryptor,
            mut decryptor,
        } = noise::handshake(&mut write, &mut read, &identity).await?;
        let server_hello = match noise::recv_encrypted(&mut read, &mut decryptor).await? {
            Message::ServerHello(server_hello) => server_hello,
            other => {
                return Err(Error::Protocol(format!(
                    "expected server/hello, got {other:?}"
                )))
            }
        };
        log::info!("Connected to server: {} ({})", server_hello.name, server_id);
        noise::send_encrypted(&mut write, &mut encryptor, &Message::ClientHello(hello)).await?;
        let activation = match noise::recv_encrypted(&mut read, &mut decryptor).await? {
            Message::ServerActivate(activation) => activation,
            other => {
                return Err(Error::Protocol(format!(
                    "expected server/activate, got {other:?}"
                )))
            }
        };
        let activation = match admit(None, activation, psk_category, unpaired_access) {
            Ok(activation) => activation,
            Err(reason) => {
                log::warn!("Rejecting the server's activation: {reason:?}");
                let _ = perform_goodbye(&mut write, &mut encryptor, reason.clone()).await;
                return Err(Error::Connection(format!(
                    "rejected server/activate ({reason:?})"
                )));
            }
        };
        let (activation_tx, activation_rx) = watch::channel(activation.clone());

        let (out_tx, out_rx) = unbounded_channel::<WriteCommand>();
        let initial_controls = crate::audio::AppliedControls {
            volume: initial_state
                .player
                .as_ref()
                .and_then(|p| p.volume)
                .unwrap_or(100),
            muted: initial_state
                .player
                .as_ref()
                .and_then(|p| p.muted)
                .unwrap_or(false),
            static_delay_ms: initial_state
                .player
                .as_ref()
                .and_then(|p| p.static_delay_ms)
                .unwrap_or(0),
        };
        let (sinks, player, artwork, visualizer, message_rx) =
            crate::protocol::routing::ReceiveSinks::channel(initial_controls);
        let controls = player.controls();
        let (report_tx, report_rx) = unbounded_channel();
        let clock_sync = Arc::new(Mutex::new(ClockSync::new(Arc::clone(&clock))));
        let stream_state = Arc::new(StreamState::default());
        let sender = WsSender {
            tx: out_tx.clone(),
            reporter: (automatic_player_state && initial_state.player.is_some())
                .then_some(report_tx),
            controls: controls.clone(),
            stream_state: Arc::clone(&stream_state),
        };

        let writer_handle =
            tokio::spawn(writer_task(write, out_rx, encryptor, activation_rx.clone()));
        let (synced_tx, synced_rx) = oneshot::channel();
        let (time_reply_tx, time_reply_rx) = unbounded_channel();
        let router = Router {
            decryptor,
            sinks,
            time_replies: time_reply_tx,
            clock: Arc::clone(&clock),
            stream_state: Arc::clone(&stream_state),
            activation: activation_tx,
            psk_category,
            unpaired_access,
            sender: sender.clone(),
        };
        // The router task handle is used by ConnectionGuard::closed() observers.
        let router_handle = tokio::spawn(router.run(read));

        let sync_handle = tokio::spawn(time_sync(
            sender.clone(),
            clock,
            Arc::clone(&clock_sync),
            time_reply_rx,
            Some(synced_tx),
        ));

        // Dropped on an early return, which aborts the tasks spawned above.
        let mut guard = ConnectionGuard {
            sender: sender.clone(),
            router_handle: Some(router_handle),
            sync_handle: Some(sync_handle),
            writer_handle: Some(writer_handle),
            reporter_handle: None,
        };

        if activation.is_active(PLAYER_V1) {
            match tokio::time::timeout(HANDSHAKE_TIMEOUT, synced_rx).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) => {
                    return Err(Error::Connection(
                        "connection closed before the clock synchronized".to_string(),
                    ))
                }
                Err(_) => {
                    return Err(Error::Connection(
                        "the clock did not synchronize in time".to_string(),
                    ))
                }
            }
        }
        sender
            .send_raw(Message::ClientState(initial_state.clone()))
            .await?;
        if sender.reporter.is_some() {
            guard.reporter_handle = Some(tokio::spawn(crate::protocol::state_reporter::run(
                crate::protocol::state_reporter::Config {
                    commands: report_rx,
                    controls,
                    initial: initial_state,
                    output: out_tx.clone(),
                },
            )));
        }

        Ok(Self {
            out_tx,
            player,
            artwork,
            visualizer,
            message_rx,
            clock_sync,
            server_id,
            server_hello,
            first_activation: activation,
            activation: activation_rx,
            stream_state,
            guard,
        })
    }

    /// Gracefully disconnect: sends `client/goodbye`, closes the WebSocket,
    /// and aborts background tasks.
    pub async fn disconnect(self, reason: GoodbyeReason) -> Result<(), Error> {
        self.guard.disconnect(reason).await
    }

    /// See [`WsSender::enter_external_source`].
    pub async fn enter_external_source(&self) -> Result<(), Error> {
        WsSender {
            tx: self.out_tx.clone(),
            reporter: self.guard.sender.reporter.clone(),
            controls: self.guard.sender.controls.clone(),
            stream_state: Arc::clone(&self.stream_state),
        }
        .enter_external_source()
        .await
    }

    /// See [`WsSender::exit_external_source`].
    pub async fn exit_external_source(&self, player: Option<PlayerState>) -> Result<(), Error> {
        WsSender {
            tx: self.out_tx.clone(),
            reporter: self.guard.sender.reporter.clone(),
            controls: self.guard.sender.controls.clone(),
            stream_state: Arc::clone(&self.stream_state),
        }
        .exit_external_source(player)
        .await
    }

    /// Get reference to clock sync
    pub fn clock_sync(&self) -> Arc<Mutex<ClockSync>> {
        Arc::clone(&self.clock_sync)
    }

    /// The server's `server_id` from `server/init`, its static public key.
    pub fn server_id(&self) -> &str {
        &self.server_id
    }

    /// The `server/hello` received during the handshake.
    pub fn server_hello(&self) -> &ServerHello {
        &self.server_hello
    }

    /// The first `server/activate`, with `active_roles` filled in.
    pub fn activation(&self) -> &ServerActivate {
        &self.first_activation
    }

    /// Split into separate receivers for concurrent processing.
    ///
    /// This allows using `tokio::select!` to process messages and binary
    /// data concurrently. Dropping an unused role receiver disables delivery
    /// and avoids unread buffering; retain the connection guard until shutdown.
    pub fn split(self) -> Connection {
        let sender = WsSender {
            tx: self.out_tx,
            reporter: self.guard.sender.reporter.clone(),
            controls: self.guard.sender.controls.clone(),
            stream_state: self.stream_state,
        };
        let controller = Controller {
            sender: sender.clone(),
            activation: self.activation,
        };
        Connection {
            messages: self.message_rx,
            player: self.player,
            artwork: self.artwork,
            visualizer: self.visualizer,
            clock_sync: self.clock_sync,
            sender,
            controller,
            server_id: self.server_id,
            server_hello: self.server_hello,
            activation: self.first_activation,
            guard: self.guard,
        }
    }
}

/// Reads the server's messages, decrypting them and applying protocol state
/// before handing them to the role sinks.
struct Router {
    decryptor: Decryptor,
    sinks: crate::protocol::routing::ReceiveSinks,
    time_replies: UnboundedSender<(ServerTime, i64)>,
    clock: Arc<dyn Clock>,
    stream_state: Arc<StreamState>,
    activation: watch::Sender<ServerActivate>,
    psk_category: PskCategory,
    unpaired_access: bool,
    sender: WsSender,
}

impl Router {
    async fn run<S>(mut self, mut read: SplitStream<WebSocketStream<S>>)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let mut audio_chunk_count = 0u64;
        let mut visualizer_chunk_count = 0u64;

        while let Some(msg) = read.next().await {
            let data = match msg {
                Ok(WsMessage::Binary(data)) => data,
                Ok(WsMessage::Text(_)) => {
                    log::warn!("Closing the connection: cleartext message after the handshake");
                    self.sender.close();
                    break;
                }
                Ok(WsMessage::Close(_)) => {
                    log::info!("Server closed connection");
                    break;
                }
                Err(e) => {
                    log::error!("WebSocket error: {}", e);
                    break;
                }
                Ok(_) => continue,
            };
            let plaintext = match self.decryptor.open(&data) {
                Ok(Some(plaintext)) => plaintext,
                Ok(None) => continue,
                Err(e) => {
                    log::warn!("Closing the connection: {e}");
                    self.sender.close();
                    break;
                }
            };
            if plaintext[0] == MESSAGE_TYPE_JSON {
                if !self.handle_json(&plaintext[1..], self.clock.now_micros()) {
                    break;
                }
                continue;
            }
            match BinaryFrame::from_bytes(&plaintext) {
                Ok(BinaryFrame::Audio(chunk)) => {
                    audio_chunk_count = match audio_chunk_count.checked_add(1) {
                        Some(count) => count,
                        None => {
                            log::error!("Audio chunk counter exhausted; closing router");
                            break;
                        }
                    };
                    if should_log_sample(audio_chunk_count) {
                        log::trace!(
                            "Received audio chunk: chunk={}, timestamp={}µs, payload_bytes={}, wire_bytes={}",
                            audio_chunk_count,
                            chunk.timestamp,
                            chunk.data.len(),
                            data.len()
                        );
                    }
                    self.sinks.dispatch_binary(BinaryFrame::Audio(chunk));
                }
                Ok(BinaryFrame::Artwork(chunk)) => {
                    // Artwork arrives in short bursts on track changes, so
                    // every chunk is worth a line; audio and visualizer
                    // chunks stream continuously and are sampled instead.
                    log::trace!(
                        "Received artwork chunk: channel={}, timestamp={}µs, payload_bytes={}",
                        chunk.channel,
                        chunk.timestamp,
                        chunk.data.len()
                    );
                    self.sinks.dispatch_binary(BinaryFrame::Artwork(chunk));
                }
                Ok(BinaryFrame::Visualizer(chunk)) => {
                    visualizer_chunk_count = match visualizer_chunk_count.checked_add(1) {
                        Some(count) => count,
                        None => {
                            log::error!("Visualizer chunk counter exhausted; closing router");
                            break;
                        }
                    };
                    if should_log_sample(visualizer_chunk_count) {
                        log::trace!(
                            "Received visualizer chunk: chunk={}, timestamp={}µs, payload_bytes={}",
                            visualizer_chunk_count,
                            chunk.timestamp,
                            chunk.data.len()
                        );
                    }
                    self.sinks.dispatch_binary(BinaryFrame::Visualizer(chunk));
                }
                Ok(BinaryFrame::Unknown { type_id, .. }) => {
                    log::warn!("Received unknown binary type: {}", type_id);
                }
                Err(e) => {
                    log::warn!("Failed to parse binary frame: {}", e);
                }
            }
        }
        self.sinks.close();
        log::debug!("Message router: WebSocket stream ended");
    }

    /// Handle one JSON message. Returns `false` when the connection is closing.
    fn handle_json(&mut self, json: &[u8], arrival: i64) -> bool {
        let msg = match serde_json::from_slice::<Message>(json) {
            Ok(msg) => msg,
            Err(e) => {
                log::warn!(
                    "Failed to parse message: {} (payload: {})",
                    e,
                    String::from_utf8_lossy(json)
                );
                return true;
            }
        };
        match msg {
            // ServerTime is consumed here for clock sync and intentionally
            // NOT forwarded to message_rx consumers, since it's an internal
            // protocol detail. It also arrives at 1Hz for as long as the
            // connection lives, so it stays out of the debug view.
            // ClockSync::update logs the computed sync state instead.
            Message::ServerTime(st) => {
                let _ = self.time_replies.send((st, arrival));
            }
            Message::ServerActivate(activation) => return self.apply_activation(activation),
            msg => {
                log::debug!("Received message: {:?}", msg);
                // Settle the request-format gate before forwarding, so a
                // consumer reacting to this stream/start or stream/end sees
                // current state.
                match &msg {
                    Message::StreamStart(start) => self.stream_state.note_stream_start(start),
                    Message::StreamEnd(end) => self.stream_state.note_stream_end(end),
                    _ => {}
                }
                self.sinks.dispatch_message(msg);
            }
        }
        true
    }

    /// Apply a later `server/activate`. Returns `false` when it was rejected
    /// and the connection is closing.
    fn apply_activation(&mut self, activation: ServerActivate) -> bool {
        log::debug!("Received server/activate: {:?}", activation);
        let current = self.activation.borrow().clone();
        let next = match admit(
            current.active_roles.as_deref(),
            activation,
            self.psk_category,
            self.unpaired_access,
        ) {
            Ok(next) => next,
            Err(reason) => {
                log::warn!("Rejecting the server's activation: {reason:?}");
                let _ = self.sender.send_goodbye(reason);
                return false;
            }
        };
        // Removing a role stops its stream even if no stream/end preceded it.
        let ended: Vec<String> = [
            (PLAYER_V1, ROLE_PLAYER, self.stream_state.is_player_active()),
            (
                ARTWORK_V1,
                ROLE_ARTWORK,
                self.stream_state.is_artwork_active(),
            ),
            (
                VISUALIZER_V1,
                ROLE_VISUALIZER,
                self.stream_state.is_visualizer_active(),
            ),
        ]
        .into_iter()
        .filter(|(role, _, streaming)| *streaming && !next.is_active(role))
        .map(|(_, stream, _)| stream.to_string())
        .collect();
        if !ended.is_empty() {
            let end = StreamEnd { roles: Some(ended) };
            self.stream_state.note_stream_end(&end);
            self.sinks.dispatch_message(Message::StreamEnd(end));
        }
        let player_added = next.is_active(PLAYER_V1) && !current.is_active(PLAYER_V1);
        let artwork_added = next.is_active(ARTWORK_V1) && !current.is_active(ARTWORK_V1);
        self.activation.send_replace(next.clone());
        if artwork_added {
            let _ = self.sender.tx.send(WriteCommand::ReportArtwork);
        }
        if player_added {
            self.sender.queue_player_report();
        }
        self.sinks.dispatch_message(Message::ServerActivate(next));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AppliedControls;
    use crate::sync::raw_clock::Clock;
    use futures_util::{SinkExt, StreamExt};
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::time::Duration;
    use tokio::io::duplex;
    use tokio_tungstenite::tungstenite::protocol::Role;

    #[test]
    fn artwork_state_is_sent_only_while_artwork_is_active() {
        let artwork = serde_json::from_value(serde_json::json!({
            "channels": [{"source": "album", "format": "jpeg", "width": 256, "height": 256}]
        }))
        .unwrap();
        let state = ClientState {
            artwork: Some(artwork),
            ..Default::default()
        };
        let mut activation = ServerActivate {
            activities: vec![],
            active_roles: Some(vec![ARTWORK_V1.into()]),
        };
        let Some(Message::ClientState(filtered)) =
            strip_inactive_roles(Message::ClientState(state.clone()), &activation)
        else {
            panic!("active artwork state was dropped");
        };
        assert_eq!(filtered.artwork, state.artwork);
        activation.active_roles = Some(vec![PLAYER_V1.into()]);
        assert!(strip_inactive_roles(Message::ClientState(state.clone()), &activation).is_none());
        let Some(Message::ClientState(filtered)) = strip_inactive_roles(
            Message::ClientState(ClientState {
                state: Some(ClientSyncState::Synchronized),
                ..state
            }),
            &activation,
        ) else {
            panic!("client-level state was dropped");
        };
        assert!(filtered.artwork.is_none());
    }

    struct FakeClock(AtomicI64);

    impl FakeClock {
        fn new(now: i64) -> Self {
            Self(AtomicI64::new(now))
        }
    }

    impl Clock for FakeClock {
        fn now_micros(&self) -> i64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    #[tokio::test]
    async fn router_consumes_server_time_after_all_endpoints_are_dropped() {
        let (left, right) = duplex(4096);
        let mut server = WebSocketStream::from_raw_socket(left, Role::Server, None).await;
        let client = WebSocketStream::from_raw_socket(right, Role::Client, None).await;
        let (_, read) = client.split();
        let (sinks, player, artwork, visualizer, messages) =
            crate::protocol::routing::ReceiveSinks::channel(AppliedControls {
                volume: 100,
                muted: false,
                static_delay_ms: 0,
            });
        drop(player);
        drop(artwork);
        drop(visualizer);
        drop(messages);

        let (mut encryptor, decryptor) = noise::test_session_pair();
        let clock: Arc<dyn Clock> = Arc::new(FakeClock::new(3_000));
        let (time_reply_tx, mut time_reply_rx) = unbounded_channel();
        let stream_state = Arc::new(StreamState::default());
        let (tx, _) = unbounded_channel();
        let router = Router {
            decryptor,
            sinks,
            time_replies: time_reply_tx,
            clock,
            stream_state: Arc::clone(&stream_state),
            activation: watch::channel(ServerActivate {
                activities: vec![],
                active_roles: Some(vec![PLAYER_V1.to_string()]),
            })
            .0,
            psk_category: PskCategory::Sentinel,
            unpaired_access: true,
            sender: WsSender {
                tx,
                reporter: None,
                controls: watch::channel(AppliedControls {
                    volume: 100,
                    muted: false,
                    static_delay_ms: 0,
                })
                .1,
                stream_state,
            },
        };
        let mut router = tokio::spawn(router.run(read));

        let mut audio = 42_i64.to_be_bytes().to_vec();
        audio.extend_from_slice(&[0, 0x10, 0, 0xf0]);
        let mut frames = encryptor
            .seal_json(&Message::StreamStart(StreamStart {
                player: Some(crate::protocol::messages::StreamPlayerConfig {
                    codec: "pcm".into(),
                    sample_rate: 48_000,
                    channels: 2,
                    bit_depth: 16,
                    codec_header: None,
                }),
                artwork: None,
                visualizer: None,
            }))
            .unwrap();
        frames.extend(encryptor.seal(binary_types::PLAYER_AUDIO, &audio).unwrap());
        frames.extend(
            encryptor
                .seal_json(&Message::ServerCommand(
                    crate::protocol::messages::ServerCommand { player: None },
                ))
                .unwrap(),
        );
        frames.extend(
            encryptor
                .seal_json(&Message::ServerTime(
                    crate::protocol::messages::ServerTime {
                        client_transmitted: 1_000,
                        server_received: 2_000,
                        server_transmitted: 3_000,
                    },
                ))
                .unwrap(),
        );
        for frame in frames {
            server.send(frame).await.unwrap();
        }
        server.send(WsMessage::Close(None)).await.unwrap();
        match tokio::time::timeout(Duration::from_secs(5), &mut router).await {
            Ok(result) => {
                result.expect("router task panicked while processing dropped destinations")
            }
            Err(_) => {
                router.abort();
                panic!(
                    "router did not terminate after all destinations were dropped and peer close"
                );
            }
        }
        let (reply, t4) = time_reply_rx.recv().await.expect("server/time forwarded");
        assert_eq!(reply.client_transmitted, 1_000);
        assert_eq!(t4, 3_000);
    }

    #[tokio::test]
    async fn burst_sync_notifies_only_after_selected_samples_synchronize() {
        let clock = Arc::new(FakeClock::new(1_000_000));
        let sync_clock: Arc<dyn Clock> = clock.clone();
        let clock_sync = Arc::new(Mutex::new(ClockSync::new(sync_clock.clone())));
        let (tx, mut commands) = unbounded_channel();
        let (replies, reply_rx) = unbounded_channel();
        let (synced_tx, mut synced_rx) = oneshot::channel();
        let sender = WsSender {
            tx,
            reporter: None,
            controls: watch::channel(AppliedControls {
                volume: 100,
                muted: false,
                static_delay_ms: 0,
            })
            .1,
            stream_state: Arc::new(StreamState::default()),
        };
        let task = tokio::spawn(time_sync(
            sender,
            sync_clock,
            Arc::clone(&clock_sync),
            reply_rx,
            Some(synced_tx),
        ));
        tokio::time::timeout(Duration::from_secs(5), async {
            for index in 0..TIME_BURST_SIZE * 2 {
                let Some(WriteCommand::Send { msg, ack }) = commands.recv().await else {
                    panic!("expected client/time");
                };
                let Message::ClientTime(time) = *msg else {
                    panic!("expected client/time");
                };
                // `synced` fires once the first accepted burst update lands,
                // which happens after the count >= 1 startup gate is reached.
                if index < TIME_BURST_SIZE {
                    assert!(synced_rx.try_recv().is_err());
                    assert_eq!(clock_sync.lock().rtt_micros(), None);
                } else {
                    assert_eq!(clock_sync.lock().rtt_micros(), Some(100));
                }
                let rtt = if index % TIME_BURST_SIZE == 0 {
                    100
                } else {
                    200
                };
                let t1 = time.client_transmitted;
                clock.0.store(t1 + 1_000, Ordering::Relaxed);
                ack.send(Ok(())).unwrap();
                replies
                    .send((
                        ServerTime {
                            client_transmitted: t1,
                            server_received: t1 + rtt / 2,
                            server_transmitted: t1 + rtt / 2,
                        },
                        t1 + rtt,
                    ))
                    .unwrap();
            }
            (&mut synced_rx).await.unwrap();
            // `synced` fires on the first accepted update; the second
            // burst's update lands only when the task consumes the last
            // reply. A new burst's first send proves that happened.
            let Some(WriteCommand::Send { msg, .. }) = commands.recv().await else {
                panic!("expected the next client/time burst");
            };
            let Message::ClientTime(_) = *msg else {
                panic!("expected client/time");
            };
        })
        .await
        .unwrap();
        assert!(clock_sync.lock().is_synchronized());
        assert_eq!(clock_sync.lock().rtt_micros(), Some(100));
        task.abort();
        let _ = task.await;
    }
}
