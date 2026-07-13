// ABOUTME: Per-client connection actor for the server role: drives the
// ABOUTME: server-side handshake, time-sync echo, and message dispatch.

use crate::error::Error;
use crate::protocol::messages::{
    ClientHello, ConnectionReason, Message, PlayerCommand, ServerCommand, ServerHello, ServerTime,
    StreamClear, StreamEnd, StreamPlayerConfig, StreamStart,
};
use crate::server::binary::encode_audio_frame;
use crate::sync::raw_clock::Clock;
use futures_util::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio_tungstenite::{tungstenite::Message as WsMessage, WebSocketStream};

/// The only role this server negotiates in v1. See the crate-level server
/// docs for the list of roles deferred for a later contribution
/// (color/visualizer/artwork/controller/metadata).
const PLAYER_ROLE: &str = "player@v1";

enum WriteCommand {
    Send {
        msg: WsMessage,
        ack: tokio::sync::oneshot::Sender<Result<(), Error>>,
    },
    /// `server/time` reply: `server_transmitted` is stamped from the clock
    /// immediately before the frame reaches the wire, not when this command
    /// was enqueued — queueing delay would otherwise leak into the client's
    /// clock filter as measurement error (this is why it's its own variant
    /// rather than a pre-built `Send`).
    TimeReply {
        client_transmitted: i64,
        server_received: i64,
        ack: tokio::sync::oneshot::Sender<Result<(), Error>>,
    },
    Close {
        ack: tokio::sync::oneshot::Sender<Result<(), Error>>,
    },
}

async fn writer_task<S>(
    mut sink: SplitSink<WebSocketStream<S>, WsMessage>,
    mut rx: UnboundedReceiver<WriteCommand>,
    clock: Arc<dyn Clock>,
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
            WriteCommand::TimeReply {
                client_transmitted,
                server_received,
                ack,
            } => {
                let reply = Message::ServerTime(ServerTime {
                    client_transmitted,
                    server_received,
                    server_transmitted: clock.now_micros(),
                });
                let result = match serde_json::to_string(&reply) {
                    Ok(json) => sink
                        .send(WsMessage::Text(json.into()))
                        .await
                        .map_err(|e| Error::WebSocket(e.to_string())),
                    Err(e) => Err(Error::Protocol(e.to_string())),
                };
                let failed = result.is_err();
                let _ = ack.send(result);
                if failed {
                    break;
                }
            }
            WriteCommand::Close { ack } => {
                let result = sink
                    .close()
                    .await
                    .map_err(|e| Error::WebSocket(e.to_string()));
                let _ = ack.send(result);
                break;
            }
        }
    }
    log::debug!("Server connection writer task exiting");
}

/// Sender half of a server-role connection. Cheap to clone.
#[derive(Debug, Clone)]
pub struct ServerSender {
    tx: UnboundedSender<WriteCommand>,
}

impl ServerSender {
    async fn send_message(&self, msg: Message) -> Result<(), Error> {
        let json = serde_json::to_string(&msg).map_err(|e| Error::Protocol(e.to_string()))?;
        log::debug!("Sending message: {}", json);
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(WriteCommand::Send {
                msg: WsMessage::Text(json.into()),
                ack: ack_tx,
            })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
        ack_rx
            .await
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?
    }

    /// Announce the start of a player audio stream. Send this once before
    /// the first [`Self::send_audio_chunk`].
    pub async fn send_stream_start(&self, player: StreamPlayerConfig) -> Result<(), Error> {
        self.send_message(Message::StreamStart(StreamStart {
            player: Some(player),
            artwork: None,
            visualizer: None,
        }))
        .await
    }

    /// Push one player audio chunk. `timestamp_us` is the intended playback
    /// time in this server's clock domain (see [`crate::sync::raw_clock::Clock`]);
    /// the client converts it to its own domain using the offset/drift it
    /// tracks from `server/time` replies.
    pub async fn send_audio_chunk(&self, timestamp_us: i64, payload: &[u8]) -> Result<(), Error> {
        let frame = encode_audio_frame(timestamp_us, payload);
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(WriteCommand::Send {
                msg: WsMessage::Binary(frame.into()),
                ack: ack_tx,
            })
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
        ack_rx
            .await
            .map_err(|_| Error::WebSocket("connection closed".to_string()))?
    }

    /// End the player audio stream.
    pub async fn send_stream_end(&self) -> Result<(), Error> {
        self.send_message(Message::StreamEnd(StreamEnd {
            roles: Some(vec!["player".to_string()]),
        }))
        .await
    }

    /// Ask the client to discard any buffered-but-unplayed audio (e.g. after
    /// a seek), without ending the stream.
    pub async fn send_stream_clear(&self) -> Result<(), Error> {
        self.send_message(Message::StreamClear(StreamClear {
            roles: Some(vec!["player".to_string()]),
        }))
        .await
    }

    /// Send a player command (volume, mute, static delay) to the client.
    pub async fn send_player_command(&self, command: PlayerCommand) -> Result<(), Error> {
        self.send_message(Message::ServerCommand(ServerCommand {
            player: Some(command),
        }))
        .await
    }
}

/// Aborts background tasks on drop. Hold this alive for the lifetime of the
/// connection — mirrors [`crate::protocol::client::ConnectionGuard`].
pub struct ServerConnectionGuard {
    sender: ServerSender,
    router_handle: Option<tokio::task::JoinHandle<()>>,
    writer_handle: Option<tokio::task::JoinHandle<()>>,
}

impl ServerConnectionGuard {
    /// Close the connection. Unlike the client role, the server has no
    /// `goodbye` message of its own to send — it just closes the socket
    /// (optionally after the caller has already sent `stream/end`).
    pub async fn disconnect(mut self) -> Result<(), Error> {
        let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
        let close_result = self
            .sender
            .tx
            .send(WriteCommand::Close { ack: ack_tx })
            .map_err(|_| Error::WebSocket("connection closed".to_string()));
        let result = match close_result {
            Ok(()) => ack_rx
                .await
                .map_err(|_| Error::WebSocket("connection closed".to_string()))?,
            Err(e) => Err(e),
        };
        if let Some(h) = self.writer_handle.take() {
            let _ = h.await;
        }
        if let Some(h) = self.router_handle.take() {
            h.abort();
        }
        result
    }
}

impl Drop for ServerConnectionGuard {
    fn drop(&mut self) {
        if let Some(h) = self.router_handle.take() {
            h.abort();
        }
        if let Some(h) = self.writer_handle.take() {
            h.abort();
        }
    }
}

/// A single accepted client, past the handshake. Returned by
/// [`crate::server::ServerListener::accept`].
pub struct ServerConnection {
    /// The client's `client/hello` payload — identity, declared capabilities,
    /// device info. Kept in full so callers can read `player@v1_support`
    /// (supported formats, buffer capacity) before starting a stream.
    hello: ClientHello,
    /// Roles this server granted this client (currently always `["player@v1"]`
    /// if the client declared support for it, else empty).
    active_roles: Vec<String>,
    /// `client/state`, `client/command`, and `client/goodbye` messages,
    /// forwarded as received. `client/time` is consumed internally (time-sync
    /// echo) and never forwarded here — same convention as
    /// [`crate::protocol::client::Connection::messages`].
    messages: UnboundedReceiver<Message>,
    sender: ServerSender,
    guard: ServerConnectionGuard,
}

impl ServerConnection {
    /// The client's `client/hello` payload.
    pub fn hello(&self) -> &ClientHello {
        &self.hello
    }

    /// Convenience accessor for `hello().client_id`.
    pub fn client_id(&self) -> &str {
        &self.hello.client_id
    }

    /// Roles granted to this client.
    pub fn active_roles(&self) -> &[String] {
        &self.active_roles
    }

    /// A cheap-to-clone sender for pushing stream control and audio messages
    /// to this client, usable independently of `&mut self`.
    pub fn sender(&self) -> ServerSender {
        self.sender.clone()
    }

    /// Receive the next `client/state`, `client/command`, or `client/goodbye`
    /// message. Returns `None` once the connection has closed.
    pub async fn recv_message(&mut self) -> Option<Message> {
        self.messages.recv().await
    }

    /// Close the connection.
    pub async fn disconnect(self) -> Result<(), Error> {
        self.guard.disconnect().await
    }

    /// Drive the server-side handshake and message loop over an
    /// already-handshaked WebSocket stream. Shared by
    /// [`crate::server::ServerListener::accept`] and tests.
    pub(crate) async fn drive<S>(
        ws_stream: WebSocketStream<S>,
        server_id: &str,
        server_name: &str,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, Error>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let (mut write, mut read) = ws_stream.split();

        log::debug!("Waiting for client/hello...");
        let hello = loop {
            let Some(result) = read.next().await else {
                return Err(Error::Connection(
                    "connection closed before client/hello".to_string(),
                ));
            };
            match result {
                Ok(WsMessage::Text(text)) => {
                    let msg: Message = serde_json::from_str(&text).map_err(|e| {
                        log::warn!("Failed to parse client message: {} (payload: {})", e, text);
                        Error::Protocol(e.to_string())
                    })?;
                    match msg {
                        Message::ClientHello(hello) => {
                            if hello.version != 1 {
                                return Err(Error::Protocol(format!(
                                    "unsupported protocol version {} (only 1 is supported)",
                                    hello.version
                                )));
                            }
                            break hello;
                        }
                        other => {
                            return Err(Error::Protocol(format!(
                                "expected client/hello, got {:?}",
                                other
                            )))
                        }
                    }
                }
                Ok(WsMessage::Ping(_)) | Ok(WsMessage::Pong(_)) => continue,
                Ok(WsMessage::Close(_)) => {
                    return Err(Error::Connection("client closed connection".to_string()))
                }
                Ok(_) => continue,
                Err(e) => return Err(Error::WebSocket(e.to_string())),
            }
        };
        log::debug!("Received client/hello: {:?}", hello);

        let active_roles: Vec<String> = if hello.supported_roles.iter().any(|r| r == PLAYER_ROLE) {
            vec![PLAYER_ROLE.to_string()]
        } else {
            Vec::new()
        };

        let server_hello = Message::ServerHello(ServerHello {
            server_id: server_id.to_string(),
            name: server_name.to_string(),
            version: 1,
            active_roles: active_roles.clone(),
            connection_reason: ConnectionReason::Discovery,
        });
        let json =
            serde_json::to_string(&server_hello).map_err(|e| Error::Protocol(e.to_string()))?;
        write
            .send(WsMessage::Text(json.into()))
            .await
            .map_err(|e| Error::WebSocket(e.to_string()))?;

        let (out_tx, out_rx) = unbounded_channel::<WriteCommand>();
        let (message_tx, message_rx) = unbounded_channel();

        let writer_handle = tokio::spawn(writer_task(write, out_rx, Arc::clone(&clock)));

        let out_tx_router = out_tx.clone();
        let router_handle = tokio::spawn(async move {
            Self::message_router(read, message_tx, out_tx_router, clock).await;
        });

        Ok(Self {
            hello,
            active_roles,
            messages: message_rx,
            sender: ServerSender { tx: out_tx.clone() },
            guard: ServerConnectionGuard {
                sender: ServerSender { tx: out_tx },
                router_handle: Some(router_handle),
                writer_handle: Some(writer_handle),
            },
        })
    }

    async fn message_router<S>(
        mut read: SplitStream<WebSocketStream<S>>,
        message_tx: UnboundedSender<Message>,
        out_tx: UnboundedSender<WriteCommand>,
        clock: Arc<dyn Clock>,
    ) where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let mut message_closed = false;

        while let Some(msg) = read.next().await {
            match msg {
                Ok(WsMessage::Text(text)) => {
                    // Capture receive time before deserialization so
                    // `server_received` is as close to true arrival as possible.
                    let server_received = clock.now_micros();
                    match serde_json::from_str::<Message>(&text) {
                        Ok(Message::ClientTime(t)) => {
                            let (ack_tx, _ack_rx) = tokio::sync::oneshot::channel();
                            if out_tx
                                .send(WriteCommand::TimeReply {
                                    client_transmitted: t.client_transmitted,
                                    server_received,
                                    ack: ack_tx,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        Ok(Message::ClientHello(_)) => {
                            log::warn!("Ignoring unexpected client/hello after handshake");
                        }
                        Ok(msg) => {
                            log::debug!("Received message: {:?}", msg);
                            if !message_closed && message_tx.send(msg).is_err() {
                                log::error!(
                                    "Message receiver dropped — messages will be discarded"
                                );
                                message_closed = true;
                            }
                        }
                        Err(e) => {
                            log::warn!("Failed to parse message: {} (payload: {})", e, text);
                        }
                    }
                }
                Ok(WsMessage::Binary(_)) => {
                    // A client never sends binary frames in the current
                    // protocol (audio/artwork/visualizer are server->client
                    // only); log and ignore rather than erroring the
                    // connection over a forward-compatible future frame.
                    log::warn!("Ignoring unexpected binary frame from client");
                }
                Ok(WsMessage::Ping(_)) | Ok(WsMessage::Pong(_)) => {}
                Ok(WsMessage::Close(_)) => {
                    log::info!("Client closed connection");
                    break;
                }
                Err(e) => {
                    log::warn!("WebSocket error: {}", e);
                    break;
                }
                _ => {}
            }
        }
        log::debug!("Message router: WebSocket stream ended");
    }
}
