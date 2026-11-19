mod common;

use common::TestServer;
use sendspin::audio::{AudioBuffer, OutputError, PlayerEvent, PlayerOutput};
use sendspin::error::Error;
use sendspin::protocol::client::binary_types;
use sendspin::protocol::messages::{
    ArtworkFormatRequest, ArtworkSource, AudioFormatSpec, ClientCommand, ClientSyncState,
    ControllerCommandType, GoodbyeReason, ImageFormat, Message, PlayerCommand, PlayerCommandType,
    PlayerFormatRequest, PlayerState, PlayerV1Support, RepeatMode, ServerActivate, ServerCommand,
    ServerTime, StreamArtworkChannelConfig, StreamArtworkConfig, StreamClear, StreamEnd,
    StreamPlayerConfig, StreamRequestFormat, StreamStart, VisualizerDataType,
    VisualizerFormatRequest,
};
use sendspin::roles::{ArtworkEvent, ArtworkOutput, VisualizerEvent, VisualizerOutput};
use sendspin::{MemoryStore, ProtocolClientBuilder};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message as WsMessage};

fn player_support() -> PlayerV1Support {
    let format = |codec: &str, bit_depth| AudioFormatSpec {
        codec: codec.into(),
        channels: 2,
        sample_rate: 48_000,
        bit_depth,
    };
    PlayerV1Support {
        supported_formats: vec![format("opus", 16), format("pcm", 24), format("pcm", 16)],
        buffer_capacity: 50 * 1024 * 1024,
        supported_commands: vec![],
    }
}

#[derive(Clone, Default)]
struct TestOutput;
impl PlayerOutput for TestOutput {
    fn commit(&self, _: AudioBuffer) -> Result<(), OutputError> {
        Ok(())
    }
    fn clear(&self) -> Result<(), OutputError> {
        Ok(())
    }
    fn set_volume(&self, _: u8) -> Result<(), OutputError> {
        Ok(())
    }
    fn set_mute(&self, _: bool) -> Result<(), OutputError> {
        Ok(())
    }
    fn set_static_delay(&self, _: u16) -> Result<(), OutputError> {
        Ok(())
    }
}
impl ArtworkOutput for TestOutput {
    type Prepared = Vec<u8>;
    fn commit(&self, _: u8, _: Self::Prepared) -> Result<(), OutputError> {
        Ok(())
    }
    fn end(&self) -> Result<(), OutputError> {
        Ok(())
    }
}
impl VisualizerOutput for TestOutput {
    type Prepared = Vec<u8>;
    fn commit(&self, _: Self::Prepared) -> Result<(), OutputError> {
        Ok(())
    }
    fn clear(&self) -> Result<(), OutputError> {
        Ok(())
    }
    fn end(&self) -> Result<(), OutputError> {
        Ok(())
    }
}

/// Start a local server for one client. It activates `granted_roles`, or every
/// role the client supports when `None`, and then serves the connection.
async fn start_server(
    granted_roles: Option<Vec<String>>,
) -> (
    String,
    tokio::sync::mpsc::UnboundedReceiver<String>,
    tokio::sync::mpsc::UnboundedSender<WsMessage>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let (client_tx, client_rx) = tokio::sync::mpsc::unbounded_channel();
    let (server_tx, server_rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let (mut server, hello) = TestServer::handshake(accept_async(stream).await.unwrap()).await;
        server
            .activate(granted_roles.unwrap_or(hello.supported_roles))
            .await;
        server.serve(client_tx, server_rx).await;
    });
    (url, client_rx, server_tx, handle)
}

/// Start a local server that activates every role the client supports and
/// returns the client messages after the handshake, except `client/time`.
async fn start_test_server() -> (
    String,
    tokio::sync::mpsc::UnboundedReceiver<String>,
    tokio::task::JoinHandle<()>,
) {
    let (url, rx, _, handle) = start_server(None).await;
    (url, rx, handle)
}

/// Serialize a protocol message into a WebSocket text frame.
fn text_message(msg: &Message) -> WsMessage {
    WsMessage::Text(serde_json::to_string(msg).unwrap().into())
}

/// A server→client `stream/start` that starts a player stream.
fn player_stream_start() -> WsMessage {
    text_message(&Message::StreamStart(StreamStart {
        player: Some(StreamPlayerConfig {
            codec: "pcm".to_string(),
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        }),
        artwork: None,
        visualizer: None,
    }))
}

/// A server→client `stream/start` that starts an artwork stream.
fn artwork_stream_start() -> WsMessage {
    text_message(&Message::StreamStart(StreamStart {
        player: None,
        artwork: Some(StreamArtworkConfig {
            channels: vec![StreamArtworkChannelConfig {
                source: ArtworkSource::Artist,
                format: ImageFormat::Png,
                width: 640,
                height: 640,
            }],
        }),
        visualizer: None,
    }))
}

/// A server→client `stream/start` that starts a visualizer stream.
fn visualizer_stream_start() -> WsMessage {
    text_message(&Message::StreamStart(StreamStart {
        player: None,
        artwork: None,
        visualizer: Some(sendspin::protocol::messages::StreamVisualizerConfig {
            types: vec![VisualizerDataType::Spectrum],
            rate_max: 60,
            tracks_downbeats: None,
            spectrum: None,
        }),
    }))
}

/// A server→client `stream/end` for the given roles (all streams if `None`).
fn stream_end(roles: Option<Vec<String>>) -> WsMessage {
    text_message(&Message::StreamEnd(StreamEnd { roles }))
}

/// Discard the `client/state` the client emits immediately after handshake.
async fn discard_initial_state(rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>) {
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for initial state")
        .expect("channel closed");
}

async fn recv_stream_request_format(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>,
) -> StreamRequestFormat {
    loop {
        let msg_text = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("timed out waiting for stream/request-format")
            .expect("channel closed");
        match serde_json::from_str::<Message>(&msg_text).unwrap() {
            Message::StreamRequestFormat(request) => break request,
            Message::ClientTime(_) => continue,
            other => panic!("expected StreamRequestFormat, got {:?}", other),
        }
    }
}

#[tokio::test]
async fn test_connect_sends_initial_state() {
    let (url, mut rx, _handle) = start_test_server().await;

    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(PlayerV1Support {
            supported_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 1024,
            supported_commands: vec![],
        })
        .initial_player_state(PlayerState {
            volume: Some(75),
            muted: Some(false),
            static_delay_ms: Some(100),
            required_lead_time_ms: Some(500),
            min_buffer_ms: Some(500),
            supported_commands: None,
        })
        .build();

    let client = builder.connect(&url).await.unwrap();

    // Regression guard: the activation needed for multi-server arbitration
    // must survive `connect()` just as it does `accept()`.
    assert!(client.activation().is_active("player@v1"));

    // First message after handshake should be client/state
    let first_msg = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for message")
        .expect("channel closed");

    let parsed: Message = serde_json::from_str(&first_msg).unwrap();
    match parsed {
        Message::ClientState(cs) => {
            assert_eq!(cs.state, Some(ClientSyncState::Synchronized));
            let player = cs.player.expect("expected player state");
            assert_eq!(player.volume, Some(75));
            assert_eq!(player.muted, Some(false));
            assert_eq!(player.static_delay_ms, Some(100));
        }
        other => panic!("expected ClientState, got {:?}", other),
    }
}

#[tokio::test]
async fn test_connect_without_player_state_sends_state_without_player() {
    let (url, mut rx, _handle) = start_test_server().await;

    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build();

    let _client = builder.connect(&url).await.unwrap();

    // Builder always sends client/state (with state=Synchronized), even without player state
    let first_msg = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for message")
        .expect("channel closed");

    let parsed: Message = serde_json::from_str(&first_msg).unwrap();
    match parsed {
        Message::ClientState(cs) => {
            assert_eq!(cs.state, Some(ClientSyncState::Synchronized));
            assert!(cs.player.is_none(), "expected no player state");
        }
        other => panic!("expected ClientState, got {:?}", other),
    }
}

#[tokio::test]
async fn test_no_player_sender_can_send_sync_state_without_reporter() {
    let (url, mut rx, _handle) = start_test_server().await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let split = client.split();
    discard_initial_state(&mut rx).await;
    split
        .sender
        .send_sync_state(ClientSyncState::ExternalSource)
        .await
        .unwrap();
    loop {
        let message = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("timed out waiting for sync state")
            .expect("server channel closed");
        match serde_json::from_str::<Message>(&message).unwrap() {
            Message::ClientState(cs) => {
                assert_eq!(cs.state, Some(ClientSyncState::ExternalSource));
                assert!(cs.player.is_none());
                break;
            }
            Message::ClientTime(_) => {}
            other => panic!("expected sync state, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn test_connect_can_send_initial_external_source_state() {
    let (url, mut rx, _handle) = start_test_server().await;

    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .initial_sync_state(ClientSyncState::ExternalSource)
        .build();

    let _client = builder.connect(&url).await.unwrap();

    let first_msg = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for message")
        .expect("channel closed");

    let parsed: Message = serde_json::from_str(&first_msg).unwrap();
    match parsed {
        Message::ClientState(cs) => {
            assert_eq!(cs.state, Some(ClientSyncState::ExternalSource));
            assert!(cs.player.is_none(), "expected no player state");
        }
        other => panic!("expected ClientState, got {:?}", other),
    }
}

#[tokio::test]
async fn test_connect_initial_external_source_preserves_player_state() {
    let (url, mut rx, _handle) = start_test_server().await;

    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .initial_sync_state(ClientSyncState::ExternalSource)
        .initial_player_state(PlayerState {
            volume: Some(23),
            muted: Some(true),
            static_delay_ms: Some(250),
            required_lead_time_ms: Some(500),
            min_buffer_ms: Some(500),
            supported_commands: None,
        })
        .build();

    let _client = builder.connect(&url).await.unwrap();

    let first_msg = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for message")
        .expect("channel closed");

    let parsed: Message = serde_json::from_str(&first_msg).unwrap();
    match parsed {
        Message::ClientState(cs) => {
            assert_eq!(cs.state, Some(ClientSyncState::ExternalSource));
            let player = cs.player.expect("expected player state");
            assert_eq!(player.volume, Some(23));
            assert_eq!(player.muted, Some(true));
            assert_eq!(player.static_delay_ms, Some(250));
        }
        other => panic!("expected ClientState, got {:?}", other),
    }
}

#[tokio::test]
async fn test_disconnect_sends_goodbye() {
    let (url, mut rx, _handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();

    client.disconnect(GoodbyeReason::Shutdown).await.unwrap();

    // Drain messages until we find client/goodbye
    let mut found_goodbye = false;
    while let Ok(Some(msg_text)) =
        tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv()).await
    {
        if let Ok(Message::ClientGoodbye(goodbye)) = serde_json::from_str::<Message>(&msg_text) {
            assert_eq!(goodbye.reason, GoodbyeReason::Shutdown);
            found_goodbye = true;
            break;
        }
    }
    assert!(found_goodbye, "never received client/goodbye");
}

#[tokio::test]
async fn test_sender_emits_stream_request_format() {
    let (url, mut rx, server_tx, _handle) = start_server(None).await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let mut player = conn.player.attach(TestOutput);

    discard_initial_state(&mut rx).await;

    server_tx.send(player_stream_start()).unwrap();
    assert!(matches!(player.recv().await, Some(PlayerEvent::Start(_))));

    conn.sender
        .request_player_format(PlayerFormatRequest {
            codec: Some("pcm".to_string()),
            channels: Some(2),
            sample_rate: Some(48_000),
            bit_depth: Some(16),
        })
        .await
        .unwrap();

    let request = recv_stream_request_format(&mut rx).await;

    let player = request.player.expect("expected player format request");
    assert_eq!(player.codec.as_deref(), Some("pcm"));
    assert_eq!(player.channels, Some(2));
    assert_eq!(player.sample_rate, Some(48_000));
    assert_eq!(player.bit_depth, Some(16));
    assert!(request.artwork.is_none());
}

#[tokio::test]
async fn test_sender_emits_artwork_stream_request_format() {
    let (url, mut rx, server_tx, _handle) = start_server(None).await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let mut artwork = conn.artwork.attach(TestOutput);

    discard_initial_state(&mut rx).await;

    server_tx.send(artwork_stream_start()).unwrap();
    assert!(matches!(artwork.recv().await, Some(ArtworkEvent::Start(_))));

    conn.sender
        .request_artwork_format(ArtworkFormatRequest {
            channel: 1,
            source: Some(ArtworkSource::Artist),
            format: Some(ImageFormat::Png),
            media_width: Some(400),
            media_height: Some(300),
        })
        .await
        .unwrap();

    let request = recv_stream_request_format(&mut rx).await;

    assert!(request.player.is_none());
    let artwork = request.artwork.expect("expected artwork format request");
    assert_eq!(artwork.channel, 1);
    assert_eq!(artwork.source, Some(ArtworkSource::Artist));
    assert_eq!(artwork.format, Some(ImageFormat::Png));
    assert_eq!(artwork.media_width, Some(400));
    assert_eq!(artwork.media_height, Some(300));
}

#[tokio::test]
async fn test_sender_rejects_empty_stream_request_format() {
    let (url, mut rx, _handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();

    // Discard the initial client/state emitted immediately after handshake.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for initial state")
        .expect("channel closed");

    let result = conn.sender.request_stream_format(None, None).await;

    match result {
        Err(Error::Protocol(msg)) => assert!(
            msg.contains("requires a player, artwork, or visualizer request"),
            "unexpected protocol error: {msg}"
        ),
        other => panic!("expected protocol error, got {:?}", other),
    }
}

#[tokio::test]
async fn test_sender_rejects_request_format_before_stream_start() {
    // No stream/start: the server never starts a stream, so the gate stays shut.
    let (url, mut rx, _handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();

    discard_initial_state(&mut rx).await;

    let result = conn
        .sender
        .request_player_format(PlayerFormatRequest {
            codec: Some("pcm".to_string()),
            channels: Some(2),
            sample_rate: Some(48_000),
            bit_depth: Some(16),
        })
        .await;

    match result {
        Err(Error::Protocol(msg)) => assert!(
            msg.contains("active player stream"),
            "unexpected protocol error: {msg}"
        ),
        other => panic!("expected protocol error, got {:?}", other),
    }
}

#[tokio::test]
async fn test_sender_visualizer_request_requires_active_stream() {
    let (url, mut rx, server_tx, _handle) = start_server(None).await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .visualizer_v1_support(sendspin::protocol::messages::VisualizerV1Support {
            types: vec![VisualizerDataType::Spectrum],
            buffer_capacity: 4096,
            rate_max: 60,
            spectrum: None,
        })
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let mut visualizer = conn.visualizer.attach(TestOutput);
    discard_initial_state(&mut rx).await;

    let request = VisualizerFormatRequest {
        types: Some(vec![VisualizerDataType::Loudness]),
        rate_max: Some(30),
        spectrum: None,
    };
    let empty = VisualizerFormatRequest {
        types: None,
        rate_max: None,
        spectrum: None,
    };
    let result = conn.sender.request_visualizer_format(empty).await;
    assert!(matches!(result, Err(Error::Protocol(msg)) if msg.contains("at least one field")));

    let result = conn.sender.request_visualizer_format(request.clone()).await;
    assert!(
        matches!(result, Err(Error::Protocol(msg)) if msg.contains("active visualizer stream"))
    );

    server_tx.send(visualizer_stream_start()).unwrap();
    assert!(matches!(
        visualizer.recv().await,
        Some(VisualizerEvent::Start(_))
    ));

    conn.sender
        .request_visualizer_format(request)
        .await
        .unwrap();
    let sent = recv_stream_request_format(&mut rx).await;
    assert_eq!(
        sent.visualizer.expect("visualizer request").rate_max,
        Some(30)
    );
}

#[tokio::test]
async fn test_sender_rejects_player_request_after_stream_end() {
    // The player stream starts then ends, so the gate closes again.
    let (url, mut rx, server_tx, _handle) = start_server(None).await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let mut player = conn.player.attach(TestOutput);

    discard_initial_state(&mut rx).await;

    server_tx.send(player_stream_start()).unwrap();
    assert!(matches!(player.recv().await, Some(PlayerEvent::Start(_))));

    server_tx.send(stream_end(None)).unwrap();
    assert!(matches!(player.recv().await, Some(PlayerEvent::End)));

    let result = conn
        .sender
        .request_player_format(PlayerFormatRequest {
            codec: Some("pcm".to_string()),
            channels: Some(2),
            sample_rate: Some(48_000),
            bit_depth: Some(16),
        })
        .await;

    match result {
        Err(Error::Protocol(msg)) => assert!(
            msg.contains("active player stream"),
            "unexpected protocol error: {msg}"
        ),
        other => panic!("expected protocol error, got {:?}", other),
    }
}

#[tokio::test]
async fn test_sender_rejects_artwork_request_without_artwork_stream() {
    // Only a player stream is active; an artwork request has nothing to
    // renegotiate and must be rejected for its own role.
    let (url, mut rx, server_tx, _handle) = start_server(None).await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let mut player = conn.player.attach(TestOutput);

    discard_initial_state(&mut rx).await;

    server_tx.send(player_stream_start()).unwrap();
    assert!(matches!(player.recv().await, Some(PlayerEvent::Start(_))));

    let result = conn
        .sender
        .request_artwork_format(ArtworkFormatRequest {
            channel: 1,
            source: Some(ArtworkSource::Artist),
            format: Some(ImageFormat::Png),
            media_width: Some(400),
            media_height: Some(300),
        })
        .await;

    match result {
        Err(Error::Protocol(msg)) => assert!(
            msg.contains("active artwork stream"),
            "unexpected protocol error: {msg}"
        ),
        other => panic!("expected protocol error, got {:?}", other),
    }
}

#[tokio::test]
async fn test_disconnect_closes_socket_and_stops_background_tasks() {
    let (url, mut rx, handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();

    // Let clock sync send at least one client/time
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    client.disconnect(GoodbyeReason::Shutdown).await.unwrap();

    // Drain remaining messages — should find goodbye then channel closes
    let mut found_goodbye = false;
    let mut messages_after_goodbye = 0;
    while let Ok(Some(msg_text)) =
        tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv()).await
    {
        if found_goodbye {
            messages_after_goodbye += 1;
        }
        if let Ok(Message::ClientGoodbye(_)) = serde_json::from_str::<Message>(&msg_text) {
            found_goodbye = true;
        }
    }

    assert!(found_goodbye, "never received client/goodbye");

    assert_eq!(
        messages_after_goodbye, 0,
        "no messages should arrive after goodbye"
    );

    // Server task should have exited (socket closed)
    let server_exited = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
        .await
        .is_ok();
    assert!(
        server_exited,
        "server task did not exit — socket not closed"
    );
}

#[tokio::test]
async fn test_connection_disconnect_sends_goodbye_and_closes() {
    let (url, mut rx, handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let guard = conn.guard;

    // Let clock sync send at least one client/time
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    guard.disconnect(GoodbyeReason::Shutdown).await.unwrap();

    // Drain remaining messages — should find goodbye then channel closes
    let mut found_goodbye = false;
    let mut messages_after_goodbye = 0;
    while let Ok(Some(msg_text)) =
        tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv()).await
    {
        if found_goodbye {
            messages_after_goodbye += 1;
        }
        if let Ok(Message::ClientGoodbye(_)) = serde_json::from_str::<Message>(&msg_text) {
            found_goodbye = true;
        }
    }

    assert!(found_goodbye, "never received client/goodbye");

    assert_eq!(
        messages_after_goodbye, 0,
        "no messages should arrive after goodbye"
    );

    // Server task should have exited (socket closed)
    let server_exited = tokio::time::timeout(std::time::Duration::from_secs(2), handle)
        .await
        .is_ok();
    assert!(
        server_exited,
        "server task did not exit — socket not closed"
    );
}

/// Helper: connect with controller role, return the rx channel and a Controller handle
async fn connect_with_controller() -> (
    tokio::sync::mpsc::UnboundedReceiver<String>,
    sendspin::protocol::client::Controller,
    sendspin::protocol::client::Connection,
    tokio::task::JoinHandle<()>,
) {
    let (url, rx, handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(PlayerV1Support {
            supported_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 1024,
            supported_commands: vec![],
        })
        .controller()
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let controller = conn.controller.clone();

    (rx, controller, conn, handle)
}

/// Helper: drain client/time messages and return the next client/command
async fn next_client_command(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>,
) -> ClientCommand {
    loop {
        let msg_text = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("timed out waiting for message")
            .expect("channel closed");

        let parsed: Message = serde_json::from_str(&msg_text).unwrap();
        if let Message::ClientCommand(cmd) = parsed {
            return cmd;
        }
    }
}

async fn next_client_state(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>,
) -> sendspin::protocol::messages::ClientState {
    loop {
        let msg_text = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("timed out waiting for message")
            .expect("channel closed");

        let parsed: Message = serde_json::from_str(&msg_text).unwrap();
        if let Message::ClientState(state) = parsed {
            return state;
        }
    }
}

#[tokio::test]
async fn test_enter_external_source_sends_external_source_state() {
    let (url, mut rx, _handle) = start_test_server().await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();

    let initial = next_client_state(&mut rx).await;
    assert_eq!(initial.state, Some(ClientSyncState::Synchronized));

    client.enter_external_source().await.unwrap();

    let state = next_client_state(&mut rx).await;
    assert_eq!(state.state, Some(ClientSyncState::ExternalSource));
    assert!(state.player.is_none());
}

#[tokio::test]
async fn test_exit_external_source_sends_synchronized_state() {
    let (url, mut rx, _handle) = start_test_server().await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();

    let initial = next_client_state(&mut rx).await;
    assert_eq!(initial.state, Some(ClientSyncState::Synchronized));

    client.exit_external_source(None).await.unwrap();

    let state = next_client_state(&mut rx).await;
    assert_eq!(state.state, Some(ClientSyncState::Synchronized));
    assert!(
        state.player.is_none(),
        "opted-out exit must not fabricate player fields"
    );
}

#[tokio::test]
async fn test_exit_external_source_sends_full_player_state() {
    let (url, mut rx, _handle) = start_test_server().await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();

    let _initial = next_client_state(&mut rx).await;

    client
        .exit_external_source(Some(PlayerState {
            volume: None,
            muted: None,
            static_delay_ms: Some(3),
            ..Default::default()
        }))
        .await
        .unwrap();

    let state = next_client_state(&mut rx).await;
    assert_eq!(state.state, Some(ClientSyncState::Synchronized));
    let player = state.player.expect("player state should be present");
    assert_eq!(player.volume, None);
    assert_eq!(player.muted, None);
    assert_eq!(player.static_delay_ms, Some(3));
}

// Macro to generate tests for simple controller commands (no parameters).
// Defined once to prevent copy-paste typo errors across the six methods.
macro_rules! test_simple_controller_command {
    ($name:ident, $method:ident, $expected:expr) => {
        #[tokio::test]
        async fn $name() {
            let (mut rx, controller, _conn, _handle) = connect_with_controller().await;
            controller.$method().await.unwrap();

            let cmd = next_client_command(&mut rx).await;
            let ctrl = cmd.controller.expect("expected controller command");
            assert_eq!(ctrl.command, $expected);
        }
    };
}

test_simple_controller_command!(test_controller_play, play, ControllerCommandType::Play);
test_simple_controller_command!(test_controller_pause, pause, ControllerCommandType::Pause);
test_simple_controller_command!(test_controller_stop, stop, ControllerCommandType::Stop);
test_simple_controller_command!(test_controller_next, next, ControllerCommandType::Next);
test_simple_controller_command!(
    test_controller_previous,
    previous,
    ControllerCommandType::Previous
);
test_simple_controller_command!(
    test_controller_switch,
    switch,
    ControllerCommandType::Switch
);

#[tokio::test]
async fn test_controller_set_volume_sends_value() {
    let (mut rx, controller, _conn, _handle) = connect_with_controller().await;
    controller.set_volume(25).await.unwrap();

    let cmd = next_client_command(&mut rx).await;
    let ctrl = cmd.controller.expect("expected controller command");
    assert_eq!(ctrl.command, ControllerCommandType::Volume);
    assert_eq!(ctrl.volume, Some(25));
}

#[tokio::test]
async fn test_controller_set_volume_clamps_above_100() {
    let (mut rx, controller, _conn, _handle) = connect_with_controller().await;
    controller.set_volume(200).await.unwrap();

    let cmd = next_client_command(&mut rx).await;
    let ctrl = cmd.controller.expect("expected controller command");
    assert_eq!(ctrl.volume, Some(100));
}

#[tokio::test]
async fn test_controller_set_mute_sends_value() {
    let (mut rx, controller, _conn, _handle) = connect_with_controller().await;
    controller.set_mute(true).await.unwrap();

    let cmd = next_client_command(&mut rx).await;
    let ctrl = cmd.controller.expect("expected controller command");
    assert_eq!(ctrl.command, ControllerCommandType::Mute);
    assert_eq!(ctrl.mute, Some(true));
}

#[tokio::test]
async fn test_controller_repeat_sends_mode() {
    let (mut rx, controller, _conn, _handle) = connect_with_controller().await;
    controller.repeat(RepeatMode::All).await.unwrap();

    let cmd = next_client_command(&mut rx).await;
    let ctrl = cmd.controller.expect("expected controller command");
    assert_eq!(ctrl.command, ControllerCommandType::RepeatAll);
}

#[tokio::test]
async fn test_controller_shuffle_sends_correct_command() {
    let (mut rx, controller, _conn, _handle) = connect_with_controller().await;
    controller.shuffle(true).await.unwrap();

    let cmd = next_client_command(&mut rx).await;
    let ctrl = cmd.controller.expect("expected controller command");
    assert_eq!(ctrl.command, ControllerCommandType::Shuffle);
}

#[tokio::test]
async fn test_controller_unshuffle_sends_correct_command() {
    let (mut rx, controller, _conn, _handle) = connect_with_controller().await;
    controller.shuffle(false).await.unwrap();

    let cmd = next_client_command(&mut rx).await;
    let ctrl = cmd.controller.expect("expected controller command");
    assert_eq!(ctrl.command, ControllerCommandType::Unshuffle);
}

#[tokio::test]
async fn test_controller_seek_sends_position() {
    let (mut rx, controller, _conn, _handle) = connect_with_controller().await;
    controller.seek(123_456).await.unwrap();

    let cmd = next_client_command(&mut rx).await;
    let ctrl = cmd.controller.expect("expected controller command");
    assert_eq!(ctrl.command, ControllerCommandType::Seek);
    assert_eq!(ctrl.position_ms, Some(123_456));
    assert!(ctrl.offset_ms.is_none());
}

#[tokio::test]
async fn test_controller_seek_relative_sends_offset() {
    let (mut rx, controller, _conn, _handle) = connect_with_controller().await;
    controller.seek_relative(-15_000).await.unwrap();

    let cmd = next_client_command(&mut rx).await;
    let ctrl = cmd.controller.expect("expected controller command");
    assert_eq!(ctrl.command, ControllerCommandType::SeekRelative);
    assert_eq!(ctrl.offset_ms, Some(-15_000));
    assert!(ctrl.position_ms.is_none());
}

#[tokio::test]
async fn test_no_controller_when_server_denies_role() {
    let (url, _rx, _, _handle) = start_server(Some(vec!["player@v1".to_string()])).await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(PlayerV1Support {
            supported_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 1024,
            supported_commands: vec![],
        })
        .controller()
        .build()
        .connect(&url)
        .await
        .unwrap();

    let conn = client.split();
    assert!(
        matches!(conn.controller.play().await, Err(Error::Protocol(_))),
        "controller must fail when the server denies the role"
    );
    // The same activation gates the controller (above) and feeds the
    // multi-server arbitration policy.
    assert_eq!(
        conn.activation.active_roles,
        Some(vec!["player@v1".to_string()])
    );
}

#[tokio::test]
async fn test_no_controller_when_role_not_declared() {
    let (url, _rx, _handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();

    let conn = client.split();
    assert!(
        matches!(conn.controller.play().await, Err(Error::Protocol(_))),
        "controller must fail without the role"
    );
}

#[tokio::test]
async fn test_clock_sync_getter_returns_shared_handle() {
    // `clock_sync()` must return the *same* `Arc<Mutex<ClockSync>>` instance
    // on every call — callers rely on shared state with the background time
    // sync task. Two calls returning pointer-distinct handles would compile
    // fine but silently break sync state sharing; `Arc::ptr_eq` pins this.
    let (url, _rx, _handle) = start_test_server().await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();

    let a = client.clock_sync();
    let b = client.clock_sync();
    assert!(
        std::sync::Arc::ptr_eq(&a, &b),
        "clock_sync() must return the same shared handle on every call"
    );
}

#[tokio::test]
async fn test_message_router_forwards_binary_audio() {
    // End-to-end: a binary audio frame routes into a player session.
    let (url, _rx, server_tx, _handle) = start_server(None).await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(PlayerV1Support {
            supported_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 1024,
            supported_commands: vec![],
        })
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let mut session = conn.player.attach(TestOutput);

    // Audio chunk: type byte, 8-byte big-endian timestamp, then payload.
    let audio_frame: Vec<u8> = vec![
        binary_types::PLAYER_AUDIO,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x2A,
        0xDE,
        0xAD,
    ];
    server_tx
        .send(WsMessage::Binary(audio_frame.into()))
        .unwrap();

    let event = tokio::time::timeout(std::time::Duration::from_secs(2), session.recv())
        .await
        .expect("timed out waiting for audio event")
        .expect("player session closed");
    assert!(
        matches!(event, PlayerEvent::Audio(work) if work.timestamp() == 42 && work.data() == [0xDE, 0xAD])
    );
}

#[tokio::test]
async fn test_message_router_forwards_binary_artwork() {
    // Mirror of the audio test for the artwork path. Each binary channel has
    // its own closed-flag and send wiring, so each needs separate coverage.
    let (url, _rx, server_tx, _handle) = start_server(None).await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let mut artwork = conn.artwork.attach(TestOutput);
    server_tx.send(artwork_stream_start()).unwrap();
    assert!(matches!(artwork.recv().await, Some(ArtworkEvent::Start(_))));

    let artwork_frame: Vec<u8> = vec![
        binary_types::ARTWORK_CHANNEL_2,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x64,
        0xFF,
        0xD8,
        0xFF,
        0xE0,
    ];
    server_tx
        .send(WsMessage::Binary(artwork_frame.into()))
        .unwrap();
    let work = match artwork.recv().await {
        Some(ArtworkEvent::Data(work)) => work,
        _ => panic!("expected artwork data"),
    };
    assert_eq!(work.timestamp(), 100);
    assert_eq!(work.channel(), 2);
    assert_eq!(work.data(), &[0xFF, 0xD8, 0xFF, 0xE0]);
    work.commit(vec![1]).unwrap();
}

#[tokio::test]
async fn test_message_router_forwards_binary_visualizer() {
    // Mirror of the audio test for the visualizer path (binary type 16).
    let (url, _rx, server_tx, _handle) = start_server(None).await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let mut visualizer = conn.visualizer.attach(TestOutput);
    server_tx.send(visualizer_stream_start()).unwrap();
    assert!(matches!(
        visualizer.recv().await,
        Some(VisualizerEvent::Start(_))
    ));

    for (type_id, timestamp, expected) in [
        (
            binary_types::VISUALIZER_LOUDNESS,
            200_i64,
            vec![0x01, 0x02, 0x03],
        ),
        (binary_types::VISUALIZER_BEAT, 0_i64, vec![0x01]),
        (binary_types::VISUALIZER_SPECTRUM, 0_i64, vec![0x02, 0x03]),
    ] {
        server_tx
            .send(WsMessage::Binary(
                [
                    vec![type_id],
                    timestamp.to_be_bytes().to_vec(),
                    expected.clone(),
                ]
                .concat()
                .into(),
            ))
            .unwrap();
        let work = match visualizer.recv().await {
            Some(VisualizerEvent::Data(work)) => work,
            _ => panic!("expected visualizer data"),
        };
        assert_eq!(work.type_id(), type_id);
        assert_eq!(work.timestamp(), timestamp);
        assert_eq!(work.data(), expected.as_slice());
        work.commit(vec![type_id]).unwrap();
    }
}

#[tokio::test]
async fn test_public_websocket_clear_commands_and_fallback() {
    let (url, mut rx, server_tx, _handle) = start_server(None).await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(PlayerV1Support {
            supported_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48_000,
                bit_depth: 24,
            }],
            buffer_capacity: 1024,
            supported_commands: vec!["volume".to_string()],
        })
        .initial_player_state(PlayerState {
            volume: Some(100),
            ..PlayerState::default()
        })
        .build()
        .connect(&url)
        .await
        .unwrap();
    let mut conn = client.split();
    let mut player = conn.player.attach(TestOutput);
    discard_initial_state(&mut rx).await;
    server_tx.send(player_stream_start()).unwrap();
    assert!(matches!(player.recv().await, Some(PlayerEvent::Start(_))));

    server_tx
        .send(text_message(&Message::StreamClear(StreamClear {
            roles: Some(vec!["player".to_string()]),
        })))
        .unwrap();
    assert!(matches!(player.recv().await, Some(PlayerEvent::Clear)));

    server_tx
        .send(text_message(&Message::ServerCommand(ServerCommand {
            player: Some(PlayerCommand {
                command: PlayerCommandType::Volume,
                volume: Some(73),
                mute: None,
                static_delay_ms: None,
            }),
        })))
        .unwrap();
    let mut controls = player.controls();
    controls.changed().await.unwrap();
    assert_eq!(controls.borrow().volume, 73);

    let invalid = Message::ServerCommand(ServerCommand {
        player: Some(PlayerCommand {
            command: PlayerCommandType::Volume,
            volume: None,
            mute: None,
            static_delay_ms: None,
        }),
    });
    server_tx.send(text_message(&invalid)).unwrap();
    let fallback = tokio::time::timeout(std::time::Duration::from_secs(2), conn.messages.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        fallback,
        Message::ServerCommand(ServerCommand {
            player: Some(PlayerCommand { volume: None, .. })
        })
    ));
}

#[tokio::test]
async fn test_message_router_forwards_text_messages() {
    // End-to-end: a server text message routes into `conn.messages`.
    let (url, _rx, server_tx, _handle) = start_server(None).await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let mut conn = client.split();

    let server_state = serde_json::to_string(&Message::ServerState(
        sendspin::protocol::messages::ServerState {
            metadata: None,
            controller: None,
            color: None,
        },
    ))
    .unwrap();
    server_tx
        .send(WsMessage::Text(server_state.into()))
        .unwrap();

    let msg = tokio::time::timeout(std::time::Duration::from_secs(2), conn.messages.recv())
        .await
        .expect("timed out waiting for server message")
        .expect("message channel closed");
    match msg {
        Message::ServerState(_) => {}
        other => panic!("expected ServerState, got {other:?}"),
    }
}

#[test]
#[ignore] // Requires running server
fn test_client_receives_stream_start() {
    // Test that client can receive stream/start message
    // Will implement when we have full client
}

#[test]
#[ignore] // Requires running server
fn test_client_handles_audio_chunks() {
    // Test that client can receive binary audio chunks
    // Will implement when we have full client
}

/// Contract: `send_message` returns `Err` once the writer task is gone.
/// Without this, callers could see `Ok` for sends that never reached the wire.
#[tokio::test]
async fn test_send_message_fails_after_disconnect() {
    let (url, mut rx, _handle) = start_test_server().await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .controller()
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let controller = conn.controller;
    let guard = conn.guard;

    let _ = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await;

    guard.disconnect(GoodbyeReason::Shutdown).await.unwrap();

    let result = controller.play().await;
    assert!(result.is_err(), "expected Err, got {:?}", result);
}

/// `disconnect` must not return until the writer has actually exited,
/// across every clone of the sender handle.
#[tokio::test]
async fn test_disconnect_observes_writer_completion() {
    let (url, mut rx, _handle) = start_test_server().await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .controller()
        .build()
        .connect(&url)
        .await
        .unwrap();

    let conn = client.split();
    let controller = conn.controller;
    let sender_clone = conn.sender.clone();
    let guard = conn.guard;

    let _ = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await;

    guard.disconnect(GoodbyeReason::Shutdown).await.unwrap();

    assert!(controller.play().await.is_err());
    assert!(sender_clone
        .send_message(Message::ClientCommand(ClientCommand { controller: None }))
        .await
        .is_err());
}

/// When the peer sends a WS Close mid-stream, the router must terminate the
/// message stream, and sends must eventually fail once the writer hits the
/// dead socket. This is the primary real-world disconnect path.
#[tokio::test]
async fn test_peer_close_midstream_ends_stream_and_fails_sends() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("ws://{}", addr);

    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let (mut server, _) = TestServer::handshake(accept_async(stream).await.unwrap()).await;
        server.activate(vec!["player@v1".to_string()]).await;
        loop {
            match server.recv().await {
                Some(Message::ClientTime(time)) => server.answer_time(&time).await,
                Some(Message::ClientState(_)) => break,
                other => panic!("expected the initial client/state, got {other:?}"),
            }
        }
        server.send_raw(WsMessage::Close(None)).await;
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    });

    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let mut conn = client.split();

    let terminal = tokio::time::timeout(std::time::Duration::from_secs(2), conn.messages.recv())
        .await
        .expect("message stream did not terminate after peer close");
    assert!(
        terminal.is_none(),
        "expected message channel to close (None) after peer WS close, got {terminal:?}"
    );

    // The first send may race ahead of the close registering locally; retry a
    // bounded number of times until the writer observes the dead socket.
    let mut last = Ok(());
    for _ in 0..20 {
        last = conn
            .sender
            .send_message(Message::ClientCommand(ClientCommand { controller: None }))
            .await;
        if last.is_err() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        last.is_err(),
        "send eventually fails after peer close, got {last:?}"
    );
}

/// A send racing a disconnect must resolve cleanly — never deadlock, never
/// panic. disconnect succeeds (peer is alive, goodbye flushes); the racing
/// send either lands before the goodbye (Ok) or is rejected because the
/// writer already exited (Err). Both are correct.
#[tokio::test]
async fn test_concurrent_send_during_disconnect_is_clean() {
    let (mut rx, controller, conn, _handle) = connect_with_controller().await;
    let guard = conn.guard;

    let _ = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv()).await;

    let (send_res, disc_res) =
        tokio::join!(controller.play(), guard.disconnect(GoodbyeReason::Shutdown),);

    disc_res.expect("disconnect should succeed against a live peer");
    // send_res is intentionally not asserted: Ok (landed before goodbye) and
    // Err (writer already exited) are both correct. The contract under test is
    // that join! returned at all — neither future deadlocked.
    let _ = send_res;
}

/// Accept one client on a fresh port and run the server/init, Noise and
/// hello exchange with it.
async fn accept_one() -> (
    String,
    tokio::task::JoinHandle<TestServer<tokio::net::TcpStream>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        TestServer::handshake(accept_async(stream).await.unwrap())
            .await
            .0
    });
    (url, server)
}

#[tokio::test]
async fn test_client_waits_for_activation_and_clock_sync() {
    let (url, server) = accept_one().await;
    let server = tokio::spawn(async move {
        let mut server = server.await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(300), server.recv())
                .await
                .is_err(),
            "client sent a message before server/activate"
        );
        server.activate(vec!["player@v1".to_string()]).await;
        // Two complete 8-message bursts are required before ClockSync::is_synchronized() becomes true.
        // Keep this in sync with src/protocol/client.rs:TIME_BURST_SIZE
        const STARTUP_REQUESTS: usize = 2 * 8;
        for sample in 0..STARTUP_REQUESTS {
            let time = match server.recv().await {
                Some(Message::ClientTime(time)) => time,
                other => panic!("expected client/time, got {other:?}"),
            };
            let received = Instant::now();
            if sample == 0 {
                assert!(
                    tokio::time::timeout(Duration::from_millis(300), server.recv())
                        .await
                        .is_err(),
                    "client sent a message before receiving its first clock reply"
                );
            }
            // Report the time each request was held so the round trip stays short.
            let t = time.client_transmitted;
            server
                .send(&Message::ServerTime(ServerTime {
                    client_transmitted: t,
                    server_received: t,
                    server_transmitted: t + received.elapsed().as_micros() as i64,
                }))
                .await;
        }
        match server.recv().await {
            Some(Message::ClientState(_)) => {}
            other => panic!("expected client/state, got {other:?}"),
        }
    });
    let _client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn test_unpaired_playback_without_unpaired_access_is_refused() {
    let (url, server) = accept_one().await;
    let server = tokio::spawn(async move {
        let mut server = server.await.unwrap();
        server.activate(vec!["player@v1".to_string()]).await;
        match server.recv().await {
            Some(Message::ClientGoodbye(goodbye)) => {
                assert_eq!(goodbye.reason, GoodbyeReason::PairingRequired)
            }
            other => panic!("expected client/goodbye, got {other:?}"),
        }
        assert!(server.recv().await.is_none(), "client kept the connection");
    });
    let result = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await;
    assert!(matches!(result, Err(Error::Connection(_))));
    server.await.unwrap();
}

#[tokio::test]
async fn test_role_removal_ends_stream_and_forwards_activation() {
    let (url, _rx, server_tx, _handle) = start_server(None).await;
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let mut conn = client.split();
    let mut session = conn.player.attach(TestOutput);
    server_tx.send(player_stream_start()).unwrap();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(2), session.recv()).await,
        Ok(Some(PlayerEvent::Start(_)))
    ));

    server_tx
        .send(text_message(&Message::ServerActivate(ServerActivate {
            activities: vec![],
            active_roles: Some(vec![]),
        })))
        .unwrap();
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(2), session.recv()).await,
        Ok(Some(PlayerEvent::End))
    ));
    loop {
        let message = tokio::time::timeout(Duration::from_secs(2), conn.messages.recv())
            .await
            .expect("timed out waiting for server/activate")
            .expect("message channel closed");
        if let Message::ServerActivate(activation) = message {
            assert_eq!(activation.active_roles, Some(vec![]));
            break;
        }
    }
}

#[tokio::test]
async fn test_cleartext_after_handshake_closes_without_goodbye() {
    let (url, server) = accept_one().await;
    let server = tokio::spawn(async move {
        let mut server = server.await.unwrap();
        server.activate_with(vec![], vec![]).await;
        server
            .send_raw(text_message(&Message::ServerCommand(ServerCommand {
                player: None,
            })))
            .await;
        while let Some(message) = server.recv().await {
            assert!(
                !matches!(message, Message::ClientGoodbye(_)),
                "client said goodbye after cleartext"
            );
        }
    });
    let client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .name("Test".to_string())
        .metadata()
        .build()
        .connect(&url)
        .await
        .unwrap();
    let mut conn = client.split();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), conn.messages.recv())
            .await
            .expect("connection stayed open after cleartext")
            .is_none()
    );
    tokio::time::timeout(Duration::from_secs(2), server)
        .await
        .expect("client did not close the connection")
        .unwrap();
}

#[tokio::test]
async fn test_player_state_follows_player_activation() {
    let (url, mut rx, server_tx, _handle) = start_server(Some(vec![])).await;
    let _client = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .initial_player_state(PlayerState {
            volume: Some(40),
            muted: Some(false),
            static_delay_ms: Some(0),
            required_lead_time_ms: Some(500),
            min_buffer_ms: Some(500),
            supported_commands: None,
        })
        .build()
        .connect(&url)
        .await
        .unwrap();
    let mut next_state = async || loop {
        let text = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("timed out waiting for client/state")
            .expect("server channel closed");
        if let Message::ClientState(state) = serde_json::from_str(&text).unwrap() {
            break state;
        }
    };
    assert!(next_state().await.player.is_none());

    server_tx
        .send(text_message(&Message::ServerActivate(ServerActivate {
            activities: vec![],
            active_roles: Some(vec!["player@v1".to_string()]),
        })))
        .unwrap();
    assert_eq!(next_state().await.player.and_then(|p| p.volume), Some(40));
}
