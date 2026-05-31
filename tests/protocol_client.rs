use futures_util::{SinkExt, StreamExt};
use sendspin::protocol::messages::{
    AudioFormatSpec, ClientCommand, ClientSyncState, ControllerCommandType, GoodbyeReason, Message,
    PlayerState, PlayerV1Support, RepeatMode,
};
use sendspin::ProtocolClientBuilder;
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message as WsMessage};

/// Start a local WebSocket server that performs the sendspin handshake
/// and returns the listener address and a channel to receive messages
/// the client sends after the handshake.
async fn start_test_server() -> (
    String,
    tokio::sync::mpsc::UnboundedReceiver<String>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("ws://{}", addr);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        // Receive client/hello and echo back its roles
        let msg = ws.next().await.unwrap().unwrap();
        let active_roles = if let WsMessage::Text(ref text) = msg {
            match serde_json::from_str::<Message>(text).unwrap() {
                Message::ClientHello(hello) => hello.supported_roles,
                other => panic!("First message should be client/hello, got {:?}", other),
            }
        } else {
            panic!("Expected text message for client/hello");
        };

        // Send server/hello with the roles the client requested
        let server_hello = serde_json::to_string(&Message::ServerHello(
            sendspin::protocol::messages::ServerHello {
                server_id: "test-server".to_string(),
                name: "Test Server".to_string(),
                version: 1,
                active_roles,
                connection_reason: sendspin::protocol::messages::ConnectionReason::Playback,
            },
        ))
        .unwrap();
        ws.send(WsMessage::Text(server_hello)).await.unwrap();

        // Forward all subsequent messages to the channel
        while let Some(Ok(msg)) = ws.next().await {
            let text = match msg {
                WsMessage::Text(text) => text,
                WsMessage::Close(_) => break,
                _ => continue,
            };
            if tx.send(text).is_err() {
                break;
            }
        }
    });

    (url, rx, handle)
}

#[tokio::test]
async fn test_connect_sends_initial_state() {
    let (url, mut rx, _handle) = start_test_server().await;

    let builder = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
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
            supported_commands: None,
        })
        .build();

    let _client = builder.connect(&url).await.unwrap();

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
        .client_id("test-client".to_string())
        .name("Test".to_string())
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
async fn test_disconnect_sends_goodbye() {
    let (url, mut rx, _handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
        .name("Test".to_string())
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
async fn test_disconnect_closes_socket_and_stops_background_tasks() {
    let (url, mut rx, handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
        .name("Test".to_string())
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
        .client_id("test-client".to_string())
        .name("Test".to_string())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let conn = client.split();
    let sender = conn.sender;
    let guard = conn.guard;

    // Let clock sync send at least one client/time
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    guard
        .disconnect(&sender, GoodbyeReason::Shutdown)
        .await
        .unwrap();

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
        .client_id("test-client".to_string())
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
    let mut conn = client.split();
    let controller = conn
        .controller
        .take()
        .expect("should have controller when role declared");

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

/// Start a test server that only grants specific roles, ignoring what the client requests.
async fn start_test_server_with_roles(
    granted_roles: Vec<String>,
) -> (
    String,
    tokio::sync::mpsc::UnboundedReceiver<String>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("ws://{}", addr);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    let handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        // Receive and discard client/hello
        let _ = ws.next().await.unwrap().unwrap();

        // Send server/hello with only the granted roles
        let server_hello = serde_json::to_string(&Message::ServerHello(
            sendspin::protocol::messages::ServerHello {
                server_id: "test-server".to_string(),
                name: "Test Server".to_string(),
                version: 1,
                active_roles: granted_roles,
                connection_reason: sendspin::protocol::messages::ConnectionReason::Playback,
            },
        ))
        .unwrap();
        ws.send(WsMessage::Text(server_hello)).await.unwrap();

        while let Some(Ok(msg)) = ws.next().await {
            let text = match msg {
                WsMessage::Text(text) => text,
                WsMessage::Close(_) => break,
                _ => continue,
            };
            if tx.send(text).is_err() {
                break;
            }
        }
    });

    (url, rx, handle)
}

#[tokio::test]
async fn test_no_controller_when_server_denies_role() {
    let (url, _rx, _handle) = start_test_server_with_roles(vec!["player@v1".to_string()]).await;

    let client = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
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
        conn.controller.is_none(),
        "should not have controller when server denies the role"
    );
}

#[tokio::test]
async fn test_no_controller_when_role_not_declared() {
    let (url, _rx, _handle) = start_test_server().await;

    let client = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
        .name("Test".to_string())
        .build()
        .connect(&url)
        .await
        .unwrap();

    let conn = client.split();
    assert!(
        conn.controller.is_none(),
        "should not have controller without role"
    );
}

/// Variant of [`start_test_server`] that drives the server side directly —
/// exposes both the send half (for server-initiated messages) and the receive
/// half (for client-sent messages).
async fn start_test_server_with_sender() -> (
    String,
    tokio::sync::mpsc::UnboundedReceiver<String>,
    tokio::sync::mpsc::UnboundedSender<WsMessage>,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("ws://{}", addr);

    let (client_tx, client_rx) = tokio::sync::mpsc::unbounded_channel();
    let (server_send_tx, mut server_send_rx) = tokio::sync::mpsc::unbounded_channel::<WsMessage>();

    let handle = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = accept_async(stream).await.unwrap();

        // Drive handshake: consume client/hello, echo roles back.
        let msg = ws.next().await.unwrap().unwrap();
        let active_roles = if let WsMessage::Text(ref text) = msg {
            match serde_json::from_str::<Message>(text).unwrap() {
                Message::ClientHello(hello) => hello.supported_roles,
                other => panic!("First message should be client/hello, got {:?}", other),
            }
        } else {
            panic!("Expected text message for client/hello");
        };
        let server_hello = serde_json::to_string(&Message::ServerHello(
            sendspin::protocol::messages::ServerHello {
                server_id: "test-server".to_string(),
                name: "Test Server".to_string(),
                version: 1,
                active_roles,
                connection_reason: sendspin::protocol::messages::ConnectionReason::Playback,
            },
        ))
        .unwrap();
        ws.send(WsMessage::Text(server_hello)).await.unwrap();

        // Fan in: forward client messages to `client_tx`, forward outgoing
        // `server_send_rx` messages to the socket.
        loop {
            tokio::select! {
                msg = ws.next() => {
                    let Some(Ok(msg)) = msg else { break };
                    let text = match msg {
                        WsMessage::Text(text) => text,
                        WsMessage::Close(_) => break,
                        _ => continue,
                    };
                    if client_tx.send(text).is_err() { break }
                }
                outgoing = server_send_rx.recv() => {
                    let Some(out) = outgoing else { break };
                    if ws.send(out).await.is_err() { break }
                }
            }
        }
    });

    (url, client_rx, server_send_tx, handle)
}

#[tokio::test]
async fn test_clock_sync_getter_returns_shared_handle() {
    // `clock_sync()` must return the *same* `Arc<Mutex<ClockSync>>` instance
    // on every call — callers rely on shared state with the background time
    // sync task. Two calls returning pointer-distinct handles would compile
    // fine but silently break sync state sharing; `Arc::ptr_eq` pins this.
    let (url, _rx, _handle) = start_test_server().await;
    let client = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
        .name("Test".to_string())
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
    // End-to-end: server sends a binary audio frame, it should land in
    // `conn.audio`. Exercises the router's `WsMessage::Binary` arm, the
    // audio-channel dispatch, and the `!audio_closed && send.is_err()`
    // guard (which must still forward successfully when the channel is
    // open). A regression anywhere along the path manifests as
    // `conn.audio.recv()` timing out.
    let (url, _rx, server_tx, _handle) = start_test_server_with_sender().await;
    let client = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
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
    let mut conn = client.split();

    // Audio chunk: type 4, 8-byte big-endian timestamp, then payload.
    let audio_frame: Vec<u8> = vec![
        0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2A, 0xDE, 0xAD,
    ];
    server_tx.send(WsMessage::Binary(audio_frame)).unwrap();

    let chunk = tokio::time::timeout(std::time::Duration::from_secs(2), conn.audio.recv())
        .await
        .expect("timed out waiting for audio chunk")
        .expect("audio channel closed");
    assert_eq!(chunk.timestamp, 42);
    assert_eq!(&*chunk.data, &[0xDE, 0xAD]);
}

#[tokio::test]
async fn test_message_router_forwards_binary_artwork() {
    // Mirror of the audio routing test for the artwork path (binary
    // types 8-11). The artwork channel-dispatch and its
    // `!artwork_closed && send.is_err()` guard are structurally distinct
    // from the audio path — each channel has its own closed-flag and send
    // wiring, so each needs its own end-to-end coverage.
    let (url, _rx, server_tx, _handle) = start_test_server_with_sender().await;
    let client = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
        .name("Test".to_string())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let mut conn = client.split();

    // Artwork channel 2: type 0x0A, 8-byte big-endian timestamp, then payload.
    let artwork_frame: Vec<u8> = vec![
        0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x64, 0xFF, 0xD8, 0xFF, 0xE0,
    ];
    server_tx.send(WsMessage::Binary(artwork_frame)).unwrap();

    let chunk = tokio::time::timeout(std::time::Duration::from_secs(2), conn.artwork.recv())
        .await
        .expect("timed out waiting for artwork chunk")
        .expect("artwork channel closed");
    assert_eq!(chunk.channel, 2);
    assert_eq!(chunk.timestamp, 100);
    assert_eq!(&*chunk.data, &[0xFF, 0xD8, 0xFF, 0xE0]);
}

#[tokio::test]
async fn test_message_router_forwards_binary_visualizer() {
    // Mirror of the audio routing test for the visualizer path (binary
    // type 16 / 0x10). Like artwork, visualizer has its own
    // channel-dispatch and `!visualizer_closed && send.is_err()` guard,
    // independent of the audio and artwork paths.
    let (url, _rx, server_tx, _handle) = start_test_server_with_sender().await;
    let client = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
        .name("Test".to_string())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let mut conn = client.split();

    // Visualizer: type 0x10, 8-byte big-endian timestamp, then FFT data.
    let vis_frame: Vec<u8> = vec![
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC8, 0x01, 0x02, 0x03,
    ];
    server_tx.send(WsMessage::Binary(vis_frame)).unwrap();

    let chunk = tokio::time::timeout(std::time::Duration::from_secs(2), conn.visualizer.recv())
        .await
        .expect("timed out waiting for visualizer chunk")
        .expect("visualizer channel closed");
    assert_eq!(chunk.timestamp, 200);
    assert_eq!(&*chunk.data, &[0x01, 0x02, 0x03]);
}

#[tokio::test]
async fn test_message_router_forwards_text_messages() {
    // End-to-end: server sends a protocol text message, it should land in
    // `conn.messages`. Exercises the router's `WsMessage::Text` arm, the
    // JSON deserialization step, and the `!message_closed && send.is_err()`
    // guard. A regression in any of those paths manifests as
    // `conn.messages.recv()` timing out.
    let (url, _rx, server_tx, _handle) = start_test_server_with_sender().await;
    let client = ProtocolClientBuilder::builder()
        .client_id("test-client".to_string())
        .name("Test".to_string())
        .build()
        .connect(&url)
        .await
        .unwrap();
    let mut conn = client.split();

    let server_state = serde_json::to_string(&Message::ServerState(
        sendspin::protocol::messages::ServerState {
            metadata: None,
            controller: None,
        },
    ))
    .unwrap();
    server_tx.send(WsMessage::Text(server_state)).unwrap();

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
