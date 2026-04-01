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
            match msg {
                WsMessage::Text(text) => {
                    if tx.send(text).is_err() {
                        break;
                    }
                }
                WsMessage::Close(_) => break,
                _ => {}
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
            match msg {
                WsMessage::Text(text) => {
                    if tx.send(text).is_err() {
                        break;
                    }
                }
                WsMessage::Close(_) => break,
                _ => {}
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
