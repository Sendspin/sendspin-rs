use futures_util::{SinkExt, StreamExt};
use sendspin::protocol::client::ProtocolClient;
use sendspin::protocol::messages::{
    ClientHello, ClientState, ClientSyncState, GoodbyeReason, Message, PlayerState,
};
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

        // Receive client/hello
        let msg = ws.next().await.unwrap().unwrap();
        if let WsMessage::Text(text) = msg {
            let parsed: Message = serde_json::from_str(&text).unwrap();
            assert!(
                matches!(parsed, Message::ClientHello(_)),
                "First message should be client/hello"
            );
        }

        // Send server/hello
        let server_hello = serde_json::to_string(&Message::ServerHello(
            sendspin::protocol::messages::ServerHello {
                server_id: "test-server".to_string(),
                name: "Test Server".to_string(),
                version: 1,
                active_roles: vec!["player@v1".to_string()],
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

    let initial_state = ClientState {
        state: Some(ClientSyncState::Synchronized),
        player: Some(PlayerState {
            volume: Some(75),
            muted: Some(false),
            static_delay_ms: Some(100),
            supported_commands: None,
        }),
    };

    let hello = ClientHello {
        client_id: "test-client".to_string(),
        name: "Test".to_string(),
        version: 1,
        supported_roles: vec!["player@v1".to_string()],
        device_info: None,
        player_v1_support: None,
        artwork_v1_support: None,
        visualizer_v1_support: None,
    };

    let _client = ProtocolClient::connect(&url, hello, Some(initial_state))
        .await
        .unwrap();

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
async fn test_connect_without_initial_state_sends_nothing() {
    let (url, mut rx, _handle) = start_test_server().await;

    let hello = ClientHello {
        client_id: "test-client".to_string(),
        name: "Test".to_string(),
        version: 1,
        supported_roles: vec!["player@v1".to_string()],
        device_info: None,
        player_v1_support: None,
        artwork_v1_support: None,
        visualizer_v1_support: None,
    };

    let _client = ProtocolClient::connect(&url, hello, None).await.unwrap();

    // First message should be client/time (from the clock sync task), not client/state
    let first_msg = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for message")
        .expect("channel closed");

    let parsed: Message = serde_json::from_str(&first_msg).unwrap();
    assert!(
        matches!(parsed, Message::ClientTime(_)),
        "expected ClientTime, got {:?}",
        parsed
    );
}

#[tokio::test]
async fn test_disconnect_sends_goodbye() {
    let (url, mut rx, _handle) = start_test_server().await;

    let hello = ClientHello {
        client_id: "test-client".to_string(),
        name: "Test".to_string(),
        version: 1,
        supported_roles: vec!["player@v1".to_string()],
        device_info: None,
        player_v1_support: None,
        artwork_v1_support: None,
        visualizer_v1_support: None,
    };

    let client = ProtocolClient::connect(&url, hello, None).await.unwrap();

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

    let hello = ClientHello {
        client_id: "test-client".to_string(),
        name: "Test".to_string(),
        version: 1,
        supported_roles: vec!["player@v1".to_string()],
        device_info: None,
        player_v1_support: None,
        artwork_v1_support: None,
        visualizer_v1_support: None,
    };

    let client = ProtocolClient::connect(&url, hello, None).await.unwrap();

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
