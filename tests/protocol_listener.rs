// ABOUTME: Integration tests for ProtocolListener — inbound (server-initiated)
// ABOUTME: WebSocket acceptor that drives the protocol-client state machine.

mod common;

use common::TestServer;
use futures_util::StreamExt;
use sendspin::audio::decode::{Decoder, PcmDecoder};
use sendspin::audio::{AudioBuffer, AudioFormat, Codec, OutputError, PlayerEvent, PlayerOutput};
use sendspin::protocol::client::binary_types;
use sendspin::protocol::messages::{
    AudioFormatSpec, ClientCommand, ClientSyncState, GoodbyeReason, Message, PlayerState,
    PlayerV1Support, StreamPlayerConfig, StreamStart,
};
use sendspin::{MemoryStore, ProtocolClientBuilder};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Default)]
struct ListenerOutput {
    commits: Arc<Mutex<Vec<AudioBuffer>>>,
}
impl PlayerOutput for ListenerOutput {
    fn commit(&self, buffer: AudioBuffer) -> Result<(), OutputError> {
        self.commits.lock().unwrap().push(buffer);
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
use tokio::net::TcpStream;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::{
    connect_async, tungstenite::Message as WsMessage, MaybeTlsStream, WebSocketStream,
};

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

/// The server end of an inbound connection after the handshake and activation.
struct Peer {
    supported_roles: Vec<String>,
    server_id: String,
    /// Every client JSON message except `client/time`.
    rx: UnboundedReceiver<String>,
    /// Finishes once the client closes the connection.
    reader: JoinHandle<()>,
    writer: UnboundedSender<WsMessage>,
}

async fn run_peer(
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    active_roles: Vec<String>,
) -> Peer {
    let (mut server, hello) = TestServer::handshake(ws).await;
    server.activate(active_roles).await;
    let server_id = server.server_id.clone();
    let (tx, rx) = unbounded_channel();
    let (writer, writer_rx) = unbounded_channel();
    let reader = tokio::spawn(server.serve(tx, writer_rx));
    Peer {
        supported_roles: hello.supported_roles,
        server_id,
        rx,
        reader,
        writer,
    }
}

async fn drive_peer_until_state(url: &str, active_roles: Vec<String>) -> Peer {
    let (ws, _) = connect_async(url).await.expect("WS connect failed");
    run_peer(ws, active_roles).await
}

async fn bind_listener(builder: ProtocolClientBuilder) -> (String, sendspin::ProtocolListener) {
    let listener = builder.listen("127.0.0.1:0").await.expect("listen failed");
    let addr = listener.local_addr().expect("local_addr");
    let url = format!("ws://{addr}");
    (url, listener)
}

/// `true` once a `client/goodbye` arrives; `false` if the peer goes quiet for 2s.
async fn wait_for_goodbye(rx: &mut UnboundedReceiver<String>) -> bool {
    while let Ok(Some(text)) = timeout(Duration::from_secs(2), rx.recv()).await {
        if matches!(
            serde_json::from_str::<Message>(&text),
            Ok(Message::ClientGoodbye(_))
        ) {
            return true;
        }
    }
    false
}

async fn wait_for_peer_close(reader: JoinHandle<()>) {
    timeout(Duration::from_secs(2), reader)
        .await
        .expect("peer did not observe WebSocket close")
        .expect("peer reader task failed");
}

#[tokio::test]
async fn test_listen_role_session_delivers_player_events() {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Listener Role".to_string())
        .player_v1_support(player_support())
        .build();
    let (url, listener) = bind_listener(builder).await;
    let peer = tokio::spawn(async move {
        let (ws, _) = connect_async(&url).await.expect("connect");
        run_peer(ws, vec!["player@v1".to_string()]).await
    });
    let (client, _) = timeout(Duration::from_secs(5), listener.accept())
        .await
        .expect("accept timeout")
        .expect("accept");
    let peer = peer.await.expect("peer task");
    let split = client.split();
    let output = ListenerOutput::default();
    let mut session = split.player.attach(output.clone());
    let start = Message::StreamStart(StreamStart {
        player: Some(StreamPlayerConfig {
            codec: "pcm".into(),
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        }),
        artwork: None,
        visualizer: None,
    });
    peer.writer
        .send(WsMessage::Text(
            serde_json::to_string(&start).unwrap().into(),
        ))
        .unwrap();
    let mut frame = vec![binary_types::PLAYER_AUDIO];
    frame.extend_from_slice(&77i64.to_be_bytes());
    frame.extend_from_slice(&[0, 0x10, 0, 0xf0]);
    peer.writer.send(WsMessage::Binary(frame.into())).unwrap();
    match timeout(Duration::from_secs(5), session.recv())
        .await
        .unwrap()
        .unwrap()
    {
        PlayerEvent::Start(config) => assert_eq!(config.codec, "pcm"),
        other => panic!("expected start, got {other:?}"),
    }
    match timeout(Duration::from_secs(5), session.recv())
        .await
        .unwrap()
        .unwrap()
    {
        PlayerEvent::Audio(work) => {
            assert_eq!(work.timestamp(), 77);
            assert_eq!(work.data(), &[0, 0x10, 0, 0xf0]);
            let samples = PcmDecoder::new(16).decode(work.data()).unwrap();
            assert_eq!(
                work.commit(AudioBuffer {
                    timestamp: 77,
                    samples,
                    format: AudioFormat {
                        codec: Codec::Pcm,
                        sample_rate: 48_000,
                        channels: 2,
                        bit_depth: 16,
                        codec_header: None,
                    },
                }),
                Ok(())
            );
            let records = output.commits.lock().unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].timestamp, 77);
            assert_eq!(records[0].samples.as_ref(), &[268_435_456, -268_435_456]);
            assert_eq!(records[0].format.codec, Codec::Pcm);
        }
        other => panic!("expected audio, got {other:?}"),
    }
}

#[tokio::test]
async fn test_listen_accepts_and_drives_handshake() {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test Inbound".to_string())
        .player_v1_support(player_support())
        .initial_player_state(PlayerState {
            volume: Some(60),
            muted: Some(false),
            static_delay_ms: Some(10),
            required_lead_time_ms: Some(500),
            min_buffer_ms: Some(500),
            supported_commands: None,
        })
        .build();

    let (url, listener) = bind_listener(builder).await;

    let peer =
        tokio::spawn(
            async move { drive_peer_until_state(&url, vec!["player@v1".to_string()]).await },
        );

    let (client, peer_addr) = timeout(Duration::from_secs(5), listener.accept())
        .await
        .expect("accept timed out")
        .expect("accept failed");
    assert!(peer_addr.ip().is_loopback(), "peer addr should be loopback");

    let mut peer = peer.await.expect("peer task panicked");
    assert_eq!(peer.supported_roles, vec!["player@v1".to_string()]);
    // The application uses these for the multi-server keep-or-switch policy
    // described on `ProtocolListener`.
    assert_eq!(client.server_id(), peer.server_id);
    assert!(client.activation().is_active("player@v1"));
    let rx = &mut peer.rx;

    let first_after_hello = timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("timed out waiting for client/state")
        .expect("peer channel closed");
    let parsed: Message =
        serde_json::from_str(&first_after_hello).expect("client/state must deserialize");
    match parsed {
        Message::ClientState(cs) => {
            assert_eq!(cs.state, Some(ClientSyncState::Synchronized));
            let player = cs.player.expect("expected initial player state");
            assert_eq!(player.volume, Some(60));
            assert_eq!(player.static_delay_ms, Some(10));
        }
        other => panic!("expected ClientState as first post-hello frame, got {other:?}"),
    }

    client
        .disconnect(GoodbyeReason::Shutdown)
        .await
        .expect("disconnect");

    assert!(
        wait_for_goodbye(rx).await,
        "never received client/goodbye over inbound link"
    );
}

#[tokio::test]
async fn test_listen_path_match_accepts() {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test Path".to_string())
        .player_v1_support(player_support())
        .build();

    let listener = builder
        .listen("127.0.0.1:0")
        .await
        .expect("listen")
        .path("/sendspin");
    let addr = listener.local_addr().unwrap();
    let url = format!("ws://{addr}/sendspin");

    let peer =
        tokio::spawn(
            async move { drive_peer_until_state(&url, vec!["player@v1".to_string()]).await },
        );
    let (client, _peer_addr) = timeout(Duration::from_secs(5), listener.accept())
        .await
        .expect("accept timed out")
        .expect("accept");

    let mut peer = peer.await.expect("peer task");

    client
        .disconnect(GoodbyeReason::Shutdown)
        .await
        .expect("disconnect");

    assert!(
        wait_for_goodbye(&mut peer.rx).await,
        "matched-path connection never delivered client/goodbye"
    );
}

/// A path given without a leading slash must still match the `/sendspin` the
/// peer connects to — `request.uri().path()` always starts with `/`, so the
/// un-normalized form would silently reject everyone.
#[tokio::test]
async fn test_listen_path_normalizes_missing_leading_slash() {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test Path Norm".to_string())
        .player_v1_support(player_support())
        .build();

    let listener = builder
        .listen("127.0.0.1:0")
        .await
        .expect("listen")
        .path("sendspin"); // no leading slash
    let addr = listener.local_addr().unwrap();
    let url = format!("ws://{addr}/sendspin");

    let peer =
        tokio::spawn(
            async move { drive_peer_until_state(&url, vec!["player@v1".to_string()]).await },
        );
    let (client, _peer_addr) = timeout(Duration::from_secs(5), listener.accept())
        .await
        .expect("accept timed out")
        .expect("accept");
    let mut peer = peer.await.expect("peer task");

    client
        .disconnect(GoodbyeReason::Shutdown)
        .await
        .expect("disconnect");
    assert!(
        wait_for_goodbye(&mut peer.rx).await,
        "normalized-path connection never delivered client/goodbye"
    );
}

#[tokio::test]
async fn test_listen_path_mismatch_rejects_but_listener_survives() {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test Path Reject".to_string())
        .player_v1_support(player_support())
        .build();

    let listener = builder
        .listen("127.0.0.1:0")
        .await
        .expect("listen")
        .path("/sendspin");
    let addr = listener.local_addr().unwrap();
    let bad_url = format!("ws://{addr}/wrong-path");
    let good_url = format!("ws://{addr}/sendspin");

    let bad_peer = tokio::spawn(async move {
        let result = connect_async(&bad_url).await;
        assert!(result.is_err(), "expected handshake to fail on bad path");
    });
    let accept_err = timeout(Duration::from_secs(5), listener.accept()).await;
    assert!(accept_err.is_ok(), "listener.accept did not return");
    assert!(
        accept_err.unwrap().is_err(),
        "expected per-peer Err on bad path"
    );
    bad_peer.await.expect("bad peer task");

    let good_peer = tokio::spawn(async move {
        drive_peer_until_state(&good_url, vec!["player@v1".to_string()]).await
    });
    let (client, _peer_addr) = timeout(Duration::from_secs(5), listener.accept())
        .await
        .expect("second accept timed out")
        .expect("second accept");
    let mut peer = good_peer.await.expect("good peer task");

    client
        .disconnect(GoodbyeReason::Shutdown)
        .await
        .expect("disconnect");
    assert!(
        wait_for_goodbye(&mut peer.rx).await,
        "matched-path connection never delivered client/goodbye"
    );
    wait_for_peer_close(peer.reader).await;
}

#[tokio::test]
async fn test_listen_local_addr_resolves_ephemeral_port() {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build();
    let listener = builder.listen("127.0.0.1:0").await.expect("listen");
    let addr = listener.local_addr().expect("local_addr");
    assert!(
        addr.port() != 0,
        "ephemeral port should resolve to non-zero"
    );
    assert!(addr.ip().is_loopback());
}

/// The send-after-disconnect contract must hold on the inbound path too,
/// not just the outbound `connect()` path.
#[tokio::test]
async fn test_listen_send_message_fails_after_disconnect() {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build();
    let (url, listener) = bind_listener(builder).await;

    let peer =
        tokio::spawn(
            async move { drive_peer_until_state(&url, vec!["player@v1".to_string()]).await },
        );
    let (client, _addr) = timeout(Duration::from_secs(5), listener.accept())
        .await
        .expect("accept timed out")
        .expect("accept");
    let peer = peer.await.expect("peer task");

    let conn = client.split();
    // `split()` must carry the arbitration inputs through to `Connection`,
    // the post-split handle most apps hold onto.
    assert_eq!(conn.server_id, peer.server_id);
    assert!(conn.activation.is_active("player@v1"));
    let sender = conn.sender;
    let guard = conn.guard;
    guard
        .disconnect(GoodbyeReason::Shutdown)
        .await
        .expect("disconnect");

    let result = sender
        .send_message(Message::ClientCommand(ClientCommand { controller: None }))
        .await;
    assert!(
        result.is_err(),
        "expected Err after disconnect, got {result:?}"
    );
}

/// A peer that completes the WebSocket upgrade but never sends `server/init`
/// must leave `accept()` pending — it must not spuriously resolve. Callers
/// guard against this with a timeout, which this test stands in for.
#[tokio::test]
async fn test_listen_accept_pending_when_peer_stalls_handshake() {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build();
    let (url, listener) = bind_listener(builder).await;

    let _peer = tokio::spawn(async move {
        let (mut ws, _) = connect_async(&url).await.expect("connect");
        // Read client/init but deliberately never reply with server/init,
        // holding the socket open so the SDK blocks awaiting it.
        let _ = ws.next().await;
        tokio::time::sleep(Duration::from_secs(30)).await;
        drop(ws);
    });

    let timed_out = timeout(Duration::from_millis(500), listener.accept())
        .await
        .is_err();
    assert!(timed_out, "accept() resolved before server/init arrived");
}

/// Browsers always send `Origin`, so rejecting it keeps web pages from
/// reaching the client.
#[tokio::test]
async fn test_listen_rejects_upgrade_with_origin() {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Test".to_string())
        .player_v1_support(player_support())
        .build();
    let (url, listener) = bind_listener(builder).await;

    let peer = tokio::spawn(async move {
        let mut request = url.into_client_request().unwrap();
        request
            .headers_mut()
            .insert("Origin", HeaderValue::from_static("https://example.com"));
        connect_async(request).await
    });
    let accepted = timeout(Duration::from_secs(5), listener.accept())
        .await
        .expect("accept did not return");
    assert!(accepted.is_err(), "upgrade with Origin was accepted");
    match peer.await.unwrap() {
        Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
            assert_eq!(response.status(), 403)
        }
        other => panic!("expected HTTP 403, got {other:?}"),
    }
}
