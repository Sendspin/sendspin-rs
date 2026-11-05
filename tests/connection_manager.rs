// ABOUTME: Integration tests for ConnectionManager — concurrent inbound
// ABOUTME: handshakes, multi-server arbitration, and automatic goodbyes.

mod common;

use common::{server_id, TestServer};
use sendspin::audio::decode::{Decoder, PcmDecoder};
use sendspin::audio::{AudioBuffer, AudioFormat, Codec, OutputError, PlayerEvent, PlayerOutput};
use sendspin::protocol::client::binary_types;
use sendspin::protocol::manager::{ConnectionManager, ManagerConfig};
use sendspin::protocol::messages::{
    Activity, AudioFormatSpec, GoodbyeReason, Message, PlayerV1Support,
};
use sendspin::{MemoryStore, ProtocolClientBuilder};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Default)]
struct ManagerOutput {
    commits: Arc<Mutex<Vec<AudioBufferRecord>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AudioBufferRecord {
    timestamp: i64,
    samples: Arc<[i32]>,
    format: AudioFormat,
}

impl PlayerOutput for ManagerOutput {
    fn commit(&self, buffer: AudioBuffer) -> Result<(), OutputError> {
        self.commits.lock().unwrap().push(AudioBufferRecord {
            timestamp: buffer.timestamp,
            samples: buffer.samples,
            format: buffer.format,
        });
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

use tokio::time::timeout;
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};

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

/// A test protocol-server: dials the manager's listener, activates the
/// player role as the server called `name`, and forwards every later client
/// message except `client/time` to `messages`.
struct Peer {
    messages: tokio::sync::mpsc::UnboundedReceiver<String>,
    /// Send frames to the client, including role hello/state traffic.
    writer: tokio::sync::mpsc::UnboundedSender<WsMessage>,
}

/// Whether a server activates the client to play or only keeps it idle.
#[derive(Clone, Copy)]
enum Intent {
    Playback,
    Idle,
}

async fn connect_peer(url: &str, name: &str, intent: Intent) -> Peer {
    let (ws, _) = timeout(Duration::from_secs(5), connect_async(url))
        .await
        .expect("peer establishment timed out")
        .expect("WS connect failed");
    let (mut server, _) = TestServer::handshake_as(ws, name).await;
    let activities = match intent {
        Intent::Playback => vec![Activity::Playback],
        Intent::Idle => vec![],
    };
    server
        .activate_with(activities, vec!["player@v1".to_string()])
        .await;
    let (msg_tx, msg_rx) = tokio::sync::mpsc::unbounded_channel();
    let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(server.serve(msg_tx, out_rx));
    Peer {
        messages: msg_rx,
        writer: out_tx,
    }
}

async fn manager(config: Option<ManagerConfig>) -> (String, ConnectionManager) {
    let builder = ProtocolClientBuilder::builder()
        .persistence(Arc::new(MemoryStore::new()))
        .unpaired_access(true)
        .name("Managed Client".to_string())
        .player_v1_support(player_support())
        .build();
    let listener = builder.listen("127.0.0.1:0").await.expect("listen failed");
    let mgr = match config {
        Some(c) => ConnectionManager::with_config(listener, c),
        None => ConnectionManager::new(listener),
    };
    let addr = mgr.local_addr().expect("local_addr");
    (format!("ws://{addr}"), mgr)
}

/// Waits until the peer receives `client/goodbye` and returns its reason;
/// panics if the peer goes quiet for 2s first.
async fn expect_goodbye(peer: &mut Peer) -> GoodbyeReason {
    loop {
        let text = timeout(Duration::from_secs(2), peer.messages.recv())
            .await
            .expect("timed out waiting for client/goodbye")
            .expect("peer channel closed without a goodbye");
        if let Ok(Message::ClientGoodbye(g)) = serde_json::from_str::<Message>(&text) {
            return g.reason;
        }
    }
}

/// Asserts the peer receives no `client/goodbye` within `window`.
async fn assert_no_goodbye(peer: &mut Peer, window: Duration) {
    let deadline = tokio::time::Instant::now() + window;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        match timeout(remaining, peer.messages.recv()).await {
            Err(_) => return, // window elapsed quietly
            Ok(None) => panic!("peer connection closed unexpectedly"),
            Ok(Some(text)) => {
                if matches!(
                    serde_json::from_str::<Message>(&text),
                    Ok(Message::ClientGoodbye(_))
                ) {
                    panic!("unexpected client/goodbye");
                }
            }
        }
    }
}

/// Drains a yielded connection's message channel until it closes; panics if
/// it stays open past 2s.
async fn expect_channels_close(conn: &mut sendspin::ManagedConnection) {
    loop {
        match timeout(Duration::from_secs(2), conn.messages.recv()).await {
            Ok(None) => return,
            Ok(Some(_)) => continue,
            Err(_) => panic!("displaced connection's channels never closed"),
        }
    }
}

#[tokio::test]
async fn manager_role_session_handoff_and_displacement_fences_audio() {
    let (url, mut mgr) = manager(None).await;
    let mut peer_a = connect_peer(&url, "server-a", Intent::Idle).await;
    let conn_a = timeout(Duration::from_secs(5), mgr.next_connection())
        .await
        .unwrap()
        .unwrap();
    let output_a = ManagerOutput::default();
    let mut session_a = conn_a.player.attach(output_a.clone());
    peer_a
        .writer
        .send(WsMessage::Text(
            serde_json::to_string(&Message::StreamStart(
                sendspin::protocol::messages::StreamStart {
                    player: Some(sendspin::protocol::messages::StreamPlayerConfig {
                        codec: "pcm".into(),
                        sample_rate: 48_000,
                        channels: 2,
                        bit_depth: 16,
                        codec_header: None,
                    }),
                    artwork: None,
                    visualizer: None,
                },
            ))
            .unwrap()
            .into(),
        ))
        .unwrap();
    let mut frame_a = vec![binary_types::PLAYER_AUDIO];
    frame_a.extend_from_slice(&11i64.to_be_bytes());
    frame_a.extend_from_slice(&[0, 1, 0, 2]);
    peer_a
        .writer
        .send(WsMessage::Binary(frame_a.into()))
        .unwrap();
    match timeout(Duration::from_secs(5), session_a.recv())
        .await
        .expect("start event timed out")
        .expect("session ended before start")
    {
        PlayerEvent::Start(config) => {
            assert_eq!(config.codec, "pcm");
            assert_eq!(config.sample_rate, 48_000);
            assert_eq!(config.channels, 2);
            assert_eq!(config.bit_depth, 16);
        }
        other => panic!("expected start event, got {other:?}"),
    }
    let held = match timeout(Duration::from_secs(5), session_a.recv())
        .await
        .expect("audio event timed out")
        .expect("session ended before audio")
    {
        PlayerEvent::Audio(work) => {
            assert_eq!(work.timestamp(), 11);
            assert_eq!(work.data(), &[0, 1, 0, 2]);
            work
        }
        other => panic!("expected audio event, got {other:?}"),
    };
    let peer_b = connect_peer(&url, "server-b", Intent::Playback).await;
    assert_eq!(
        expect_goodbye(&mut peer_a).await,
        GoodbyeReason::AnotherServer
    );
    assert!(timeout(Duration::from_secs(5), session_a.recv())
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        held.commit(AudioBuffer {
            timestamp: 11,
            samples: PcmDecoder::new(16).decode(&[0, 1, 0, 2]).unwrap(),
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            },
        }),
        Err(sendspin::audio::CommitRejected::Closed)
    );
    assert!(output_a.commits.lock().unwrap().is_empty());
    let conn_b = timeout(Duration::from_secs(5), mgr.next_connection())
        .await
        .unwrap()
        .unwrap();
    let output_b = ManagerOutput::default();
    let mut session_b = conn_b.player.attach(output_b.clone());
    peer_b
        .writer
        .send(WsMessage::Text(
            serde_json::to_string(&Message::StreamStart(
                sendspin::protocol::messages::StreamStart {
                    player: Some(sendspin::protocol::messages::StreamPlayerConfig {
                        codec: "pcm".into(),
                        sample_rate: 48_000,
                        channels: 2,
                        bit_depth: 16,
                        codec_header: None,
                    }),
                    artwork: None,
                    visualizer: None,
                },
            ))
            .unwrap()
            .into(),
        ))
        .unwrap();
    let mut frame_b = vec![binary_types::PLAYER_AUDIO];
    frame_b.extend_from_slice(&22i64.to_be_bytes());
    frame_b.extend_from_slice(&[3, 4, 5, 6]);
    peer_b
        .writer
        .send(WsMessage::Binary(frame_b.into()))
        .unwrap();
    match timeout(Duration::from_secs(5), session_b.recv())
        .await
        .unwrap()
        .unwrap()
    {
        PlayerEvent::Start(_) => {}
        other => panic!("expected replacement start, got {other:?}"),
    }
    match timeout(Duration::from_secs(5), session_b.recv())
        .await
        .unwrap()
        .unwrap()
    {
        PlayerEvent::Audio(work) => {
            assert_eq!(work.timestamp(), 22);
            assert_eq!(work.data(), &[3, 4, 5, 6]);
            let samples = PcmDecoder::new(16).decode(work.data()).unwrap();
            assert_eq!(
                work.commit(AudioBuffer {
                    timestamp: 22,
                    samples: samples.clone(),
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
            assert_eq!(
                output_b.commits.lock().unwrap().as_slice(),
                &[AudioBufferRecord {
                    timestamp: 22,
                    samples,
                    format: AudioFormat {
                        codec: Codec::Pcm,
                        sample_rate: 48_000,
                        channels: 2,
                        bit_depth: 16,
                        codec_header: None,
                    },
                }]
            );
            assert!(output_a.commits.lock().unwrap().is_empty());
        }
        other => panic!("expected replacement audio, got {other:?}"),
    }
}

#[tokio::test]
async fn test_first_server_is_promoted() {
    let (url, mut mgr) = manager(None).await;

    let _peer = connect_peer(&url, "server-a", Intent::Idle).await;

    let conn = timeout(Duration::from_secs(5), mgr.next_connection())
        .await
        .expect("next_connection timed out")
        .expect("manager stopped");
    assert_eq!(conn.server_id, server_id("server-a"));
    assert!(conn.activation.activities.is_empty());
    assert!(conn.peer.ip().is_loopback());
}

#[tokio::test]
async fn test_playback_displaces_idle_incumbent() {
    let (url, mut mgr) = manager(None).await;

    let mut peer_a = connect_peer(&url, "server-a", Intent::Idle).await;
    let mut conn_a = mgr.next_connection().await.expect("manager stopped");
    assert_eq!(conn_a.server_id, server_id("server-a"));

    let _peer_b = connect_peer(&url, "server-b", Intent::Playback).await;

    // The displaced incumbent gets an automatic goodbye(another_server)...
    assert_eq!(
        expect_goodbye(&mut peer_a).await,
        GoodbyeReason::AnotherServer
    );
    // ...the app sees its channels close...
    expect_channels_close(&mut conn_a).await;
    // ...and the winner is yielded.
    let conn_b = timeout(Duration::from_secs(5), mgr.next_connection())
        .await
        .expect("next_connection timed out")
        .expect("manager stopped");
    assert_eq!(conn_b.server_id, server_id("server-b"));
}

#[tokio::test]
async fn test_idle_never_displaces_playback() {
    let (url, mut mgr) = manager(None).await;

    let mut peer_a = connect_peer(&url, "server-a", Intent::Playback).await;
    let conn_a = mgr.next_connection().await.expect("manager stopped");
    assert_eq!(conn_a.server_id, server_id("server-a"));

    // Even the last-played server cannot displace active playback.
    mgr.set_last_played(Some(server_id("server-b")));
    let mut peer_b = connect_peer(&url, "server-b", Intent::Idle).await;

    // The rejected newcomer gets the goodbye; the incumbent stays quiet.
    assert_eq!(
        expect_goodbye(&mut peer_b).await,
        GoodbyeReason::AnotherServer
    );
    assert_no_goodbye(&mut peer_a, Duration::from_millis(300)).await;
}

#[tokio::test]
async fn test_last_played_breaks_idle_tie() {
    let (url, mut mgr) = manager(None).await;
    mgr.set_last_played(Some(server_id("server-b")));

    let mut peer_a = connect_peer(&url, "server-a", Intent::Idle).await;
    let conn_a = mgr.next_connection().await.expect("manager stopped");
    assert_eq!(conn_a.server_id, server_id("server-a"));

    let _peer_b = connect_peer(&url, "server-b", Intent::Idle).await;
    assert_eq!(
        expect_goodbye(&mut peer_a).await,
        GoodbyeReason::AnotherServer
    );
    let conn_b = timeout(Duration::from_secs(5), mgr.next_connection())
        .await
        .expect("next_connection timed out")
        .expect("manager stopped");
    assert_eq!(conn_b.server_id, server_id("server-b"));
}

#[tokio::test]
async fn test_idle_tie_defaults_to_keep() {
    let (url, mut mgr) = manager(None).await;

    let mut peer_a = connect_peer(&url, "server-a", Intent::Idle).await;
    let conn_a = mgr.next_connection().await.expect("manager stopped");
    assert_eq!(conn_a.server_id, server_id("server-a"));

    let mut peer_b = connect_peer(&url, "server-b", Intent::Idle).await;
    assert_eq!(
        expect_goodbye(&mut peer_b).await,
        GoodbyeReason::AnotherServer
    );
    assert_no_goodbye(&mut peer_a, Duration::from_millis(300)).await;
}

#[tokio::test]
async fn test_incumbent_death_frees_slot() {
    let (url, mut mgr) = manager(None).await;

    let peer_a = connect_peer(&url, "server-a", Intent::Idle).await;
    let mut conn_a = mgr.next_connection().await.expect("manager stopped");
    assert_eq!(conn_a.server_id, server_id("server-a"));

    // Server A goes away (socket closed, no goodbye exchange).
    drop(peer_a);
    expect_channels_close(&mut conn_a).await;

    // A plain idle connection (not last-played) must now be promoted:
    // the dead incumbent may no longer win arbitration.
    let mut peer_b = connect_peer(&url, "server-b", Intent::Idle).await;
    let conn_b = timeout(Duration::from_secs(5), mgr.next_connection())
        .await
        .expect("next_connection timed out")
        .expect("manager stopped");
    assert_eq!(conn_b.server_id, server_id("server-b"));
    assert_no_goodbye(&mut peer_b, Duration::from_millis(300)).await;
}

#[tokio::test]
async fn test_disconnect_sends_goodbye_and_keeps_listening() {
    let (url, mut mgr) = manager(None).await;

    // Disconnect with no incumbent is a no-op Ok.
    mgr.disconnect(GoodbyeReason::UserRequest)
        .await
        .expect("no-op disconnect");

    let mut peer_a = connect_peer(&url, "server-a", Intent::Playback).await;
    let mut conn_a = mgr.next_connection().await.expect("manager stopped");

    mgr.disconnect(GoodbyeReason::UserRequest)
        .await
        .expect("disconnect");
    assert_eq!(
        expect_goodbye(&mut peer_a).await,
        GoodbyeReason::UserRequest
    );
    expect_channels_close(&mut conn_a).await;

    // The manager keeps listening: a new server can still connect.
    let _peer_b = connect_peer(&url, "server-b", Intent::Idle).await;
    let conn_b = timeout(Duration::from_secs(5), mgr.next_connection())
        .await
        .expect("next_connection timed out")
        .expect("manager stopped");
    assert_eq!(conn_b.server_id, server_id("server-b"));
}

#[tokio::test]
async fn test_playback_displaces_playback() {
    let (url, mut mgr) = manager(None).await;

    let mut peer_a = connect_peer(&url, "server-a", Intent::Playback).await;
    let conn_a = mgr.next_connection().await.expect("manager stopped");
    assert_eq!(conn_a.server_id, server_id("server-a"));

    // Newest playback intent wins even against a playback incumbent.
    let _peer_b = connect_peer(&url, "server-b", Intent::Playback).await;
    assert_eq!(
        expect_goodbye(&mut peer_a).await,
        GoodbyeReason::AnotherServer
    );
    let conn_b = timeout(Duration::from_secs(5), mgr.next_connection())
        .await
        .expect("next_connection timed out")
        .expect("manager stopped");
    assert_eq!(conn_b.server_id, server_id("server-b"));
}

#[tokio::test]
async fn test_drop_aborts_inflight_handshake_promptly() {
    // A long establish deadline: without abort-on-drop tracking, the
    // in-flight handshake task would keep the peer's TCP stream (and the
    // listener) alive for the full 30s after the manager is gone.
    let (url, mgr) = manager(Some(ManagerConfig {
        establish_timeout: Duration::from_secs(30),
        max_concurrent_handshakes: 1,
        ..ManagerConfig::default()
    }))
    .await;

    let addr = url.strip_prefix("ws://").unwrap().to_string();
    let stalled = tokio::net::TcpStream::connect(&addr)
        .await
        .expect("raw TCP connect");
    // Let the accept loop admit the stalled peer into its handshake slot.
    tokio::time::sleep(Duration::from_millis(100)).await;

    drop(mgr);

    // The aborted handshake task must drop the TCP stream promptly: the
    // stalled peer observes EOF/reset well before the 30s deadline.
    let mut buf = [0u8; 1];
    let read = timeout(Duration::from_secs(2), async {
        use tokio::io::AsyncReadExt;
        let mut stream = stalled;
        stream.read(&mut buf).await
    })
    .await;
    match read {
        Ok(Ok(0)) => {}  // clean EOF
        Ok(Err(_)) => {} // reset — also fine
        Ok(Ok(n)) => panic!("unexpected {n} bytes from dropped manager"),
        Err(_) => panic!("handshake task kept the connection alive after manager drop"),
    }
}

#[tokio::test]
async fn test_stalled_handshake_is_reaped_and_frees_slot() {
    // One handshake slot and a short establish deadline: a peer that
    // connects and stalls must be reaped so the next peer can establish.
    let (url, mut mgr) = manager(Some(ManagerConfig {
        establish_timeout: Duration::from_millis(300),
        max_concurrent_handshakes: 1,
        ..ManagerConfig::default()
    }))
    .await;

    // Raw TCP connect that never speaks WebSocket, occupying the only slot.
    let addr = url.strip_prefix("ws://").unwrap().to_string();
    let stalled = tokio::net::TcpStream::connect(&addr)
        .await
        .expect("raw TCP connect");

    // Give the accept loop time to admit the stalled peer into its slot.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // A real server connecting behind it must still establish once the
    // stalled peer is reaped at the deadline.
    let _peer = connect_peer(&url, "server-a", Intent::Playback).await;
    let conn = timeout(Duration::from_secs(5), mgr.next_connection())
        .await
        .expect("next_connection timed out — stalled peer was never reaped")
        .expect("manager stopped");
    assert_eq!(conn.server_id, server_id("server-a"));

    drop(stalled);
}
