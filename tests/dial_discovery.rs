// ABOUTME: Integration test for the server-initiated connection path —
// ABOUTME: discovering a client that advertises itself over mDNS
// ABOUTME: (like ESPHome's sendspin: component) and dialing in to it.

mod common;

use futures_util::{SinkExt, StreamExt};
use mdns_sd::ServiceInfo;
use sendspin::protocol::messages::Message;
use sendspin::server::{dial_client, ClientBrowser};
use sendspin::DefaultClock;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Stands in for a real embedded client (e.g. ESPHome's sendspin: component)
/// that only ever runs its own WebSocket server and advertises itself over
/// mDNS for a real Sendspin server to discover and dial in to.
async fn run_fake_embedded_client(port: u16) -> Message {
    let tcp = TcpListener::bind(("127.0.0.1", port)).await.unwrap();
    let (stream, _) = tcp.accept().await.unwrap();
    let ws = tokio_tungstenite::accept_async(stream).await.unwrap();
    let (mut write, mut read) = ws.split();

    let hello = serde_json::to_string(&Message::ClientHello(
        sendspin::protocol::messages::ClientHello {
            client_id: "fake-embedded-client".to_string(),
            name: "Fake Embedded Client".to_string(),
            version: 1,
            supported_roles: vec!["player@v1".to_string()],
            device_info: None,
            player_v1_support: None,
            artwork_v1_support: None,
            visualizer_v1_support: None,
        },
    ))
    .unwrap();
    write.send(WsMessage::Text(hello.into())).await.unwrap();

    let msg = read.next().await.expect("no server/hello").unwrap();
    match msg {
        WsMessage::Text(text) => {
            serde_json::from_str(&text).expect("server/hello must deserialize")
        }
        other => panic!("expected text server/hello, got {other:?}"),
    }
}

#[tokio::test]
async fn discovers_and_dials_a_self_advertising_client() {
    if !common::net_tests_enabled() {
        eprintln!("skipping: set SENDSPIN_NET_TESTS=1 to run mDNS multicast tests");
        return;
    }
    // Real mDNS multicast timing (see tests/common/mod.rs) — retry rather
    // than let one transient hiccup fail CI.
    common::retry_flaky(3, discovers_and_dials_a_self_advertising_client_impl).await;
}

async fn discovers_and_dials_a_self_advertising_client_impl() {
    // A free ephemeral port, then bind our fake client to it directly (not
    // through the OS's "any free port" allocator) so the mDNS advertisement
    // below can name it explicitly, matching how a real embedded device
    // knows its own listening port.
    let port = {
        let probe = TcpListener::bind("127.0.0.1:0").await.unwrap();
        probe.local_addr().unwrap().port()
    };

    // Instance name includes the port so a retried attempt (after a
    // previous one panicked, leaving its ServiceDaemon's advertisement
    // registered — ServiceDaemon has no Drop impl to unregister it for us)
    // never collides with a still-lingering advertisement from an earlier
    // attempt in the same test run.
    let instance_name = format!("fake-embedded-client-{port}");
    let advertise_daemon = mdns_sd::ServiceDaemon::new().expect("mdns daemon");
    let service = ServiceInfo::new(
        "_sendspin._tcp.local.",
        &instance_name,
        &format!("{instance_name}.local."),
        "",
        port,
        &[("path", "/sendspin")][..],
    )
    .unwrap()
    .enable_addr_auto();
    advertise_daemon.register(service).expect("register");

    let client_task = tokio::spawn(run_fake_embedded_client(port));

    // The test host's LAN can (and, per direct observation while writing
    // this test, does) have *real* Sendspin devices also advertising
    // `_sendspin._tcp.local.` — so this loops past whatever else it sees
    // rather than assuming the first resolved client is ours.
    let browser = ClientBrowser::new().expect("browser");
    let url = timeout(Duration::from_secs(15), async {
        loop {
            let url = browser.next_client_url().await?;
            if url.contains(&port.to_string()) {
                return Some(url);
            }
        }
    })
    .await
    .expect("discovery timed out")
    .expect("browser channel closed without finding our fake client");
    assert!(
        url.contains(&port.to_string()) && url.ends_with("/sendspin"),
        "unexpected discovered URL: {url}"
    );

    let conn = dial_client(
        &url,
        "test-dial-server",
        "Test Dial Server",
        Arc::new(DefaultClock::default()),
    )
    .await
    .expect("dial_client failed");
    assert_eq!(conn.client_id(), "fake-embedded-client");
    assert_eq!(conn.active_roles(), ["player@v1".to_string()]);

    let server_hello = timeout(Duration::from_secs(5), client_task)
        .await
        .expect("fake client task timed out")
        .expect("fake client task panicked");
    match server_hello {
        Message::ServerHello(hello) => {
            assert_eq!(hello.server_id, "test-dial-server");
            assert_eq!(hello.active_roles, vec!["player@v1".to_string()]);
        }
        other => panic!("expected server/hello, got {other:?}"),
    }

    let _ = advertise_daemon.shutdown();
}
