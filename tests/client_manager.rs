// ABOUTME: Integration tests for ClientManager
// ABOUTME: Discovery/dial, retry after disconnect, and re-dial on address change
//
// These exercise real mDNS multicast, so they are wrapped in
// common::retry_flaky to absorb environmental timing noise on a shared LAN.

mod common;

use futures_util::{SinkExt, StreamExt};
use mdns_sd::ServiceInfo;
use sendspin::protocol::messages::{ClientHello, Message};
use sendspin::server::{ClientEvent, ClientManager};
use sendspin::DefaultClock;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::Message as WsMessage;

fn test_hello(client_id: &str) -> ClientHello {
    ClientHello {
        client_id: client_id.to_string(),
        name: "Fake Embedded Client".to_string(),
        version: 1,
        supported_roles: vec!["player@v1".to_string()],
        device_info: None,
        player_v1_support: None,
        artwork_v1_support: None,
        visualizer_v1_support: None,
    }
}

async fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn advertise(fullname_instance: &str, port: u16) -> mdns_sd::ServiceDaemon {
    let daemon = mdns_sd::ServiceDaemon::new().expect("mdns daemon");
    let service = ServiceInfo::new(
        "_sendspin._tcp.local.",
        fullname_instance,
        &format!("{fullname_instance}.local."),
        "",
        port,
        &[("path", "/sendspin")][..],
    )
    .unwrap()
    .enable_addr_auto();
    daemon.register(service).expect("register");
    daemon
}

/// Accepts exactly one connection, completes the handshake, then closes —
/// standing in for a real device's connection dropping.
async fn accept_one_and_close(port: u16, client_id: &str) {
    let tcp = TcpListener::bind(("127.0.0.1", port)).await.unwrap();
    let (stream, _) = tcp.accept().await.unwrap();
    let ws = tokio_tungstenite::accept_async(stream).await.unwrap();
    let (mut write, mut read) = ws.split();
    let hello = serde_json::to_string(&Message::ClientHello(test_hello(client_id))).unwrap();
    write.send(WsMessage::Text(hello.into())).await.unwrap();
    read.next().await.expect("no server/hello").unwrap();
    write.close().await.ok();
}

/// Waits for a `Connected` event whose `client_id` matches, ignoring any
/// interleaved `Disconnected`/`Message` events from other test noise on a
/// shared LAN (see tests/dial_discovery.rs for why that matters here).
async fn next_connected(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<ClientEvent>,
    expected_client_id: &str,
) {
    loop {
        match timeout(Duration::from_secs(30), rx.recv())
            .await
            .expect("timed out waiting for Connected")
            .expect("event channel closed")
        {
            ClientEvent::Connected { client_id, .. } if client_id == expected_client_id => return,
            _ => continue,
        }
    }
}

/// Same idea as [`next_connected`], for `Disconnected` — the shared LAN can
/// interleave events from other real devices, so this skips anything that
/// isn't the one we're waiting for instead of assuming the next event is it.
async fn next_disconnected(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<ClientEvent>,
    expected_client_id: &str,
) {
    loop {
        match timeout(Duration::from_secs(30), rx.recv())
            .await
            .expect("timed out waiting for Disconnected")
            .expect("event channel closed")
        {
            ClientEvent::Disconnected { client_id } if client_id == expected_client_id => return,
            _ => continue,
        }
    }
}

#[tokio::test]
async fn reconnects_after_the_client_drops() {
    if !common::net_tests_enabled() {
        eprintln!("skipping: set SENDSPIN_NET_TESTS=1 to run mDNS multicast tests");
        return;
    }
    common::retry_flaky(3, reconnects_after_the_client_drops_impl).await;
}

async fn reconnects_after_the_client_drops_impl() {
    let port = free_port().await;
    // Includes the port so a retried attempt (after a previous one
    // panicked, leaving its bare mdns_sd::ServiceDaemon's advertisement
    // registered with no Drop impl to clean it up) never collides with a
    // still-lingering advertisement from an earlier attempt.
    let instance_name = format!("manager-test-reconnect-{port}");
    let daemon = advertise(&instance_name, port);

    // Filtered to just this test's own fake client: the dev network this
    // was written against has real Sendspin devices on it too, and an
    // unfiltered manager would also try to dial (and hold reconnect loops
    // against) those — noisy for the test and, worse, actively interferes
    // with hardware you might be using for something else at the time.
    let (manager, mut events) = ClientManager::start_filtered(
        "test-server",
        "Test Server",
        Arc::new(DefaultClock::default()),
        move |fullname| fullname.starts_with(&instance_name),
    )
    .expect("start");

    accept_one_and_close(port, "reconnect-client").await;
    next_connected(&mut events, "reconnect-client").await;

    // First connection closing should surface as Disconnected...
    next_disconnected(&mut events, "reconnect-client").await;

    // ...and the manager should dial again on its own (backoff starts at
    // 1s) without anything external triggering it.
    accept_one_and_close(port, "reconnect-client").await;
    next_connected(&mut events, "reconnect-client").await;

    drop(manager);
    let _ = daemon.shutdown();
}

#[tokio::test]
async fn redials_promptly_when_the_same_device_reappears_at_a_new_address() {
    if !common::net_tests_enabled() {
        eprintln!("skipping: set SENDSPIN_NET_TESTS=1 to run mDNS multicast tests");
        return;
    }
    common::retry_flaky(
        3,
        redials_promptly_when_the_same_device_reappears_at_a_new_address_impl,
    )
    .await;
}

async fn redials_promptly_when_the_same_device_reappears_at_a_new_address_impl() {
    let _ = env_logger::builder().is_test(false).try_init();
    let old_port = free_port().await;
    let new_port = free_port().await;
    // Includes old_port (fresh per attempt) so a retried attempt never
    // collides with a still-lingering advertisement from an earlier one —
    // stable across the old_port -> new_port move within *this* attempt,
    // since that's the specific thing being tested.
    let instance_name = format!("manager-test-address-change-{old_port}");
    let daemon = advertise(&instance_name, old_port);

    let (manager, mut events) = ClientManager::start_filtered(
        "test-server",
        "Test Server",
        Arc::new(DefaultClock::default()),
        {
            let instance_name = instance_name.clone();
            move |fullname| fullname.starts_with(&instance_name)
        },
    )
    .expect("start");

    accept_one_and_close(old_port, "movable-client").await;
    next_connected(&mut events, "movable-client").await;
    next_disconnected(&mut events, "movable-client").await;

    // Re-advertise the *same* instance name at a new port before the old
    // task's backoff would naturally retry — this proves reconnection
    // happens via the new resolution, not by coincidentally retrying an
    // address that still happens to work.
    let _ = daemon.shutdown();
    let daemon2 = advertise(&instance_name, new_port);

    accept_one_and_close(new_port, "movable-client").await;
    next_connected(&mut events, "movable-client").await;

    drop(manager);
    let _ = daemon2.shutdown();
}
