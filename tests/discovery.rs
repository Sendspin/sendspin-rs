// ABOUTME: Integration test for the server role's mDNS advertisement —
// ABOUTME: proves a real browser resolves the service with the right port/path.

mod common;

use mdns_sd::{ServiceDaemon, ServiceEvent};
use sendspin::server::Advertisement;
use std::time::Duration;

#[test]
fn advertised_service_resolves_with_expected_port_and_path() {
    if !common::net_tests_enabled() {
        eprintln!("skipping: set SENDSPIN_NET_TESTS=1 to run mDNS multicast tests");
        return;
    }
    // Real mDNS multicast timing (see tests/common/mod.rs) — retry rather
    // than let one transient hiccup fail CI.
    common::retry_flaky_sync(
        3,
        advertised_service_resolves_with_expected_port_and_path_impl,
    );
}

fn advertised_service_resolves_with_expected_port_and_path_impl() {
    let ad = Advertisement::new(
        "test-discovery-server",
        "Test Discovery Server",
        18999,
        "/sendspin",
    )
    .expect("advertise");

    let browser = ServiceDaemon::new().expect("browser daemon");
    let receiver = browser
        .browse("_sendspin-server._tcp.local.")
        .expect("browse");

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut resolved = None;
    while std::time::Instant::now() < deadline {
        if let Ok(ServiceEvent::ServiceResolved(info)) =
            receiver.recv_timeout(Duration::from_secs(1))
        {
            if info.get_port() == 18999 {
                resolved = Some(info);
                break;
            }
        }
    }
    drop(ad);
    let _ = browser.shutdown();

    let info = resolved.expect("service was never resolved via mDNS within 10s");
    assert_eq!(info.get_port(), 18999);
    assert_eq!(
        info.get_property_val_str("path"),
        Some("/sendspin"),
        "path TXT record must match what clients need to connect to"
    );
    assert_eq!(
        info.get_property_val_str("name"),
        Some("Test Discovery Server")
    );
}
