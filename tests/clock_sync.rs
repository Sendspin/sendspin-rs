use sendspin::sync::ClockSync;

#[test]
fn test_clock_sync_rtt_calculation() {
    let mut sync = ClockSync::new();

    // Simulate sync: client sends at 1000µs, server receives at 500µs (server loop time)
    let t1 = 1_000_000; // Client transmitted (Unix µs)
    let t2 = 500_000; // Server received (server loop µs)
    let t3 = 500_010; // Server transmitted (server loop µs)
    let t4 = 1_000_050; // Client received (Unix µs)

    sync.update(t1, t2, t3, t4);

    // RTT = (t4 - t1) - (t3 - t2) = 50 - 10 = 40µs
    assert_eq!(sync.rtt_micros(), Some(40));
}

#[test]
fn test_server_to_local_conversion() {
    let mut sync = ClockSync::new();

    let t1 = 1_000_000;
    let t2 = 500_000;
    let t3 = 500_010;
    let t4 = 1_000_050;

    sync.update(t1, t2, t3, t4);

    // Server loop start = t4 - t3 = 1_000_050 - 500_010 = 500_040 Unix µs
    // Converting server time 520_000 should give us ~520_040 Unix µs
    let local = sync.server_to_local_instant(520_000);
    assert!(local.is_some());
}

#[test]
fn test_sync_quality() {
    let mut sync = ClockSync::new();

    // Good RTT (30µs)
    sync.update(1_000_000, 500_000, 500_010, 1_000_040);
    assert_eq!(sync.quality(), sendspin::sync::SyncQuality::Good);

    // Degraded RTT (75ms = 75,000µs)
    sync.update(2_000_000, 600_000, 600_010, 2_075_010);
    assert_eq!(sync.quality(), sendspin::sync::SyncQuality::Degraded);
}
