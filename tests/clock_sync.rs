use sendspin::sync::ClockSync;

/// Assert that two values are within tolerance (microseconds precision)
fn assert_within(actual: Option<i64>, expected: i64, tolerance: i64) {
    let actual = actual.expect("expected Some value");
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "expected {} ± {}, got {} (diff: {})",
        expected,
        tolerance,
        actual,
        diff
    );
}

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
fn test_server_to_client_conversion() {
    let mut sync = ClockSync::new();

    let t1 = 1_000_000;
    let t2 = 1_005_100;
    let t3 = 1_005_100;
    let t4 = 1_000_200;

    sync.update(t1, t2, t3, t4);
    sync.update(2_000_000, 2_005_100, 2_005_100, 2_000_200);

    let client_micros = sync.server_to_client_micros(2_005_000);
    // Kalman filter may introduce small rounding errors
    assert_within(client_micros, 2_000_000, 10);
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

#[test]
fn test_clock_drift_correction() {
    let mut sync = ClockSync::new();

    sync.update(1_000_000, 1_005_100, 1_005_100, 1_000_200);
    sync.update(2_000_000, 2_005_200, 2_005_200, 2_000_200);

    let server_time = sync.client_to_server_micros(3_000_200);
    // Kalman filter may introduce small rounding errors
    assert_within(server_time, 3_005_400, 10);
}
