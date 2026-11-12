use sendspin::sync::{ClockSync, DefaultClock, SyncQuality};
use std::sync::Arc;

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
fn test_fresh_clock_sync_initial_state() {
    let sync = ClockSync::new(Arc::new(DefaultClock::new()));

    assert_eq!(sync.rtt_micros(), None);
    assert!(!sync.is_synchronized());
    assert_eq!(sync.quality(), SyncQuality::Lost);
    assert!(sync.is_stale());
    assert_eq!(sync.server_to_client_micros(1000), None);
    assert_eq!(sync.client_to_server_micros(1000), None);
}

#[test]
fn test_single_update_not_synchronized() {
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

    // One sample isn't enough for the filter to converge (needs count >= 2)
    sync.update(1_000_000, 500_000, 500_010, 1_000_040);

    assert_eq!(sync.rtt_micros(), Some(30));
    assert!(
        !sync.is_synchronized(),
        "single update should not synchronize"
    );
    assert_eq!(
        sync.server_to_client_micros(500_000),
        None,
        "should return None when not synchronized"
    );
}

#[test]
fn test_clock_sync_rtt_calculation() {
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

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
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

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
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

    // Good RTT (30µs)
    sync.update(1_000_000, 500_000, 500_010, 1_000_040);
    assert_eq!(sync.quality(), sendspin::sync::SyncQuality::Good);

    // Degraded RTT (75ms = 75,000µs)
    sync.update(2_000_000, 600_000, 600_010, 2_075_010);
    assert_eq!(sync.quality(), SyncQuality::Degraded);
}

#[test]
fn test_sync_quality_recovery() {
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

    // Start with degraded RTT (75ms)
    sync.update(1_000_000, 600_000, 600_010, 1_075_010);
    assert_eq!(sync.quality(), SyncQuality::Degraded);

    // Recover with good RTT (20µs)
    sync.update(2_000_000, 700_000, 700_010, 2_000_030);
    assert_eq!(sync.quality(), SyncQuality::Good);
}

#[test]
fn test_high_rtt_sample_is_used() {
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

    sync.update(1_000_000, 500_000, 500_010, 1_000_040);
    assert_eq!(sync.quality(), SyncQuality::Good);

    // RTT = (3_100_020 - 3_000_000) - (700_010 - 700_000) = 100_010µs
    sync.update(3_000_000, 700_000, 700_010, 3_100_020);

    assert_eq!(sync.rtt_micros(), Some(100_010));
    assert_eq!(sync.quality(), SyncQuality::Lost);
}

#[test]
fn test_clock_drift_correction() {
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

    // Offset grows by 100ppm: 5000µs + t / 10_000, with a 20µs RTT.
    for second in 1..=4 {
        let t = second * 1_000_000;
        let server = t + 5_000 + t / 10_000 + 10;
        sync.update(t, server, server, t + 20);
    }

    let server_time = sync.client_to_server_micros(5_000_000);
    assert_within(server_time, 5_005_500, 10);
}

#[test]
fn test_first_two_samples_do_not_apply_drift() {
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

    // Offset moves +100µs → +4100µs over a 200ms baseline (drift = 0.02)
    // with a 2µs RTT. The filter seeds drift from this pair but only
    // decides whether to apply it from the third sample on.
    sync.update(1_000, 1_101, 1_101, 1_002);
    sync.update(201_000, 205_101, 205_101, 201_002);

    assert!(sync.is_synchronized());
    assert_eq!(sync.server_to_client_micros(3_120), Some(-980));
    assert_eq!(sync.client_to_server_micros(3_000), Some(7_100));
}

#[test]
fn test_negative_rtt_discarded() {
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

    // Craft timestamps where t4 < t1 (response "before" request),
    // producing negative RTT. Should be silently discarded.
    sync.update(1000, 500, 500, 900);
    assert_eq!(sync.rtt_micros(), None, "negative RTT should not be stored");
    assert!(
        !sync.is_synchronized(),
        "filter should not advance on invalid RTT"
    );
}

#[test]
fn test_zero_rtt_discarded() {
    let mut sync = ClockSync::new(Arc::new(DefaultClock::new()));

    // RTT = 0 gives max_error = 0, which the time filter discards.
    sync.update(1000, 1100, 1100, 1000);
    sync.update(2000, 2100, 2100, 2000);
    assert_eq!(sync.rtt_micros(), None);
    assert!(!sync.is_synchronized());
}
