// ABOUTME: Clock synchronization implementation
// ABOUTME: Drift-aware time sync with RTT estimation and server/client time conversion

use super::raw_clock::{Clock, DefaultClock};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Parameter values from the spec's time filter parameter table.
const PROCESS_STD_DEV: f64 = 0.0;
const DRIFT_PROCESS_STD_DEV: f64 = 1e-11;
const FORGET_FACTOR: f64 = 2.0;
const ADAPTIVE_CUTOFF: f64 = 3.0;
const MIN_SAMPLES: u32 = 100;
const DRIFT_SIGNIFICANCE_THRESHOLD: f64 = 2.0;
const MAX_ERROR_SCALE: f64 = 0.5;

/// The spec's time filter: a Kalman filter over clock offset and drift.
#[derive(Debug)]
struct TimeFilter {
    offset: f64,
    drift: f64,
    p_oo: f64,
    p_od: f64,
    p_dd: f64,
    last_update: i64,
    count: u32,
    use_drift: bool,
}

impl TimeFilter {
    fn new() -> Self {
        Self {
            offset: 0.0,
            drift: 0.0,
            p_oo: f64::INFINITY,
            p_od: 0.0,
            p_dd: 0.0,
            last_update: 0,
            count: 0,
            use_drift: false,
        }
    }

    /// Returns false when the measurement is discarded.
    fn update(&mut self, measurement: i64, max_error: i64, time_added: i64) -> bool {
        if time_added <= self.last_update || max_error <= 0 {
            return false;
        }
        let dt = (time_added - self.last_update) as f64;
        self.last_update = time_added;

        let measurement = measurement as f64;
        let max_error = max_error as f64;
        let r = (MAX_ERROR_SCALE * max_error) * (MAX_ERROR_SCALE * max_error);

        if self.count == 0 {
            self.offset = measurement;
            self.p_oo = r;
            self.drift = 0.0;
            self.count = 1;
            return true;
        }

        if self.count == 1 {
            self.drift = (measurement - self.offset) / dt;
            self.p_dd = (self.p_oo + r) / (dt * dt);
            self.offset = measurement;
            self.p_oo = r;
            self.count = 2;
            return true;
        }

        let offset_p = self.offset + self.drift * dt;
        let mut p_oo_p = self.p_oo
            + 2.0 * self.p_od * dt
            + self.p_dd * (dt * dt)
            + PROCESS_STD_DEV * PROCESS_STD_DEV * dt;
        let mut p_od_p = self.p_od + self.p_dd * dt;
        let mut p_dd_p = self.p_dd + DRIFT_PROCESS_STD_DEV * DRIFT_PROCESS_STD_DEV * dt;

        let y = measurement - offset_p;
        if self.count < MIN_SAMPLES {
            self.count += 1;
        } else if y.abs() > ADAPTIVE_CUTOFF * max_error {
            log::debug!(
                "TimeFilter: residual={y:.1}µs exceeds {ADAPTIVE_CUTOFF} × max_error={max_error}µs, inflating covariance"
            );
            let forget = FORGET_FACTOR * FORGET_FACTOR;
            p_oo_p *= forget;
            p_od_p *= forget;
            p_dd_p *= forget;
        }

        let s = p_oo_p + r;
        let k_o = p_oo_p / s;
        let k_d = p_od_p / s;
        self.offset = offset_p + k_o * y;
        self.drift += k_d * y;
        self.p_oo = p_oo_p - k_o * p_oo_p;
        self.p_od = p_od_p - k_d * p_oo_p;
        self.p_dd = p_dd_p - k_d * p_od_p;

        self.use_drift = self.drift * self.drift
            > DRIFT_SIGNIFICANCE_THRESHOLD * DRIFT_SIGNIFICANCE_THRESHOLD * self.p_dd;
        true
    }

    fn effective_drift(&self) -> f64 {
        if self.use_drift {
            self.drift
        } else {
            0.0
        }
    }

    fn compute_server_time(&self, client_time: i64) -> i64 {
        let dt = (client_time - self.last_update) as f64;
        client_time + (self.offset + self.effective_drift() * dt).round() as i64
    }

    fn compute_client_time(&self, server_time: i64) -> i64 {
        let d = self.effective_drift();
        ((server_time as f64 - self.offset + d * self.last_update as f64) / (1.0 + d)).round()
            as i64
    }

    fn is_synchronized(&self) -> bool {
        self.count >= 2 && self.p_oo.is_finite()
    }

    /// Samples required before the estimate is trusted for fine-grained
    /// playback corrections. Bursts start 1s apart until then (see
    /// `TIME_BURST_INTERVAL_SETTLING`), so ten samples settle ~8s after
    /// connect.
    const SETTLE_MIN_SAMPLES: u32 = 10;

    fn is_settled(&self) -> bool {
        self.count >= Self::SETTLE_MIN_SAMPLES
    }
}

/// Half the round-trip delay of one `client/time` exchange, the bound on
/// its offset error.
pub(crate) fn max_error(t1: i64, t2: i64, t3: i64, t4: i64) -> i64 {
    ((t4 - t1) - (t3 - t2)) / 2
}

/// Clock synchronization quality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncQuality {
    /// Good synchronization (RTT < 50ms)
    Good,
    /// Degraded synchronization (RTT 50-100ms)
    Degraded,
    /// Lost synchronization (RTT > 100ms or no sync)
    Lost,
}

/// Clock synchronization state
pub struct ClockSync {
    /// Last known RTT in microseconds
    rtt_micros: Option<i64>,
    /// When we computed this (for staleness detection)
    last_update: Option<Instant>,
    /// Drift-aware time filter
    filter: TimeFilter,
    /// Raw monotonic clock used for timestamps
    clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for ClockSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClockSync")
            .field("rtt_micros", &self.rtt_micros)
            .field("last_update", &self.last_update)
            .field("filter", &self.filter)
            .finish()
    }
}

impl ClockSync {
    /// Create a new clock synchronization instance with the given clock.
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            rtt_micros: None,
            last_update: None,
            filter: TimeFilter::new(),
            clock,
        }
    }

    /// Get a clone of the underlying clock.
    ///
    /// Useful for generating timestamps (t1, t4) in the same timebase
    /// that the filter operates on, without holding the `ClockSync` lock.
    pub fn clock(&self) -> Arc<dyn Clock> {
        Arc::clone(&self.clock)
    }

    /// Update clock sync with new measurement.
    ///
    /// - `t1` = client_transmitted (raw monotonic µs from [`Clock::now_micros`])
    /// - `t2` = server_received (server loop µs)
    /// - `t3` = server_transmitted (server loop µs)
    /// - `t4` = client_received (raw monotonic µs from [`Clock::now_micros`])
    ///
    /// Returns false when the filter discards the measurement.
    pub fn update(&mut self, t1: i64, t2: i64, t3: i64, t4: i64) -> bool {
        let rtt = (t4 - t1) - (t3 - t2);
        let measurement = ((t2 - t1) + (t3 - t4)) / 2;
        let was_synced = self.filter.is_synchronized();
        if !self
            .filter
            .update(measurement, max_error(t1, t2, t3, t4), t4)
        {
            log::debug!(
                "Clock sync sample discarded: rtt={rtt}µs (t1={t1}, t2={t2}, t3={t3}, t4={t4})"
            );
            return false;
        }
        self.rtt_micros = Some(rtt);
        self.last_update = Some(Instant::now());
        log::trace!(
            "Clock sync sample: offset={:.0}µs, rtt={}µs, drift={:.6}, synchronized={}",
            self.filter.offset,
            rtt,
            self.filter.drift,
            self.filter.is_synchronized(),
        );
        if !was_synced && self.filter.is_synchronized() {
            log::info!(
                "Clock sync achieved: offset={:.0}µs, rtt={}µs",
                self.filter.offset,
                rtt
            );
        }
        true
    }

    /// Get current RTT in microseconds
    pub fn rtt_micros(&self) -> Option<i64> {
        self.rtt_micros
    }

    /// Convert server loop microseconds to client clock microseconds
    pub fn server_to_client_micros(&self, server_micros: i64) -> Option<i64> {
        if !self.filter.is_synchronized() {
            return None;
        }
        Some(self.filter.compute_client_time(server_micros))
    }

    /// Convert client clock microseconds to server loop microseconds
    pub fn client_to_server_micros(&self, client_micros: i64) -> Option<i64> {
        if !self.filter.is_synchronized() {
            return None;
        }
        Some(self.filter.compute_server_time(client_micros))
    }

    /// Convert server loop microseconds to local Instant
    pub fn server_to_local_instant(&self, server_micros: i64) -> Option<Instant> {
        let client_micros = self.server_to_client_micros(server_micros)?;
        self.client_micros_to_instant(client_micros)
    }

    /// Convert server loop microseconds to local Instant, compensating for output latency
    pub fn server_to_local_instant_with_latency(
        &self,
        server_micros: i64,
        output_latency_micros: u64,
    ) -> Option<Instant> {
        let instant = self.server_to_local_instant(server_micros)?;
        instant.checked_sub(Duration::from_micros(output_latency_micros))
    }

    fn client_micros_to_instant(&self, client_micros: i64) -> Option<Instant> {
        self.clock.micros_to_instant(client_micros)
    }

    /// Convert a local Instant to client clock microseconds.
    pub fn instant_to_client_micros(&self, instant: Instant) -> i64 {
        self.clock.instant_to_micros(instant)
    }

    /// Get sync quality based on RTT
    pub fn quality(&self) -> SyncQuality {
        match self.rtt_micros {
            Some(rtt) if rtt < 50_000 => SyncQuality::Good,
            Some(rtt) if rtt < 100_000 => SyncQuality::Degraded,
            _ => SyncQuality::Lost,
        }
    }

    /// Check if sync is stale (>5 seconds since last update).
    ///
    /// Uses `Instant::now()` (not the injected [`Clock`]) because staleness
    /// is a wall-clock concept — we're measuring real elapsed time since the
    /// last successful sync round, regardless of which timebase the filter
    /// operates on. This means `is_stale()` always reflects real time even
    /// when a mock clock is injected for testing.
    pub fn is_stale(&self) -> bool {
        match self.last_update {
            Some(last) => last.elapsed() > Duration::from_secs(5),
            None => true,
        }
    }

    /// Check if clock sync has converged
    pub fn is_synchronized(&self) -> bool {
        self.filter.is_synchronized()
    }

    /// Whether the estimate has settled enough to drive fine-grained
    /// playback corrections. [`is_synchronized`](Self::is_synchronized)
    /// means conversions are available at all: playback starts when
    /// synchronized; the correction planner waits until settled.
    pub fn is_settled(&self) -> bool {
        self.filter.is_settled()
    }
}

impl Default for ClockSync {
    fn default() -> Self {
        Self::new(Arc::new(DefaultClock::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed enough samples to exit the count==0 and count==1 bootstrap paths
    /// and put the filter into the steady-state Kalman branch, where negative
    /// `dt` would actually corrupt the prediction/covariance math.
    fn primed_filter() -> TimeFilter {
        let mut f = TimeFilter::new();
        f.update(100, 10, 1_000);
        f.update(120, 10, 2_000);
        f.update(140, 10, 3_000);
        assert_eq!(f.count, 3);
        assert_eq!(f.last_update, 3_000);
        f
    }

    /// Snapshot every bit of filter state that a discarded sample could
    /// perturb.
    fn snapshot(f: &TimeFilter) -> (i64, u32, f64, f64, f64, f64, f64, bool) {
        (
            f.last_update,
            f.count,
            f.offset,
            f.drift,
            f.p_oo,
            f.p_dd,
            f.p_od,
            f.use_drift,
        )
    }

    #[test]
    fn update_rejects_backwards_time_added() {
        let mut f = primed_filter();
        let before = snapshot(&f);

        // t4 < last_update: pre-fix code would compute negative dt and
        // corrupt the Kalman prediction. Post-fix, the guard rejects it.
        assert!(!f.update(999, 10, 2_500));

        assert_eq!(snapshot(&f), before, "backwards t4 must not mutate state");
    }

    #[test]
    fn update_rejects_duplicate_time_added() {
        let mut f = primed_filter();
        let before = snapshot(&f);

        // Equal t4: original == guard already handled this; make sure the
        // <= relaxation didn't accidentally change behavior for duplicates.
        assert!(!f.update(999, 10, 3_000));

        assert_eq!(snapshot(&f), before, "duplicate t4 must not mutate state");
    }

    #[test]
    fn update_rejects_non_positive_max_error() {
        let mut f = primed_filter();
        let before = snapshot(&f);

        assert!(!f.update(160, 0, 4_000));
        assert!(!f.update(160, -5, 4_000));

        assert_eq!(snapshot(&f), before);
    }

    #[test]
    fn update_accepts_forward_time_added() {
        let mut f = primed_filter();
        let before = snapshot(&f);

        assert!(f.update(160, 10, 4_000));

        assert_ne!(snapshot(&f), before, "forward t4 must advance the filter");
        assert_eq!(f.last_update, 4_000);
    }

    #[test]
    fn backwards_sample_does_not_affect_later_predictions() {
        // End-to-end: a rejected backwards sample must leave future
        // conversions identical to a filter that never saw it at all.
        let mut with_bad = TimeFilter::new();
        let mut without_bad = TimeFilter::new();

        for (m, t) in [(100, 1_000), (120, 2_000), (140, 3_000)] {
            with_bad.update(m, 10, t);
            without_bad.update(m, 10, t);
        }

        // Inject a backwards sample into one filter only.
        with_bad.update(9_999, 10, 1_500);

        // Then drive both forward identically.
        for (m, t) in [(160, 4_000), (180, 5_000)] {
            with_bad.update(m, 10, t);
            without_bad.update(m, 10, t);
        }

        assert_eq!(
            with_bad.compute_server_time(10_000),
            without_bad.compute_server_time(10_000),
        );
        assert_eq!(
            with_bad.compute_client_time(10_000),
            without_bad.compute_client_time(10_000),
        );
    }

    #[test]
    fn measurement_and_max_error_truncate_toward_zero() {
        let mut sync = ClockSync::default();
        // (T2 - T1) + (T3 - T4) = -3 and RTT = 203: rounding would give an
        // offset of -2 and a max_error of 102.
        sync.update(0, 100, 100, 203);

        assert_eq!(sync.filter.offset, -1.0);
        assert_eq!(sync.filter.p_oo, (0.5 * 101.0) * (0.5 * 101.0));
    }

    #[test]
    fn matches_reference_implementation() {
        // Expected values come from sendspin-cpp's SendspinTimeFilter (spec
        // parameters) fed the same sequence: 50 ppm drift, bounded noise, a
        // 5ms second sample spacing and a 50ms outlier after min_samples.
        let expected = [
            (0, 3_000_150, 7_999_850),
            (1, 3_004_868, 8_005_132),
            (2, 4_999_657, 10_000_343),
            (3, 5_999_863, 11_000_137),
            (9, 11_999_425, 17_000_575),
            (50, 52_997_346, 58_002_654),
            (104, 106_994_718, 112_005_282),
            (105, 107_991_313, 113_008_687),
            (106, 108_993_137, 114_006_864),
            (119, 121_993_802, 127_006_198),
        ];
        let mut f = TimeFilter::new();
        let mut checkpoints = expected.iter();
        let mut next = checkpoints.next();
        for i in 0..120_i64 {
            let t = if i == 1 {
                1_005_000
            } else {
                1_000_000 + i * 1_000_000
            };
            let max_error = 200 + (i * 37) % 300;
            let noise = (i * 7_919) % (2 * max_error + 1) - max_error;
            let outlier = if i == 105 { 50_000 } else { 0 };
            let measurement = 5_000_000 + t / 20_000 + noise + outlier;
            assert!(f.update(measurement, max_error, t));
            if let Some(&(at, client, server)) = next.filter(|c| c.0 == i) {
                assert_eq!(f.compute_client_time(t + 7_000_000), client, "sample {at}");
                assert_eq!(f.compute_server_time(t + 2_000_000), server, "sample {at}");
                next = checkpoints.next();
            }
        }
        assert!(next.is_none());
    }

    #[test]
    fn settles_after_minimum_samples() {
        let mut f = TimeFilter::new();
        for i in 0..i64::from(TimeFilter::SETTLE_MIN_SAMPLES) {
            assert!(!f.is_settled(), "must not settle before sample {}", i + 1);
            f.update(1_000 + i, 10, 1_000 + i * 1_000_000);
        }
        assert!(f.is_settled());
    }
}
