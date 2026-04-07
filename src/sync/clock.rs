// ABOUTME: Clock synchronization implementation
// ABOUTME: Drift-aware time sync with RTT estimation and server/client time conversion

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const ADAPTIVE_FORGETTING_CUTOFF: f64 = 0.75;

#[derive(Debug, Clone, Copy)]
struct TimeElement {
    last_update: i64,
    offset: f64,
    drift: f64,
}

#[derive(Debug)]
struct TimeFilter {
    last_update: i64,
    count: u32,
    offset: f64,
    drift: f64,
    offset_covariance: f64,
    offset_drift_covariance: f64,
    drift_covariance: f64,
    process_variance: f64,
    forget_variance_factor: f64,
    current: TimeElement,
}

impl TimeFilter {
    fn new(process_std_dev: f64, forget_factor: f64) -> Self {
        let process_variance = process_std_dev * process_std_dev;
        let forget_variance_factor = forget_factor * forget_factor;
        Self {
            last_update: 0,
            count: 0,
            offset: 0.0,
            drift: 0.0,
            offset_covariance: f64::INFINITY,
            offset_drift_covariance: 0.0,
            drift_covariance: 0.0,
            process_variance,
            forget_variance_factor,
            current: TimeElement {
                last_update: 0,
                offset: 0.0,
                drift: 0.0,
            },
        }
    }

    fn update(&mut self, measurement: i64, max_error: i64, time_added: i64) {
        if time_added <= self.last_update {
            return;
        }

        let dt = (time_added - self.last_update) as f64;
        self.last_update = time_added;

        let update_std_dev = max_error as f64;
        let measurement_variance = update_std_dev * update_std_dev;

        if self.count == 0 {
            self.count = 1;
            self.offset = measurement as f64;
            self.offset_covariance = measurement_variance;
            self.drift = 0.0;
            self.current = TimeElement {
                last_update: self.last_update,
                offset: self.offset,
                drift: self.drift,
            };
            return;
        }

        if self.count == 1 {
            self.count = 2;
            self.drift = (measurement as f64 - self.offset) / dt;
            self.offset = measurement as f64;
            // Divide by dt^2, not dt: variance of (x/c) is var(x)/c^2.
            self.drift_covariance = (self.offset_covariance + measurement_variance) / (dt * dt);
            self.offset_covariance = measurement_variance;
            self.current = TimeElement {
                last_update: self.last_update,
                offset: self.offset,
                drift: self.drift,
            };
            return;
        }

        let predicted_offset = self.offset + self.drift * dt;
        let dt_squared = dt * dt;

        let drift_process_variance = 0.0;
        let mut new_drift_covariance = self.drift_covariance + drift_process_variance;

        let offset_drift_process_variance = 0.0;
        let mut new_offset_drift_covariance = self.offset_drift_covariance
            + self.drift_covariance * dt
            + offset_drift_process_variance;

        let offset_process_variance = dt * self.process_variance;
        let mut new_offset_covariance = self.offset_covariance
            + 2.0 * self.offset_drift_covariance * dt
            + self.drift_covariance * dt_squared
            + offset_process_variance;

        let residual = measurement as f64 - predicted_offset;
        let max_residual_cutoff = max_error as f64 * ADAPTIVE_FORGETTING_CUTOFF;

        if self.count < 100 {
            self.count += 1;
        } else if residual.abs() > max_residual_cutoff {
            new_drift_covariance *= self.forget_variance_factor;
            new_offset_drift_covariance *= self.forget_variance_factor;
            new_offset_covariance *= self.forget_variance_factor;
        }

        let uncertainty = 1.0 / (new_offset_covariance + measurement_variance);
        let offset_gain = new_offset_covariance * uncertainty;
        let drift_gain = new_offset_drift_covariance * uncertainty;

        self.offset = predicted_offset + offset_gain * residual;
        self.drift += drift_gain * residual;

        self.drift_covariance = new_drift_covariance - drift_gain * new_offset_drift_covariance;
        self.offset_drift_covariance =
            new_offset_drift_covariance - drift_gain * new_offset_covariance;
        self.offset_covariance = new_offset_covariance - offset_gain * new_offset_covariance;

        self.current = TimeElement {
            last_update: self.last_update,
            offset: self.offset,
            drift: self.drift,
        };
    }

    /// Maximum plausible drift magnitude. Real clock drift between
    /// hardware oscillators is measured in ppm; 20% is far beyond
    /// physical reality. This is a last-resort safety net for a
    /// diverged filter, not a quality gate.
    const MAX_DRIFT: f64 = 0.2;

    fn compute_server_time(&self, client_time: i64) -> Option<i64> {
        // Accept only drift <= threshold; reject NaN/incomparable values.
        if !matches!(
            self.current.drift.abs().partial_cmp(&Self::MAX_DRIFT),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ) {
            return None;
        }
        let dt = (client_time - self.current.last_update) as f64;
        let offset = self.current.offset + self.current.drift * dt;
        Some(client_time + offset.round() as i64)
    }

    fn compute_client_time(&self, server_time: i64) -> Option<i64> {
        // Accept only drift <= threshold; reject NaN/incomparable values.
        if !matches!(
            self.current.drift.abs().partial_cmp(&Self::MAX_DRIFT),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ) {
            return None;
        }
        let numerator = server_time as f64 - self.current.offset
            + self.current.drift * self.current.last_update as f64;
        Some((numerator / (1.0 + self.current.drift)).round() as i64)
    }

    fn is_synchronized(&self) -> bool {
        self.count >= 2 && self.offset_covariance.is_finite()
    }
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
#[derive(Debug)]
pub struct ClockSync {
    /// Last known RTT in microseconds
    rtt_micros: Option<i64>,
    /// When we computed this (for staleness detection)
    last_update: Option<Instant>,
    /// Drift-aware time filter
    filter: TimeFilter,
}

impl ClockSync {
    /// Create a new clock synchronization instance
    pub fn new() -> Self {
        Self {
            rtt_micros: None,
            last_update: None,
            filter: TimeFilter::new(0.01, 1.001),
        }
    }

    /// Update clock sync with new measurement
    /// t1 = client_transmitted (Unix µs)
    /// t2 = server_received (server loop µs)
    /// t3 = server_transmitted (server loop µs)
    /// t4 = client_received (Unix µs)
    pub fn update(&mut self, t1: i64, t2: i64, t3: i64, t4: i64) {
        // RTT = (t4 - t1) - (t3 - t2)
        let rtt = (t4 - t1) - (t3 - t2);

        // Discard negative RTT (misordered timestamps) and high RTT
        // (network congestion). Store only valid RTT so that quality()
        // doesn't report Good on a negative value.
        if !(0..=100_000).contains(&rtt) {
            return;
        }
        self.rtt_micros = Some(rtt);

        // NTP offset = ((t2 - t1) + (t3 - t4)) / 2
        // Use f64 division to avoid systematic ±0.5µs bias from integer truncation.
        let measurement = (((t2 - t1) as f64 + (t3 - t4) as f64) / 2.0).round() as i64;
        // Floor max_error at 1µs so the Kalman filter always sees
        // nonzero measurement variance. RTT = 0 is legitimate on
        // localhost; without this floor, repeated zero-variance
        // samples would drive covariance to zero and eventually NaN.
        let max_error = (rtt / 2).max(1);

        self.filter.update(measurement, max_error, t4);
        self.last_update = Some(Instant::now());
    }

    /// Get current RTT in microseconds
    pub fn rtt_micros(&self) -> Option<i64> {
        self.rtt_micros
    }

    /// Convert server loop microseconds to client Unix microseconds
    pub fn server_to_client_micros(&self, server_micros: i64) -> Option<i64> {
        if !self.filter.is_synchronized() {
            return None;
        }
        self.filter.compute_client_time(server_micros)
    }

    /// Convert client Unix microseconds to server loop microseconds
    pub fn client_to_server_micros(&self, client_micros: i64) -> Option<i64> {
        if !self.filter.is_synchronized() {
            return None;
        }
        self.filter.compute_server_time(client_micros)
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
        let now_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_micros() as i64;
        let now_instant = Instant::now();
        let delta_micros = client_micros - now_unix;

        if delta_micros >= 0 {
            Some(now_instant + Duration::from_micros(delta_micros as u64))
        } else {
            now_instant.checked_sub(Duration::from_micros((-delta_micros) as u64))
        }
    }

    /// Convert a local Instant to client Unix microseconds.
    pub fn instant_to_client_micros(&self, instant: Instant) -> Option<i64> {
        let now_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_micros() as i64;
        let now_instant = Instant::now();
        let delta_micros = if instant >= now_instant {
            instant.duration_since(now_instant).as_micros() as i64
        } else {
            -(now_instant.duration_since(instant).as_micros() as i64)
        };
        Some(now_unix + delta_micros)
    }

    /// Get sync quality based on RTT
    pub fn quality(&self) -> SyncQuality {
        match self.rtt_micros {
            Some(rtt) if rtt < 50_000 => SyncQuality::Good,
            Some(rtt) if rtt < 100_000 => SyncQuality::Degraded,
            _ => SyncQuality::Lost,
        }
    }

    /// Check if sync is stale (>5 seconds old)
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
}

impl Default for ClockSync {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed enough samples to exit the count==0 and count==1 bootstrap paths
    /// and put the filter into the steady-state Kalman branch, where negative
    /// `dt` would actually corrupt the prediction/covariance math.
    fn primed_filter() -> TimeFilter {
        let mut f = TimeFilter::new(0.01, 1.001);
        f.update(100, 10, 1_000);
        f.update(120, 10, 2_000);
        f.update(140, 10, 3_000);
        assert_eq!(f.count, 3);
        assert_eq!(f.last_update, 3_000);
        f
    }

    /// Snapshot every bit of filter state that a monotonicity-guard bypass
    /// could perturb. If the guard works, a rejected sample leaves all of
    /// these untouched.
    fn snapshot(f: &TimeFilter) -> (i64, u32, f64, f64, f64, f64, f64) {
        (
            f.last_update,
            f.count,
            f.offset,
            f.drift,
            f.offset_covariance,
            f.drift_covariance,
            f.offset_drift_covariance,
        )
    }

    #[test]
    fn update_rejects_backwards_time_added() {
        let mut f = primed_filter();
        let before = snapshot(&f);

        // t4 < last_update: pre-fix code would compute negative dt and
        // corrupt the Kalman prediction. Post-fix, the guard rejects it.
        f.update(999, 10, 2_500);

        assert_eq!(snapshot(&f), before, "backwards t4 must not mutate state");
    }

    #[test]
    fn update_rejects_duplicate_time_added() {
        let mut f = primed_filter();
        let before = snapshot(&f);

        // Equal t4: original == guard already handled this; make sure the
        // <= relaxation didn't accidentally change behavior for duplicates.
        f.update(999, 10, 3_000);

        assert_eq!(snapshot(&f), before, "duplicate t4 must not mutate state");
    }

    #[test]
    fn update_accepts_forward_time_added() {
        let mut f = primed_filter();
        let before = snapshot(&f);

        f.update(160, 10, 4_000);

        assert_ne!(snapshot(&f), before, "forward t4 must advance the filter");
        assert_eq!(f.last_update, 4_000);
    }

    #[test]
    fn backwards_sample_does_not_affect_later_predictions() {
        // End-to-end: a rejected backwards sample must leave future
        // conversions identical to a filter that never saw it at all.
        let mut with_bad = TimeFilter::new(0.01, 1.001);
        let mut without_bad = TimeFilter::new(0.01, 1.001);

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
}
