// ABOUTME: Sync correction planner for drop/insert cadence, plus the
// ABOUTME: measurement filter and engage gate that keep it off phantom errors

use std::time::{Duration, Instant};

/// Correction schedule for drop/insert cadence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CorrectionSchedule {
    /// Insert one frame every N frames (0 disables).
    pub insert_every_n_frames: u32,
    /// Drop one frame every N frames (0 disables).
    pub drop_every_n_frames: u32,
    /// True when re-anchoring is required.
    pub reanchor: bool,
}

impl CorrectionSchedule {
    /// True when any correction (insert, drop, or reanchor) is active.
    pub fn is_correcting(&self) -> bool {
        self.insert_every_n_frames > 0 || self.drop_every_n_frames > 0 || self.reanchor
    }
}

/// Time after a disturbance by which the spec's steady state requires the
/// sync error back within ±1ms.
const STEADY_STATE_DEADLINE: Duration = Duration::from_secs(10);

/// Part of [`STEADY_STATE_DEADLINE`] left for soft correction once the error
/// window and engage streak have seen an error, keeping half a second spare.
pub(crate) const SOFT_CORRECTION_BUDGET: Duration = STEADY_STATE_DEADLINE
    .saturating_sub(SYNC_ERROR_WINDOW_DURATION)
    .saturating_sub(ENGAGE_STREAK_DURATION)
    .saturating_sub(Duration::from_millis(500));

/// Error a correction racing [`SOFT_CORRECTION_BUDGET`] aims for, half the
/// ±1ms steady-state bound.
pub(crate) const SOFT_CORRECTION_TARGET_US: i64 = 500;

/// Planner that converts sync error into a correction schedule.
///
/// Uses hysteresis to prevent oscillation at the deadband boundary:
/// correction engages at `engage_us` and disengages at `deadband_us`.
#[derive(Debug, Clone, Copy)]
pub struct CorrectionPlanner {
    deadband_us: i64,
    engage_us: i64,
    target_seconds: f64,
    max_speed_correction: f64,
}

impl CorrectionPlanner {
    /// Create a planner with default thresholds.
    pub fn new() -> Self {
        Self {
            deadband_us: 100,
            engage_us: 300,
            target_seconds: 2.0,
            // Under the spec's 0.5% so that any 150ms window, which can hold
            // one correction more than the average rate, stays within it.
            max_speed_correction: 0.0045,
        }
    }

    /// Largest error the speed cap can soft-correct within `time`.
    pub(crate) fn max_soft_correction_us(&self, time: Duration) -> i64 {
        (self.max_speed_correction * time.as_secs_f64() * 1_000_000.0) as i64
    }

    /// Plan a correction schedule from sync error and sample rate.
    ///
    /// `error_us` is the sync error in microseconds (positive = ahead, negative = behind).
    /// `currently_correcting` controls hysteresis: when already correcting, the lower
    /// deadband threshold is used instead of the engage threshold.
    pub fn plan(
        &self,
        error_us: i64,
        sample_rate: u32,
        currently_correcting: bool,
    ) -> CorrectionSchedule {
        self.plan_within(
            error_us,
            sample_rate,
            currently_correcting,
            SOFT_CORRECTION_BUDGET,
        )
    }

    /// Like [`plan`](Self::plan), but fast enough to bring the error within
    /// [`SOFT_CORRECTION_TARGET_US`] by the end of `remaining`.
    pub(crate) fn plan_within(
        &self,
        error_us: i64,
        sample_rate: u32,
        currently_correcting: bool,
        remaining: Duration,
    ) -> CorrectionSchedule {
        let abs_error = error_us.saturating_abs();

        // Hysteresis: use lower threshold to keep correcting, higher to start.
        let threshold = if currently_correcting {
            self.deadband_us
        } else {
            self.engage_us
        };

        if abs_error <= threshold {
            return CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: 0,
                reanchor: false,
            };
        }

        if abs_error >= self.max_soft_correction_us(SOFT_CORRECTION_BUDGET) {
            return CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: 0,
                reanchor: true,
            };
        }

        let sample_rate_f = sample_rate as f64;
        let frames_error = (error_us as f64 * sample_rate_f) / 1_000_000.0;
        let desired_corrections_per_sec = frames_error.abs() / self.target_seconds;
        let excess_us = abs_error - SOFT_CORRECTION_TARGET_US;
        let deadline_corrections_per_sec = if excess_us > 0 {
            excess_us as f64 * sample_rate_f / 1_000_000.0 / remaining.as_secs_f64()
        } else {
            0.0
        };
        let max_corrections_per_sec = sample_rate_f * self.max_speed_correction;
        let corrections_per_sec = desired_corrections_per_sec
            .max(deadline_corrections_per_sec)
            .min(max_corrections_per_sec);

        if corrections_per_sec <= 0.0 {
            return CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: 0,
                reanchor: false,
            };
        }

        let interval_frames = (sample_rate_f / corrections_per_sec).round() as u32;

        if error_us > 0 {
            CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: interval_frames.max(1),
                reanchor: false,
            }
        } else {
            CorrectionSchedule {
                insert_every_n_frames: interval_frames.max(1),
                drop_every_n_frames: 0,
                reanchor: false,
            }
        }
    }
}

impl Default for CorrectionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Measurements kept by [`SyncErrorFilter`]. 101 callbacks is ~1s at the
/// common 10ms WASAPI period (longer periods are cut at
/// [`SYNC_ERROR_WINDOW_DURATION`]): long enough that the
/// floor mode gets sampled even through dense flapping, short enough that a
/// real displacement reaches the output within about a second.
pub(crate) const SYNC_ERROR_WINDOW: usize = 101;

/// Longest span of [`SyncErrorFilter`]'s window, so a long callback period
/// does not delay a real displacement by several seconds.
pub(crate) const SYNC_ERROR_WINDOW_DURATION: Duration = Duration::from_secs(1);

/// Windowed minimum (floor tracker) over recent sync-error measurements.
///
/// Audio presentation timestamps are quantized to the engine period: under
/// scheduler jitter the wake-time padding snapshot jumps *up* by whole
/// periods (observed as a 2ms <-> 12ms alternation on shared-mode WASAPI)
/// while the endpoint FIFO plays gaplessly. The noise is one-sided — extra
/// queued padding can only make a reading later, never earlier — so the
/// window floor is the honest alignment estimate: it holds regardless of
/// which mode has the majority, so long as the floor mode is sampled at
/// least once per window. A real displacement (engine starvation inserting
/// silence, a cursor jump) shifts every reading *including the floor*, so it
/// still reaches the planner within one window.
///
/// The response is deliberately asymmetric: a drop in error propagates
/// immediately (a new low sample becomes the floor at once, so a running
/// correction converges without overshoot), while a rise propagates only
/// once the old floor ages out of the window — exactly the conservatism an
/// engage decision wants. Frames the player inserted or dropped shift the
/// stored readings with them, so its own corrections propagate at once in
/// both directions. A spuriously *low* outlier would bias toward
/// under-correction, the safe direction. The padding term never produces
/// one (padding cannot go below empty); the clock filter normally moves by
/// microseconds once settled, though an outlier time sample can still step
/// the estimate by low single-digit milliseconds in either direction — a
/// negative step costs at most one window of under-correction.
///
/// Allocation-free, O(window) scan per update; sized for the audio callback.
#[derive(Debug, Clone)]
pub(crate) struct SyncErrorFilter {
    samples: [i64; SYNC_ERROR_WINDOW],
    times: [Instant; SYNC_ERROR_WINDOW],
    len: usize,
    next: usize,
    first_at: Option<Instant>,
    warm: bool,
    corrected_us: f64,
}

impl SyncErrorFilter {
    /// Create an empty (cold) filter.
    pub fn new() -> Self {
        Self {
            samples: [0; SYNC_ERROR_WINDOW],
            times: [Instant::now(); SYNC_ERROR_WINDOW],
            len: 0,
            next: 0,
            first_at: None,
            warm: false,
            corrected_us: 0.0,
        }
    }

    /// Discard all samples, e.g. after a reanchor or playback reset made
    /// prior measurements meaningless.
    pub fn reset(&mut self) {
        self.len = 0;
        self.next = 0;
        self.first_at = None;
        self.warm = false;
        self.corrected_us = 0.0;
    }

    /// Record net frames the player inserted, negative for dropped frames.
    pub fn record_corrections(&mut self, frames: i64, sample_rate: u32) {
        self.corrected_us += frames as f64 * 1_000_000.0 / f64::from(sample_rate);
    }

    /// True once the window is fully populated or spans its full duration.
    /// A floor over a partial window is still returned by
    /// [`SyncErrorFilter::update`], but engage decisions should wait for
    /// warmth (see [`EngageGate`]).
    pub fn is_warm(&self) -> bool {
        self.warm
    }

    /// Record a raw error measurement (µs) and return the windowed minimum.
    pub fn update(&mut self, now: Instant, raw_error_us: i64) -> i64 {
        let corrected_us = self.corrected_us.round() as i64;
        self.samples[self.next] = raw_error_us - corrected_us;
        self.times[self.next] = now;
        self.next = (self.next + 1) % SYNC_ERROR_WINDOW;
        if self.len < SYNC_ERROR_WINDOW {
            self.len += 1;
        }
        let first_at = *self.first_at.get_or_insert(now);
        self.warm = self.len == SYNC_ERROR_WINDOW
            || now.saturating_duration_since(first_at) >= SYNC_ERROR_WINDOW_DURATION;
        // While filling, the live samples are the contiguous prefix
        // (next == len until the first wrap); afterwards the whole array.
        // Stale slots past `len` are never read. `len >= 1` here, so the
        // fold always sees at least one real sample.
        self.samples[..self.len]
            .iter()
            .zip(&self.times[..self.len])
            .filter(|(_, at)| now.saturating_duration_since(**at) < SYNC_ERROR_WINDOW_DURATION)
            .map(|(sample, _)| *sample)
            .fold(i64::MAX, i64::min)
            + corrected_us
    }
}

impl Default for SyncErrorFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Consecutive correcting plans required before a correction may engage from
/// idle: ~0.5s at a 10ms callback period. Filtered errors already move at
/// window timescale, so a genuine displacement holds the streak trivially;
/// the gate keeps engagement evidence-backed across filter resets and
/// warm-up, when the estimate is least trustworthy.
pub(crate) const ENGAGE_STREAK_PLANS: u32 = 50;

/// Streak span that also engages, so a long callback period waits no longer
/// than [`ENGAGE_STREAK_PLANS`] take at 10ms.
pub(crate) const ENGAGE_STREAK_DURATION: Duration = Duration::from_millis(500);

/// Gate between the planner and the applied schedule: an idle stream only
/// starts correcting after a sustained run of correcting plans over a warm
/// filter. Every engaged correction mutates real audio (dropped or repeated
/// frames are audible as ticks), so engagement must be evidence-backed;
/// disengagement and replanning of a running correction stay immediate
/// because stopping or retuning is never audible.
#[derive(Debug, Clone)]
pub(crate) struct EngageGate {
    streak: u32,
    streak_since: Option<Instant>,
}

impl EngageGate {
    /// Create a gate with no accumulated streak.
    pub fn new() -> Self {
        Self {
            streak: 0,
            streak_since: None,
        }
    }

    /// Clear the accumulated streak.
    pub fn reset(&mut self) {
        self.streak = 0;
        self.streak_since = None;
    }

    #[cfg(test)]
    pub(crate) fn pending_plans(&self) -> u32 {
        self.streak
    }

    /// Admit or suppress a planned schedule.
    ///
    /// `currently_correcting` is whether a correction episode is already
    /// running; `filter_warm` is whether the error filter window is fully
    /// populated. Reanchor plans skip the streak but wait for a warm filter,
    /// so a single reading carrying period noise cannot jump the cursor.
    pub fn admit(
        &mut self,
        now: Instant,
        planned: CorrectionSchedule,
        currently_correcting: bool,
        filter_warm: bool,
    ) -> CorrectionSchedule {
        if planned.reanchor {
            self.reset();
            return if filter_warm {
                planned
            } else {
                CorrectionSchedule::default()
            };
        }
        if currently_correcting || !planned.is_correcting() {
            self.reset();
            return planned;
        }
        self.streak = self.streak.saturating_add(1);
        let since = *self.streak_since.get_or_insert(now);
        if filter_warm
            && (self.streak >= ENGAGE_STREAK_PLANS
                || now.saturating_duration_since(since) >= ENGAGE_STREAK_DURATION)
        {
            planned
        } else {
            CorrectionSchedule::default()
        }
    }
}

impl Default for EngageGate {
    fn default() -> Self {
        Self::new()
    }
}

/// Observation window for the minimum reported output delay.
pub(crate) const OUTPUT_DELAY_SETTLE_WINDOW: Duration = Duration::from_millis(500);

/// Maximum range of window floors over the entire stability interval.
pub(crate) const OUTPUT_DELAY_SETTLE_TOLERANCE_US: u64 = 10_000;

/// Time required to confirm a baseline floor. Short cold-start plateaus must
/// not enable corrections before the output-delay estimate finishes relaxing.
pub(crate) const OUTPUT_DELAY_SETTLE_STABLE_DURATION: Duration = Duration::from_secs(2);

// One baseline window followed by the stability interval's comparison windows.
const OUTPUT_DELAY_SETTLE_FLOORS: usize = 1
    + (OUTPUT_DELAY_SETTLE_STABLE_DURATION.as_millis() / OUTPUT_DELAY_SETTLE_WINDOW.as_millis())
        as usize;

/// Uses monotonic-time windows so callback cadence cannot shorten settling.
/// Window minima ignore upward period flapping; their full range must fit the
/// tolerance so successive small steps cannot hide a ramp. Allocation-free.
#[derive(Debug, Clone)]
pub(crate) struct OutputDelaySettle {
    window_min_us: u64,
    window_started_at: Option<Instant>,
    last_observation_at: Option<Instant>,
    floors: [u64; OUTPUT_DELAY_SETTLE_FLOORS],
    floors_len: usize,
    next_floor: usize,
}

impl OutputDelaySettle {
    pub fn new() -> Self {
        Self {
            window_min_us: u64::MAX,
            window_started_at: None,
            last_observation_at: None,
            floors: [0; OUTPUT_DELAY_SETTLE_FLOORS],
            floors_len: 0,
            next_floor: 0,
        }
    }

    /// Forget retained delay measurements.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn is_settled(&self) -> bool {
        self.settled_floor_us().is_some()
    }

    /// The observed floor, available only after the stability interval agrees.
    /// Reanchors must use this floor rather than a possibly high callback sample.
    pub fn settled_floor_us(&self) -> Option<u64> {
        if self.floors_len < OUTPUT_DELAY_SETTLE_FLOORS {
            return None;
        }
        let min = *self.floors.iter().min().unwrap();
        let max = *self.floors.iter().max().unwrap();
        (max - min <= OUTPUT_DELAY_SETTLE_TOLERANCE_US).then_some(min)
    }

    /// Record one callback's output delay (µs). A gap of a full observation
    /// window invalidates the evidence: unobserved time is not stable time.
    pub fn update(&mut self, now: Instant, delay_us: u64) {
        if self
            .last_observation_at
            .is_some_and(|last| now.saturating_duration_since(last) >= OUTPUT_DELAY_SETTLE_WINDOW)
        {
            self.reset();
        }
        self.last_observation_at = Some(now);

        if let Some(start) = self.window_started_at {
            if now.saturating_duration_since(start) >= OUTPUT_DELAY_SETTLE_WINDOW {
                self.floors[self.next_floor] = self.window_min_us;
                self.next_floor = (self.next_floor + 1) % OUTPUT_DELAY_SETTLE_FLOORS;
                self.floors_len = (self.floors_len + 1).min(OUTPUT_DELAY_SETTLE_FLOORS);
                // Begin at the actual observation, not an earlier nominal
                // boundary: overshoot must never shorten the next window.
                self.window_started_at = Some(now);
                self.window_min_us = delay_us;
                return;
            }
        } else {
            self.window_started_at = Some(now);
        }
        self.window_min_us = self.window_min_us.min(delay_us);
    }
}

impl Default for OutputDelaySettle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_correction_within_engage_threshold() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(250, 48_000, false);
        assert!(!schedule.is_correcting(), "should not engage below 0.3ms");
    }

    #[test]
    fn test_correction_engages_above_threshold() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(400, 48_000, false);
        assert!(schedule.is_correcting(), "should engage above 0.3ms");
        assert!(schedule.drop_every_n_frames > 0, "positive error = drop");
    }

    #[test]
    fn test_hysteresis_keeps_correcting_above_deadband() {
        let planner = CorrectionPlanner::new();
        // 0.2ms error: below engage (0.3ms) but above deadband (0.1ms).
        // Should keep correcting if already active.
        let schedule = planner.plan(200, 48_000, true);
        assert!(
            schedule.is_correcting(),
            "should keep correcting above 0.1ms deadband"
        );
    }

    #[test]
    fn test_hysteresis_stops_below_deadband() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(50, 48_000, true);
        assert!(
            !schedule.is_correcting(),
            "should stop below 0.1ms deadband"
        );
    }

    #[test]
    fn test_negative_error_inserts() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(-5_000, 48_000, false);
        assert!(
            schedule.insert_every_n_frames > 0,
            "negative error = insert"
        );
        assert_eq!(schedule.drop_every_n_frames, 0);
    }

    #[test]
    fn test_reanchor_only_above_what_the_budget_can_soft_correct() {
        let planner = CorrectionPlanner::new();
        assert!(!planner.plan(35_000, 48_000, false).reanchor);
        assert!(planner.plan(36_000, 48_000, false).reanchor);
    }

    #[test]
    fn test_exact_engage_threshold_does_not_engage() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(300, 48_000, false);
        assert!(
            !schedule.is_correcting(),
            "exactly at engage threshold should not engage (<=)"
        );
    }

    #[test]
    fn test_exact_deadband_threshold_disengages() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(100, 48_000, true);
        assert!(
            !schedule.is_correcting(),
            "exactly at deadband threshold should disengage (<=)"
        );
    }

    #[test]
    fn test_negative_hysteresis_keeps_inserting_above_deadband() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(-200, 48_000, true);
        assert!(
            schedule.is_correcting(),
            "should keep correcting negative error above deadband"
        );
        assert!(
            schedule.insert_every_n_frames > 0,
            "negative error = insert"
        );
        assert_eq!(schedule.drop_every_n_frames, 0);
    }

    // -- Exact interval values --
    //
    // The tests above only check *which* field is set (drop vs insert).
    // These pin down the actual interval arithmetic: `frames_error`, the
    // `desired` / `max` corrections_per_sec cap, and the `interval_frames`
    // rounding. Checking a single "correction engaged" bit lets the
    // arithmetic drift silently; checking the computed cadence doesn't.

    #[test]
    fn test_drop_interval_below_speed_cap_matches_spec() {
        // 5ms error at 48 kHz, already correcting (deadband threshold):
        //   frames_error                = 5000 * 48000 / 1_000_000 = 240 frames
        //   desired_corrections_per_sec = 240 / 2.0                = 120
        //   max_corrections_per_sec     = 48000 * 0.0045           = 216 (not binding)
        //   interval_frames             = round(48000 / 120)       = 400
        let planner = CorrectionPlanner::new();
        let s = planner.plan(5_000, 48_000, true);
        assert_eq!(
            s.drop_every_n_frames, 400,
            "5ms drift should produce drop every 400 frames"
        );
        assert_eq!(s.insert_every_n_frames, 0);
    }

    #[test]
    fn test_insert_interval_below_speed_cap_matches_spec() {
        // Mirror of the drop test with a negative error.
        let planner = CorrectionPlanner::new();
        let s = planner.plan(-5_000, 48_000, true);
        assert_eq!(
            s.insert_every_n_frames, 400,
            "-5ms drift should produce insert every 400 frames"
        );
        assert_eq!(s.drop_every_n_frames, 0);
    }

    #[test]
    fn test_drop_interval_hits_max_speed_cap() {
        // 15ms error, large enough that the desired rate exceeds the cap:
        //   frames_error                = 15000 * 48000 / 1_000_000 = 720
        //   desired_corrections_per_sec = 720 / 2.0                 = 360
        //   max_corrections_per_sec     = 48000 * 0.0045            = 216 (binding)
        //   interval_frames             = round(48000 / 216)        = 222
        let planner = CorrectionPlanner::new();
        let s = planner.plan(15_000, 48_000, true);
        assert_eq!(
            s.drop_every_n_frames, 222,
            "large drift should be capped at max speed correction (interval = 222 frames)"
        );
    }

    #[test]
    fn test_speed_cap_holds_over_any_150ms_window() {
        let planner = CorrectionPlanner::new();
        for sample_rate in [
            8_000, 16_000, 22_050, 32_000, 44_100, 48_000, 96_000, 192_000,
        ] {
            let interval = planner.plan(19_999, sample_rate, true).drop_every_n_frames;
            let window_frames = sample_rate * 150 / 1_000;
            let max_corrections = (window_frames - 1) / interval + 1;
            assert!(
                f64::from(max_corrections) <= 0.005 * f64::from(window_frames),
                "{sample_rate} Hz: {max_corrections} corrections in {window_frames} frames"
            );
        }
    }

    #[test]
    fn test_filter_constant_input_passes_through() {
        let now = Instant::now();
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            assert_eq!(filter.update(now, 4_000), 4_000);
        }
        assert!(filter.is_warm());
    }

    #[test]
    fn test_filter_warms_only_after_full_window() {
        let now = Instant::now();
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW - 1 {
            filter.update(now, 0);
            assert!(!filter.is_warm());
        }
        filter.update(now, 0);
        assert!(filter.is_warm());
    }

    #[test]
    fn test_filter_floor_ignores_flap_at_any_duty() {
        // Period-quantized flapping is one-sided (+one period). Even a 90%
        // majority in the high mode — enough to drag any central-tendency
        // estimate — must not lift the floor while the quiet mode is sampled
        // at least once per window.
        let now = Instant::now();
        let mut filter = SyncErrorFilter::new();
        for i in 0..SYNC_ERROR_WINDOW * 3 {
            let raw = if i % 10 == 0 { 0 } else { 10_000 };
            let floor = filter.update(now, raw);
            if i >= 1 {
                assert_eq!(floor, 0, "flap majority must not lift the floor");
            }
        }
    }

    #[test]
    fn test_filter_floor_follows_persistent_rise_after_full_window() {
        // A real displacement shifts every reading including the floor; the
        // output must follow once the old floor ages out of the window.
        let now = Instant::now();
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            filter.update(now, 0);
        }
        let mut floor = 0;
        for i in 0..SYNC_ERROR_WINDOW {
            floor = filter.update(now, 10_000);
            if i < SYNC_ERROR_WINDOW - 1 {
                assert_eq!(floor, 0, "rise must wait for the window to age out");
            }
        }
        assert_eq!(floor, 10_000, "persistent rise must reach the output");
    }

    #[test]
    fn test_filter_floor_follows_drop_immediately() {
        // A running correction shrinks the error; the floor must track the
        // new low on the very next sample so the correction converges
        // without overshoot.
        let now = Instant::now();
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            filter.update(now, 10_000);
        }
        assert_eq!(
            filter.update(now, 2_000),
            2_000,
            "new low becomes the floor"
        );
    }

    #[test]
    fn test_filter_handles_negative_errors() {
        // Negative true error (audio early, insert side) with one-sided
        // positive excursions on top.
        let now = Instant::now();
        let mut filter = SyncErrorFilter::new();
        for i in 0..SYNC_ERROR_WINDOW {
            let raw = if i % 3 == 0 { 10_000 } else { -5_000 };
            filter.update(now, raw);
        }
        assert_eq!(filter.update(now, -5_000), -5_000);
    }

    #[test]
    fn test_filter_single_low_evicted_exactly_after_window() {
        // One low among highs must stop being the floor exactly WINDOW
        // updates after it was recorded.
        let now = Instant::now();
        let mut filter = SyncErrorFilter::new();
        filter.update(now, 1_000);
        for i in 0..SYNC_ERROR_WINDOW - 1 {
            assert_eq!(
                filter.update(now, 10_000),
                1_000,
                "low still in window at {i}"
            );
        }
        assert_eq!(
            filter.update(now, 10_000),
            10_000,
            "low must age out exactly one window after it was recorded"
        );
    }

    #[test]
    fn test_filter_tracks_min_across_multiple_wraps() {
        // Distinct values across several ring cycles: the floor must always
        // equal the true minimum of the last WINDOW samples.
        let now = Instant::now();
        let mut filter = SyncErrorFilter::new();
        let mut recent: Vec<i64> = Vec::new();
        for i in 0..(SYNC_ERROR_WINDOW as i64 * 5) {
            let value = (i * 37) % 1_000 + if i % 13 == 0 { -500 } else { 0 };
            recent.push(value);
            if recent.len() > SYNC_ERROR_WINDOW {
                recent.remove(0);
            }
            let expected = *recent.iter().min().unwrap();
            assert_eq!(
                filter.update(now, value),
                expected,
                "mismatch at update {i}"
            );
        }
    }

    #[test]
    fn test_filter_reset_clears_window() {
        // Fill with a LOW value so stale slots would drag the floor down if
        // reset failed to fence them off.
        let now = Instant::now();
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            filter.update(now, -10_000);
        }
        filter.reset();
        assert!(!filter.is_warm());
        assert_eq!(
            filter.update(now, 5_000),
            5_000,
            "floor after reset must reflect only new samples"
        );
    }

    #[test]
    fn test_filter_floor_follows_inserted_frames_at_once() {
        let start = Instant::now();
        let mut filter = SyncErrorFilter::new();
        filter.update(start, -5_000);
        filter.record_corrections(48, 48_000);
        assert_eq!(
            filter.update(start + Duration::from_millis(10), -4_000),
            -4_000
        );
    }

    #[test]
    fn test_filter_window_is_bounded_in_time_on_long_periods() {
        // 40ms callbacks: a full count window would span 4s.
        let start = Instant::now();
        let period = Duration::from_millis(40);
        let mut filter = SyncErrorFilter::new();
        filter.update(start, 0);
        let mut i = 1;
        while period * i < SYNC_ERROR_WINDOW_DURATION {
            assert_eq!(filter.update(start + period * i, 10_000), 0);
            assert!(!filter.is_warm(), "warm after {i} readings");
            i += 1;
        }
        assert_eq!(filter.update(start + period * i, 10_000), 10_000);
        assert!(filter.is_warm());
    }

    fn correcting_plan() -> CorrectionSchedule {
        CorrectionSchedule {
            insert_every_n_frames: 0,
            drop_every_n_frames: 200,
            reanchor: false,
        }
    }

    #[test]
    fn test_gate_requires_sustained_streak() {
        let now = Instant::now();
        let mut gate = EngageGate::new();
        for _ in 0..ENGAGE_STREAK_PLANS - 1 {
            let admitted = gate.admit(now, correcting_plan(), false, true);
            assert!(!admitted.is_correcting(), "streak not yet sustained");
        }
        let admitted = gate.admit(now, correcting_plan(), false, true);
        assert!(admitted.is_correcting(), "sustained streak must engage");
    }

    #[test]
    fn test_gate_streak_engages_after_its_duration_on_long_periods() {
        let start = Instant::now();
        let period = Duration::from_millis(40);
        let mut gate = EngageGate::new();
        let mut i = 0;
        while period * i < ENGAGE_STREAK_DURATION {
            let admitted = gate.admit(start + period * i, correcting_plan(), false, true);
            assert!(!admitted.is_correcting(), "engaged after {i} plans");
            i += 1;
        }
        let admitted = gate.admit(start + period * i, correcting_plan(), false, true);
        assert!(admitted.is_correcting());
    }

    #[test]
    fn test_gate_streak_resets_on_idle_plan() {
        let now = Instant::now();
        let mut gate = EngageGate::new();
        for _ in 0..ENGAGE_STREAK_PLANS - 1 {
            gate.admit(now, correcting_plan(), false, true);
        }
        gate.admit(now, CorrectionSchedule::default(), false, true);
        let admitted = gate.admit(now, correcting_plan(), false, true);
        assert!(
            !admitted.is_correcting(),
            "an idle plan must reset the streak"
        );
    }

    #[test]
    fn test_gate_cold_filter_never_engages() {
        let now = Instant::now();
        let mut gate = EngageGate::new();
        for _ in 0..ENGAGE_STREAK_PLANS * 2 {
            let admitted = gate.admit(now, correcting_plan(), false, false);
            assert!(!admitted.is_correcting(), "cold filter must not engage");
        }
    }

    #[test]
    fn test_gate_reanchor_skips_streak_but_waits_for_warm_filter() {
        let now = Instant::now();
        let mut gate = EngageGate::new();
        let reanchor = CorrectionSchedule {
            insert_every_n_frames: 0,
            drop_every_n_frames: 0,
            reanchor: true,
        };
        let cold = gate.admit(now, reanchor, false, false);
        assert!(!cold.is_correcting(), "cold filter must not reanchor");
        let warm = gate.admit(now, reanchor, false, true);
        assert!(warm.reanchor, "reanchor must not wait for a streak");
    }

    #[test]
    fn test_gate_running_correction_replans_freely() {
        let now = Instant::now();
        let mut gate = EngageGate::new();
        let admitted = gate.admit(now, correcting_plan(), true, true);
        assert!(
            admitted.is_correcting(),
            "a running correction must replan without gating"
        );
        let disengage = gate.admit(now, CorrectionSchedule::default(), true, true);
        assert!(
            !disengage.is_correcting(),
            "disengage must pass immediately"
        );
    }

    const SETTLE_TIME: Duration =
        OUTPUT_DELAY_SETTLE_WINDOW.saturating_add(OUTPUT_DELAY_SETTLE_STABLE_DURATION);

    fn callback_periods() -> [Duration; 4] {
        [
            Duration::from_secs_f64(64.0 / 48_000.0),
            Duration::from_secs_f64(128.0 / 48_000.0),
            Duration::from_millis(10),
            Duration::from_millis(40),
        ]
    }

    fn replay(
        period: Duration,
        duration: Duration,
        delay: impl Fn(Duration, usize) -> u64,
        mut observe: impl FnMut(Duration, &OutputDelaySettle),
    ) {
        let start = Instant::now();
        let mut tracker = OutputDelaySettle::new();
        let mut elapsed = Duration::ZERO;
        let mut callback = 0;
        while elapsed <= duration {
            tracker.update(start + elapsed, delay(elapsed, callback));
            observe(elapsed, &tracker);
            elapsed += period;
            callback += 1;
        }
    }

    #[test]
    fn test_delay_settle_uses_elapsed_time_at_all_cadences() {
        for period in callback_periods() {
            let mut first_settled = None;
            replay(
                period,
                SETTLE_TIME + OUTPUT_DELAY_SETTLE_WINDOW,
                |_, _| 250_000,
                |elapsed, tracker| {
                    if tracker.is_settled() && first_settled.is_none() {
                        first_settled = Some(elapsed);
                    }
                },
            );
            let elapsed = first_settled.expect("constant delay must settle");
            assert!(elapsed >= SETTLE_TIME, "must observe the whole interval");
            assert!(elapsed <= SETTLE_TIME + period * OUTPUT_DELAY_SETTLE_FLOORS as u32);
        }
    }

    #[test]
    fn test_delay_settle_ignores_period_flapping_and_returns_floor() {
        for period in callback_periods() {
            let mut final_floor = None;
            replay(
                period,
                SETTLE_TIME + OUTPUT_DELAY_SETTLE_WINDOW,
                |_, i| if i % 2 == 0 { 280_000 } else { 290_600 },
                |_, tracker| {
                    final_floor = tracker.settled_floor_us();
                },
            );
            assert_eq!(final_floor, Some(280_000));
        }
    }

    #[test]
    fn test_delay_settle_rejects_ramps_at_all_cadences() {
        for period in callback_periods() {
            // 20ms/s moves only 10ms per nominal window, but 40ms over
            // the comparison interval. Adjacent comparisons are insufficient.
            for rate in [20_000, 50_000] {
                for rising in [false, true] {
                    replay(
                        period,
                        Duration::from_secs(5),
                        |elapsed, _| {
                            let change = (elapsed.as_secs_f64() * rate as f64) as u64;
                            if rising {
                                550_000 + change
                            } else {
                                550_000 - change
                            }
                        },
                        |_, tracker| assert!(!tracker.is_settled(), "a ramp must not settle"),
                    );
                }
            }
        }
    }

    #[test]
    fn test_delay_settle_rejects_plateau_then_recovers_after_relaxation() {
        for period in callback_periods() {
            let mut settled = false;
            replay(
                period,
                Duration::from_secs(7),
                |elapsed, _| {
                    let seconds = elapsed.as_secs_f64();
                    if seconds < 1.0 {
                        475_000
                    } else if seconds < 4.0 {
                        475_000 - ((seconds - 1.0) * 50_000.0) as u64
                    } else {
                        325_000
                    }
                },
                |elapsed, tracker| {
                    if elapsed < Duration::from_secs(4) {
                        assert!(!tracker.is_settled(), "plateau and ramp are transient");
                    }
                    settled = tracker.is_settled();
                },
            );
            assert!(settled, "must recover once the floor stops moving");
        }
    }

    #[test]
    fn test_delay_settle_unsettles_and_settles_again() {
        let period = Duration::from_millis(10);
        let step_at = SETTLE_TIME + OUTPUT_DELAY_SETTLE_WINDOW;
        let mut transitions = 0;
        let mut was_settled = false;
        let mut unsettled_after_step = false;
        replay(
            period,
            step_at + SETTLE_TIME + OUTPUT_DELAY_SETTLE_WINDOW,
            |elapsed, _| if elapsed < step_at { 300_000 } else { 400_000 },
            |elapsed, tracker| {
                let settled = tracker.is_settled();
                if settled && !was_settled {
                    transitions += 1;
                }
                if elapsed > step_at && !settled {
                    unsettled_after_step = true;
                }
                if settled && elapsed > step_at + SETTLE_TIME {
                    assert_eq!(tracker.settled_floor_us(), Some(400_000));
                }
                was_settled = settled;
            },
        );
        assert!(unsettled_after_step);
        assert_eq!(transitions, 2);
        assert!(was_settled);
    }

    #[test]
    fn test_delay_settle_tolerance_boundary() {
        for range in [
            OUTPUT_DELAY_SETTLE_TOLERANCE_US,
            OUTPUT_DELAY_SETTLE_TOLERANCE_US + 1,
        ] {
            let start = Instant::now();
            let period = Duration::from_millis(10);
            let callbacks = (SETTLE_TIME.as_nanos() / period.as_nanos()) as u32;
            let mut tracker = OutputDelaySettle::new();
            for i in 0..=callbacks {
                let elapsed = period * i;
                let delay = if elapsed < OUTPUT_DELAY_SETTLE_WINDOW {
                    250_000
                } else {
                    250_000 + range
                };
                tracker.update(start + elapsed, delay);
            }
            assert_eq!(
                tracker.is_settled(),
                range == OUTPUT_DELAY_SETTLE_TOLERANCE_US
            );
        }
    }

    #[test]
    fn test_delay_settle_jitter_does_not_shorten_windows() {
        let start = Instant::now();
        let mut tracker = OutputDelaySettle::new();
        let mut elapsed = Duration::ZERO;
        let mut first_settled = None;
        let mut i = 0;
        while elapsed <= SETTLE_TIME + OUTPUT_DELAY_SETTLE_WINDOW {
            tracker.update(start + elapsed, 250_000);
            if tracker.is_settled() && first_settled.is_none() {
                first_settled = Some(elapsed);
            }
            elapsed += if i % 2 == 0 {
                Duration::from_millis(7)
            } else {
                Duration::from_millis(19)
            };
            i += 1;
        }
        assert!(first_settled.expect("jittered observations must settle") >= SETTLE_TIME);
    }

    #[test]
    fn test_delay_settle_gap_and_reset_require_fresh_evidence() {
        for gap in [OUTPUT_DELAY_SETTLE_WINDOW, Duration::from_secs(10)] {
            let start = Instant::now();
            let period = Duration::from_millis(10);
            let mut tracker = OutputDelaySettle::new();
            let callbacks = (SETTLE_TIME.as_nanos() / period.as_nanos()) as u32;
            for i in 0..=callbacks {
                tracker.update(start + period * i, 250_000);
            }
            assert!(tracker.is_settled());
            let resumed_at = start + SETTLE_TIME + gap;
            tracker.update(resumed_at, 250_000);
            assert!(!tracker.is_settled(), "gap must discard stale evidence");
            for i in 1..callbacks {
                tracker.update(resumed_at + period * i, 250_000);
                assert!(!tracker.is_settled());
            }
            tracker.update(resumed_at + SETTLE_TIME, 250_000);
            assert!(tracker.is_settled());
            tracker.reset();
            assert_eq!(tracker.settled_floor_us(), None);
            tracker.update(resumed_at + SETTLE_TIME + period, 250_000);
            assert!(!tracker.is_settled());
        }
    }
}
