// ABOUTME: Sync correction planner for drop/insert cadence, plus the
// ABOUTME: measurement filter and engage gate that keep it off phantom errors

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

/// Planner that converts sync error into a correction schedule.
///
/// Uses hysteresis to prevent oscillation at the deadband boundary:
/// correction engages at `engage_us` and disengages at `deadband_us`.
#[derive(Debug, Clone, Copy)]
pub struct CorrectionPlanner {
    deadband_us: i64,
    engage_us: i64,
    reanchor_threshold_us: i64,
    target_seconds: f64,
    max_speed_correction: f64,
}

impl CorrectionPlanner {
    /// Create a planner with default thresholds.
    pub fn new() -> Self {
        Self {
            deadband_us: 1_500,
            engage_us: 3_000,
            reanchor_threshold_us: 500_000,
            target_seconds: 2.0,
            max_speed_correction: 0.04,
        }
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

        if abs_error >= self.reanchor_threshold_us {
            return CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: 0,
                reanchor: true,
            };
        }

        let sample_rate_f = sample_rate as f64;
        let frames_error = (error_us as f64 * sample_rate_f) / 1_000_000.0;
        let desired_corrections_per_sec = frames_error.abs() / self.target_seconds;
        let max_corrections_per_sec = sample_rate_f * self.max_speed_correction;
        let corrections_per_sec = desired_corrections_per_sec.min(max_corrections_per_sec);

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
/// common 10ms WASAPI period (2-4s on 20-40ms periods): long enough to
/// out-vote period-quantized flapping, short enough that a real displacement
/// shifts the median within half a window.
pub const SYNC_ERROR_WINDOW: usize = 101;

/// Windowed median over recent sync-error measurements.
///
/// Audio presentation timestamps are quantized to the engine period: under
/// scheduler jitter the wake-time padding snapshot flaps by a whole period
/// (observed as a 2ms <-> 12ms alternation on shared-mode WASAPI) while the
/// endpoint FIFO plays gaplessly. The instantaneous error is therefore
/// bimodal measurement noise around the true alignment. The median tracks
/// the majority mode and only moves when a shift *persists* — real
/// displacement or clock drift — never on single-callback excursions, which
/// makes it safe to feed to [`CorrectionPlanner`].
///
/// Allocation-free and O(window) per update; sized for the audio callback.
#[derive(Debug, Clone)]
pub struct SyncErrorFilter {
    samples: [i64; SYNC_ERROR_WINDOW],
    len: usize,
    next: usize,
}

impl SyncErrorFilter {
    /// Create an empty (cold) filter.
    pub fn new() -> Self {
        Self {
            samples: [0; SYNC_ERROR_WINDOW],
            len: 0,
            next: 0,
        }
    }

    /// Discard all samples, e.g. after a reanchor or generation change made
    /// prior measurements meaningless.
    pub fn reset(&mut self) {
        self.len = 0;
        self.next = 0;
    }

    /// True once the window is fully populated. Medians over a partial
    /// window are still returned by [`SyncErrorFilter::update`], but engage
    /// decisions should wait for warmth (see [`EngageGate`]).
    pub fn is_warm(&self) -> bool {
        self.len == SYNC_ERROR_WINDOW
    }

    /// Record a raw error measurement (µs) and return the windowed median.
    pub fn update(&mut self, raw_error_us: i64) -> i64 {
        self.samples[self.next] = raw_error_us;
        self.next = (self.next + 1) % SYNC_ERROR_WINDOW;
        if self.len < SYNC_ERROR_WINDOW {
            self.len += 1;
        }
        // While filling, the live samples are the contiguous prefix
        // (next == len until the first wrap); afterwards the whole array.
        let mut scratch = [0i64; SYNC_ERROR_WINDOW];
        let filled = &mut scratch[..self.len];
        filled.copy_from_slice(&self.samples[..self.len]);
        filled.sort_unstable();
        filled[self.len / 2]
    }
}

impl Default for SyncErrorFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Consecutive correcting plans required before a correction may engage from
/// idle: ~0.5s at a 10ms callback period. A phantom error (measurement flap)
/// reverses within a callback or two and never builds a streak; drift and
/// real displacement hold the streak trivially.
pub const ENGAGE_STREAK_PLANS: u32 = 50;

/// Gate between the planner and the applied schedule: an idle stream only
/// starts correcting after a sustained run of correcting plans over a warm
/// filter. Every engaged correction mutates real audio (dropped or repeated
/// frames are audible as ticks), so engagement must be evidence-backed;
/// disengagement and replanning of a running correction stay immediate
/// because stopping or retuning is never audible.
#[derive(Debug, Clone, Default)]
pub struct EngageGate {
    streak: u32,
}

impl EngageGate {
    /// Create a gate with no accumulated streak.
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the accumulated streak.
    pub fn reset(&mut self) {
        self.streak = 0;
    }

    /// Admit or suppress a planned schedule.
    ///
    /// `currently_correcting` is whether a correction episode is already
    /// running; `filter_warm` is whether the error filter window is fully
    /// populated. Reanchor plans always pass: they indicate gross real
    /// displacement where a delayed reaction is worse than a rare false
    /// positive.
    pub fn admit(
        &mut self,
        planned: CorrectionSchedule,
        currently_correcting: bool,
        filter_warm: bool,
    ) -> CorrectionSchedule {
        if planned.reanchor || currently_correcting || !planned.is_correcting() {
            self.streak = 0;
            return planned;
        }
        self.streak = self.streak.saturating_add(1);
        if filter_warm && self.streak >= ENGAGE_STREAK_PLANS {
            planned
        } else {
            CorrectionSchedule::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_correction_within_engage_threshold() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(2_500, 48_000, false);
        assert!(!schedule.is_correcting(), "should not engage below 3ms");
    }

    #[test]
    fn test_correction_engages_above_threshold() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(3_500, 48_000, false);
        assert!(schedule.is_correcting(), "should engage above 3ms");
        assert!(schedule.drop_every_n_frames > 0, "positive error = drop");
    }

    #[test]
    fn test_hysteresis_keeps_correcting_above_deadband() {
        let planner = CorrectionPlanner::new();
        // 2ms error: below engage (3ms) but above deadband (1.5ms).
        // Should keep correcting if already active.
        let schedule = planner.plan(2_000, 48_000, true);
        assert!(
            schedule.is_correcting(),
            "should keep correcting above 1.5ms deadband"
        );
    }

    #[test]
    fn test_hysteresis_stops_below_deadband() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(1_000, 48_000, true);
        assert!(
            !schedule.is_correcting(),
            "should stop below 1.5ms deadband"
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
    fn test_reanchor_at_large_error() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(500_000, 48_000, false);
        assert!(schedule.reanchor);
    }

    #[test]
    fn test_exact_engage_threshold_does_not_engage() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(3_000, 48_000, false);
        assert!(
            !schedule.is_correcting(),
            "exactly at engage threshold should not engage (<=)"
        );
    }

    #[test]
    fn test_exact_deadband_threshold_disengages() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(1_500, 48_000, true);
        assert!(
            !schedule.is_correcting(),
            "exactly at deadband threshold should disengage (<=)"
        );
    }

    #[test]
    fn test_negative_hysteresis_keeps_inserting_above_deadband() {
        let planner = CorrectionPlanner::new();
        let schedule = planner.plan(-2_000, 48_000, true);
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
        //   max_corrections_per_sec     = 48000 * 0.04             = 1920 (not binding)
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
        // 200ms error — large enough that the desired rate exceeds the cap:
        //   frames_error                = 200000 * 48000 / 1_000_000 = 9600
        //   desired_corrections_per_sec = 9600 / 2.0                 = 4800
        //   max_corrections_per_sec     = 48000 * 0.04               = 1920 (binding)
        //   interval_frames             = round(48000 / 1920)        = 25
        let planner = CorrectionPlanner::new();
        let s = planner.plan(200_000, 48_000, true);
        assert_eq!(
            s.drop_every_n_frames, 25,
            "large drift should be capped at max speed correction (interval = 25 frames)"
        );
    }

    #[test]
    fn test_filter_constant_input_passes_through() {
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            assert_eq!(filter.update(4_000), 4_000);
        }
        assert!(filter.is_warm());
    }

    #[test]
    fn test_filter_warms_only_after_full_window() {
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW - 1 {
            filter.update(0);
            assert!(!filter.is_warm());
        }
        filter.update(0);
        assert!(filter.is_warm());
    }

    #[test]
    fn test_filter_median_ignores_minority_flap() {
        // Period-quantized flapping: one +10ms excursion every third sample
        // (the observed WASAPI padding alternation) must not move the median.
        let mut filter = SyncErrorFilter::new();
        let mut median = 0;
        for i in 0..SYNC_ERROR_WINDOW * 2 {
            let raw = if i % 3 == 0 { 10_000 } else { 0 };
            median = filter.update(raw);
        }
        assert_eq!(median, 0, "minority mode must not shift the median");
    }

    #[test]
    fn test_filter_median_follows_persistent_step() {
        // A real displacement shifts every subsequent measurement; the
        // median must follow within ~half a window.
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            filter.update(0);
        }
        let mut median = 0;
        for _ in 0..(SYNC_ERROR_WINDOW / 2 + 1) {
            median = filter.update(10_000);
        }
        assert_eq!(median, 10_000, "persistent step must reach the median");
    }

    #[test]
    fn test_filter_reset_clears_window() {
        let mut filter = SyncErrorFilter::new();
        for _ in 0..SYNC_ERROR_WINDOW {
            filter.update(10_000);
        }
        filter.reset();
        assert!(!filter.is_warm());
        assert_eq!(
            filter.update(0),
            0,
            "median after reset must reflect only new samples"
        );
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
        let mut gate = EngageGate::new();
        for _ in 0..ENGAGE_STREAK_PLANS - 1 {
            let admitted = gate.admit(correcting_plan(), false, true);
            assert!(!admitted.is_correcting(), "streak not yet sustained");
        }
        let admitted = gate.admit(correcting_plan(), false, true);
        assert!(admitted.is_correcting(), "sustained streak must engage");
    }

    #[test]
    fn test_gate_streak_resets_on_idle_plan() {
        let mut gate = EngageGate::new();
        for _ in 0..ENGAGE_STREAK_PLANS - 1 {
            gate.admit(correcting_plan(), false, true);
        }
        gate.admit(CorrectionSchedule::default(), false, true);
        let admitted = gate.admit(correcting_plan(), false, true);
        assert!(
            !admitted.is_correcting(),
            "an idle plan must reset the streak"
        );
    }

    #[test]
    fn test_gate_cold_filter_never_engages() {
        let mut gate = EngageGate::new();
        for _ in 0..ENGAGE_STREAK_PLANS * 2 {
            let admitted = gate.admit(correcting_plan(), false, false);
            assert!(!admitted.is_correcting(), "cold filter must not engage");
        }
    }

    #[test]
    fn test_gate_reanchor_bypasses() {
        let mut gate = EngageGate::new();
        let reanchor = CorrectionSchedule {
            insert_every_n_frames: 0,
            drop_every_n_frames: 0,
            reanchor: true,
        };
        let admitted = gate.admit(reanchor, false, false);
        assert!(admitted.reanchor, "reanchor must bypass the gate");
    }

    #[test]
    fn test_gate_running_correction_replans_freely() {
        let mut gate = EngageGate::new();
        let admitted = gate.admit(correcting_plan(), true, true);
        assert!(
            admitted.is_correcting(),
            "a running correction must replan without gating"
        );
        let disengage = gate.admit(CorrectionSchedule::default(), true, true);
        assert!(
            !disengage.is_correcting(),
            "disengage must pass immediately"
        );
    }
}
