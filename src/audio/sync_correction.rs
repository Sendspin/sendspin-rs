// ABOUTME: Sync correction planner for drop/insert cadence
// ABOUTME: Computes correction schedule from sync error and sample rate

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
}
