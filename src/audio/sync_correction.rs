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

/// Planner that converts sync error into a correction schedule.
#[derive(Debug, Clone, Copy)]
pub struct CorrectionPlanner {
    deadband_us: i64,
    reanchor_threshold_us: i64,
    target_seconds: f64,
    max_speed_correction: f64,
}

impl CorrectionPlanner {
    /// Create a planner with default thresholds.
    pub fn new() -> Self {
        Self {
            deadband_us: 2_000,
            reanchor_threshold_us: 500_000,
            target_seconds: 2.0,
            max_speed_correction: 0.04,
        }
    }

    /// Plan a correction schedule from sync error and sample rate.
    pub fn plan(&self, error_us: i64, sample_rate: u32) -> CorrectionSchedule {
        if error_us.saturating_abs() <= self.deadband_us {
            return CorrectionSchedule {
                insert_every_n_frames: 0,
                drop_every_n_frames: 0,
                reanchor: false,
            };
        }

        if error_us.saturating_abs() >= self.reanchor_threshold_us {
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
