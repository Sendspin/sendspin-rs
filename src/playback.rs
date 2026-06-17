// ABOUTME: Gates playback after underruns until enough audio re-buffers

//! Playback recovery coordination primitive.
//!
//! When the audio callback underruns, the player must mute output while it
//! rebuilds buffer, then resume once enough continuous audio is available.
//!
//! [`PlaybackRecoveryCoordinator`] is the audio-side handle. Its methods are
//! non-blocking and allocation-free so the real-time audio callback can report
//! underruns, query whether output should stay muted, and feed buffered-duration
//! observations without taking a mutex.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

#[derive(Debug)]
struct PlaybackRecoveryInner {
    recovering: AtomicBool,
    recovery_buffer_us: AtomicI64,
}

/// Shared underrun recovery coordinator (audio-side handle).
///
/// Clone this handle freely; all methods are non-blocking and allocation-free
/// after construction, so audio callbacks can call [`Self::report_underrun`],
/// [`Self::is_recovering`], and [`Self::observe_buffered_duration`] without
/// taking a mutex.
#[derive(Debug, Clone)]
pub struct PlaybackRecoveryCoordinator {
    inner: Arc<PlaybackRecoveryInner>,
}

impl PlaybackRecoveryCoordinator {
    /// Create a coordinator with an explicit recovery-buffer threshold.
    pub fn new(recovery_buffer_ms: u32) -> Self {
        Self {
            inner: Arc::new(PlaybackRecoveryInner {
                recovering: AtomicBool::new(false),
                recovery_buffer_us: AtomicI64::new(i64::from(recovery_buffer_ms) * 1_000),
            }),
        }
    }

    /// Create a coordinator from the player's timing parameters.
    ///
    /// The recovery threshold is `max(required_lead_time_ms, min_buffer_ms)`,
    /// matching the Sendspin server's live/realtime send-ahead floor.
    pub fn from_player_timing(required_lead_time_ms: u32, min_buffer_ms: u32) -> Self {
        Self::new(required_lead_time_ms.max(min_buffer_ms))
    }

    /// Configure the recovery-buffer threshold from explicit milliseconds.
    pub fn set_recovery_buffer_ms(&self, recovery_buffer_ms: u32) {
        self.inner
            .recovery_buffer_us
            .store(i64::from(recovery_buffer_ms) * 1_000, Ordering::Release);
    }

    /// Configure the recovery-buffer threshold from player timing parameters.
    pub fn set_player_timing(&self, required_lead_time_ms: u32, min_buffer_ms: u32) {
        self.set_recovery_buffer_ms(required_lead_time_ms.max(min_buffer_ms));
    }

    /// Mark that playback underrun occurred.
    ///
    /// This immediately gates playback. Repeated underruns while already
    /// recovering keep playback gated.
    pub fn report_underrun(&self) {
        self.inner.recovering.store(true, Ordering::Release);
    }

    /// Whether audio output should remain muted while buffering after underrun.
    pub fn is_recovering(&self) -> bool {
        self.inner.recovering.load(Ordering::Acquire)
    }

    /// Observe the currently buffered continuous audio duration.
    ///
    /// While recovering, playback is released as soon as the buffered duration
    /// reaches the configured threshold. If the buffer drops back below the
    /// threshold after recovery, the audio callback will underrun again and
    /// re-trigger [`Self::report_underrun`].
    ///
    /// Returns `true` when this call releases recovery.
    pub fn observe_buffered_duration(&self, buffered_duration_us: Option<i64>) -> bool {
        if !self.is_recovering() {
            return false;
        }

        let recovered =
            buffered_duration_us.is_some_and(|duration| duration >= self.recovery_buffer_us());

        if recovered {
            self.inner.recovering.store(false, Ordering::Release);
            return true;
        }

        false
    }

    /// Recovery threshold in microseconds.
    pub fn recovery_buffer_us(&self) -> i64 {
        self.inner.recovery_buffer_us.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn underrun_gates_playback() {
        let recovery = PlaybackRecoveryCoordinator::new(500);

        assert!(!recovery.is_recovering());

        recovery.report_underrun();
        assert!(recovery.is_recovering());
    }

    #[test]
    fn recovery_releases_on_first_above_threshold_observation() {
        let recovery = PlaybackRecoveryCoordinator::new(500);
        recovery.report_underrun();

        // Below threshold: no release
        assert!(!recovery.observe_buffered_duration(Some(499_999)));
        assert!(recovery.is_recovering());

        // At or above threshold: immediate release
        assert!(recovery.observe_buffered_duration(Some(500_000)));
        assert!(!recovery.is_recovering());
    }

    #[test]
    fn fast_underrun_recovery_cycle_gates_then_releases() {
        // A short underrun immediately followed by recovery still gates and
        // then releases playback.
        let recovery = PlaybackRecoveryCoordinator::new(500);

        recovery.report_underrun();
        assert!(recovery.is_recovering());
        assert!(recovery.observe_buffered_duration(Some(500_000)));
        assert!(!recovery.is_recovering());
    }

    #[test]
    fn insufficient_buffer_keeps_recovery_gated() {
        let recovery = PlaybackRecoveryCoordinator::new(500);
        recovery.report_underrun();

        // Multiple below-threshold observations don't accumulate
        assert!(!recovery.observe_buffered_duration(Some(499_999)));
        assert!(!recovery.observe_buffered_duration(None));
        assert!(recovery.is_recovering());

        // Once above threshold, recovery releases immediately
        assert!(recovery.observe_buffered_duration(Some(500_000)));
        assert!(!recovery.is_recovering());
    }

    #[test]
    fn fresh_underrun_gates_recovered_playback() {
        // If recovery completes and then a new underrun occurs, playback must
        // gate again.
        let recovery = PlaybackRecoveryCoordinator::new(500);

        // Drive a full recovery.
        recovery.report_underrun();
        assert!(recovery.observe_buffered_duration(Some(500_000)));

        // New underrun gates playback again.
        recovery.report_underrun();
        assert!(recovery.is_recovering());
    }

    #[test]
    fn set_player_timing_updates_recovery_threshold_mid_recovery() {
        let recovery = PlaybackRecoveryCoordinator::new(500);
        recovery.report_underrun();

        // Below original threshold: no release
        assert!(!recovery.observe_buffered_duration(Some(300_000)));

        // Client reports tighter timing; threshold drops to 300ms
        recovery.set_player_timing(300, 200);
        assert_eq!(recovery.recovery_buffer_us(), 300_000);

        // Same buffered amount now meets the updated threshold
        assert!(recovery.observe_buffered_duration(Some(300_000)));
        assert!(!recovery.is_recovering());
    }
}
