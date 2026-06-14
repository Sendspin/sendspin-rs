// ABOUTME: Playback recovery coordination shared by audio and protocol layers
// ABOUTME: Gates playback after underruns and exposes state transitions to protocol

//! Playback recovery coordination primitives.
//!
//! The Sendspin spec requires an underrunning player to report `error`, mute
//! output while it rebuilds buffer, then report `synchronized` when playback can
//! safely resume.
//!
//! Two handles share one piece of state:
//!
//! - [`PlaybackRecoveryCoordinator`] is the **audio-side** handle. The audio
//!   callback reports underruns, queries whether output should stay muted, and
//!   feeds buffered-duration observations. All of its methods are non-blocking
//!   and allocation-free so they are safe to call from a real-time audio
//!   callback.
//! - [`PlaybackRecoveryMonitor`] is the **protocol-side** handle, obtained via
//!   [`PlaybackRecoveryCoordinator::monitor`]. It awaits transitions and drains
//!   them for `client/state` messages, but cannot drive recovery itself.
//!
//! Splitting the API this way keeps each side honest: protocol code can't fake
//! an underrun, and audio code can't accidentally consume a transition the
//! protocol task is responsible for sending.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

// Pending-transition bit encoding. `Error` is always drained before
// `Synchronized`, so a complete underrun/recovery cycle is reported even when it
// happens before the protocol task wakes. A fresh underrun clears any undelivered
// `Synchronized`, preventing stale recovery reports after playback is gated again.
const PENDING_NONE: u8 = 0;
const PENDING_ERROR: u8 = 1 << 0;
const PENDING_SYNCHRONIZED: u8 = 1 << 1;

/// Playback synchronization state transitions emitted by the recovery
/// coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackRecoveryState {
    /// Playback underrun occurred; output should remain muted while buffering.
    Error,
    /// Enough audio has buffered to resume synchronized playback.
    Synchronized,
}

#[derive(Debug)]
struct PlaybackRecoveryInner {
    recovering: AtomicBool,
    /// Undelivered protocol transitions (`PENDING_*` bits). Error is drained
    /// before synchronized; a new error clears stale synchronized.
    pending: AtomicU8,
    recovery_buffer_us: AtomicI64,
    /// Wakes the protocol monitor task when `pending` changes.
    transition: Notify,
}

impl PlaybackRecoveryInner {
    fn queue_error(&self) {
        self.pending.store(PENDING_ERROR, Ordering::Release);
        self.transition.notify_one();
    }

    fn queue_synchronized(&self) {
        self.pending
            .fetch_or(PENDING_SYNCHRONIZED, Ordering::AcqRel);
        self.transition.notify_one();
    }
}

/// Shared underrun recovery coordinator (audio-side handle).
///
/// Clone this handle freely; all methods are non-blocking and allocation-free
/// after construction, so audio callbacks can call [`Self::report_underrun`],
/// [`Self::is_recovering`], and [`Self::observe_buffered_duration`] without
/// taking a mutex. Obtain the protocol-side handle with [`Self::monitor`].
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
                pending: AtomicU8::new(PENDING_NONE),
                recovery_buffer_us: AtomicI64::new(i64::from(recovery_buffer_ms) * 1_000),
                transition: Notify::new(),
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

    /// Obtain the protocol-side handle for emitting `client/state` transitions.
    pub fn monitor(&self) -> PlaybackRecoveryMonitor {
        PlaybackRecoveryMonitor {
            inner: Arc::clone(&self.inner),
        }
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
    /// This immediately gates playback and queues an `Error` transition for
    /// protocol reporting, replacing any undelivered `Synchronized`. A subsequent
    /// recovery queues `Synchronized` behind the undelivered `Error`, so short
    /// underrun/recovery cycles are still fully reported. Repeated underruns
    /// while already recovering keep playback gated.
    pub fn report_underrun(&self) {
        self.inner.recovering.store(true, Ordering::Release);
        self.inner.queue_error();
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
            self.inner.queue_synchronized();
            return true;
        }

        false
    }

    /// Recovery threshold in microseconds.
    pub fn recovery_buffer_us(&self) -> i64 {
        self.inner.recovery_buffer_us.load(Ordering::Acquire)
    }
}

/// Protocol-side handle for emitting `client/state` transitions.
///
/// Obtain one with [`PlaybackRecoveryCoordinator::monitor`]. It shares the same
/// underlying state as the audio callback's coordinator, so protocol
/// `client/state` transitions and playback gating cannot diverge.
#[derive(Debug, Clone)]
pub struct PlaybackRecoveryMonitor {
    inner: Arc<PlaybackRecoveryInner>,
}

impl PlaybackRecoveryMonitor {
    /// Wait until a transition is pending.
    ///
    /// A wakeup is delivered even if it arrives before this future is awaited,
    /// so transitions cannot be lost between calls. After waking, drain with
    /// [`Self::take_pending_transition`]. Spurious wakeups are possible; treat a
    /// `None` from `take_pending_transition` as "nothing to send".
    pub async fn notified(&self) {
        self.inner.transition.notified().await;
    }

    /// Return and clear the next pending state transition for protocol reporting.
    pub fn take_pending_transition(&self) -> Option<PlaybackRecoveryState> {
        loop {
            let pending = self.inner.pending.load(Ordering::Acquire);
            let (state, new_pending) = if pending & PENDING_ERROR != 0 {
                (PlaybackRecoveryState::Error, pending & !PENDING_ERROR)
            } else if pending & PENDING_SYNCHRONIZED != 0 {
                (
                    PlaybackRecoveryState::Synchronized,
                    pending & !PENDING_SYNCHRONIZED,
                )
            } else {
                return None;
            };

            if self
                .inner
                .pending
                .compare_exchange(pending, new_pending, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn underrun_gates_playback_and_queues_error_once() {
        let recovery = PlaybackRecoveryCoordinator::new(500);
        let monitor = recovery.monitor();

        assert!(!recovery.is_recovering());
        assert_eq!(monitor.take_pending_transition(), None);

        recovery.report_underrun();
        assert!(recovery.is_recovering());
        assert_eq!(
            monitor.take_pending_transition(),
            Some(PlaybackRecoveryState::Error)
        );
        assert_eq!(monitor.take_pending_transition(), None);
    }

    #[test]
    fn recovery_releases_on_first_above_threshold_observation() {
        let recovery = PlaybackRecoveryCoordinator::new(500);
        let monitor = recovery.monitor();
        recovery.report_underrun();
        assert_eq!(
            monitor.take_pending_transition(),
            Some(PlaybackRecoveryState::Error)
        );

        // Below threshold: no release
        assert!(!recovery.observe_buffered_duration(Some(499_999)));
        assert!(recovery.is_recovering());
        assert_eq!(monitor.take_pending_transition(), None);

        // At or above threshold: immediate release
        assert!(recovery.observe_buffered_duration(Some(500_000)));
        assert!(!recovery.is_recovering());
        assert_eq!(
            monitor.take_pending_transition(),
            Some(PlaybackRecoveryState::Synchronized)
        );
        assert_eq!(monitor.take_pending_transition(), None);
    }

    #[test]
    fn fast_recovery_reports_error_before_synchronized() {
        let recovery = PlaybackRecoveryCoordinator::new(500);
        let monitor = recovery.monitor();

        recovery.report_underrun();
        assert!(recovery.observe_buffered_duration(Some(500_000)));

        assert_eq!(
            monitor.take_pending_transition(),
            Some(PlaybackRecoveryState::Error),
            "a short underrun must still report error before recovery"
        );
        assert_eq!(
            monitor.take_pending_transition(),
            Some(PlaybackRecoveryState::Synchronized)
        );
        assert_eq!(monitor.take_pending_transition(), None);
    }

    #[test]
    fn insufficient_buffer_keeps_recovery_gated() {
        let recovery = PlaybackRecoveryCoordinator::new(500);
        let monitor = recovery.monitor();
        recovery.report_underrun();
        assert_eq!(
            monitor.take_pending_transition(),
            Some(PlaybackRecoveryState::Error)
        );

        // Multiple below-threshold observations don't accumulate
        assert!(!recovery.observe_buffered_duration(Some(499_999)));
        assert!(!recovery.observe_buffered_duration(None));
        assert!(recovery.is_recovering());

        // Once above threshold, recovery releases immediately
        assert!(recovery.observe_buffered_duration(Some(500_000)));
        assert!(!recovery.is_recovering());
    }

    #[test]
    fn fresh_underrun_overrides_undelivered_synchronized() {
        // If recovery completes and then a new underrun occurs before the
        // protocol task drains the transition, the monitor must observe the
        // current state (Error), never the stale Synchronized.
        let recovery = PlaybackRecoveryCoordinator::new(500);
        let monitor = recovery.monitor();

        // Drive a full recovery so Synchronized becomes pending.
        recovery.report_underrun();
        assert!(recovery.observe_buffered_duration(Some(500_000)));

        // New underrun before the protocol side drained the transition.
        recovery.report_underrun();

        assert_eq!(
            monitor.take_pending_transition(),
            Some(PlaybackRecoveryState::Error),
            "a fresh underrun must replace an undelivered synchronized transition"
        );
        assert_eq!(monitor.take_pending_transition(), None);
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
