//! Private invalidation counters for ordered stream work and local playback resets.

/// Identifies the stream boundary that invalidates queued or in-flight work.
/// Advancement is checked: wrapping could admit work from an older boundary.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StreamGeneration(u64);

#[allow(dead_code)]
impl StreamGeneration {
    pub(crate) const fn initial() -> Self {
        Self(0)
    }

    pub(crate) fn checked_advance(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }

    #[cfg(test)]
    pub(crate) const fn max_for_test() -> Self {
        Self(u64::MAX)
    }
}

/// Identifies a local playback queue reset.
/// Wrapping is safe because only equality is tested; keep this type distinct
/// from [`StreamGeneration`] even when their values coincide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlaybackResetGeneration(u64);

impl PlaybackResetGeneration {
    pub(crate) const fn initial() -> Self {
        Self(0)
    }

    pub(crate) fn advance(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

#[cfg(test)]
mod tests {
    use super::{PlaybackResetGeneration, StreamGeneration};

    #[test]
    fn within_stream_advancement_is_ordered_and_playback_reset_advances() {
        assert!(
            StreamGeneration::initial() < StreamGeneration::initial().checked_advance().unwrap()
        );
        assert_ne!(
            PlaybackResetGeneration::initial().advance(),
            PlaybackResetGeneration::initial()
        );
    }

    #[test]
    fn stream_generation_checked_advance_stops_at_maximum() {
        let maximum = StreamGeneration(u64::MAX);
        assert_eq!(maximum.checked_advance(), None);
    }

    #[test]
    fn playback_reset_generation_advance_wraps_at_maximum() {
        let maximum = PlaybackResetGeneration(u64::MAX);
        assert_eq!(maximum.advance(), PlaybackResetGeneration::initial());
    }
}
