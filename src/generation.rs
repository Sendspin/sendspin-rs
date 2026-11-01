//! Private invalidation counters for ordered stream work and local playback resets.

/// Identifies the stream boundary that invalidates queued or in-flight work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct StreamGeneration(u64);

impl StreamGeneration {
    pub(crate) const fn initial() -> Self {
        Self(0)
    }

    pub(crate) fn advance(self) -> Self {
        Self(self.0 + 1)
    }
}

/// Identifies a local playback queue reset.
/// Wrapping is safe because only equality is tested; keep this type distinct
/// from [`StreamGeneration`] even when their values coincide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlaybackResetGeneration(u64);

/// Identifies one installed output lifetime within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputIncarnation(u64);

impl OutputIncarnation {
    pub(crate) const fn initial() -> Self {
        Self(0)
    }

    pub(crate) fn advance(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

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
        assert!(StreamGeneration::initial() < StreamGeneration::initial().advance());
        assert_ne!(
            PlaybackResetGeneration::initial().advance(),
            PlaybackResetGeneration::initial()
        );
    }

    #[test]
    fn playback_reset_generation_advance_wraps_at_maximum() {
        let maximum = PlaybackResetGeneration(u64::MAX);
        assert_eq!(maximum.advance(), PlaybackResetGeneration::initial());
    }
}
