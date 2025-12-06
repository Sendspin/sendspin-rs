// ABOUTME: Audio decoder implementations
// ABOUTME: PCM, Opus, FLAC decoders (Phase 1: PCM only)

/// PCM decoder implementation
pub mod pcm;

pub use pcm::{PcmDecoder, PcmEndian};

use crate::audio::Sample;
use crate::error::Error;
use std::sync::Arc;

/// Decoder trait for audio codecs
pub trait Decoder {
    /// Decode raw audio data into samples
    fn decode(&self, data: &[u8]) -> Result<Arc<[Sample]>, Error>;
}
