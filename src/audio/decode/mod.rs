// ABOUTME: Audio decoder implementations
// ABOUTME: PCM, Opus, and FLAC decoders

//! Audio decoder implementations and stream-configuration selection.
//!
//! The module validates stream formats, decodes PCM/Opus/FLAC payloads, and exposes
//! individual codec decoders when applications need codec-specific construction.

/// FLAC decoder implementation
pub mod flac;
/// Opus decoder implementation
pub mod opus;
/// PCM decoder implementation
pub mod pcm;
/// Decoder selected from a stream configuration
pub mod stream;

pub use flac::FlacDecoder;
pub use opus::OpusDecoder;
pub use pcm::{PcmDecoder, PcmEndian};
pub use stream::StreamDecoder;

use crate::error::Error;
use std::sync::Arc;

/// Decoder trait for audio codecs
pub trait Decoder {
    /// Decode raw audio data into samples
    fn decode(&self, data: &[u8]) -> Result<Arc<[i32]>, Error>;
}
