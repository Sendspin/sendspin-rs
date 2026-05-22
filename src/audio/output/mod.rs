// ABOUTME: Audio output trait and implementations
// ABOUTME: Provides abstraction over platform audio APIs (cpal, ALSA, etc.)

/// cpal-based audio output implementation
pub mod cpal_output;

pub use cpal_output::CpalOutput;

use crate::audio::{AudioFormat, Sample};
use crate::error::Error;
use std::sync::Arc;

/// Audio output trait for playing audio samples
pub trait AudioOutput {
    /// Write samples to the audio output
    fn write(&mut self, samples: &Arc<[Sample]>) -> Result<(), Error>;

    /// Get the current output latency in microseconds
    fn latency_micros(&self) -> u64;

    /// Get the audio format this output expects
    fn format(&self) -> &AudioFormat;
}
