// ABOUTME: PCM decoder implementation
// ABOUTME: Supports 16-bit and 24-bit PCM decoding with zero-copy where possible

use crate::audio::decode::Decoder;
use crate::audio::Sample;
use crate::error::Error;
use std::sync::Arc;

/// PCM endianness
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmEndian {
    /// Little-endian byte order
    Little,
    /// Big-endian byte order
    Big,
}

/// PCM audio decoder supporting 16-bit and 24-bit formats
#[derive(Clone)]
pub struct PcmDecoder {
    bit_depth: u8,
    endian: PcmEndian,
}

impl PcmDecoder {
    /// Create a new PCM decoder with the specified bit depth (16 or 24), defaulting to little-endian
    pub fn new(bit_depth: u8) -> Self {
        Self {
            bit_depth,
            endian: PcmEndian::Little,
        }
    }

    /// Create a new PCM decoder with explicit endianness
    pub fn with_endian(bit_depth: u8, endian: PcmEndian) -> Self {
        Self { bit_depth, endian }
    }
}

impl Decoder for PcmDecoder {
    fn decode(&self, data: &[u8]) -> Result<Arc<[Sample]>, Error> {
        match (self.bit_depth, self.endian) {
            (16, PcmEndian::Little) => {
                // Convert 16-bit little-endian PCM to Sample
                let samples: Vec<Sample> = data
                    .chunks_exact(2)
                    .map(|c| {
                        let i16_val = i16::from_le_bytes([c[0], c[1]]);
                        Sample::from_i16(i16_val)
                    })
                    .collect();
                Ok(Arc::from(samples.into_boxed_slice()))
            }
            (16, PcmEndian::Big) => {
                // Convert 16-bit big-endian PCM to Sample
                let samples: Vec<Sample> = data
                    .chunks_exact(2)
                    .map(|c| {
                        let i16_val = i16::from_be_bytes([c[0], c[1]]);
                        Sample::from_i16(i16_val)
                    })
                    .collect();
                Ok(Arc::from(samples.into_boxed_slice()))
            }
            (24, PcmEndian::Little) => {
                // Convert 24-bit little-endian PCM to Sample
                let samples: Vec<Sample> = data
                    .chunks_exact(3)
                    .map(|c| Sample::from_i24_le([c[0], c[1], c[2]]))
                    .collect();
                Ok(Arc::from(samples.into_boxed_slice()))
            }
            (24, PcmEndian::Big) => {
                // Convert 24-bit big-endian PCM to Sample
                let samples: Vec<Sample> = data
                    .chunks_exact(3)
                    .map(|c| Sample::from_i24_be([c[0], c[1], c[2]]))
                    .collect();
                Ok(Arc::from(samples.into_boxed_slice()))
            }
            _ => Err(Error::Protocol(format!(
                "Unsupported bit depth: {}",
                self.bit_depth
            ))),
        }
    }
}
