use super::{Decoder, FlacDecoder, OpusDecoder, PcmDecoder};
use crate::audio::{AudioFormat, Codec, StreamPlayerConfig};
use crate::error::Error;
use base64::Engine;
use std::sync::Arc;

enum DecoderImpl {
    Pcm(PcmDecoder),
    Opus(Box<OpusDecoder>),
    Flac(FlacDecoder),
}

/// Decoder selected by one `stream/start` configuration for PCM, Opus, or FLAC.
/// Keep one per stream: Opus carries decoding state across chunks.
/// A supplied codec header must be nonempty standard base64; only FLAC interprets it.
pub struct StreamDecoder {
    format: AudioFormat,
    decoder: DecoderImpl,
}

impl StreamDecoder {
    /// Builds a decoder for PCM 16/24-bit, Opus, or FLAC from a stream configuration.
    /// Opus accepts 8/12/16/24/48 kHz and 1-2 channels; FLAC accepts 4-32 bits and 1-8 channels.
    /// FLAC headers must match the config; [`FlacDecoder`] documents frame-header limitations.
    pub fn new(config: &StreamPlayerConfig) -> Result<Self, Error> {
        if config.sample_rate == 0 {
            return Err(Error::Protocol("Sample rate must be non-zero".into()));
        }
        if config.channels == 0 {
            return Err(Error::Protocol("Channel count must be non-zero".into()));
        }
        let codec = match config.codec.as_str() {
            "pcm" => Codec::Pcm,
            "opus" => Codec::Opus,
            "flac" => Codec::Flac,
            "mp3" => return Err(Error::Protocol("MP3 decoding is not supported".into())),
            other => return Err(Error::Protocol(format!("Unsupported audio codec: {other}"))),
        };
        let format = AudioFormat {
            codec,
            sample_rate: config.sample_rate,
            channels: config.channels,
            bit_depth: config.bit_depth,
            codec_header: decode_header(config.codec_header.as_deref())?,
        };
        let decoder = match codec {
            Codec::Pcm => {
                if !matches!(config.bit_depth, 16 | 24) {
                    return Err(Error::Protocol(format!(
                        "Unsupported PCM bit depth: {}",
                        config.bit_depth
                    )));
                }
                DecoderImpl::Pcm(PcmDecoder::new(config.bit_depth))
            }
            Codec::Opus => DecoderImpl::Opus(Box::new(OpusDecoder::new(
                config.sample_rate,
                config.channels,
            )?)),
            Codec::Flac => {
                if !(4..=32).contains(&config.bit_depth) {
                    return Err(Error::Protocol(format!(
                        "Unsupported FLAC bit depth: {}",
                        config.bit_depth
                    )));
                }
                if config.channels > 8 {
                    return Err(Error::Protocol(format!(
                        "Unsupported FLAC channel count: {}",
                        config.channels
                    )));
                }
                if let Some(header) = format.codec_header.as_deref() {
                    let info = flac_codec::metadata::read_info(header)
                        .map_err(|e| Error::Protocol(format!("Invalid FLAC codec header: {e}")))?;
                    if info.sample_rate != format.sample_rate
                        || info.channels.get() != format.channels
                        || u32::from(info.bits_per_sample) != u32::from(format.bit_depth)
                    {
                        return Err(Error::Protocol(
                            "FLAC header format does not match expected stream format".into(),
                        ));
                    }
                }
                DecoderImpl::Flac(FlacDecoder::from_format(&format))
            }
            Codec::Mp3 => unreachable!(),
        };
        Ok(Self { format, decoder })
    }

    /// Return the validated audio format used by this decoder.
    pub fn format(&self) -> &AudioFormat {
        &self.format
    }
}

fn decode_header(header: Option<&str>) -> Result<Option<Vec<u8>>, Error> {
    header
        .map(|header| {
            if header.is_empty() {
                return Err(Error::Protocol("Codec header must not be empty".into()));
            }
            base64::engine::general_purpose::STANDARD
                .decode(header)
                .map_err(|e| Error::Protocol(format!("Invalid codec header base64: {e}")))
        })
        .transpose()
}

impl Decoder for StreamDecoder {
    fn decode(&self, data: &[u8]) -> Result<Arc<[i32]>, Error> {
        let samples = match &self.decoder {
            DecoderImpl::Pcm(decoder) => decoder.decode(data),
            DecoderImpl::Opus(decoder) => decoder.decode(data),
            DecoderImpl::Flac(decoder) => decoder.decode(data),
        }?;
        let channels = usize::from(self.format.channels);
        if !samples.len().is_multiple_of(channels) {
            return Err(Error::Protocol(format!(
                "Decoded audio has {} samples, not a whole number of {}-channel frames",
                samples.len(),
                channels
            )));
        }
        Ok(samples)
    }
}
