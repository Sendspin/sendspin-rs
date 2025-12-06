use sendspin::audio::output::{AudioOutput, CpalOutput};
use sendspin::audio::{AudioFormat, Codec, Sample};
use std::sync::Arc;

#[test]
fn test_audio_output_creation() {
    let format = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 48000,
        channels: 2,
        bit_depth: 24,
        codec_header: None,
    };

    // CpalOutput::new() should succeed
    let output = CpalOutput::new(format);
    assert!(output.is_ok());
}

#[test]
fn test_audio_output_write() {
    let format = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 48000,
        channels: 2,
        bit_depth: 24,
        codec_header: None,
    };

    let mut output = CpalOutput::new(format).unwrap();

    // Create some test samples (silence)
    let samples: Vec<Sample> = vec![Sample::ZERO; 960]; // 10ms at 48kHz stereo
    let samples_arc = Arc::from(samples.into_boxed_slice());

    // Should be able to write without error
    let result = output.write(&samples_arc);
    assert!(result.is_ok());
}
