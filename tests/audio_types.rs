use sendspin::audio::{AudioFormat, Codec, Sample};

#[test]
fn test_sample_from_i16() {
    let sample = Sample::from_i16(1000);
    assert_eq!(sample.to_i16(), 1000);
}

#[test]
fn test_sample_from_i24() {
    let bytes = [0x00, 0x10, 0x00]; // 4096 in 24-bit little-endian
    let sample = Sample::from_i24_le(bytes);
    assert_eq!(sample.0, 4096);
}

#[test]
fn test_sample_clamp() {
    let over_max = Sample(10_000_000);
    assert_eq!(over_max.clamp().0, Sample::MAX.0);

    let under_min = Sample(-10_000_000);
    assert_eq!(under_min.clamp().0, Sample::MIN.0);
}

#[test]
fn test_audio_format_creation() {
    let format = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 48000,
        channels: 2,
        bit_depth: 24,
        codec_header: None,
    };

    assert_eq!(format.sample_rate, 48000);
    assert_eq!(format.channels, 2);
}
