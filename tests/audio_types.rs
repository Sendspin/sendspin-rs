use sendspin::audio::{AudioFormat, Codec, SendspinSample};

#[test]
fn test_sample_from_i16_roundtrip() {
    assert_eq!(SendspinSample::from_i16(1000).to_i16(), 1000);
}

#[test]
fn test_sample_from_i16_boundary_values() {
    assert_eq!(SendspinSample::from_i16(i16::MAX).to_i16(), i16::MAX);
    assert_eq!(SendspinSample::from_i16(i16::MIN).to_i16(), i16::MIN);
    assert_eq!(SendspinSample::from_i16(0).to_i16(), 0);
    assert_eq!(SendspinSample::from_i16(-1).to_i16(), -1);
    assert_eq!(SendspinSample::from_i16(1).to_i16(), 1);
}

#[test]
fn test_sample_from_i24_le() {
    let bytes = [0x00, 0x10, 0x00]; // 4096 in 24-bit little-endian
    let sample = SendspinSample::from_i24_le(bytes);
    assert_eq!(sample.to_f32(), 4096.0 / 8_388_608.0);
}

#[test]
fn test_sample_from_i24_le_negative() {
    // -1 in 24-bit LE: 0xFF 0xFF 0xFF
    let sample = SendspinSample::from_i24_le([0xFF, 0xFF, 0xFF]);
    assert_eq!(sample, SendspinSample(-1));
}

#[test]
fn test_sample_from_i24_le_boundary_values() {
    // Max 24-bit: 0x7FFFFF = 8388607
    let max_sample = SendspinSample::from_i24_le([0xFF, 0xFF, 0x7F]);
    assert_eq!(max_sample, SendspinSample::MAX);

    // Min 24-bit: 0x800000 = -8388608
    let min_sample = SendspinSample::from_i24_le([0x00, 0x00, 0x80]);
    assert_eq!(min_sample, SendspinSample::MIN);

    // Zero
    let zero = SendspinSample::from_i24_le([0x00, 0x00, 0x00]);
    assert_eq!(zero, SendspinSample::ZERO);
}

#[test]
fn test_sample_from_i24_be_roundtrip() {
    // 4096 in 24-bit BE: 0x00 0x10 0x00
    let sample = SendspinSample::from_i24_be([0x00, 0x10, 0x00]);
    assert_eq!(sample, SendspinSample(4096));

    // -1 in 24-bit BE
    let neg = SendspinSample::from_i24_be([0xFF, 0xFF, 0xFF]);
    assert_eq!(neg, SendspinSample(-1));
}

#[test]
fn test_sample_from_i24_be_boundary_values() {
    // Max 24-bit: 0x7FFFFF = 8388607
    let max_sample = SendspinSample::from_i24_be([0x7F, 0xFF, 0xFF]);
    assert_eq!(max_sample, SendspinSample::MAX);

    // Min 24-bit: 0x800000 = -8388608
    let min_sample = SendspinSample::from_i24_be([0x80, 0x00, 0x00]);
    assert_eq!(min_sample, SendspinSample::MIN);

    // Zero
    let zero = SendspinSample::from_i24_be([0x00, 0x00, 0x00]);
    assert_eq!(zero, SendspinSample::ZERO);
}

#[test]
fn test_sample_clamp_out_of_range() {
    let over_max = SendspinSample(10_000_000);
    assert_eq!(over_max.clamp(), SendspinSample::MAX);

    let under_min = SendspinSample(-10_000_000);
    assert_eq!(under_min.clamp(), SendspinSample::MIN);
}

#[test]
fn test_sample_clamp_within_range() {
    let in_range = SendspinSample(42);
    assert_eq!(in_range.clamp(), SendspinSample(42));

    assert_eq!(SendspinSample::ZERO.clamp(), SendspinSample::ZERO);
    assert_eq!(SendspinSample::MAX.clamp(), SendspinSample::MAX);
    assert_eq!(SendspinSample::MIN.clamp(), SendspinSample::MIN);
}

#[test]
fn test_sample_to_f32_range() {
    // MIN should map to exactly -1.0
    assert_eq!(SendspinSample::MIN.to_f32(), -1.0);
    // ZERO should map to 0.0
    assert_eq!(SendspinSample::ZERO.to_f32(), 0.0);
    // MAX should be close to but not exceed 1.0
    let max_f32 = SendspinSample::MAX.to_f32();
    assert!(max_f32 > 0.999 && max_f32 < 1.0);
}

#[test]
fn test_sample_to_f32_typical_values() {
    // Half of max should be ~0.5
    let half = SendspinSample(8_388_608 / 2).to_f32();
    assert!((half - 0.5).abs() < 0.001);

    // Negative half should be ~-0.5
    let neg_half = SendspinSample(-8_388_608 / 2).to_f32();
    assert!((neg_half - (-0.5)).abs() < 0.001);

    // Small positive value from i16 conversion
    let from_i16 = SendspinSample::from_i16(1000);
    let f = from_i16.to_f32();
    assert!(f > 0.0 && f < 1.0);
}

#[test]
fn test_audio_format_equality() {
    let format1 = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 48000,
        channels: 2,
        bit_depth: 24,
        codec_header: None,
    };
    let format2 = format1.clone();
    assert_eq!(format1, format2);

    let different = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 44100,
        channels: 2,
        bit_depth: 16,
        codec_header: None,
    };
    assert_ne!(format1, different);
}
