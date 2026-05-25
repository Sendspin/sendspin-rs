use sendspin::audio::{AudioFormat, Codec, Sample};

#[test]
fn test_sample_from_i16_roundtrip() {
    assert_eq!(Sample::from_i16(1000).to_i16(), 1000);
}

#[test]
fn test_sample_from_i16_boundary_values() {
    assert_eq!(Sample::from_i16(i16::MAX).to_i16(), i16::MAX);
    assert_eq!(Sample::from_i16(i16::MIN).to_i16(), i16::MIN);
    assert_eq!(Sample::from_i16(0).to_i16(), 0);
    assert_eq!(Sample::from_i16(-1).to_i16(), -1);
    assert_eq!(Sample::from_i16(1).to_i16(), 1);
}

#[test]
fn test_sample_from_i24_le() {
    let bytes = [0x00, 0x10, 0x00]; // 4096 in 24-bit little-endian
    let sample = Sample::from_i24_le(bytes);
    assert_eq!(sample.to_f32(), 4096.0 / 8_388_608.0);
}

#[test]
fn test_sample_from_i24_le_negative() {
    // -1 in 24-bit LE: 0xFF 0xFF 0xFF
    let sample = Sample::from_i24_le([0xFF, 0xFF, 0xFF]);
    assert_eq!(sample, Sample(-1));
}

#[test]
fn test_sample_from_i24_le_boundary_values() {
    // Max 24-bit: 0x7FFFFF = 8388607
    let max_sample = Sample::from_i24_le([0xFF, 0xFF, 0x7F]);
    assert_eq!(max_sample, Sample::MAX);

    // Min 24-bit: 0x800000 = -8388608
    let min_sample = Sample::from_i24_le([0x00, 0x00, 0x80]);
    assert_eq!(min_sample, Sample::MIN);

    // Zero
    let zero = Sample::from_i24_le([0x00, 0x00, 0x00]);
    assert_eq!(zero, Sample::ZERO);
}

#[test]
fn test_sample_from_i24_be_roundtrip() {
    // 4096 in 24-bit BE: 0x00 0x10 0x00
    let sample = Sample::from_i24_be([0x00, 0x10, 0x00]);
    assert_eq!(sample, Sample(4096));

    // -1 in 24-bit BE
    let neg = Sample::from_i24_be([0xFF, 0xFF, 0xFF]);
    assert_eq!(neg, Sample(-1));
}

#[test]
fn test_sample_from_i24_be_boundary_values() {
    // Max 24-bit: 0x7FFFFF = 8388607
    let max_sample = Sample::from_i24_be([0x7F, 0xFF, 0xFF]);
    assert_eq!(max_sample, Sample::MAX);

    // Min 24-bit: 0x800000 = -8388608
    let min_sample = Sample::from_i24_be([0x80, 0x00, 0x00]);
    assert_eq!(min_sample, Sample::MIN);

    // Zero
    let zero = Sample::from_i24_be([0x00, 0x00, 0x00]);
    assert_eq!(zero, Sample::ZERO);
}

#[test]
fn test_sample_clamp_out_of_range() {
    let over_max = Sample(10_000_000);
    assert_eq!(over_max.clamp(), Sample::MAX);

    let under_min = Sample(-10_000_000);
    assert_eq!(under_min.clamp(), Sample::MIN);
}

#[test]
fn test_sample_clamp_within_range() {
    let in_range = Sample(42);
    assert_eq!(in_range.clamp(), Sample(42));

    assert_eq!(Sample::ZERO.clamp(), Sample::ZERO);
    assert_eq!(Sample::MAX.clamp(), Sample::MAX);
    assert_eq!(Sample::MIN.clamp(), Sample::MIN);
}

#[test]
fn test_sample_to_f32_range() {
    // MIN should map to exactly -1.0
    assert_eq!(Sample::MIN.to_f32(), -1.0);
    // ZERO should map to 0.0
    assert_eq!(Sample::ZERO.to_f32(), 0.0);
    // MAX should be close to but not exceed 1.0
    let max_f32 = Sample::MAX.to_f32();
    assert!(max_f32 > 0.999 && max_f32 < 1.0);
}

#[test]
fn test_sample_to_f32_typical_values() {
    // Half of max should be ~0.5
    let half = Sample(8_388_608 / 2).to_f32();
    assert!((half - 0.5).abs() < 0.001);

    // Negative half should be ~-0.5
    let neg_half = Sample(-8_388_608 / 2).to_f32();
    assert!((neg_half - (-0.5)).abs() < 0.001);

    // Small positive value from i16 conversion
    let from_i16 = Sample::from_i16(1000);
    let f = from_i16.to_f32();
    assert!(f > 0.0 && f < 1.0);
}

#[test]
fn test_audio_format_duration_us_rounds_half_up() {
    // 1 frame at 48 kHz is 20.833 µs — the implementation adds `rate / 2`
    // before dividing by `rate` so that duration rounds to the nearest whole
    // microsecond (21) instead of truncating to 20. This keeps duration_us
    // consistent with the remainder-tracking cursor advance in synced_player.
    let format = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 48_000,
        channels: 1,
        bit_depth: 24,
        codec_header: None,
    };
    assert_eq!(
        format.duration_us(1),
        21,
        "48 kHz / 1 frame should round to 21 µs"
    );
    // 1000 frames at 48 kHz is 20833.33 µs → rounds to 20833.
    assert_eq!(format.duration_us(1000), 20_833);
    // Exact divisions stay exact.
    assert_eq!(format.duration_us(48_000), 1_000_000);
}

#[test]
fn test_audio_format_duration_us_accounts_for_channels() {
    // `num_samples` is the *total* interleaved sample count. Stereo halves
    // the frame count, so duration should be half of what a mono format of
    // the same rate reports.
    let stereo = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 48_000,
        channels: 2,
        bit_depth: 24,
        codec_header: None,
    };
    // 96 samples stereo = 48 frames at 48 kHz = 1000 µs exactly.
    assert_eq!(stereo.duration_us(96), 1_000);
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
