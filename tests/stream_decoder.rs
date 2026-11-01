use base64::Engine;
use sendspin::audio::decode::{Decoder, StreamDecoder};
use sendspin::audio::{Codec, StreamPlayerConfig};

fn config(
    codec: &str,
    rate: u32,
    channels: u8,
    depth: u8,
    header: Option<String>,
) -> StreamPlayerConfig {
    StreamPlayerConfig {
        codec: codec.into(),
        sample_rate: rate,
        channels,
        bit_depth: depth,
        codec_header: header,
    }
}

fn split_flac(bytes: &[u8]) -> (&[u8], &[u8]) {
    assert_eq!(&bytes[..4], b"fLaC");
    let mut pos = 4;
    loop {
        let last = bytes[pos] & 0x80 != 0;
        let len = u32::from_be_bytes([0, bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        pos += 4 + len;
        if last {
            return bytes.split_at(pos);
        }
    }
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn stream_decoder_is_send_sync_and_preserves_pcm16_format() {
    assert_send_sync::<StreamDecoder>();
    let decoder = StreamDecoder::new(&config("pcm", 44_100, 1, 16, None)).unwrap();
    assert_eq!(decoder.format().codec, Codec::Pcm);
    assert_eq!(decoder.format().sample_rate, 44_100);
    assert_eq!(
        decoder.decode(&[0, 0, 0xFF, 0x7F]).unwrap().as_ref(),
        &[0, 2_147_418_112]
    );
}

#[test]
fn stereo_pcm_partial_frames_are_rejected() {
    for depth in [16, 24] {
        let decoder = StreamDecoder::new(&config("pcm", 44_100, 2, depth, None)).unwrap();
        let bytes = if depth == 16 { vec![0; 6] } else { vec![0; 9] };
        assert!(decoder.decode(&bytes).is_err());
    }
    let decoder = StreamDecoder::new(&config("pcm", 44_100, 1, 24, None)).unwrap();
    assert!(decoder.decode(&[0, 0, 0]).is_ok());
}

#[test]
fn decodes_pcm24_known_samples() {
    let decoder = StreamDecoder::new(&config("pcm", 44_100, 1, 24, None)).unwrap();
    let samples = decoder
        .decode(&[0, 0, 0, 0, 0, 0x80, 0xFF, 0xFF, 0x7F])
        .unwrap();
    assert_eq!(samples.as_ref(), &[0, -2_147_483_648, 2_147_483_392]);
}

#[test]
fn decodes_real_opus_packet() {
    let mut encoder = opus_rs::OpusEncoder::new(48_000, 1, opus_rs::Application::Audio).unwrap();
    encoder.bitrate_bps = 64_000;
    let input = vec![0.0f32; 960];
    let mut packet = vec![0u8; 1275];
    let length = encoder.encode(&input, 960, &mut packet).unwrap();
    let decoder = StreamDecoder::new(&config("opus", 48_000, 1, 16, None)).unwrap();
    assert_eq!(decoder.decode(&packet[..length]).unwrap().len(), 960);
}

#[test]
fn flac_header_and_headerless_decoders_match_fixture() {
    let flac = std::fs::read("tests/data/48k_16bit_stereo.flac").unwrap();
    let (header, frames) = split_flac(&flac);
    let encoded = base64::engine::general_purpose::STANDARD.encode(header);
    let with_header = StreamDecoder::new(&config("flac", 48_000, 2, 16, Some(encoded))).unwrap();
    assert_eq!(with_header.format().codec_header.as_deref(), Some(header));
    let headerless = StreamDecoder::new(&config("flac", 48_000, 2, 16, None)).unwrap();
    assert_eq!(
        with_header.decode(frames).unwrap(),
        headerless.decode(frames).unwrap()
    );
}

#[test]
fn flac_header_mismatch_is_rejected_early() {
    let flac = std::fs::read("tests/data/48k_16bit_stereo.flac").unwrap();
    let (header, _) = split_flac(&flac);
    let encoded = base64::engine::general_purpose::STANDARD.encode(header);
    for (rate, channels, depth) in [(44_100, 2, 16), (48_000, 1, 16), (48_000, 2, 24)] {
        assert!(StreamDecoder::new(&config(
            "flac",
            rate,
            channels,
            depth,
            Some(encoded.clone())
        ))
        .is_err());
    }
}

#[test]
fn flac_headerless_frame_mismatch_is_rejected_on_decode() {
    let flac = std::fs::read("tests/data/48k_16bit_stereo.flac").unwrap();
    let (_, frames) = split_flac(&flac);
    for (rate, channels, depth) in [(44_100, 2, 16), (48_000, 1, 16), (48_000, 2, 24)] {
        let decoder = StreamDecoder::new(&config("flac", rate, channels, depth, None)).unwrap();
        assert!(decoder.decode(frames).is_err());
    }
}

#[test]
fn invalid_stream_configurations_are_rejected() {
    let cases = [
        ("unknown", 48_000, 2, 16, None),
        ("mp3", 48_000, 2, 16, None),
        ("pcm", 0, 2, 16, None),
        ("pcm", 48_000, 0, 16, None),
        ("pcm", 48_000, 2, 8, None),
        ("pcm", 48_000, 2, 32, None),
        ("opus", 44_100, 2, 16, None),
        ("opus", 48_000, 3, 16, None),
        ("flac", 48_000, 9, 16, None),
        ("flac", 48_000, 2, 3, None),
        ("flac", 48_000, 2, 33, None),
        ("flac", 48_000, 2, 16, Some("%%%")),
        ("flac", 48_000, 2, 16, Some("")),
    ];
    for (codec, rate, channels, depth, header) in cases {
        assert!(
            StreamDecoder::new(&config(
                codec,
                rate,
                channels,
                depth,
                header.map(str::to_owned)
            ))
            .is_err(),
            "accepted invalid {codec} configuration"
        );
    }
}
