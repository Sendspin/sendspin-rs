use sendspin::audio::decode::{Decoder, PcmDecoder};

#[test]
fn test_decode_pcm_16bit() {
    let decoder = PcmDecoder::new(16);

    // 4 samples (8 bytes) of 16-bit PCM
    let data = vec![
        0x00, 0x04, // 1024 in little-endian
        0x00, 0x08, // 2048
        0xFF, 0xFF, // -1
        0x00, 0x00, // 0
    ];

    let samples = decoder.decode(&data).unwrap();

    assert_eq!(samples.len(), 4);
    assert_eq!(samples[0].to_i16(), 1024);
    assert_eq!(samples[1].to_i16(), 2048);
    assert_eq!(samples[2].to_i16(), -1);
    assert_eq!(samples[3].to_i16(), 0);
}

#[test]
fn test_decode_pcm_24bit() {
    let decoder = PcmDecoder::new(24);

    // 2 samples (6 bytes) of 24-bit PCM
    let data = vec![
        0x00, 0x10, 0x00, // 4096 in little-endian 24-bit
        0xFF, 0xFF, 0xFF, // -1 in 24-bit
    ];

    let samples = decoder.decode(&data).unwrap();

    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].0, 4096);
    assert_eq!(samples[1].0, -1);
}
