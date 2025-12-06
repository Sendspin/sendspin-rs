use sendspin::protocol::client::{
    binary_types, ArtworkChunk, AudioChunk, BinaryFrame, VisualizerChunk,
};

// =============================================================================
// Binary Type Constants Tests
// =============================================================================

#[test]
fn test_binary_type_constants() {
    assert_eq!(binary_types::PLAYER_AUDIO, 0x04);
    assert_eq!(binary_types::ARTWORK_CHANNEL_0, 0x08);
    assert_eq!(binary_types::ARTWORK_CHANNEL_1, 0x09);
    assert_eq!(binary_types::ARTWORK_CHANNEL_2, 0x0A);
    assert_eq!(binary_types::ARTWORK_CHANNEL_3, 0x0B);
    assert_eq!(binary_types::VISUALIZER, 0x10);
}

#[test]
fn test_is_artwork() {
    assert!(!binary_types::is_artwork(0x04)); // Audio
    assert!(!binary_types::is_artwork(0x07)); // Reserved
    assert!(binary_types::is_artwork(0x08)); // Artwork channel 0
    assert!(binary_types::is_artwork(0x09)); // Artwork channel 1
    assert!(binary_types::is_artwork(0x0A)); // Artwork channel 2
    assert!(binary_types::is_artwork(0x0B)); // Artwork channel 3
    assert!(!binary_types::is_artwork(0x0C)); // Out of range
    assert!(!binary_types::is_artwork(0x10)); // Visualizer
}

#[test]
fn test_artwork_channel() {
    assert_eq!(binary_types::artwork_channel(0x08), Some(0));
    assert_eq!(binary_types::artwork_channel(0x09), Some(1));
    assert_eq!(binary_types::artwork_channel(0x0A), Some(2));
    assert_eq!(binary_types::artwork_channel(0x0B), Some(3));
    assert_eq!(binary_types::artwork_channel(0x04), None); // Audio
    assert_eq!(binary_types::artwork_channel(0x10), None); // Visualizer
}

// =============================================================================
// Audio Chunk Tests
// =============================================================================

#[test]
fn test_audio_chunk_parsing() {
    // Type 4, timestamp 1000000 (big-endian), some audio data
    let frame: Vec<u8> = vec![
        0x04, // Type: player audio
        0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x42, 0x40, // Timestamp: 1000000
        0xDE, 0xAD, 0xBE, 0xEF, // Audio data
    ];

    let chunk = AudioChunk::from_bytes(&frame).unwrap();
    assert_eq!(chunk.timestamp, 1000000);
    assert_eq!(chunk.data.len(), 4);
    assert_eq!(&*chunk.data, &[0xDE, 0xAD, 0xBE, 0xEF]);
}

#[test]
fn test_audio_chunk_wrong_type() {
    let frame: Vec<u8> = vec![
        0x08, // Wrong type (artwork)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, // Timestamp
        0x00, // Data
    ];

    let result = AudioChunk::from_bytes(&frame);
    assert!(result.is_err());
}

#[test]
fn test_audio_chunk_too_short() {
    let frame: Vec<u8> = vec![0x04, 0x00, 0x00]; // Only 3 bytes
    let result = AudioChunk::from_bytes(&frame);
    assert!(result.is_err());
}

// =============================================================================
// Artwork Chunk Tests
// =============================================================================

#[test]
fn test_artwork_chunk_channel_0() {
    let frame: Vec<u8> = vec![
        0x08, // Type: artwork channel 0
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xE8, // Timestamp: 1000
        0xFF, 0xD8, 0xFF, 0xE0, // JPEG magic bytes
    ];

    let chunk = ArtworkChunk::from_bytes(&frame).unwrap();
    assert_eq!(chunk.channel, 0);
    assert_eq!(chunk.timestamp, 1000);
    assert_eq!(chunk.data.len(), 4);
    assert!(!chunk.is_clear());
}

#[test]
fn test_artwork_chunk_channel_3() {
    let frame: Vec<u8> = vec![
        0x0B, // Type: artwork channel 3
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0xD0, // Timestamp: 2000
        0x89, 0x50, 0x4E, 0x47, // PNG magic bytes
    ];

    let chunk = ArtworkChunk::from_bytes(&frame).unwrap();
    assert_eq!(chunk.channel, 3);
    assert_eq!(chunk.timestamp, 2000);
}

#[test]
fn test_artwork_chunk_clear() {
    // Empty payload = clear artwork
    let frame: Vec<u8> = vec![
        0x09, // Type: artwork channel 1
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Timestamp: 0
        // No data - clear command
    ];

    let chunk = ArtworkChunk::from_bytes(&frame).unwrap();
    assert_eq!(chunk.channel, 1);
    assert!(chunk.is_clear());
    assert!(chunk.data.is_empty());
}

#[test]
fn test_artwork_chunk_wrong_type() {
    let frame: Vec<u8> = vec![
        0x04, // Wrong type (audio)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x00,
    ];

    let result = ArtworkChunk::from_bytes(&frame);
    assert!(result.is_err());
}

// =============================================================================
// Visualizer Chunk Tests
// =============================================================================

#[test]
fn test_visualizer_chunk_parsing() {
    let frame: Vec<u8> = vec![
        0x10, // Type: visualizer
        0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x86, 0xA0, // Timestamp: 100000
        0x01, 0x02, 0x03, 0x04, 0x05, // FFT data
    ];

    let chunk = VisualizerChunk::from_bytes(&frame).unwrap();
    assert_eq!(chunk.timestamp, 100000);
    assert_eq!(chunk.data.len(), 5);
}

#[test]
fn test_visualizer_chunk_wrong_type() {
    let frame: Vec<u8> = vec![
        0x04, // Wrong type
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x00,
    ];

    let result = VisualizerChunk::from_bytes(&frame);
    assert!(result.is_err());
}

// =============================================================================
// Binary Frame Dispatch Tests
// =============================================================================

#[test]
fn test_binary_frame_audio() {
    let frame: Vec<u8> = vec![
        0x04, // Audio
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0xAB, 0xCD,
    ];

    match BinaryFrame::from_bytes(&frame).unwrap() {
        BinaryFrame::Audio(chunk) => {
            assert_eq!(chunk.timestamp, 1);
        }
        _ => panic!("Expected Audio frame"),
    }
}

#[test]
fn test_binary_frame_artwork() {
    let frame: Vec<u8> = vec![
        0x0A, // Artwork channel 2
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
        0x12, 0x34,
    ];

    match BinaryFrame::from_bytes(&frame).unwrap() {
        BinaryFrame::Artwork(chunk) => {
            assert_eq!(chunk.channel, 2);
            assert_eq!(chunk.timestamp, 2);
        }
        _ => panic!("Expected Artwork frame"),
    }
}

#[test]
fn test_binary_frame_visualizer() {
    let frame: Vec<u8> = vec![
        0x10, // Visualizer
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
        0x56, 0x78,
    ];

    match BinaryFrame::from_bytes(&frame).unwrap() {
        BinaryFrame::Visualizer(chunk) => {
            assert_eq!(chunk.timestamp, 3);
        }
        _ => panic!("Expected Visualizer frame"),
    }
}

#[test]
fn test_binary_frame_unknown() {
    let frame: Vec<u8> = vec![
        0xFF, // Unknown type
        0x00, 0x01, 0x02,
    ];

    match BinaryFrame::from_bytes(&frame).unwrap() {
        BinaryFrame::Unknown { type_id, data } => {
            assert_eq!(type_id, 0xFF);
            assert_eq!(data.len(), 3);
        }
        _ => panic!("Expected Unknown frame"),
    }
}

#[test]
fn test_binary_frame_empty() {
    let frame: Vec<u8> = vec![];
    let result = BinaryFrame::from_bytes(&frame);
    assert!(result.is_err());
}
