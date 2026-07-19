// ABOUTME: Binary audio frame encoding for the server role (the encode-side
// ABOUTME: mirror of protocol::client::AudioChunk::from_bytes, which only parses).

use crate::protocol::client::binary_types;

/// Build a binary WebSocket frame carrying one player audio chunk:
/// `[type_id: u8][timestamp: i64 big-endian, µs][payload]`.
///
/// A client never sends audio, so [`crate::protocol::client::AudioChunk`] only
/// ever parses this layout — the server role needs the builder that side never
/// required. `timestamp_us` is the intended playback time in the server's
/// clock domain; the receiving client converts it to its own clock domain
/// using the offset/drift it tracks from `server/time` replies.
pub fn encode_audio_frame(timestamp_us: i64, payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(9 + payload.len());
    frame.push(binary_types::PLAYER_AUDIO);
    frame.extend_from_slice(&timestamp_us.to_be_bytes());
    frame.extend_from_slice(payload);
    frame
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::client::AudioChunk;

    #[test]
    fn round_trips_through_the_client_side_parser() {
        let payload = [1u8, 2, 3, 4, 5, 255, 0];
        let frame = encode_audio_frame(123_456_789, &payload);
        let chunk = AudioChunk::from_bytes(&frame).expect("client parser must accept our frame");
        assert_eq!(chunk.timestamp, 123_456_789);
        assert_eq!(&*chunk.data, &payload[..]);
    }

    #[test]
    fn negative_timestamps_round_trip() {
        // Clock domains are implementation-defined epochs (see sync::raw_clock::Clock),
        // so a timestamp can legitimately be negative relative to process start.
        let frame = encode_audio_frame(-42, &[9, 9]);
        let chunk = AudioChunk::from_bytes(&frame).unwrap();
        assert_eq!(chunk.timestamp, -42);
    }
}
