#[path = "../../src/generation.rs"]
mod generation;

use generation::{PlaybackResetGeneration, StreamGeneration};

fn main() {
    let stream = StreamGeneration::initial();
    let next_stream = stream.checked_advance().unwrap();
    assert!(next_stream > stream);

    let playback = PlaybackResetGeneration::initial();
    assert_eq!(playback.advance(), PlaybackResetGeneration::initial().advance());
}
