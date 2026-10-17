#[path = "../../src/generation.rs"]
mod generation;

use generation::{PlaybackResetGeneration, StreamGeneration};

fn main() {
    let stream = StreamGeneration::initial();
    let _wrong: PlaybackResetGeneration = stream;
}
