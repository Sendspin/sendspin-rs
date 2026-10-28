use sendspin::audio::{AudioBuffer, AudioWork, PlaybackControl};

fn commit_into_other_session(
    work: AudioWork,
    _other: &PlaybackControl,
    buffer: AudioBuffer,
) {
    let _ = work.commit_for(_other, buffer);
}

fn main() {}
