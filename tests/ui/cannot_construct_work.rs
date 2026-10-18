use sendspin::audio::AudioWork;

fn main() {
    let _ = AudioWork {
        commit: None,
        generation: panic!(),
        chunk: panic!(),
        fence: panic!(),
    };
}
