use sendspin::audio::{AudioBuffer, AudioFormat, Codec, Sample};
use sendspin::scheduler::AudioScheduler;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn test_scheduler_creation() {
    let scheduler = AudioScheduler::new();
    assert!(scheduler.is_empty());
}

#[test]
fn test_scheduler_schedule_and_ready() {
    let scheduler = AudioScheduler::new();

    let format = AudioFormat {
        codec: Codec::Pcm,
        sample_rate: 48000,
        channels: 2,
        bit_depth: 24,
        codec_header: None,
    };

    let samples = vec![Sample::ZERO; 960];
    let buffer = AudioBuffer {
        timestamp: 0,
        play_at: Instant::now() + Duration::from_millis(10),
        samples: Arc::from(samples.into_boxed_slice()),
        format,
    };

    scheduler.schedule(buffer);
    assert!(!scheduler.is_empty());

    // Buffer shouldn't be ready yet (10ms in future)
    assert!(scheduler.next_ready().is_none());

    // Sleep and check if ready
    std::thread::sleep(Duration::from_millis(15));
    let ready = scheduler.next_ready();
    assert!(ready.is_some());
}
