// ABOUTME: Synced audio player with drift correction
// ABOUTME: Uses DAC callback timestamps to drop/insert frames for alignment

use crate::audio::gain::{GainControl, GainRamp};
use crate::audio::sync_correction::{CorrectionPlanner, CorrectionSchedule};
use crate::audio::{AudioBuffer, AudioFormat, Sample};
use crate::error::Error;
use crate::sync::ClockSync;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Callback for post-processing audio samples before output.
///
/// Receives `&mut [f32]` (interleaved, after gain is applied).
///
/// The callback is invoked on **every** audio callback, including during
/// pre-start silence when the buffer is all zeros. This allows consumers
/// (e.g. VU meters) to observe the silence rather than missing callbacks.
///
/// # Thread Safety
///
/// This closure runs on the **audio callback thread**. It must:
/// - Not block (no locks, I/O, or sleeping)
/// - Not allocate (no `Vec::push`, `Box::new`, etc.)
/// - Not panic (would abort the audio thread)
///
/// # Why `Box<dyn>`?
///
/// Using dynamic dispatch (`Box<dyn FnMut>`) keeps `SyncedPlayer` a concrete,
/// non-generic type. This simplifies storage, trait object compatibility, and
/// downstream usage at the cost of one vtable indirect call per audio callback
/// (~1 ns vs the ~200 us callback budget).
pub type ProcessCallback = Box<dyn FnMut(&mut [f32]) + Send + 'static>;

struct PlaybackQueue {
    queue: VecDeque<AudioBuffer>,
    current: Option<AudioBuffer>,
    index: usize,
    /// Current playback position in **server-time microseconds**. Periodically
    /// reanchored to the server's clock during clock-sync correction, so this
    /// represents "what server timestamp is playing right now", not how much
    /// audio content has been consumed.
    cursor_us: i64,
    cursor_remainder: i64,
    initialized: bool,
    generation: u64,
    force_reanchor: bool,
}

impl PlaybackQueue {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            current: None,
            index: 0,
            cursor_us: 0,
            cursor_remainder: 0,
            initialized: false,
            generation: 0,
            force_reanchor: true,
        }
    }

    fn clear(&mut self) {
        self.queue.clear();
        self.current = None;
        self.index = 0;
        self.cursor_us = 0;
        self.cursor_remainder = 0;
        self.initialized = false;
        self.generation = self.generation.wrapping_add(1);
        self.force_reanchor = true;
    }

    fn push(&mut self, buffer: AudioBuffer) {
        // Initialize the cursor from the first enqueued buffer so the audio
        // callback can see a valid cursor_us before it starts reading. Without
        // this, the callback's pre-start gate can't evaluate timestamps and
        // outputs silence indefinitely. Use the minimum timestamp seen so far
        // since buffers may arrive out of order.
        if !self.initialized {
            self.cursor_us = buffer.timestamp;
            self.cursor_remainder = 0;
            self.initialized = true;
        } else if buffer.timestamp < self.cursor_us {
            self.cursor_us = buffer.timestamp;
            self.cursor_remainder = 0;
        }

        // When the server rebases its timeline backward (e.g. after event loop
        // starvation), new chunks arrive with timestamps that overlap chunks
        // already in the queue. Remove all overlapping buffers to prevent
        // duplicate audio that causes audible stuttering (~500ms of audio
        // played twice). The server will send fresh buffers for any gaps.
        //
        // Two buffers overlap when their time ranges intersect:
        //   start_a < end_b && start_b < end_a
        // This works for any chunk size (the sendspin spec allows arbitrarily
        // small chunks), unlike the previous fixed-threshold approach.
        let new_end = buffer.timestamp + buffer.duration_us();
        self.queue.retain(|b| {
            let existing_end = b.timestamp + b.duration_us();
            !(buffer.timestamp < existing_end && b.timestamp < new_end)
        });

        let pos = self
            .queue
            .iter()
            .position(|b| b.timestamp > buffer.timestamp);
        if let Some(pos) = pos {
            self.queue.insert(pos, buffer);
        } else {
            self.queue.push_back(buffer);
        }
    }

    fn next_frame(&mut self, channels: usize, sample_rate: u32) -> Option<&[Sample]> {
        if self.current.is_none() || self.index + channels > self.current.as_ref()?.samples.len() {
            // Drop stale buffers that are entirely before the cursor.
            if self.initialized {
                while let Some(front) = self.queue.front() {
                    if front.timestamp + front.duration_us() < self.cursor_us {
                        let _ = self.queue.pop_front();
                        continue;
                    }
                    break;
                }
            }
            self.current = self.queue.pop_front();
            self.index = 0;

            // Skip past samples that are behind the cursor. This handles buffers
            // that partially overlap with the current playback position, e.g. from
            // backward timestamp jumps during server timeline rebases. Without this,
            // playing from the start of such a buffer repeats audio the cursor has
            // already passed, causing an audible stutter.
            if self.initialized {
                if let Some(ref current) = self.current {
                    if current.timestamp < self.cursor_us {
                        let skip_us = self.cursor_us - current.timestamp;
                        let skip_frames = (skip_us * sample_rate as i64 / 1_000_000) as usize;
                        if skip_frames > 0 {
                            self.index = (skip_frames * channels).min(current.samples.len());
                        }
                    }
                }
            }
        }

        if !self.initialized {
            if let Some(current) = self.current.as_ref() {
                self.cursor_us = current.timestamp;
                self.cursor_remainder = 0;
                self.initialized = true;
            }
        }

        // Bail before advancing cursor/index when the queue is empty.
        // Without this the cursor races ahead during underruns, causing
        // the stale-buffer-dropping logic to discard valid buffers when
        // audio resumes.
        self.current.as_ref()?;

        let start = self.index;
        let end = self.index + channels;
        self.index = end;
        self.advance_cursor(sample_rate);

        Some(&self.current.as_ref()?.samples[start..end])
    }

    fn advance_cursor(&mut self, sample_rate: u32) {
        self.cursor_remainder += 1_000_000;
        let advance = self.cursor_remainder / sample_rate as i64;
        self.cursor_remainder %= sample_rate as i64;
        self.cursor_us += advance;
    }
}

/// Bundles gain and post-processing parameters for the data callback.
struct CallbackConfig {
    gain_control: GainControl,
    process_callback: Option<ProcessCallback>,
}

/// Synced audio output with drift correction.
pub struct SyncedPlayer {
    format: AudioFormat,
    _stream: Stream,
    queue: Arc<Mutex<PlaybackQueue>>,
    /// Last error from the audio stream callback, if any.
    last_error: Arc<Mutex<Option<String>>>,
    gain: GainControl,
}

impl SyncedPlayer {
    /// Create a new synced player using the provided clock sync and optional device.
    ///
    /// The player starts at `volume` (0-100) and `muted` state. These are
    /// applied immediately — the first audio callback uses the correct gain
    /// with no ramp from a default value.
    pub fn new(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        device: Option<Device>,
        volume: u8,
        muted: bool,
    ) -> Result<Self, Error> {
        Self::build(format, clock_sync, device, None, volume, muted)
    }

    /// Create a player with a process callback for post-gain audio processing.
    ///
    /// The callback receives samples **after** gain/mute processing has been
    /// applied. See [`ProcessCallback`] for thread-safety requirements.
    ///
    /// # Example (requires physical audio hardware to run)
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use parking_lot::Mutex;
    /// # use sendspin::audio::{AudioFormat, Codec, SyncedPlayer};
    /// # use sendspin::sync::ClockSync;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let format = AudioFormat {
    ///     codec: Codec::Pcm,
    ///     sample_rate: 48_000,
    ///     channels: 2,
    ///     bit_depth: 24,
    ///     codec_header: None,
    /// };
    /// let clock_sync = Arc::new(Mutex::new(ClockSync::new()));
    /// let player = SyncedPlayer::with_process_callback(
    ///     format, clock_sync, None,
    ///     100, false,
    ///     Box::new(|data| { /* e.g. feed a VU meter or visualizer */ }),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_process_callback(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        device: Option<Device>,
        volume: u8,
        muted: bool,
        callback: ProcessCallback,
    ) -> Result<Self, Error> {
        Self::build(format, clock_sync, device, Some(callback), volume, muted)
    }

    fn build(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        device: Option<Device>,
        process_callback: Option<ProcessCallback>,
        volume: u8,
        muted: bool,
    ) -> Result<Self, Error> {
        if format.channels == 0 {
            return Err(Error::Output("channels must be > 0".to_string()));
        }
        let host = cpal::default_host();
        let device = match device {
            Some(device) => device,
            None => host
                .default_output_device()
                .ok_or_else(|| Error::Output("No output device available".to_string()))?,
        };

        let config = StreamConfig {
            channels: format.channels as u16,
            sample_rate: cpal::SampleRate(format.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        let queue_clone = Arc::clone(&queue);
        let format_clone = format.clone();
        let last_error = Arc::new(Mutex::new(None));
        let gain = GainControl::new(volume, muted);

        let cb_config = CallbackConfig {
            gain_control: gain.clone(),
            process_callback,
        };
        let error_clone = Arc::clone(&last_error);

        let stream = Self::build_stream(
            &device,
            &config,
            queue_clone,
            clock_sync,
            format_clone,
            cb_config,
            error_clone,
        )?;
        stream.play().map_err(|e| Error::Output(e.to_string()))?;

        Ok(Self {
            format,
            _stream: stream,
            queue,
            last_error,
            gain,
        })
    }

    /// Enqueue a decoded buffer for playback.
    ///
    /// Scheduling uses `buffer.timestamp` (server time in microseconds) for
    /// drift-corrected playback. The `play_at` field is ignored.
    pub fn enqueue(&self, buffer: AudioBuffer) {
        self.queue.lock().push(buffer);
    }

    /// Clear queued audio and reset playback state.
    pub fn clear(&self) {
        self.queue.lock().clear();
    }

    /// Return the configured audio format.
    pub fn format(&self) -> &AudioFormat {
        &self.format
    }

    /// Check if the audio stream has encountered an error.
    ///
    /// Returns the error message if one occurred, clearing it in the process.
    pub fn take_error(&self) -> Option<String> {
        self.last_error.lock().take()
    }

    /// Check if the audio stream has an error without clearing it.
    pub fn has_error(&self) -> bool {
        self.last_error.lock().is_some()
    }

    /// Get a reference to the volume/mute control.
    ///
    /// Call `.clone()` if you need an owned handle to share across threads
    /// (cloning is cheap — single `Arc` increment, no data copy).
    pub fn gain_control(&self) -> &GainControl {
        &self.gain
    }

    // -- Volume/mute convenience methods --
    //
    // These promote the most common operations for ergonomics in simple
    // use-cases. For full control, use `gain_control()` directly.

    /// Current volume as 0-100.
    pub fn volume(&self) -> u8 {
        self.gain.volume()
    }

    /// Whether playback is currently muted.
    pub fn is_muted(&self) -> bool {
        self.gain.is_muted()
    }

    /// Set playback volume (0-100).
    pub fn set_volume(&self, volume: u8) {
        self.gain.set_volume(volume);
    }

    /// Set mute state.
    pub fn set_mute(&self, muted: bool) {
        self.gain.set_mute(muted);
    }

    fn build_stream(
        device: &Device,
        config: &StreamConfig,
        queue: Arc<Mutex<PlaybackQueue>>,
        clock_sync: Arc<Mutex<ClockSync>>,
        format: AudioFormat,
        mut cb_config: CallbackConfig,
        error_sink: Arc<Mutex<Option<String>>>,
    ) -> Result<Stream, Error> {
        let channels = format.channels as usize;
        let sample_rate = format.sample_rate;
        let planner = CorrectionPlanner::new();
        let mut last_frame = vec![Sample::ZERO; channels];
        let mut schedule = CorrectionSchedule::default();
        let mut insert_counter = 0u32;
        let mut drop_counter = 0u32;
        let mut started = false;
        let mut last_generation = 0u64;
        let initial_gain = cb_config.gain_control.target_gain();
        let mut gain_ramp = GainRamp::new(sample_rate, initial_gain);

        let stream = device
            .build_output_stream(
                config,
                move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    // Read all queue state in a single lock to avoid
                    // a TOCTOU window between generation and cursor reads.
                    // After clear(), force_reanchor is true but cursor_us
                    // is None (initialized=false), so the reanchor block
                    // is normally skipped and the flag persists. If clear()
                    // races with an in-flight reanchor, the flag can be
                    // cleared early; next_frame() re-anchors from the
                    // first buffer's timestamp in that case.
                    let (generation, cursor_us, force_reanchor) = {
                        let queue = queue.lock();
                        let cursor = if queue.initialized {
                            Some(queue.cursor_us)
                        } else {
                            None
                        };
                        (queue.generation, cursor, queue.force_reanchor)
                    };
                    if generation != last_generation {
                        last_generation = generation;
                        started = false;
                        schedule = CorrectionSchedule::default();
                        insert_counter = 0;
                        drop_counter = 0;
                        for sample in last_frame.iter_mut() {
                            *sample = Sample::ZERO;
                        }
                    }

                    let callback_instant = Instant::now();
                    let ts = info.timestamp();
                    let playback_delta = ts
                        .playback
                        .duration_since(&ts.callback)
                        .unwrap_or(Duration::ZERO);
                    let playback_instant = callback_instant + playback_delta;

                    // try_lock: skip sync if contended rather than blocking
                    // the audio thread. force_reanchor is sticky in the
                    // queue, so it will be retried on the next callback.
                    if let (Some(cursor_us), Some(sync)) = (cursor_us, clock_sync.try_lock()) {
                        if let Some(expected_instant) = sync.server_to_local_instant(cursor_us) {
                            let early_window = Duration::from_millis(1);
                            if !started && playback_instant + early_window < expected_instant {
                                for sample in data.iter_mut() {
                                    *sample = 0.0;
                                }
                                let target = cb_config.gain_control.target_gain();
                                let frames = data.len() / channels;
                                gain_ramp.advance(frames, target);
                                if let Some(ref mut cb) = cb_config.process_callback {
                                    cb(data);
                                }
                                return;
                            }
                            started = true;

                            let error_us = if playback_instant >= expected_instant {
                                playback_instant
                                    .duration_since(expected_instant)
                                    .as_micros() as i64
                            } else {
                                -(expected_instant
                                    .duration_since(playback_instant)
                                    .as_micros() as i64)
                            };
                            let new_schedule =
                                planner.plan(error_us, sample_rate, schedule.is_correcting());
                            if new_schedule != schedule {
                                if new_schedule.is_correcting() != schedule.is_correcting() {
                                    if new_schedule.is_correcting() {
                                        log::debug!(
                                            "Sync correction engaged: \
                                             error={:.1}ms, insert_every={}, drop_every={}",
                                            error_us as f64 / 1000.0,
                                            new_schedule.insert_every_n_frames,
                                            new_schedule.drop_every_n_frames,
                                        );
                                    } else {
                                        log::debug!(
                                            "Sync correction disengaged: \
                                             error={:.1}ms",
                                            error_us as f64 / 1000.0,
                                        );
                                    }
                                }
                                schedule = new_schedule;
                                insert_counter = schedule.insert_every_n_frames;
                                drop_counter = schedule.drop_every_n_frames;
                            }

                            if schedule.reanchor || force_reanchor {
                                if let Some(client_micros) =
                                    sync.instant_to_client_micros(playback_instant)
                                {
                                    if let Some(server_time) =
                                        sync.client_to_server_micros(client_micros)
                                    {
                                        let mut queue = queue.lock();
                                        queue.cursor_us = server_time;
                                        queue.cursor_remainder = 0;
                                        queue.force_reanchor = false;
                                    }
                                }
                                schedule = CorrectionSchedule::default();
                                insert_counter = 0;
                                drop_counter = 0;
                            }
                        }
                    }

                    // If playback hasn't started yet (clock sync not converged,
                    // lock contention, or pre-start gate active), output silence.
                    // Audio data stays in the ring buffer for when sync converges
                    // and reanchor positions the cursor correctly.
                    if !started {
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                        let target = cb_config.gain_control.target_gain();
                        let frames = data.len() / channels;
                        gain_ramp.advance(frames, target);
                        if let Some(ref mut cb) = cb_config.process_callback {
                            cb(data);
                        }
                        return;
                    }

                    {
                        let mut queue = queue.lock();
                        let frames = data.len() / channels;
                        let mut out_index = 0;

                        for _ in 0..frames {
                            if schedule.drop_every_n_frames > 0 {
                                drop_counter = drop_counter.saturating_sub(1);
                                if drop_counter == 0 {
                                    // Discard one frame to catch up
                                    let _ = queue.next_frame(channels, sample_rate);
                                    drop_counter = schedule.drop_every_n_frames;
                                    // Get and output the next frame (don't repeat last_frame)
                                    if let Some(frame) = queue.next_frame(channels, sample_rate) {
                                        last_frame.copy_from_slice(frame);
                                        for sample in frame {
                                            data[out_index] = sample.to_f32();
                                            out_index += 1;
                                        }
                                    } else {
                                        for sample in &last_frame {
                                            data[out_index] = sample.to_f32();
                                            out_index += 1;
                                        }
                                    }
                                    continue;
                                }
                            }

                            if schedule.insert_every_n_frames > 0 {
                                insert_counter = insert_counter.saturating_sub(1);
                                if insert_counter == 0 {
                                    insert_counter = schedule.insert_every_n_frames;
                                    for sample in &last_frame {
                                        data[out_index] = sample.to_f32();
                                        out_index += 1;
                                    }
                                    continue;
                                }
                            }

                            if let Some(frame) = queue.next_frame(channels, sample_rate) {
                                last_frame.copy_from_slice(frame);
                                for sample in frame {
                                    data[out_index] = sample.to_f32();
                                    out_index += 1;
                                }
                            } else {
                                for _ in 0..channels {
                                    data[out_index] = 0.0;
                                    out_index += 1;
                                }
                            }
                        }
                    } // queue lock dropped before user callback

                    // Apply gain with per-frame ramping
                    let target = cb_config.gain_control.target_gain();
                    gain_ramp.apply(data, channels, target);

                    if let Some(ref mut cb) = cb_config.process_callback {
                        cb(data);
                    }
                },
                move |err| {
                    eprintln!("Audio stream error: {}", err);
                    *error_sink.lock() = Some(err.to_string());
                },
                None,
            )
            .map_err(|e| Error::Output(e.to_string()))?;

        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    // Note: SyncedPlayer's convenience methods (volume, is_muted, set_volume,
    // set_mute, gain_control) delegate to GainControl which is thoroughly tested
    // in gain.rs. Wiring tests require a real audio device (cpal Stream) and
    // cannot run in CI.

    use super::PlaybackQueue;
    use crate::audio::{AudioBuffer, AudioFormat, Codec, Sample};
    use std::sync::Arc;
    use std::time::Instant;

    #[test]
    fn test_queue_clear_bumps_generation() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };
        let samples = vec![Sample::ZERO; 96];
        queue.push(AudioBuffer {
            timestamp: 1234,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        let before = queue.generation;
        queue.clear();
        assert_ne!(queue.generation, before);
        assert!(queue.queue.is_empty());
        assert!(!queue.initialized);
    }

    #[test]
    fn test_queue_drops_stale_buffers() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };

        // Use distinct sample values so we can verify which buffer was returned.
        // 4800 stereo frames at 48kHz = 100ms per buffer.
        // With cursor at 150ms, the first buffer (ts=0, ends at 100ms) is stale.
        let stale_samples: Vec<Sample> = (0..4800 * 2).map(|_| Sample(111)).collect();
        let fresh_samples: Vec<Sample> = (0..4800 * 2).map(|_| Sample(222)).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(stale_samples.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 200_000,
            play_at: Instant::now(),
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        queue.cursor_us = 150_000;
        queue.initialized = true;

        // Copy the frame data so we can release the mutable borrow on queue.
        let frame_data: Vec<Sample> = queue
            .next_frame(2, 48_000)
            .expect("expected a frame")
            .to_vec();
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);
        // Verify we got the fresh buffer's data, not the stale one
        assert_eq!(frame_data[0], Sample(222));
        assert_eq!(frame_data[1], Sample(222));
    }

    #[test]
    fn test_queue_push_sorts_by_timestamp() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 1,
            bit_depth: 24,
            codec_header: None,
        };

        // Push out of order: 300, 100, 200
        for ts in [300_000i64, 100_000, 200_000] {
            let samples = vec![Sample(ts as i32); 48]; // 1ms of mono
            queue.push(AudioBuffer {
                timestamp: ts,
                play_at: Instant::now(),
                samples: Arc::from(samples.into_boxed_slice()),
                format: format.clone(),
            });
        }

        // Drain and verify sorted order
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 100_000);

        // Exhaust the first buffer (48 frames)
        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        // Next frame should come from the second buffer
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);

        // Exhaust the second buffer
        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        // Next frame should come from the third buffer
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 300_000);
    }

    #[test]
    fn test_queue_cursor_advances_correctly() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };

        // 480 stereo frames = 10ms at 48kHz
        let num_frames = 480;
        let samples = vec![Sample::ZERO; num_frames * 2];
        let start_ts = 1_000_000i64; // 1 second
        queue.push(AudioBuffer {
            timestamp: start_ts,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        // Consume all frames
        for _ in 0..num_frames {
            let _ = queue.next_frame(2, 48_000);
        }

        // 480 frames at 48kHz = 10,000us = 10ms
        let expected_end = start_ts + 10_000;
        assert_eq!(
            queue.cursor_us,
            expected_end,
            "cursor should advance by exactly 10ms (10000us), got delta={}",
            queue.cursor_us - start_ts
        );
    }

    #[test]
    fn test_cursor_does_not_advance_during_underrun() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };

        // Push one buffer to initialize the cursor
        let samples = vec![Sample::ZERO; 480 * 2]; // 10ms stereo
        let start_ts = 1_000_000i64;
        queue.push(AudioBuffer {
            timestamp: start_ts,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format: format.clone(),
        });

        // Consume all frames
        for _ in 0..480 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        let cursor_after_drain = queue.cursor_us;

        // Queue is now empty. Calling next_frame should return None
        // and NOT advance the cursor.
        for _ in 0..1000 {
            assert!(queue.next_frame(2, 48_000).is_none());
        }
        assert_eq!(
            queue.cursor_us,
            cursor_after_drain,
            "cursor must not advance during underrun; advanced by {}us",
            queue.cursor_us - cursor_after_drain
        );

        // Push a new buffer after the underrun. It should NOT be
        // dropped as stale — the cursor hasn't raced ahead.
        let fresh_samples: Vec<Sample> = (0..480 * 2).map(|_| Sample(999)).collect();
        queue.push(AudioBuffer {
            timestamp: cursor_after_drain, // starts right where we left off
            play_at: Instant::now(),
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("buffer should not be dropped as stale");
        assert_eq!(
            frame[0],
            Sample(999),
            "should get the fresh buffer, not stale data"
        );
    }

    #[test]
    fn test_push_initializes_cursor_from_first_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };

        assert!(!queue.initialized);
        assert_eq!(queue.cursor_us, 0);

        let samples = vec![Sample::ZERO; 96];
        queue.push(AudioBuffer {
            timestamp: 500_000,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert!(queue.initialized);
        assert_eq!(queue.cursor_us, 500_000);
        assert_eq!(queue.cursor_remainder, 0);
    }

    #[test]
    fn test_push_lowers_cursor_for_earlier_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };
        let samples = vec![Sample::ZERO; 96];

        // First buffer at 500ms
        queue.push(AudioBuffer {
            timestamp: 500_000,
            play_at: Instant::now(),
            samples: Arc::from(samples.clone().into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.cursor_us, 500_000);

        // Second buffer at 200ms -- cursor should move back
        queue.push(AudioBuffer {
            timestamp: 200_000,
            play_at: Instant::now(),
            samples: Arc::from(samples.clone().into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.cursor_us, 200_000);

        // Third buffer at 300ms -- cursor should NOT move (300 > 200)
        queue.push(AudioBuffer {
            timestamp: 300_000,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });
        assert_eq!(queue.cursor_us, 200_000);
    }

    #[test]
    fn test_next_frame_skips_into_overlapping_buffer() {
        // Simulates a backward timestamp jump from a server timeline rebase.
        // Buffer A is 50ms (2400 frames stereo at 48kHz). After consuming A
        // the cursor is at 50ms. Buffer B arrives at 25ms — the skip logic
        // should jump 25ms (1200 frames) into B.
        //
        // B is pushed AFTER A is consumed so dedup doesn't apply (A is no
        // longer in the queue).
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };

        let buf_a: Vec<Sample> = (0..2400 * 2).map(|_| Sample(111)).collect();
        let buf_b: Vec<Sample> = (0..2400 * 2)
            .map(|i| {
                // First half (1200 frames) = 222, second half = 333
                if i < 2400 {
                    Sample(222)
                } else {
                    Sample(333)
                }
            })
            .collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(buf_a.into_boxed_slice()),
            format: format.clone(),
        });

        // Consume all of buffer A (2400 frames). Cursor advances to 50000µs.
        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        assert_eq!(queue.cursor_us, 50_000);

        // Push B after A is consumed — no dedup, tests skip logic only.
        // push() lowers cursor_us to 25000 (min-tracking), so restore it
        // to 50000 to simulate the real scenario where cursor has advanced.
        queue.push(AudioBuffer {
            timestamp: 25_000,
            play_at: Instant::now(),
            samples: Arc::from(buf_b.into_boxed_slice()),
            format,
        });
        queue.cursor_us = 50_000;
        queue.cursor_remainder = 0;

        // Buffer B starts at 25ms but cursor is at 50ms, so 25ms (1200 frames)
        // should be skipped. First returned frame should be Sample(333).
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get a frame from buffer B");
        assert_eq!(
            frame[0],
            Sample(333),
            "expected skip into second half of buffer B (past the overlap), \
             got first half — backward-timestamped audio was replayed"
        );
    }

    #[test]
    fn test_next_frame_no_skip_when_buffer_starts_at_or_after_cursor() {
        // Verify that the skip logic doesn't activate for normal (non-overlapping)
        // buffers — only for buffers that start before the cursor.
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };

        // 2400 frames = 50ms per buffer. Adjacent, non-overlapping.
        let samples_a: Vec<Sample> = (0..2400 * 2).map(|_| Sample(111)).collect();
        let samples_b: Vec<Sample> = (0..2400 * 2).map(|_| Sample(222)).collect();

        // Two consecutive, non-overlapping buffers.
        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 50_000, // starts exactly where A ends
            play_at: Instant::now(),
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Consume all of buffer A.
        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }

        // Buffer B starts at cursor (50000µs) — no skip should occur.
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get first frame of buffer B");
        assert_eq!(
            frame[0],
            Sample(222),
            "buffer B should play from the start (no skip needed)"
        );
    }

    #[test]
    fn test_push_dedup_replaces_overlapping_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };

        // Buffer A: 10ms at ts=0 (480 stereo frames)
        let samples_a: Vec<Sample> = (0..480 * 2).map(|_| Sample(111)).collect();
        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 1);

        // Buffer B: 10ms at ts=5000 (5ms) — overlaps A's range [0, 10000)
        let samples_b: Vec<Sample> = (0..480 * 2).map(|_| Sample(222)).collect();
        queue.push(AudioBuffer {
            timestamp: 5_000,
            play_at: Instant::now(),
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Should replace A, not add a second entry
        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].samples[0], Sample(222));
    }

    #[test]
    fn test_push_dedup_no_false_positive_small_chunks() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };

        // Two adjacent 5ms chunks (240 stereo frames each).
        // Chunk A: [0, 5000), Chunk B: [5000, 10000) — no overlap.
        let samples_a: Vec<Sample> = (0..240 * 2).map(|_| Sample(111)).collect();
        let samples_b: Vec<Sample> = (0..240 * 2).map(|_| Sample(222)).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 5_000,
            play_at: Instant::now(),
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Both should be kept — they're adjacent, not overlapping
        assert_eq!(queue.queue.len(), 2);
        assert_eq!(queue.queue[0].timestamp, 0);
        assert_eq!(queue.queue[1].timestamp, 5_000);
    }

    #[test]
    fn test_push_dedup_removes_all_overlapping() {
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        };

        // Buffer A: 10ms at ts=0 — range [0, 10000)
        let samples_a: Vec<Sample> = (0..480 * 2).map(|_| Sample(111)).collect();
        // Buffer B: 10ms at ts=12000 — range [12000, 22000). No overlap with A.
        let samples_b: Vec<Sample> = (0..480 * 2).map(|_| Sample(222)).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 12_000,
            play_at: Instant::now(),
            samples: Arc::from(samples_b.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 2);

        // Buffer C: 20ms at ts=9000 — range [9000, 29000).
        // Overlaps both A (9000 < 10000 && 0 < 29000) and B (9000 < 22000 && 12000 < 29000).
        // Both stale buffers should be removed — the server will send fresh
        // data for any gaps. Keeping either would cause duplicate audio.
        let samples_c: Vec<Sample> = (0..960 * 2).map(|_| Sample(333)).collect();
        queue.push(AudioBuffer {
            timestamp: 9_000,
            play_at: Instant::now(),
            samples: Arc::from(samples_c.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].timestamp, 9_000);
        assert_eq!(queue.queue[0].samples[0], Sample(333));
    }
}
