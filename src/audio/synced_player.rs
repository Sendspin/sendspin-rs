// ABOUTME: Synced audio player with drift correction
// ABOUTME: Uses DAC callback timestamps to drop/insert frames for alignment

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

struct PlaybackQueue {
    queue: VecDeque<AudioBuffer>,
    current: Option<AudioBuffer>,
    index: usize,
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
                    let frames = front.samples.len() / channels.max(1);
                    let duration_us =
                        (frames as i64 * 1_000_000) / sample_rate as i64;
                    if front.timestamp + duration_us < self.cursor_us {
                        let _ = self.queue.pop_front();
                        continue;
                    }
                    break;
                }
            }
            self.current = self.queue.pop_front();
            self.index = 0;
        }

        if !self.initialized {
            if let Some(current) = self.current.as_ref() {
                self.cursor_us = current.timestamp;
                self.cursor_remainder = 0;
                self.initialized = true;
            }
        }

        let start = self.index;
        let end = self.index + channels;
        self.index = end;
        self.advance_cursor(sample_rate);

        let current = self.current.as_ref()?;
        Some(&current.samples[start..end])
    }

    fn advance_cursor(&mut self, sample_rate: u32) {
        self.cursor_remainder += 1_000_000;
        let advance = self.cursor_remainder / sample_rate as i64;
        self.cursor_remainder %= sample_rate as i64;
        self.cursor_us += advance;
    }
}

#[cfg(test)]
mod tests {
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

        let samples = vec![Sample::ZERO; 4800 * 2];
        queue.push(AudioBuffer {
            timestamp: 0,
            play_at: Instant::now(),
            samples: Arc::from(samples.clone().into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 200_000,
            play_at: Instant::now(),
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        queue.cursor_us = 150_000;
        queue.initialized = true;

        let frame = queue.next_frame(2, 48_000);
        assert!(frame.is_some());
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);
    }
}

/// Synced audio output with drift correction.
pub struct SyncedPlayer {
    format: AudioFormat,
    _stream: Stream,
    queue: Arc<Mutex<PlaybackQueue>>,
    /// Last error from the audio stream callback, if any.
    last_error: Arc<Mutex<Option<String>>>,
}

impl SyncedPlayer {
    /// Create a new synced player using the provided clock sync and optional device.
    pub fn new(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        device: Option<Device>,
    ) -> Result<Self, Error> {
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
        let error_clone = Arc::clone(&last_error);

        let stream =
            Self::build_stream(&device, &config, queue_clone, clock_sync, format_clone, error_clone)?;
        stream.play().map_err(|e| Error::Output(e.to_string()))?;

        Ok(Self {
            format,
            _stream: stream,
            queue,
            last_error,
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

    fn build_stream(
        device: &Device,
        config: &StreamConfig,
        queue: Arc<Mutex<PlaybackQueue>>,
        clock_sync: Arc<Mutex<ClockSync>>,
        format: AudioFormat,
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

        let stream = device
            .build_output_stream(
                config,
                move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    let generation = {
                        let queue = queue.lock();
                        queue.generation
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
                    let (cursor_us, force_reanchor) = {
                        let queue = queue.lock();
                        if queue.initialized {
                            (Some(queue.cursor_us), queue.force_reanchor)
                        } else {
                            (None, queue.force_reanchor)
                        }
                    };

                    if let (Some(cursor_us), Some(sync)) = (cursor_us, clock_sync.try_lock()) {
                        if let Some(expected_instant) = sync.server_to_local_instant(cursor_us) {
                            let early_window = Duration::from_millis(1);
                            if !started && playback_instant + early_window < expected_instant {
                                for sample in data.iter_mut() {
                                    *sample = 0.0;
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
                            let new_schedule = planner.plan(error_us, sample_rate);
                            if new_schedule != schedule {
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
                                        data[out_index] = sample.0 as f32 / 8_388_607.0;
                                        out_index += 1;
                                    }
                                } else {
                                    for sample in &last_frame {
                                        data[out_index] = sample.0 as f32 / 8_388_607.0;
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
                                    data[out_index] = sample.0 as f32 / 8_388_607.0;
                                    out_index += 1;
                                }
                                continue;
                            }
                        }

                        if let Some(frame) = queue.next_frame(channels, sample_rate) {
                            last_frame.copy_from_slice(frame);
                            for sample in frame {
                                data[out_index] = sample.0 as f32 / 8_388_607.0;
                                out_index += 1;
                            }
                        } else {
                            for _ in 0..channels {
                                data[out_index] = 0.0;
                                out_index += 1;
                            }
                        }

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
