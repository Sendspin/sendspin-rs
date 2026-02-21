// ABOUTME: cpal-based audio output implementation
// ABOUTME: Cross-platform audio output using the cpal library

use crate::audio::output::AudioOutput;
use crate::audio::{AudioFormat, Sample};
use crate::error::Error;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

/// cpal-based audio output
pub struct CpalOutput {
    format: AudioFormat,
    _stream: Stream,
    sample_tx: SyncSender<Arc<[Sample]>>,
    latency_micros: Arc<AtomicU64>,
}

impl CpalOutput {
    /// Create a new cpal audio output
    pub fn new(format: AudioFormat) -> Result<Self, Error> {
        if format.channels == 0 {
            return Err(Error::Output("channels must be > 0".to_string()));
        }
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| Error::Output("No output device available".to_string()))?;

        // Log device's default supported config to catch format mismatches
        if let Ok(def) = device.default_output_config() {
            eprintln!(
                "Device default: {:?} {}Hz {}ch",
                def.sample_format(),
                def.sample_rate().0,
                def.channels()
            );
            if def.sample_rate().0 != format.sample_rate
                || def.channels() != format.channels as u16
            {
                eprintln!(
                    "WARN: requested {}Hz/{}ch; device default is {}Hz/{}ch (OS may resample)",
                    format.sample_rate, format.channels, def.sample_rate().0, def.channels()
                );
            }
        }

        let config = StreamConfig {
            channels: format.channels as u16,
            sample_rate: cpal::SampleRate(format.sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        // Use bounded channel for backpressure (10 buffers max = ~200ms at 20ms chunks)
        let (sample_tx, sample_rx) = sync_channel::<Arc<[Sample]>>(10);
        let latency_micros = Arc::new(AtomicU64::new(0));
        let latency_clone = Arc::clone(&latency_micros);

        let stream = Self::build_stream(&device, &config, sample_rx, latency_clone)?;
        stream.play().map_err(|e| Error::Output(e.to_string()))?;

        Ok(Self {
            format,
            _stream: stream,
            sample_tx,
            latency_micros,
        })
    }

    fn build_stream(
        device: &Device,
        config: &StreamConfig,
        sample_rx: Receiver<Arc<[Sample]>>,
        latency_micros: Arc<AtomicU64>,
    ) -> Result<Stream, Error> {
        let sample_rx = Arc::new(Mutex::new(sample_rx));
        let mut current_buffer: Option<Arc<[Sample]>> = None;
        let mut buffer_pos = 0;

        let stream = device
            .build_output_stream(
                config,
                move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    let ts = info.timestamp();
                    if let Some(latency) = ts.playback.duration_since(&ts.callback) {
                        latency_micros.store(latency.as_micros() as u64, Ordering::Relaxed);
                    }
                    for sample_out in data.iter_mut() {
                        // Get next sample from current buffer or receive new buffer
                        if current_buffer.is_none()
                            || buffer_pos >= current_buffer.as_ref().unwrap().len()
                        {
                            // Try to get new buffer
                            if let Ok(rx) = sample_rx.lock() {
                                if let Ok(buf) = rx.try_recv() {
                                    current_buffer = Some(buf);
                                    buffer_pos = 0;
                                }
                            }
                        }

                        // Output sample or silence
                        if let Some(ref buf) = current_buffer {
                            if buffer_pos < buf.len() {
                                let sample = buf[buffer_pos];
                                // Convert 24-bit sample to f32 (-1.0 to 1.0)
                                *sample_out = sample.0 as f32 / 8388607.0;
                                buffer_pos += 1;
                            } else {
                                *sample_out = 0.0; // Silence
                            }
                        } else {
                            *sample_out = 0.0; // Silence
                        }
                    }
                },
                |err| eprintln!("Audio stream error: {}", err),
                None,
            )
            .map_err(|e| Error::Output(e.to_string()))?;

        Ok(stream)
    }
}

impl AudioOutput for CpalOutput {
    fn write(&mut self, samples: &Arc<[Sample]>) -> Result<(), Error> {
        self.sample_tx
            .send(Arc::clone(samples))
            .map_err(|_| Error::Output("Failed to send samples to audio thread".to_string()))
    }

    fn latency_micros(&self) -> u64 {
        self.latency_micros.load(Ordering::Relaxed)
    }

    fn format(&self) -> &AudioFormat {
        &self.format
    }
}
