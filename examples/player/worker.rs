// ABOUTME: Dedicated audio worker and backend lifecycle management
// ABOUTME: Serializes stream commands while preserving the installed output

use base64::prelude::*;

use cpal::traits::{DeviceTrait, HostTrait};
use sendspin::audio::decode::{Decoder, FlacDecoder, OpusDecoder, PcmDecoder, PcmEndian};
use sendspin::audio::{
    AudioBuffer, AudioFormat, AudioWork, Codec, OutputInstallPermit, StreamPlayerConfig,
    SyncedPlayer, SyncedPlayerConfig,
};
use sendspin::sync::ClockSync;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::thread::{self, JoinHandle};
use tokio::sync::OwnedSemaphorePermit;

pub enum Command<P = OutputInstallPermit> {
    Start(StreamPlayerConfig, OwnedSemaphorePermit),
    Audio(AudioWork, OwnedSemaphorePermit, Option<P>),
    Clear(OwnedSemaphorePermit),
    End(OwnedSemaphorePermit),
    Shutdown,
}

pub struct Worker {
    tx: mpsc::Sender<Command>,
    join: Option<JoinHandle<()>>,
    stopping: Arc<AtomicBool>,
}

impl Worker {
    pub fn validate_device(device_query: Option<&str>) -> Result<(), String> {
        let Some(query) = device_query else {
            return Ok(());
        };
        find_device(query)
            .map(|_| ())
            .ok_or_else(|| format!("requested audio device '{query}' is unavailable"))
    }

    pub fn start(device_query: Option<String>, clock: Arc<parking_lot::Mutex<ClockSync>>) -> Self {
        let (tx, rx) = mpsc::channel();
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = stopping.clone();
        let join = thread::Builder::new()
            .name("sendspin-player".to_string())
            .spawn(move || {
                // Keep an explicit request distinct from the default-device fallback.
                let requested_device = device_query.clone();
                let device = device_query.as_deref().and_then(find_device);
                run(
                    rx,
                    ProductionBackend {
                        device,
                        requested_device,
                        clock,
                    },
                    worker_stopping,
                );
            })
            .expect("failed to start player worker");
        Self {
            tx,
            join: Some(join),
            stopping,
        }
    }

    pub fn send(&self, command: Command) -> Result<(), mpsc::SendError<Command>> {
        self.tx.send(command)
    }

    pub async fn shutdown(mut self) {
        self.stopping.store(true, Ordering::Release);
        let _ = self.tx.send(Command::Shutdown);
        if let Some(join) = self.join.take() {
            let _ = tokio::task::spawn_blocking(move || join.join()).await;
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        let _ = self.tx.send(Command::Shutdown);
        // A destructor may run on Tokio's executor; do not join a device thread here.
        let _ = self.join.take();
    }
}

trait BackendFactory {
    type Output;
    type InstallPermit;

    fn create(&mut self, format: &AudioFormat) -> Result<Self::Output, String>;
    fn install(
        &mut self,
        output: &mut Self::Output,
        permit: Self::InstallPermit,
    ) -> Result<(), String>;
}

struct ProductionBackend {
    device: Option<cpal::Device>,
    requested_device: Option<String>,
    clock: Arc<parking_lot::Mutex<ClockSync>>,
}

struct ProductionOutput(SyncedPlayer);

impl BackendFactory for ProductionBackend {
    type Output = ProductionOutput;
    type InstallPermit = OutputInstallPermit;

    fn create(&mut self, format: &AudioFormat) -> Result<Self::Output, String> {
        if let Some(query) = &self.requested_device {
            if self.device.is_none() {
                return Err(format!("requested audio device '{query}' is unavailable"));
            }
        }
        SyncedPlayer::new(
            format.clone(),
            Arc::clone(&self.clock),
            SyncedPlayerConfig {
                device: self.device.clone(),
                volume: 100,
                muted: false,
                buffer_size: None,
            },
        )
        .map(ProductionOutput)
        .map_err(|error| error.to_string())
    }

    fn install(
        &mut self,
        output: &mut Self::Output,
        permit: OutputInstallPermit,
    ) -> Result<(), String> {
        permit
            .install(output.0.control())
            .map_err(|error| format!("{error:?}"))
    }
}

struct State<F: BackendFactory> {
    decoder: Option<Box<dyn Decoder>>,
    // The decoder format may change before output exists; output_format is the installed shape.
    decoder_format: Option<AudioFormat>,
    output_format: Option<AudioFormat>,
    output: Option<F::Output>,
    backend: F,
    output_failed: bool,
    input_blocked: bool,
}

fn run<F: BackendFactory>(
    rx: mpsc::Receiver<Command<F::InstallPermit>>,
    backend: F,
    stopping: Arc<AtomicBool>,
) -> usize {
    let mut processed = 0;
    let mut state = State {
        decoder: None,
        decoder_format: None,
        output_format: None,
        output: None,
        backend,
        output_failed: false,
        input_blocked: false,
    };
    while let Ok(command) = rx.recv() {
        // Check the stop flag before dispatch so shutdown wins over queued work.
        if stopping.load(Ordering::Acquire) {
            break;
        }
        processed += 1;
        match command {
            Command::Start(config, _permit) => state.start(config),
            Command::Audio(work, _permit, install) => state.audio(work, install),
            Command::Clear(_permit) => state.clear(),
            Command::End(_permit) => state.end(),
            Command::Shutdown => break,
        }
    }
    processed
}

impl<F: BackendFactory> State<F> {
    fn start(&mut self, config: StreamPlayerConfig) {
        match decoder_for_start(&config) {
            Ok((decoder, format)) => {
                if self.output.is_some() {
                    let old = self
                        .output_format
                        .as_ref()
                        .expect("installed output has a format");
                    if old.sample_rate != format.sample_rate || old.channels != format.channels {
                        self.input_blocked = true;
                        eprintln!(
                            "Ignoring audio format change to {}Hz {}ch; output remains at {}Hz {}ch",
                            format.sample_rate,
                            format.channels,
                            old.sample_rate,
                            old.channels
                        );
                        return;
                    }
                }
                self.decoder = decoder;
                self.decoder_format = Some(format);
                self.input_blocked = false;
                self.output_failed = false;
            }
            Err(error) => {
                // Keep the installed output shape so a later incompatible start cannot reopen it.
                self.decoder = None;
                self.decoder_format = None;
                self.input_blocked = true;
                eprintln!("ERROR: {error}");
            }
        }
    }

    fn audio(&mut self, work: AudioWork, install: Option<F::InstallPermit>) {
        self.audio_with_install(work, install);
    }

    fn audio_with_install(&mut self, work: AudioWork, install: Option<F::InstallPermit>) {
        if self.input_blocked {
            return;
        }
        let Some(format) = self.decoder_format.clone() else {
            return;
        };
        if format.codec == Codec::Pcm
            && !validate_pcm_frame(work.data().len(), format.bit_depth, format.channels)
        {
            eprintln!(
                "Invalid PCM frame: {} bytes for {}-bit {}-channel audio",
                work.data().len(),
                format.bit_depth,
                format.channels
            );
            return;
        }
        if format.codec == Codec::Pcm && self.decoder.is_none() {
            self.decoder = Some(Box::new(PcmDecoder::with_endian(
                format.bit_depth,
                PcmEndian::Little,
            )));
        }
        let Some(decoder) = self.decoder.as_ref() else {
            return;
        };
        let samples = match decoder.decode(work.data()) {
            Ok(samples) => samples,
            Err(error) => {
                eprintln!("Decode error: {error}");
                return;
            }
        };
        if self.output.is_none() && !self.ensure_output(&format, install) {
            return;
        }
        let buffer = AudioBuffer {
            timestamp: work.timestamp(),
            samples,
            format,
        };
        if let Err(error) = work.commit(buffer) {
            eprintln!("Audio commit rejected: {error:?}");
        }
    }

    fn ensure_output(&mut self, format: &AudioFormat, install: Option<F::InstallPermit>) -> bool {
        let Some(permit) = install else {
            eprintln!("Dropping audio until an output installation permit is available");
            return false;
        };
        if self.output_failed {
            return false;
        }
        let mut output = match self.backend.create(format) {
            Ok(output) => output,
            Err(error) => {
                self.output_failed = true;
                eprintln!("Failed to initialize audio output: {error}");
                return false;
            }
        };
        if let Err(error) = self.backend.install(&mut output, permit) {
            self.output_failed = true;
            eprintln!("Failed to install audio output: {error}");
            return false;
        }
        self.output_format = Some(format.clone());
        self.output = Some(output);
        true
    }

    fn clear(&mut self) {
        // Clear is informational here: the library clears buffered audio, so keep the stream configuration and output ready.
    }

    fn end(&mut self) {
        // End resets decoding for the next stream while keeping the installed output shape.
        self.decoder = None;
        self.decoder_format = None;
        self.input_blocked = false;
        self.output_failed = false;
    }
}

fn validate_pcm_frame(data_len: usize, bit_depth: u8, channels: u8) -> bool {
    let bytes_per_sample = match bit_depth {
        16 => 2,
        24 => 3,
        _ => return false,
    };
    channels != 0
        && data_len != 0
        && data_len.is_multiple_of(bytes_per_sample * usize::from(channels))
}

fn find_device(query: &str) -> Option<cpal::Device> {
    let index = query.parse::<usize>().ok();
    let mut current = 0;
    for host_id in cpal::available_hosts() {
        let host = cpal::host_from_id(host_id).ok()?;
        for device in host.devices().ok()? {
            let id = device
                .id()
                .map_or_else(|_| "Unknown ID".to_string(), |id| id.to_string());
            let usable = device
                .supported_output_configs()
                .is_ok_and(|mut configs| configs.next().is_some());
            if usable {
                if index == Some(current) || index.is_none() && id == query {
                    return Some(device);
                }
                current += 1;
            }
        }
    }
    None
}

fn decoder_for_start(
    config: &StreamPlayerConfig,
) -> Result<(Option<Box<dyn Decoder>>, AudioFormat), String> {
    let codec = match config.codec.as_str() {
        "pcm" => Codec::Pcm,
        "opus" => Codec::Opus,
        "flac" => Codec::Flac,
        other => return Err(format!("Unsupported codec '{other}'")),
    };
    if !matches!(config.bit_depth, 16 | 24) {
        return Err(format!("Unsupported bit depth {}", config.bit_depth));
    }
    let codec_header = config
        .codec_header
        .as_deref()
        .map(|value| BASE64_STANDARD.decode(value))
        .transpose()
        .map_err(|e| format!("Invalid base64 codec header: {e}"))?;
    let decoder = match codec {
        Codec::Pcm => None,
        Codec::Opus => Some(Box::new(
            OpusDecoder::new(config.sample_rate, config.channels)
                .map_err(|e| format!("Invalid Opus stream format: {e}"))?,
        ) as Box<dyn Decoder>),
        Codec::Flac => Some(Box::new(match codec_header.as_deref() {
            Some(header) => FlacDecoder::with_header(header)
                .map_err(|e| format!("Invalid FLAC codec header: {e}"))?,
            None => FlacDecoder::new(),
        }) as Box<dyn Decoder>),
        Codec::Mp3 => return Err("Unsupported codec 'mp3'".to_string()),
    };
    Ok((
        decoder,
        AudioFormat {
            codec,
            sample_rate: config.sample_rate,
            channels: config.channels,
            bit_depth: config.bit_depth,
            codec_header,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeOutput;
    #[derive(Clone, Copy)]
    struct FakePermit;
    struct FakeBackend {
        creates: usize,
        fail_create: bool,
    }

    impl BackendFactory for FakeBackend {
        type Output = FakeOutput;
        type InstallPermit = FakePermit;

        fn create(&mut self, _format: &AudioFormat) -> Result<Self::Output, String> {
            self.creates += 1;
            if self.fail_create {
                Err("fake backend failure".into())
            } else {
                Ok(FakeOutput)
            }
        }

        fn install(
            &mut self,
            _output: &mut Self::Output,
            _permit: FakePermit,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    fn config(codec: &str) -> StreamPlayerConfig {
        StreamPlayerConfig {
            codec: codec.to_string(),
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        }
    }

    fn state() -> State<FakeBackend> {
        State {
            decoder: None,
            decoder_format: None,
            output_format: None,
            output: None,
            backend: FakeBackend {
                creates: 0,
                fail_create: false,
            },
            output_failed: false,
            input_blocked: false,
        }
    }

    #[test]
    fn pcm_frame_validation_requires_complete_stereo_frames() {
        assert!(!validate_pcm_frame(6, 16, 2));
        assert!(validate_pcm_frame(8, 16, 2));
        assert!(!validate_pcm_frame(9, 24, 2));
        assert!(validate_pcm_frame(12, 24, 2));
    }

    #[test]
    fn pcm_frame_validation_rejects_invalid_formats_without_panicking() {
        assert!(!validate_pcm_frame(8, 0, 2));
        assert!(!validate_pcm_frame(8, 32, 2));
        assert!(!validate_pcm_frame(8, 16, 0));
    }

    #[test]
    fn shutdown_discards_queued_lifecycle_commands() {
        let (tx, rx) = mpsc::channel();
        let stopping = Arc::new(AtomicBool::new(false));
        let admission = Arc::new(tokio::sync::Semaphore::new(2));
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let first = runtime.block_on(admission.clone().acquire_owned()).unwrap();
        let second = runtime.block_on(admission.acquire_owned()).unwrap();
        tx.send(Command::Start(config("pcm"), first)).unwrap();
        tx.send(Command::Start(config("opus"), second)).unwrap();
        stopping.store(true, Ordering::Release);
        tx.send(Command::Shutdown).unwrap();
        let handle = std::thread::spawn({
            let thread_stopping = stopping.clone();
            move || {
                run(
                    rx,
                    FakeBackend {
                        creates: 0,
                        fail_create: false,
                    },
                    thread_stopping,
                )
            }
        });
        assert_eq!(handle.join().unwrap(), 0, "queued starts ran after stop");
    }

    #[test]
    fn codec_change_reuses_output_and_swaps_decoder() {
        let mut state = state();
        state.start(config("pcm"));
        state.output = Some(FakeOutput);
        state.output_format = state.decoder_format.clone();
        state.start(config("opus"));
        assert!(state.output.is_some());
        assert_eq!(
            state.decoder_format.as_ref().map(|f| f.codec),
            Some(Codec::Opus)
        );
        assert_eq!(state.backend.creates, 0);
    }

    #[test]
    fn end_start_reactivates_decoder_without_reopening_output() {
        let mut state = state();
        state.start(config("opus"));
        state.output = Some(FakeOutput);
        state.output_format = state.decoder_format.clone();
        state.end();
        assert!(state.decoder.is_none());
        assert!(state.decoder_format.is_none());
        assert!(state.output.is_some());
        state.start(config("opus"));
        assert!(state.decoder.is_some());
        assert!(state.output.is_some());
        assert_eq!(state.backend.creates, 0);
    }

    #[test]
    fn clear_preserves_decoder_format_and_output_for_pcm_and_opus() {
        for codec in ["pcm", "opus"] {
            let mut state = state();
            state.start(config(codec));
            state.output = Some(FakeOutput);
            state.output_format = state.decoder_format.clone();
            let decoder_format = state.decoder_format.clone();
            let output_format = state.output_format.clone();
            let had_decoder = state.decoder.is_some();

            state.clear();

            assert_eq!(state.decoder_format, decoder_format);
            assert_eq!(state.output_format, output_format);
            assert!(state.output.is_some());
            assert_eq!(state.decoder.is_some(), had_decoder);
            assert!(!state.input_blocked);
            assert!(!state.output_failed);
            assert!(state.decoder_format.is_some(), "{codec} remains decodable");
        }
    }

    #[test]
    fn incompatible_start_sets_input_blocked() {
        let mut state = state();
        state.start(config("pcm"));
        state.output = Some(FakeOutput);
        state.output_format = state.decoder_format.clone();
        let mut changed = config("pcm");
        changed.channels = 1;
        state.start(changed);
        assert!(state.output.is_some());
        assert!(state.input_blocked);
        assert_eq!(state.decoder_format.as_ref().map(|f| f.channels), Some(2));
    }

    #[test]
    fn invalid_start_clears_decoder_but_keeps_output_shape() {
        let mut state = state();
        state.start(config("opus"));
        state.output = Some(FakeOutput);
        state.output_format = state.decoder_format.clone();
        state.start(config("mp3"));
        assert!(state.decoder.is_none());
        assert!(state.decoder_format.is_none());
        assert!(state.output.is_some());
        assert!(state.output_format.is_some());
    }

    #[test]
    fn end_then_incompatible_start_keeps_installed_output() {
        let mut state = state();
        state.start(config("pcm"));
        state.output = Some(FakeOutput);
        state.output_format = state.decoder_format.clone();
        state.end();
        let mut changed = config("pcm");
        changed.channels = 1;
        state.start(changed);
        assert!(state.input_blocked);
        assert!(state.output.is_some());
        assert_eq!(state.backend.creates, 0);
    }

    #[test]
    fn ensure_output_without_permit_does_not_create_output() {
        let mut state = state();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };
        assert!(!state.ensure_output(&format, None));
        assert!(state.output.is_none());
        assert_eq!(state.backend.creates, 0);
    }

    #[test]
    fn failed_output_is_not_retried_until_valid_start() {
        let mut state = state();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };
        state.backend.fail_create = true;
        assert!(!state.ensure_output(&format, Some(FakePermit)));
        assert_eq!(state.backend.creates, 1);
        state.backend.fail_create = false;
        assert!(!state.ensure_output(&format, Some(FakePermit)));
        assert_eq!(state.backend.creates, 1);
        state.start(config("pcm"));
        assert!(state.ensure_output(&format, Some(FakePermit)));
        assert_eq!(state.backend.creates, 2);
    }
}
