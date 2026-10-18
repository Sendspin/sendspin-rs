// ABOUTME: End-to-end player example
// ABOUTME: Connects to server, receives audio, and plays it back

use base64::prelude::*;
use clap::{Parser, ValueEnum};
use sendspin::audio::decode::{Decoder, FlacDecoder, OpusDecoder, PcmDecoder, PcmEndian};
use sendspin::audio::{
    AudioBuffer, AudioFormat, Codec, PlayerEvent, StreamPlayerConfig, SyncedPlayer,
    SyncedPlayerConfig,
};

fn session_event_or_terminate(event: Option<PlayerEvent>) -> Result<PlayerEvent, ()> {
    event.ok_or(())
}
use sendspin::protocol::messages::{
    AudioFormatSpec, Message, PlayerFormatRequest, PlayerState, PlayerStateCommand, PlayerV1Support,
};
use sendspin::ProtocolClientBuilder;
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait};

fn print_devices() -> Result<(), Box<dyn std::error::Error>> {
    let available_hosts = cpal::available_hosts();
    println!("Available audio output devices:\n");
    let mut usable_index = 0;
    let mut example_id: Option<String> = None;
    for host_id in available_hosts {
        let host = cpal::host_from_id(host_id)?;
        let devices = host.devices()?;
        for device in devices {
            let id = device
                .id()
                .map_or("Unknown ID".to_string(), |id| id.to_string());
            let output_configs = match device.supported_output_configs() {
                Ok(f) => f.collect(),
                Err(_) => Vec::new(),
            };
            if !output_configs.is_empty() {
                if example_id.is_none() {
                    example_id = Some(id.clone());
                }
                let index_str = format!("[{}]", usable_index);
                if let Ok(desc) = device.description() {
                    println!("{:>6} {}\n       Description: {}", index_str, id, desc);
                } else {
                    println!("{:>6} {}", index_str, id);
                }
                usable_index += 1;
            }
        }
    }
    if usable_index == 0 {
        println!("\nNo devices found");
    } else {
        println!("\nTo select an audio device by index:\n  player --audio-device 0");
        if let Some(id) = example_id {
            println!(
                "Or by device id string:\n  player --audio-device \"{}\"",
                id
            );
        }
    }
    Ok(())
}

/// Accepts either an index (as printed by print_devices) or a device id string.
fn find_device(device_query: &str) -> Result<Option<cpal::Device>, Box<dyn std::error::Error>> {
    let available_hosts = cpal::available_hosts();
    let idx_query = device_query.parse::<usize>().ok();
    let mut usable_index = 0;
    for host_id in available_hosts {
        let host = cpal::host_from_id(host_id)?;
        let devices = host.devices()?;
        for device in devices {
            let id = device
                .id()
                .map_or("Unknown ID".to_string(), |id| id.to_string());
            let output_configs = match device.supported_output_configs() {
                Ok(f) => f.collect(),
                Err(_) => Vec::new(),
            };
            if !output_configs.is_empty() {
                if let Some(idx) = idx_query {
                    if usable_index == idx {
                        return Ok(Some(device));
                    }
                } else if id == device_query {
                    return Ok(Some(device));
                }
                usable_index += 1;
            }
        }
    }
    Ok(None)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn env_bool(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

fn reset_after_clear(
    buffered_duration_us: &mut u64,
    playback_started: &mut bool,
    first_chunk_logged: &mut bool,
) {
    *buffered_duration_us = 0;
    *playback_started = false;
    *first_chunk_logged = false;
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
        .map(|b64| BASE64_STANDARD.decode(b64))
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

    fn config(codec: &str) -> StreamPlayerConfig {
        StreamPlayerConfig {
            codec: codec.to_string(),
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        }
    }

    #[test]
    fn session_none_terminates_with_controls_sender_alive() {
        let (controls_tx, _controls_rx) =
            tokio::sync::watch::channel(sendspin::audio::AppliedControls {
                volume: 50,
                muted: false,
                static_delay_ms: 0,
            });
        let event = session_event_or_terminate(None);
        drop(controls_tx);
        assert!(event.is_err());
    }

    #[test]
    fn clear_reset_preserves_stream_configuration() {
        let mut buffered = 123;
        let mut started = true;
        let mut logged = true;
        let format = config("flac");
        let (_, audio_format) = decoder_for_start(&format).unwrap();
        reset_after_clear(&mut buffered, &mut started, &mut logged);
        assert_eq!(buffered, 0);
        assert!(!started);
        assert!(!logged);
        assert_eq!(audio_format.codec, Codec::Flac);
        assert_eq!(audio_format.sample_rate, 48_000);
    }

    #[test]
    fn start_setup_creates_pcm_format_and_defers_decoder() {
        let (decoder, format) = decoder_for_start(&config("pcm")).unwrap();
        assert!(decoder.is_none());
        assert_eq!(format.codec, Codec::Pcm);
        assert_eq!(format.sample_rate, 48_000);
        assert_eq!(format.channels, 2);
    }

    #[test]
    fn start_setup_preserves_format_for_codec_only_start() {
        let (decoder, format) = decoder_for_start(&config("flac")).unwrap();
        assert!(decoder.is_some());
        assert_eq!(format.codec, Codec::Flac);
        assert_eq!(format.sample_rate, 48_000);
        assert_eq!(format.channels, 2);
    }

    #[test]
    fn start_setup_reports_unsupported_codec() {
        let result = decoder_for_start(&config("mp3"));
        match result {
            Ok(_) => panic!("mp3 must be rejected"),
            Err(error) => assert!(error.contains("Unsupported codec")),
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CodecChoice {
    Pcm,
    Opus,
    Flac,
}

#[derive(Parser, Debug)]
#[command(name = "player", about = "Connect to Sendspin server and play audio", long_about = None)]
struct Args {
    #[arg(short, long, default_value = "ws://localhost:8927/sendspin")]
    server: String,
    #[arg(short, long, default_value = "Sendspin-RS Player")]
    name: String,
    #[arg(short = 'i', long = "id")]
    id: Option<String>,
    #[arg(long, value_enum)]
    codec: Option<CodecChoice>,
    #[arg(long = "list-audio-devices")]
    list_audio_devices: bool,
    #[arg(long = "audio-device")]
    audio_device: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    if args.list_audio_devices {
        print_devices()?;
        return Ok(());
    }
    let device = if let Some(device_name) = &args.audio_device {
        match find_device(device_name) {
            Ok(Some(dev)) => Some(dev),
            Ok(None) => return Err(format!("Device '{}' not found", device_name).into()),
            Err(e) => return Err(e),
        }
    } else {
        None
    };
    let client_id = args.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    println!("Connecting to {}...", args.server);
    let required_lead_time_ms = 500;
    let min_buffer_ms = 500;
    let formats: &[(&str, u8)] = match args.codec {
        Some(CodecChoice::Pcm) => &[("pcm", 24), ("pcm", 16)],
        Some(CodecChoice::Opus) => &[("opus", 16)],
        Some(CodecChoice::Flac) => &[("flac", 24), ("flac", 16)],
        None => &[
            ("flac", 24),
            ("flac", 16),
            ("opus", 16),
            ("pcm", 24),
            ("pcm", 16),
        ],
    };
    let supported_formats = formats
        .iter()
        .map(|&(codec, bit_depth)| AudioFormatSpec {
            codec: codec.to_string(),
            channels: 2,
            sample_rate: 48000,
            bit_depth,
        })
        .collect();
    let test = ProtocolClientBuilder::builder()
        .client_id(client_id)
        .name(args.name.clone())
        .player_v1_support(PlayerV1Support {
            supported_formats,
            buffer_capacity: 50 * 1024 * 1024,
            supported_commands: vec!["volume".to_string(), "mute".to_string()],
        })
        .initial_player_state(PlayerState {
            volume: Some(100),
            muted: Some(false),
            static_delay_ms: Some(0),
            required_lead_time_ms: Some(required_lead_time_ms),
            min_buffer_ms: Some(min_buffer_ms),
            supported_commands: Some(vec![PlayerStateCommand::SetStaticDelay]),
        })
        .build();
    let client = test.connect(&args.server).await?;
    println!("Connected!");
    let conn = client.split();
    drop(conn.artwork);
    drop(conn.visualizer);
    let mut message_rx = conn.messages;
    let mut player_session = conn.player.attach_lazy(sendspin::audio::AppliedControls {
        volume: 100,
        muted: false,
        static_delay_ms: 0,
    });
    let mut controls_rx = player_session.controls();
    let clock_sync = conn.clock_sync;
    let sender = conn.sender;
    let _guard = conn.guard;
    println!("Waiting for stream to start...");
    let start_buffer_ms = env_u64("SS_PLAY_START_BUFFER_MS", 500);
    let log_lead = env_bool("SS_LOG_LEAD");
    let mut decoder: Option<Box<dyn Decoder>> = None;
    let mut audio_format: Option<AudioFormat> = None;
    let mut endian_locked: Option<PcmEndian> = None;
    let mut buffered_duration_us: u64 = 0;
    let mut playback_started = false;
    let mut first_chunk_logged = false;
    let mut synced_player: Option<SyncedPlayer> = None;
    let mut format_requested = false;
    let mut last_controls = controls_rx.borrow().clone();

    loop {
        tokio::select! {
            Some(msg) = message_rx.recv() => match msg {
                Message::StreamStart(_) | Message::StreamEnd(_) | Message::StreamClear(_) => {
                    println!("Received stream lifecycle message; audio session owns ordering");
                }
                Message::ServerCommand(_) => println!("Received player command; controls are applied by the audio session"),
                _ => println!("Received message: {:?}", msg),
            },
            changed = controls_rx.changed() => {
                if changed.is_err() { break; }
                let controls = controls_rx.borrow().clone();
                if controls != last_controls { println!("Playback controls: volume={} muted={} delay={}ms", controls.volume, controls.muted, controls.static_delay_ms); last_controls = controls; }
            }
            event = player_session.recv() => {
                let Ok(event) = session_event_or_terminate(event) else { break; };
                match event {
                PlayerEvent::Start(config) => {
                    decoder = None; audio_format = None; endian_locked = None;
                    match decoder_for_start(&config) {
                        Ok((new_decoder, format)) => {
                            println!("Stream starting: codec='{}' {}Hz {}ch {}bit", config.codec, config.sample_rate, config.channels, config.bit_depth);
                            decoder = new_decoder; audio_format = Some(format.clone());
                            buffered_duration_us = 0; playback_started = false; first_chunk_logged = false;
                            if env_bool("SS_REQUEST_FORMAT") && !format_requested {
                                format_requested = true;
                                sender.request_player_format(PlayerFormatRequest { codec: Some("pcm".to_string()), channels: Some(2), sample_rate: Some(48_000), bit_depth: Some(16) }).await?;
                            }
                        }
                        Err(error) => eprintln!("ERROR: {error}"),
                    }
                }
                PlayerEvent::Clear => {
                    println!("Audio session boundary received");
                    reset_after_clear(&mut buffered_duration_us, &mut playback_started, &mut first_chunk_logged);
                }
                PlayerEvent::End => {
                    println!("Audio session ended");
                    decoder = None; audio_format = None; endian_locked = None;
                    buffered_duration_us = 0; playback_started = false; first_chunk_logged = false;
                }
                PlayerEvent::Audio(work) => {
                    if !first_chunk_logged { println!("First audio chunk: {} bytes at {} µs", work.data().len(), work.timestamp()); first_chunk_logged = true; }
                    let Some(fmt) = audio_format.clone() else { eprintln!("Audio received before stream start"); continue; };
                    if fmt.codec == Codec::Pcm {
                        let frame_size = (fmt.bit_depth as usize / 8) * fmt.channels as usize;
                        if work.data().len() % frame_size != 0 { eprintln!("BAD FRAME: {} bytes not multiple of frame size {}", work.data().len(), frame_size); continue; }
                        if endian_locked.is_none() { endian_locked = Some(PcmEndian::Little); decoder = Some(Box::new(PcmDecoder::with_endian(fmt.bit_depth, PcmEndian::Little))); }
                    }
                    let Some(dec) = decoder.as_ref() else { eprintln!("Decoder unavailable for {:?}", fmt.codec); continue; };
                    match dec.decode(work.data()) {
                        Ok(samples) => {
                            let frames = samples.len() / fmt.channels as usize;
                            buffered_duration_us += (frames as u64 * 1_000_000) / fmt.sample_rate as u64;
                            if !playback_started && buffered_duration_us >= start_buffer_ms * 1000 { playback_started = true; println!("Prebuffering complete; starting playback"); }
                            if log_lead { println!("Decoded chunk ts={} buffered={:.1}ms", work.timestamp(), buffered_duration_us as f64 / 1000.0); }
                            if synced_player.is_none() {
                                let player = SyncedPlayer::new(fmt.clone(), Arc::clone(&clock_sync), SyncedPlayerConfig { device: device.as_ref().cloned(), volume: 100, muted: false, buffer_size: None })?;
                                player_session
                                    .install(player.control())
                                    .map_err(|error| format!("Failed to install audio output: {error:?}"))?;
                                synced_player = Some(player);
                                println!("Synced audio output initialized");
                            }
                            let buffer = AudioBuffer { timestamp: work.timestamp(), samples, format: fmt };
                            if let Err(error) = work.commit(buffer) { eprintln!("Audio commit rejected: {error:?}"); }
                        }
                        Err(error) => eprintln!("Decode error: {error}"),
                    }
                }
                }
            },
            else => break,
        }
    }
    Ok(())
}
