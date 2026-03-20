// ABOUTME: End-to-end player example
// ABOUTME: Connects to server, receives audio, and plays it back

use clap::Parser;
use sendspin::audio::decode::{Decoder, PcmDecoder, PcmEndian};
use sendspin::audio::{AudioBuffer, AudioFormat, Codec, SyncedPlayer};
use sendspin::protocol::messages::{
    ClientState, ClientSyncState, ClientTime, Message, PlayerState,
};
use sendspin::ProtocolClientBuilder;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::interval;

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
                // Construct [number] and right-justify it in a fixed-width field
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
        println!("\nTo select an audio device by index:");
        println!("  player --audio-device 0");
        if let Some(id) = example_id {
            println!("Or by device id string:");
            println!("  player --audio-device \"{}\"", id);
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

/// Environment variable helpers
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

/// Sendspin audio player
#[derive(Parser, Debug)]
#[command(name = "player")]
#[command(about = "Connect to Sendspin server and play audio", long_about = None)]
struct Args {
    /// WebSocket URL of the Sendspin server
    #[arg(short, long, default_value = "ws://localhost:8927/sendspin")]
    server: String,

    /// Client name
    #[arg(short, long, default_value = "Sendspin-RS Player")]
    name: String,

    /// Player ID (optional, generates random UUID if not provided)
    #[arg(short = 'i', long = "id")]
    id: Option<String>,

    /// List all available audio output devices and exit
    #[arg(long = "list-audio-devices")]
    list_audio_devices: bool,

    /// Audio output device ID (optional, uses default if not specified)
    #[arg(long = "audio-device")]
    audio_device: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    // Handle --list-audio-devices flag
    if args.list_audio_devices {
        print_devices()?;
        return Ok(());
    }

    // Resolve device if specified
    let device = if let Some(device_name) = &args.audio_device {
        match find_device(device_name) {
            Ok(Some(dev)) => Some(dev),
            Ok(None) => {
                return Err(format!("Device '{}' not found", device_name).into());
            }
            Err(e) => {
                eprintln!("Failed to find device: {}", e);
                return Err(e);
            }
        }
    } else {
        None
    };

    // Use provided ID or generate a random UUID
    let client_id = args.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    println!("Connecting to {}...", args.server);
    let test = ProtocolClientBuilder::builder()
        .client_id(client_id)
        .name(args.name.clone())
        .build();

    let client = test.connect(&args.server).await?;
    println!("Connected!");

    // Split client into separate receivers for concurrent processing
    let (mut message_rx, mut audio_rx, clock_sync, ws_tx, _guard) = client.split();

    // Send initial client/state message (handshake step 3)
    let client_state = Message::ClientState(ClientState {
        state: Some(ClientSyncState::Synchronized),
        player: Some(PlayerState {
            volume: Some(100),
            muted: Some(false),
            static_delay_ms: Some(0),
            supported_commands: None,
        }),
    });
    ws_tx.send_message(client_state).await?;
    println!("Sent initial client/state");

    // Send immediate initial clock sync
    let client_transmitted = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;
    let time_msg = Message::ClientTime(ClientTime { client_transmitted });
    ws_tx.send_message(time_msg).await?;
    println!("Sent initial client/time for clock sync");

    println!("Waiting for stream to start...");

    // Spawn clock sync task that sends client/time every 5 seconds
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(5));
        loop {
            interval.tick().await;

            // Get current Unix epoch microseconds
            let client_transmitted = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64;

            let time_msg = Message::ClientTime(ClientTime { client_transmitted });

            // Send time sync message
            if let Err(e) = ws_tx.send_message(time_msg).await {
                eprintln!("Failed to send time sync: {}", e);
                break;
            }
        }
    });

    // Configuration from environment variables
    let start_buffer_ms = env_u64("SS_PLAY_START_BUFFER_MS", 500);
    let log_lead = env_bool("SS_LOG_LEAD");
    println!(
        "Player config: start_buffer={}ms, log_lead={}",
        start_buffer_ms, log_lead
    );

    // Message handling variables
    let mut decoder: Option<PcmDecoder> = None;
    let mut audio_format: Option<AudioFormat> = None;
    let mut endian_locked: Option<PcmEndian> = None; // Auto-detect on first chunk
    let mut buffered_duration_us: u64 = 0; // Track buffered audio duration in microseconds
    let mut playback_started = false; // Track if we've started playback
    let mut first_chunk_logged = false; // Track if we've logged the first chunk
    let mut synced_player: Option<SyncedPlayer> = None;

    loop {
        // Process messages and audio chunks concurrently
        tokio::select! {
            Some(msg) = message_rx.recv() => {
                match msg {
                    Message::StreamStart(stream_start) => {
                        if let Some(ref player_config) = stream_start.player {
                            println!(
                                "Stream starting: codec='{}' {}Hz {}ch {}bit",
                                player_config.codec,
                                player_config.sample_rate,
                                player_config.channels,
                                player_config.bit_depth
                            );

                            // Validate codec before proceeding
                            if player_config.codec != "pcm" {
                                eprintln!("ERROR: Unsupported codec '{}' - only 'pcm' is supported!", player_config.codec);
                                eprintln!("Server is sending compressed audio that we can't decode!");
                                continue;
                            }

                            if player_config.bit_depth != 16 && player_config.bit_depth != 24 {
                                eprintln!("ERROR: Unsupported bit depth {} - only 16 or 24-bit PCM supported!", player_config.bit_depth);
                                continue;
                            }

                            audio_format = Some(AudioFormat {
                                codec: Codec::Pcm,
                                sample_rate: player_config.sample_rate,
                                channels: player_config.channels,
                                bit_depth: player_config.bit_depth,
                                codec_header: None,
                            });

                            // Decoder will be created on first chunk after auto-detecting endianness
                            decoder = None;
                            endian_locked = None;
                            buffered_duration_us = 0; // Reset on new stream
                            playback_started = false;
                            first_chunk_logged = false; // Reset for new stream
                            println!("Waiting for first audio chunk to auto-detect endianness...");
                        } else {
                            println!("Received stream/start without player config");
                        }
                    }
                    Message::ServerTime(server_time) => {
                        // Get t4 (client receive time) in Unix microseconds
                        let t4 = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_micros() as i64;

                        // Update clock sync with all four timestamps
                        let t1 = server_time.client_transmitted;
                        let t2 = server_time.server_received;
                        let t3 = server_time.server_transmitted;

                        clock_sync.lock().update(t1, t2, t3, t4);

                        // Log sync quality
                        let sync = clock_sync.lock();
                        if let Some(rtt) = sync.rtt_micros() {
                            let quality = sync.quality();
                            println!(
                                "Clock sync updated: RTT={:.2}ms, quality={:?}",
                                rtt as f64 / 1000.0,
                                quality
                            );
                        }
                    }
                    Message::StreamEnd(stream_end) => {
                        println!("Stream ended: {:?}", stream_end.roles);
                        if let Some(ref player) = synced_player {
                            player.clear();
                        }
                        buffered_duration_us = 0;
                        playback_started = false;
                        first_chunk_logged = false;
                    }
                    Message::StreamClear(stream_clear) => {
                        println!("Stream cleared: {:?}", stream_clear.roles);
                        if let Some(ref player) = synced_player {
                            player.clear();
                        }
                        buffered_duration_us = 0;
                        playback_started = false;
                        first_chunk_logged = false;
                    }
                    _ => {
                        println!("Received message: {:?}", msg);
                    }
                }
            }
            Some(chunk) = audio_rx.recv() => {
                // Log first chunk bytes for diagnostics
                if !first_chunk_logged {
                    println!("\n=== FIRST AUDIO CHUNK DIAGNOSTICS ===");
                    println!("Chunk timestamp: {} µs", chunk.timestamp);
                    println!("Chunk data length: {} bytes", chunk.data.len());
                    let preview_len = chunk.data.len().min(32);
                    print!("First {} bytes (hex): ", preview_len);
                    for byte in &chunk.data[..preview_len] {
                        print!("{:02X} ", byte);
                    }
                    println!("\n=====================================\n");
                    first_chunk_logged = true;
                }

                if let Some(ref fmt) = audio_format {
                    // Frame sanity check
                    let bytes_per_sample = match fmt.bit_depth {
                        16 => 2,
                        24 => 3,
                        _ => {
                            eprintln!("Unsupported bit depth: {}", fmt.bit_depth);
                            continue;
                        }
                    } as usize;
                    let frame_size = bytes_per_sample * fmt.channels as usize;

                    if chunk.data.len() % frame_size != 0 {
                        eprintln!(
                            "BAD FRAME: {} bytes not multiple of frame size {} ({}-bit, {}ch)",
                            chunk.data.len(), frame_size, fmt.bit_depth, fmt.channels
                        );
                        continue; // Don't decode garbage
                    }

                    // One-time endianness setup on first chunk
                    // Per spec: macOS and most systems use Little-Endian PCM
                    // Only use Big-Endian if explicitly signaled by server
                    if endian_locked.is_none() {
                        // Default to Little-Endian (standard for macOS/Windows/Linux)
                        let endian = PcmEndian::Little;
                        endian_locked = Some(endian);
                        decoder = Some(PcmDecoder::with_endian(fmt.bit_depth, endian));
                        println!("Using Little-Endian PCM (standard for modern systems)");
                    }
                }

                if let (Some(ref dec), Some(ref fmt)) = (&decoder, &audio_format) {
                    match dec.decode(&chunk.data) {
                        Ok(samples) => {
                            // Calculate chunk duration in microseconds
                            // samples.len() includes all channels
                            let frames = samples.len() / fmt.channels as usize;
                            let duration_micros = (frames as u64 * 1_000_000) / fmt.sample_rate as u64;
                            // Track buffered duration
                            buffered_duration_us += duration_micros;

                            // Check if we've buffered enough to start playback
                            if !playback_started && buffered_duration_us >= start_buffer_ms * 1000 {
                                playback_started = true;
                                println!(
                                    "Prebuffering complete ({:.1}ms buffered), starting playback!",
                                    buffered_duration_us as f64 / 1000.0
                                );
                            }

                            // Track and log lead time
                            if log_lead {
                                println!(
                                    "Enqueued chunk ts={} buffered={:.1}ms len={} bytes",
                                    chunk.timestamp,
                                    buffered_duration_us as f64 / 1000.0,
                                    chunk.data.len()
                                );
                            }

                            if synced_player.is_none() {
                                match SyncedPlayer::new(
                                    fmt.clone(),
                                    Arc::clone(&clock_sync),
                                    device.as_ref().cloned(),
                                    100,
                                    false,
                                ) {
                                    Ok(player) => {
                                        println!("Synced audio output initialized");
                                        synced_player = Some(player);
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to create synced output: {}", e);
                                    }
                                }
                            }

                            if let Some(ref player) = synced_player {
                                let buffer = AudioBuffer {
                                    timestamp: chunk.timestamp,
                                    play_at: Instant::now(),
                                    samples,
                                    format: fmt.clone(),
                                };
                                player.enqueue(buffer);
                            }
                        }
                        Err(e) => {
                            eprintln!("Decode error: {}", e);
                        }
                    }
                }
            }
            else => {
                // Both channels closed
                break;
            }
        }
    }

    Ok(())
}
