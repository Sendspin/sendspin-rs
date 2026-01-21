// ABOUTME: End-to-end player example
// ABOUTME: Connects to server, receives audio, and plays it back

use clap::Parser;
use sendspin::audio::decode::{Decoder, PcmDecoder, PcmEndian};
use sendspin::audio::{AudioBuffer, AudioFormat, AudioOutput, Codec, CpalOutput};
use sendspin::protocol::client::ProtocolClient;
use sendspin::protocol::messages::{
    AudioFormatSpec, ClientHello, ClientState, ClientTime, DeviceInfo, Message, PlayerState,
    PlayerSyncState, PlayerV1Support,
};
use sendspin::scheduler::AudioScheduler;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::interval;

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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    let hello = ClientHello {
        client_id: uuid::Uuid::new_v4().to_string(),
        name: args.name.clone(),
        version: 1,
        supported_roles: vec!["player@v1".to_string()],
        device_info: Some(DeviceInfo {
            product_name: Some(args.name.clone()),
            manufacturer: Some("Sendspin".to_string()),
            software_version: Some("0.1.0".to_string()),
        }),
        player_v1_support: Some(PlayerV1Support {
            supported_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 50 * 1024 * 1024, // 50 MB
            supported_commands: vec!["volume".to_string(), "mute".to_string()],
        }),
        artwork_v1_support: None,
        visualizer_v1_support: None,
    };

    println!("Connecting to {}...", args.server);
    let client = ProtocolClient::connect(&args.server, hello).await?;
    println!("Connected!");

    // Split client into separate receivers for concurrent processing
    let (mut message_rx, mut audio_rx, clock_sync, ws_tx) = client.split();

    // Send initial client/state message (handshake step 3)
    let client_state = Message::ClientState(ClientState {
        player: Some(PlayerState {
            state: PlayerSyncState::Synchronized,
            volume: Some(100),
            muted: Some(false),
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

    // Create shared scheduler
    let scheduler = Arc::new(AudioScheduler::new());
    let scheduler_clone = Arc::clone(&scheduler);

    // Spawn playback thread (not tokio task, since CpalOutput is !Send)
    let playback_handle = std::thread::spawn(move || {
        let mut output: Option<CpalOutput> = None;

        loop {
            if let Some(buffer) = scheduler_clone.next_ready() {
                // Lazily initialize output when first buffer arrives
                if output.is_none() {
                    match CpalOutput::new(buffer.format.clone()) {
                        Ok(out) => {
                            println!("Audio output initialized");
                            output = Some(out);
                        }
                        Err(e) => {
                            eprintln!("Failed to create audio output: {}", e);
                            break;
                        }
                    }
                }

                if let Some(ref mut out) = output {
                    if let Err(e) = out.write(&buffer.samples) {
                        eprintln!("Output error: {}", e);
                    }
                }
            }
            // Per spec: 1ms polling to reduce enqueue jitter
            std::thread::sleep(Duration::from_millis(1));
        }
    });

    // Configuration from environment variables
    let min_lead_ms = env_u64("SS_PLAY_MIN_LEAD_MS", 200);
    let start_buffer_ms = env_u64("SS_PLAY_START_BUFFER_MS", 500);
    let log_lead = env_bool("SS_LOG_LEAD");

    println!(
        "Player config: min_lead={}ms, start_buffer={}ms, log_lead={}",
        min_lead_ms, start_buffer_ms, log_lead
    );

    // Message handling variables
    let mut decoder: Option<PcmDecoder> = None;
    let mut audio_format: Option<AudioFormat> = None;
    let mut endian_locked: Option<PcmEndian> = None; // Auto-detect on first chunk
    let mut buffered_duration_us: u64 = 0; // Track buffered audio duration in microseconds
    let mut playback_started = false; // Track if we've started playback
    let mut next_play_time: Option<Instant> = None; // Track when next chunk should play
    let mut first_chunk_logged = false; // Track if we've logged the first chunk

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
                            next_play_time = None;
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

                        clock_sync.lock().await.update(t1, t2, t3, t4);

                        // Log sync quality
                        let sync = clock_sync.lock().await;
                        if let Some(rtt) = sync.rtt_micros() {
                            let quality = sync.quality();
                            println!(
                                "Clock sync updated: RTT={:.2}ms, quality={:?}",
                                rtt as f64 / 1000.0,
                                quality
                            );
                        }
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
                            let duration = Duration::from_micros(duration_micros);

                            // Try to use clock sync to determine play_at time
                            let sync = clock_sync.lock().await;
                            let play_at = if let Some(instant) = sync.server_to_local_instant(chunk.timestamp) {
                                // Clock sync is ready, use synchronized timestamp
                                instant
                            } else {
                                // No clock sync yet, fall back to continuous scheduling
                                if next_play_time.is_none() {
                                    // Start from now + initial buffer
                                    next_play_time = Some(Instant::now() + Duration::from_millis(start_buffer_ms));
                                }
                                let play_time = next_play_time.unwrap();
                                next_play_time = Some(play_time + duration);
                                play_time
                            };
                            drop(sync); // Release lock

                            // Add safety window: ensure we never schedule in the past
                            // Per spec: minimum lead (env SS_PLAY_MIN_LEAD_MS) to prevent late-chunk drops
                            let min_lead = Duration::from_millis(min_lead_ms);
                            let now = Instant::now();
                            let play_at = if play_at <= now + min_lead {
                                now + min_lead
                            } else {
                                play_at
                            };

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
                            let lead = play_at.saturating_duration_since(Instant::now());
                            let lead_us = lead.as_micros() as u64;
                            if log_lead {
                                println!(
                                    "Enqueued chunk ts={} lead={}µs ({:.1}ms) buffered={:.1}ms len={} bytes",
                                    chunk.timestamp,
                                    lead_us,
                                    lead_us as f64 / 1000.0,
                                    buffered_duration_us as f64 / 1000.0,
                                    chunk.data.len()
                                );
                            }

                            let buffer = AudioBuffer {
                                timestamp: chunk.timestamp,
                                play_at,
                                samples,
                                format: fmt.clone(),
                            };

                            scheduler.schedule(buffer);
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

    // Note: playback_handle will be cleaned up when program exits
    // We don't join() here since the thread runs an infinite loop
    drop(playback_handle);
    Ok(())
}
