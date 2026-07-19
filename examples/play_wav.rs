// ABOUTME: Example server that streams a WAV file to connected Sendspin devices
// ABOUTME: Handles both dial-in clients and clients discovered/dialed over mDNS
//
// Usage:
//   cargo run --example play_wav -- path/to/clip.wav
//   cargo run --example play_wav -- --dial ws://192.168.1.42:8928/sendspin clip.wav
//   cargo run --example play_wav -- --bind 0.0.0.0:8927 --no-advertise clip.wav
//
// Streams the file to every connected client as one synchronized group, and
// prints the client/state, client/command, and client/goodbye messages it
// receives. Handles both connection directions: clients that dial in
// (accepted via ServerListener, discoverable through this tool's mDNS
// advertisement) and clients that only run their own embedded server
// (discovered and dialed via ClientManager). `--dial` targets specific
// clients and disables discovery unless `--also-discover` is given.
//
// Only plain PCM WAV (fmt tag 1) is supported. Re-encode anything else first,
// e.g.: ffmpeg -i in.mp3 -ar 48000 -ac 2 -acodec pcm_s16le out.wav

use clap::Parser;
use sendspin::protocol::messages::Message;
use sendspin::server::{dial_client, Advertisement, ClientEvent, ClientManager, Group};
use sendspin::{DefaultClock, ServerConnection, ServerListener};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "play_wav")]
struct Args {
    /// WAV file to stream (plain PCM, any sample rate/channels/bit depth)
    file: PathBuf,

    /// Address to bind the server on, for clients that dial in
    #[arg(long, default_value = "0.0.0.0:8927")]
    bind: String,

    /// HTTP path clients connect to (fixed by the spec for real deployments)
    #[arg(long, default_value = "/sendspin")]
    path: String,

    /// Server identifier advertised to clients and over mDNS
    #[arg(long, default_value = "sendspin-rs-play-wav")]
    server_id: String,

    /// Human-readable server name
    #[arg(long, default_value = "Sendspin Rust Test Server")]
    name: String,

    /// Advertise `_sendspin-server._tcp.local.` so clients that dial in can
    /// find this server
    #[arg(long, default_value_t = true)]
    advertise: bool,

    /// Disable mDNS advertising (e.g. to avoid a device picking up this test
    /// server instead of your production one on the same network)
    #[arg(long)]
    no_advertise: bool,

    /// Disable mDNS client discovery. Discovery (of `_sendspin._tcp.local.`
    /// clients — devices that only run their own embedded server, e.g. Home
    /// Assistant Voice PE) is on by default only when no --dial is given;
    /// see --also-discover to keep it on alongside --dial.
    #[arg(long)]
    no_discover: bool,

    /// Keep mDNS discovery on even when --dial is given. By default, giving
    /// --dial turns discovery off so you can isolate a test to just the
    /// device(s) you named — otherwise every other Sendspin device on the
    /// network joins the group too.
    #[arg(long)]
    also_discover: bool,

    /// Dial a specific client URL directly (e.g.
    /// ws://192.168.1.42:8928/sendspin). Repeatable — pass it more than once
    /// to test a specific, known set of devices.
    #[arg(long = "dial")]
    dial_urls: Vec<String>,

    /// After the first client connects, wait this long for additional
    /// clients before starting playback (0 to start immediately)
    #[arg(long, default_value_t = 5)]
    wait_secs: u64,

    /// Milliseconds of audio per pushed chunk
    #[arg(long, default_value_t = 100)]
    chunk_ms: u64,

    /// How far ahead of "now" each audio chunk's timestamp is scheduled, in
    /// milliseconds — must comfortably exceed real delivery jitter or
    /// clients will receive chunks whose intended playback time already
    /// passed. Passed through to Group::with_send_ahead_us.
    #[arg(long, default_value_t = 250)]
    send_ahead_ms: u64,
}

struct WavInfo {
    channels: u8,
    sample_rate: u32,
    bit_depth: u8,
    data: Vec<u8>,
}

/// Minimal RIFF/WAVE chunk walker — just enough to find `fmt ` and `data`
/// for plain PCM. Not a general-purpose WAV reader (no WAVE_FORMAT_EXTENSIBLE,
/// no metadata chunks surfaced) — this is a manual test tool, not a decoder
/// this crate needs to ship.
fn read_wav(path: &PathBuf) -> Result<WavInfo, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{} is not a RIFF/WAVE file", path.display()));
    }

    let mut pos = 12;
    let mut fmt: Option<(u16, u8, u32, u8)> = None; // (format_tag, channels, sample_rate, bit_depth)
    let mut data: Option<Vec<u8>> = None;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + chunk_size).min(bytes.len());
        let body = &bytes[body_start..body_end];

        match chunk_id {
            b"fmt " if body.len() >= 16 => {
                let format_tag = u16::from_le_bytes(body[0..2].try_into().unwrap());
                let channels = u16::from_le_bytes(body[2..4].try_into().unwrap()) as u8;
                let sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bit_depth = u16::from_le_bytes(body[14..16].try_into().unwrap()) as u8;
                fmt = Some((format_tag, channels, sample_rate, bit_depth));
            }
            b"data" => data = Some(body.to_vec()),
            _ => {}
        }

        // Chunks are padded to even length.
        pos = body_start + chunk_size + (chunk_size % 2);
    }

    let (format_tag, channels, sample_rate, bit_depth) = fmt.ok_or("no fmt chunk found")?;
    if format_tag != 1 {
        return Err(format!(
            "unsupported WAV format tag {format_tag} (only plain PCM, tag 1, is supported) — \
             re-encode with: ffmpeg -i {} -acodec pcm_s16le out.wav",
            path.display()
        ));
    }
    let data = data.ok_or("no data chunk found")?;

    Ok(WavInfo {
        channels,
        sample_rate,
        bit_depth,
        data,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    let advertise = args.advertise && !args.no_advertise;
    let discover = !args.no_discover && (args.dial_urls.is_empty() || args.also_discover);

    let wav = read_wav(&args.file).map_err(|e| format!("failed to read WAV: {e}"))?;
    println!(
        "loaded {}: {} Hz, {} ch, {}-bit, {:.1}s",
        args.file.display(),
        wav.sample_rate,
        wav.channels,
        wav.bit_depth,
        wav.data.len() as f64
            / (wav.sample_rate as f64 * wav.channels as f64 * (wav.bit_depth as f64 / 8.0))
    );

    let listener = ServerListener::bind(&args.bind, &args.server_id, &args.name)
        .await?
        .path(&args.path);
    let port = listener.local_addr()?.port();
    println!(
        "listening on {} (path {}), for clients that dial in",
        listener.local_addr()?,
        args.path
    );

    let _advertisement = if advertise {
        println!(
            "advertising _sendspin-server._tcp.local. as {:?} on port {port}",
            args.server_id
        );
        Some(Advertisement::new(
            &args.server_id,
            &args.name,
            port,
            &args.path,
        )?)
    } else {
        None
    };

    let group = Arc::new(
        Group::new(Arc::new(DefaultClock::default()))
            .with_send_ahead_us(args.send_ahead_ms as i64 * 1000),
    );

    spawn_accept_loop(listener, Arc::clone(&group));
    // Kept alive for main()'s whole lifetime — dropping it would stop
    // discovery and abort every reconnect loop it's supervising.
    let _manager = if discover {
        Some(spawn_manager_loop(
            Arc::clone(&group),
            args.server_id.clone(),
            args.name.clone(),
        )?)
    } else {
        None
    };
    for url in &args.dial_urls {
        dial_one(&group, url, &args.server_id, &args.name).await;
    }

    println!("waiting for at least one client (inbound accept, mDNS discovery, or --dial)...");
    wait_for_first_member(&group).await;

    if args.wait_secs > 0 {
        println!(
            "waiting up to {}s for additional clients (connect more devices now for a multi-room test)...",
            args.wait_secs
        );
        tokio::time::sleep(Duration::from_secs(args.wait_secs)).await;
    }

    let member_count = group.len();
    println!("starting playback to {member_count} client(s)");

    let chunk_bytes = (wav.sample_rate as u64 * args.chunk_ms / 1000) as usize
        * wav.channels as usize
        * (wav.bit_depth as usize / 8);
    if chunk_bytes == 0 {
        return Err("computed chunk size is zero — check --chunk-ms and the WAV format".into());
    }

    group
        .start_stream(sendspin::protocol::messages::StreamPlayerConfig {
            codec: "pcm".to_string(),
            sample_rate: wav.sample_rate,
            channels: wav.channels,
            bit_depth: wav.bit_depth,
            codec_header: None,
        })
        .await;

    // push_audio enqueues without blocking and the Group anchors its own
    // timeline, so playback timing no longer depends on the exact push cadence
    // — a plain per-chunk sleep is enough to pace roughly real-time and keep
    // each member's send queue shallow. Nothing here holds a lock on the group,
    // so clients arriving mid-stream (via the accept or discovery loops) are
    // added concurrently and join the playback in progress.
    let total_chunks = wav.data.len().div_ceil(chunk_bytes);
    for (i, chunk) in wav.data.chunks(chunk_bytes).enumerate() {
        let timestamp_us = group.push_audio(chunk);
        if i % 10 == 0 || i + 1 == total_chunks {
            println!(
                "chunk {}/{total_chunks} (timestamp {timestamp_us}us)",
                i + 1
            );
        }
        tokio::time::sleep(Duration::from_millis(args.chunk_ms)).await;
    }

    println!("done sending audio, ending stream");
    group.end_stream().await;

    // Give the last chunks time to actually finish playing before hanging up.
    tokio::time::sleep(Duration::from_secs(2)).await;
    println!("done — Ctrl+C to exit (connections stay open so you can inspect further)");
    tokio::signal::ctrl_c().await.ok();
    Ok(())
}

async fn wait_for_first_member(group: &Arc<Group>) {
    loop {
        if !group.is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Accept clients that dial in to us.
fn spawn_accept_loop(listener: ServerListener, group: Arc<Group>) {
    tokio::spawn(async move {
        loop {
            match listener.accept().await {
                Ok((conn, addr)) => {
                    println!(
                        "[{addr}] client connected (inbound): id={} name={:?} roles={:?}",
                        conn.client_id(),
                        conn.hello().name,
                        conn.active_roles()
                    );
                    add_and_drain(&group, conn).await;
                }
                Err(e) => {
                    println!("accept error: {e}");
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
    });
}

/// Discover clients that only run their own embedded server, dial them, and
/// keep them connected via ClientManager (which retries with backoff on
/// disconnect and re-dials if a device reappears at a new address).
fn spawn_manager_loop(
    group: Arc<Group>,
    server_id: String,
    name: String,
) -> Result<ClientManager, Box<dyn std::error::Error>> {
    let (manager, mut events) =
        ClientManager::start(server_id, name, Arc::new(DefaultClock::default()))?;
    println!("discovering Sendspin clients via mDNS (_sendspin._tcp.local.)...");
    tokio::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                ClientEvent::Connected {
                    client_id,
                    fullname,
                    active_roles,
                    sender,
                } => {
                    println!(
                        "[{client_id}] connected (dialed via {fullname}): roles={active_roles:?}"
                    );
                    if let Err(e) = group.add_member(client_id, sender).await {
                        println!("failed to add member to group: {e}");
                    }
                }
                ClientEvent::Message { client_id, message } => match *message {
                    Message::ClientGoodbye(g) => {
                        println!("[{client_id}] client/goodbye: {:?}", g.reason);
                    }
                    other => println!("[{client_id}] <- {other:?}"),
                },
                ClientEvent::Disconnected { client_id } => {
                    println!("[{client_id}] connection closed, will retry in the background");
                    group.remove_member(&client_id);
                }
            }
        }
    });
    Ok(manager)
}

/// Dial one specific URL given via `--dial`, once, at startup.
async fn dial_one(group: &Arc<Group>, url: &str, server_id: &str, name: &str) {
    dial_and_add(group, url, server_id, name).await;
}

async fn dial_and_add(group: &Arc<Group>, url: &str, server_id: &str, name: &str) {
    match dial_client(url, server_id, name, Arc::new(DefaultClock::default())).await {
        Ok(conn) => {
            println!(
                "[{url}] client connected (dialed): id={} name={:?} roles={:?}",
                conn.client_id(),
                conn.hello().name,
                conn.active_roles()
            );
            add_and_drain(group, conn).await;
        }
        Err(e) => println!("[{url}] dial failed: {e}"),
    }
}

async fn add_and_drain(group: &Arc<Group>, conn: ServerConnection) {
    let client_id = conn.client_id().to_string();
    let sender = conn.sender();
    spawn_message_drain(client_id.clone(), conn);
    if let Err(e) = group.add_member(client_id, sender).await {
        println!("failed to add member to group: {e}");
    }
}

/// Print every client/state, client/command, and client/goodbye a connection
/// sends for as long as it lasts — this is what you want on screen when a
/// real device does something unexpected.
fn spawn_message_drain(client_id: String, mut conn: ServerConnection) {
    tokio::spawn(async move {
        while let Some(msg) = conn.recv_message().await {
            match &msg {
                Message::ClientGoodbye(g) => {
                    println!("[{client_id}] client/goodbye: {:?}", g.reason);
                }
                other => println!("[{client_id}] <- {other:?}"),
            }
        }
        println!("[{client_id}] connection closed");
    });
}
