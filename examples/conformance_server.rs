// ABOUTME: Conformance-suite driver for the server role — accepts one
// ABOUTME: connection and streams tests/conformance/fixtures/stimulus.pcm
// ABOUTME: through the identical canned stimulus oracle_server.py drives
// ABOUTME: against the real aiosendspin server. See tests/conformance/README.md.

use sendspin::protocol::messages::{PlayerCommand, PlayerCommandType, StreamPlayerConfig};
use sendspin::ServerListener;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

const SAMPLE_RATE: u32 = 48000;
const CHANNELS: u8 = 2;
const BIT_DEPTH: u8 = 16;
const CHUNK_MS: u64 = 100;
// Must match tests/conformance/oracle_server.py exactly — the whole point of
// this suite is that both servers are driven through the *same* stimulus.
const VOLUME_AT_CHUNK: usize = 5;
const VOLUME_VALUE: u8 = 42;
const MUTE_AT_CHUNK: usize = 12;

#[tokio::main]
async fn main() {
    env_logger::init();
    let port: u16 = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8927);
    let addr = format!("127.0.0.1:{port}");

    let fixture_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/conformance/fixtures/stimulus.pcm");
    let stimulus = fs::read(&fixture_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", fixture_path.display()));

    let chunk_bytes = (SAMPLE_RATE as u64 * CHUNK_MS / 1000) as usize
        * CHANNELS as usize
        * (BIT_DEPTH as usize / 8);
    let chunks: Vec<&[u8]> = stimulus.chunks(chunk_bytes).collect();

    let listener = ServerListener::bind(addr, "conformance-rust", "Conformance Rust")
        .await
        .expect("bind");
    println!("listening on {}", listener.local_addr().unwrap());

    let (conn, _peer) = tokio::time::timeout(Duration::from_secs(30), listener.accept())
        .await
        .expect("accept timed out")
        .expect("accept failed");
    println!("accepted client_id={}", conn.client_id());
    let sender = conn.sender();

    sender
        .send_stream_start(StreamPlayerConfig {
            codec: "pcm".to_string(),
            sample_rate: SAMPLE_RATE,
            channels: CHANNELS,
            bit_depth: BIT_DEPTH,
            codec_header: None,
        })
        .await
        .expect("send_stream_start");

    let mut timestamp_us: i64 = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        if i == VOLUME_AT_CHUNK {
            sender
                .send_player_command(PlayerCommand {
                    command: PlayerCommandType::Volume,
                    volume: Some(VOLUME_VALUE),
                    mute: None,
                    static_delay_ms: None,
                })
                .await
                .expect("send volume command");
        }
        if i == MUTE_AT_CHUNK {
            sender
                .send_player_command(PlayerCommand {
                    command: PlayerCommandType::Mute,
                    volume: None,
                    mute: Some(true),
                    static_delay_ms: None,
                })
                .await
                .expect("send mute command");
        }
        sender
            .send_audio_chunk(timestamp_us, chunk)
            .await
            .expect("send_audio_chunk");
        timestamp_us += (CHUNK_MS * 1000) as i64;
        tokio::time::sleep(Duration::from_millis(CHUNK_MS)).await;
    }

    sender.send_stream_end().await.expect("send_stream_end");
    tokio::time::sleep(Duration::from_secs(1)).await;

    conn.disconnect().await.expect("disconnect");
    println!(
        "rust server done: sent {} chunks, {} bytes",
        chunks.len(),
        stimulus.len()
    );
}
