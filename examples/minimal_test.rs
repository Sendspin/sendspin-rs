// ABOUTME: Minimal test to verify we receive ALL server messages
// ABOUTME: Just connects and prints everything the server sends

use clap::Parser;
use sendspin::protocol::client::ProtocolClient;
use sendspin::protocol::messages::{ClientHello, DeviceInfo, PlayerSupport, AudioFormatSpec, Message, PlayerUpdate};

/// Minimal Sendspin test client
#[derive(Parser, Debug)]
#[command(name = "minimal_test")]
struct Args {
    /// WebSocket URL of the Sendspin server
    #[arg(short, long, default_value = "ws://192.168.200.8:8927/sendspin")]
    server: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    let hello = ClientHello {
        client_id: uuid::Uuid::new_v4().to_string(),
        name: "Minimal Test Client".to_string(),
        version: 1,
        supported_roles: vec!["player".to_string()],
        device_info: DeviceInfo {
            product_name: "Minimal Test".to_string(),
            manufacturer: "Sendspin".to_string(),
            software_version: "0.1.0".to_string(),
        },
        player_support: Some(PlayerSupport {
            support_codecs: vec!["pcm".to_string()],
            support_channels: vec![2],
            support_sample_rates: vec![48000],
            support_bit_depth: vec![24],
            support_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 100,
            supported_commands: vec!["play".to_string()],
        }),
        metadata_support: None,
    };

    println!("Connecting to {}...", args.server);
    let client = ProtocolClient::connect(&args.server, hello).await?;
    println!("Connected! Server said hello.");

    // Split client
    let (mut message_rx, mut audio_rx, _clock_sync, ws_tx) = client.split();

    // Send player/update (handshake step 3)
    let player_update = Message::PlayerUpdate(PlayerUpdate {
        state: "idle".to_string(),
        volume: 100,
        muted: false,
    });
    ws_tx.send_message(player_update).await?;
    println!("Sent player/update");

    println!("\nListening for ALL messages from server...\n");

    // Just print everything we receive
    loop {
        tokio::select! {
            Some(msg) = message_rx.recv() => {
                println!("[TEXT MESSAGE] {:?}", msg);
            }
            Some(chunk) = audio_rx.recv() => {
                println!("[AUDIO CHUNK] timestamp={} size={} bytes",
                    chunk.timestamp, chunk.data.len());
            }
            else => {
                println!("Connection closed");
                break;
            }
        }
    }

    Ok(())
}
