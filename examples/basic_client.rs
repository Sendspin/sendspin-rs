// ABOUTME: Basic example demonstrating WebSocket connection and handshake
// ABOUTME: Connects to server, sends client/hello, receives server/hello

use clap::Parser;
use sendspin::protocol::client::ProtocolClient;
use sendspin::protocol::messages::{AudioFormatSpec, ClientHello, DeviceInfo, PlayerSupport};

/// Sendspin basic client
#[derive(Parser, Debug)]
#[command(name = "basic_client")]
#[command(about = "Test connection to Sendspin server", long_about = None)]
struct Args {
    /// WebSocket URL of the Sendspin server
    #[arg(short, long, default_value = "ws://localhost:8927/sendspin")]
    server: String,

    /// Client name
    #[arg(short, long, default_value = "Sendspin-RS Basic Client")]
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
        supported_roles: vec!["player".to_string()],
        device_info: DeviceInfo {
            product_name: args.name.clone(),
            manufacturer: "Sendspin".to_string(),
            software_version: "0.1.0".to_string(),
        },
        player_support: Some(PlayerSupport {
            support_codecs: vec!["pcm".to_string()],
            support_channels: vec![2], // Stereo
            support_sample_rates: vec![48000, 96000, 192000],
            support_bit_depth: vec![16, 24],
            support_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 100,
            supported_commands: vec!["play".to_string(), "pause".to_string()],
        }),
        metadata_support: None,
    };

    println!("Connecting to {}...", args.server);

    let _client = ProtocolClient::connect(&args.server, hello).await?;

    println!("Connected! Waiting for server hello...");

    // This would block waiting for messages
    // For now, just demonstrate connection works

    Ok(())
}
