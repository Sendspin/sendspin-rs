// ABOUTME: Basic example demonstrating WebSocket connection and handshake
// ABOUTME: Connects to server, sends client/hello, receives server/hello

use clap::Parser;
use sendspin::protocol::client::ProtocolClient;
use sendspin::protocol::messages::{AudioFormatSpec, ClientHello, DeviceInfo, PlayerV1Support};

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

    let _client = ProtocolClient::connect(&args.server, hello).await?;

    println!("Connected! Waiting for server hello...");

    // This would block waiting for messages
    // For now, just demonstrate connection works

    Ok(())
}
