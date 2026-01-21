// ABOUTME: Minimal test to verify we receive ALL server messages
// ABOUTME: Just connects and prints everything the server sends

use clap::Parser;
use sendspin::protocol::client::ProtocolClient;
use sendspin::protocol::messages::{
    AudioFormatSpec, ClientHello, ClientState, DeviceInfo, Message, PlayerState, PlayerSyncState,
    PlayerV1Support,
};

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
        supported_roles: vec!["player@v1".to_string()],
        device_info: Some(DeviceInfo {
            product_name: Some("Minimal Test".to_string()),
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
            buffer_capacity: 100,
            supported_commands: vec!["volume".to_string()],
        }),
        artwork_v1_support: None,
        visualizer_v1_support: None,
    };

    println!("Connecting to {}...", args.server);
    let client = ProtocolClient::connect(&args.server, hello).await?;
    println!("Connected! Server said hello.");

    // Split client
    let (mut message_rx, mut audio_rx, _clock_sync, ws_tx) = client.split();

    // Send client/state (handshake step 3)
    let client_state = Message::ClientState(ClientState {
        player: Some(PlayerState {
            state: PlayerSyncState::Synchronized,
            volume: Some(100),
            muted: Some(false),
        }),
    });
    ws_tx.send_message(client_state).await?;
    println!("Sent client/state");

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
