// ABOUTME: Minimal test to verify we receive ALL server messages
// ABOUTME: Just connects and prints everything the server sends

use clap::Parser;
use sendspin::protocol::client::ProtocolClient;
use sendspin::protocol::messages::{ClientState, Message, PlayerState, PlayerSyncState};
use sendspin::ProtocolClientBuilder;

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

    println!("Connecting to {}...", args.server);
    let test = ProtocolClientBuilder::builder()
        .client_id(uuid::Uuid::new_v4().to_string())
        .name("Minimal Test Client".to_string())
        .build();

    let client = test.connect(&args.server).await?;
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
