// ABOUTME: Basic example demonstrating secure WebSocket connection and handshake
// ABOUTME: Connects to server, sends client/hello, receives server/hello
// ABOUTME: Run with: cargo run --example secure_socket --features native-tls

use clap::Parser;
use sendspin::ProtocolClientBuilder;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

/// Sendspin basic client
#[derive(Parser, Debug)]
#[command(name = "basic_client")]
#[command(about = "Test connection to Sendspin server", long_about = None)]
struct Args {
    /// WebSocket URL of the Sendspin server
    #[arg(short, long, default_value = "wss://localhost:8927/sendspin")]
    server: String,

    /// Client name
    #[arg(short, long, default_value = "Sendspin-RS Basic Client")]
    name: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    println!("Connecting to {}...", args.server);

    // Build a request with custom headers (e.g., for auth proxy)
    let mut request = args.server.into_client_request().unwrap();
    request
        .headers_mut()
        .insert("cookie", "ingress_session=<session_token>".parse().unwrap());

    let _client = ProtocolClientBuilder::builder()
        .client_id(uuid::Uuid::new_v4().to_string())
        .name(args.name)
        .product_name(Some("Sendspin-RS Basic Client".to_string()))
        .software_version(Some("0.1.0".to_string()))
        .build()
        .connect(request)
        .await?;

    println!("Connected! Waiting for server hello...");

    // This would block waiting for messages
    // For now, just demonstrate connection works

    Ok(())
}
