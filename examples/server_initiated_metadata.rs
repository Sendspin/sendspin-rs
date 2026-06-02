// ABOUTME: Inbound listener that announces the metadata role
// ABOUTME: Advertises via mDNS and applies spec multi-server arbitration
// ABOUTME: Only partly spec-compliant: a real client would also persist the last-played server across restarts

use clap::Parser;
use mdns_sd::{ServiceDaemon, ServiceInfo};
use sendspin::protocol::client::ConnectionGuard;
use sendspin::protocol::messages::{ConnectionReason, GoodbyeReason, Message, PlaybackState};
use sendspin::ProtocolClientBuilder;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Sendspin metadata client using the inbound listener
#[derive(Parser, Debug)]
#[command(name = "server_initiated_metadata")]
struct Args {
    /// Address to bind the listener on (8928 is the recommended client port)
    #[arg(short, long, default_value = "0.0.0.0:8928")]
    bind: String,

    /// Friendly name advertised in the mDNS TXT record
    #[arg(short, long, default_value = "Metadata Listener")]
    name: String,

    /// Persistent client id (defaults to a random UUID)
    #[arg(short, long)]
    id: Option<String>,
}

/// The server we are currently connected to. The spec allows only one.
struct Active {
    peer: std::net::SocketAddr,
    server_id: String,
    reason: ConnectionReason,
    guard: ConnectionGuard,
}

#[derive(Default)]
struct State {
    active: Option<Active>,
    last_played: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();
    let client_id = args.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let listener = ProtocolClientBuilder::builder()
        .client_id(client_id)
        .name(args.name.clone())
        .metadata()
        .build()
        .listen(&args.bind)
        .await?;

    let port = listener.local_addr()?.port();

    // Advertise _sendspin._tcp.local. so servers can discover and dial in.
    let daemon = ServiceDaemon::new()?;
    let service = ServiceInfo::new(
        "_sendspin._tcp.local.",
        &args.name,
        "sendspin-metadata-listener.local.",
        "",
        port,
        &[("path", "/sendspin"), ("name", args.name.as_str())][..],
    )?
    .enable_addr_auto();
    daemon.register(service)?;

    println!("Listening on :{port}, advertising _sendspin._tcp.local. — waiting for server...");

    let state = Arc::new(Mutex::new(State::default()));

    loop {
        let (client, peer) = listener.accept().await?;
        let conn = client.split();
        let server_id = conn.server_hello.server_id.clone();
        let reason = conn.server_hello.connection_reason.clone();
        println!("[{peer}] handshake — server_id={server_id} connection_reason={reason:?}");

        let mut st = state.lock().await;
        if !keep_new(&st, &server_id, reason.clone()) {
            drop(st);
            println!("[{peer}] keeping current server — goodbye(another_server)");
            // Spawn the goodbye so a slow flush can't block the next accept().
            let guard = conn.guard;
            tokio::spawn(async move {
                guard.disconnect(GoodbyeReason::AnotherServer).await.ok();
            });
            continue;
        }

        // New server wins: displace the current one before taking over.
        let displaced = st.active.take();
        st.active = Some(Active {
            peer,
            server_id: server_id.clone(),
            reason,
            guard: conn.guard,
        });
        drop(st);

        if let Some(old) = displaced {
            println!(
                "switching {} -> {server_id} — goodbye(another_server)",
                old.server_id
            );
            tokio::spawn(async move {
                old.guard
                    .disconnect(GoodbyeReason::AnotherServer)
                    .await
                    .ok();
            });
        }

        let state = Arc::clone(&state);
        let mut messages = conn.messages;
        tokio::spawn(async move {
            while let Some(msg) = messages.recv().await {
                match msg {
                    Message::GroupUpdate(g) if g.playback_state == Some(PlaybackState::Playing) => {
                        // A spec-compliant client would also persist this to disk here.
                        state.lock().await.last_played = Some(server_id.clone());
                        println!("[{peer}] playing — recorded as last played");
                    }
                    Message::ServerState(s) => println!("[{peer}] [METADATA] {:?}", s.metadata),
                    other => println!("[{peer}] [MSG] {other:?}"),
                }
            }
            // Clear active only if this exact connection is still current.
            let mut st = state.lock().await;
            if st.active.as_ref().map(|a| a.peer) == Some(peer) {
                st.active = None;
            }
            println!("[{peer}] disconnected");
        });
    }
}

/// Spec multi-server decision (README "Multiple Servers"): does the newly
/// connected server replace the current one?
fn keep_new(st: &State, new_id: &str, new_reason: ConnectionReason) -> bool {
    let Some(active) = &st.active else {
        return true;
    };
    match (new_reason, &active.reason) {
        // Playback always wins.
        (ConnectionReason::Playback, _) => true,
        // Discovery never displaces an active playback connection.
        (ConnectionReason::Discovery, ConnectionReason::Playback) => false,
        // Both discovery: prefer the last played server, else keep existing.
        (ConnectionReason::Discovery, ConnectionReason::Discovery) => {
            st.last_played.as_deref() == Some(new_id)
        }
    }
}
