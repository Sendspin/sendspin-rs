// ABOUTME: Server-initiated connections to Sendspin clients that only run
// ABOUTME: their own embedded WebSocket server (never dialing out
// ABOUTME: themselves) — discovered via discovery::ClientBrowser, dialed here.

use crate::error::Error;
use crate::protocol::messages::ConnectionReason;
use crate::server::connection::ServerConnection;
use crate::sync::raw_clock::Clock;
use std::sync::Arc;
use tokio_tungstenite::connect_async;

/// Dial a Sendspin client's own WebSocket server (e.g. a URL discovered via
/// [`crate::server::ClientBrowser`]) and drive the server-role handshake over
/// the resulting connection.
///
/// The protocol-level roles are identical regardless of which side initiated
/// the TCP connection — the client still sends `client/hello` first, this
/// still replies `server/hello` — so this is otherwise exactly
/// [`crate::server::ServerListener::accept`]'s handshake, just dialed instead
/// of accepted. `sendspin-rs`'s own `protocol::listener::ProtocolListener` is
/// the client-role mirror of this for the reverse case (a client accepting a
/// server that dials in).
pub async fn dial_client(
    url: &str,
    server_id: &str,
    server_name: &str,
    clock: Arc<dyn Clock>,
) -> Result<ServerConnection, Error> {
    let (ws, _response) = connect_async(url)
        .await
        .map_err(|e| Error::Connection(format!("dial to {url} failed: {e}")))?;
    // The server dialed out to stream to this client, so announce Playback.
    ServerConnection::drive(
        ws,
        server_id,
        server_name,
        ConnectionReason::Playback,
        clock,
    )
    .await
}
