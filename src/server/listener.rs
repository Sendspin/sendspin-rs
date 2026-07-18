// ABOUTME: Inbound WebSocket acceptor that drives the Sendspin protocol-server
// ABOUTME: state machine (client/hello -> server/hello) on every peer that connects.

use crate::error::Error;
use crate::server::connection::ServerConnection;
use crate::sync::raw_clock::{Clock, DefaultClock};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{lookup_host, TcpListener, TcpSocket, TcpStream, ToSocketAddrs};
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::http;
use tokio_tungstenite::{accept_async, accept_hdr_async, WebSocketStream};

/// Accept inbound WebSocket peers and drive each one through the
/// protocol-**server** state machine: read `client/hello`, reply
/// `server/hello`, then hand back a [`ServerConnection`] for pushing
/// stream/audio messages and receiving state/command/goodbye.
///
/// This is the counterpart to [`crate::protocol::listener::ProtocolListener`],
/// which accepts inbound connections but drives the protocol-**client** role
/// on them (used when a server dials out to a client that runs its own tiny
/// WS listener — a reversed-topology case). `ServerListener` is what a
/// Sendspin server itself binds to accept the usual inbound player
/// connections.
///
/// [`Self::accept`] drives the full handshake before returning, so it serves
/// one inbound connection at a time — a slow handshake blocks the next
/// `accept()`. `tokio::spawn` a task per accepted connection if you need to
/// keep accepting while driving existing ones.
pub struct ServerListener {
    tcp: TcpListener,
    server_id: String,
    server_name: String,
    path: Option<String>,
    clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for ServerListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerListener")
            .field("local_addr", &self.tcp.local_addr().ok())
            .field("server_id", &self.server_id)
            .field("path", &self.path)
            .finish()
    }
}

impl ServerListener {
    /// Bind a listener. `server_id` should be stable across restarts (it's
    /// how a client recognizes "the same server" across reconnects);
    /// `server_name` is human-readable and shown to users.
    pub async fn bind(
        addr: impl ToSocketAddrs,
        server_id: impl Into<String>,
        server_name: impl Into<String>,
    ) -> Result<Self, Error> {
        // Bind with SO_REUSEADDR so a port freed by a just-torn-down server can
        // be reused immediately — otherwise recreating a group on the same port
        // can race the old socket's close and fail with EADDRINUSE.
        let sockaddr: SocketAddr = lookup_host(addr)
            .await
            .map_err(|e| Error::Connection(format!("resolve failed: {e}")))?
            .next()
            .ok_or_else(|| Error::Connection("no address to bind".to_string()))?;
        let socket = if sockaddr.is_ipv4() {
            TcpSocket::new_v4()
        } else {
            TcpSocket::new_v6()
        }
        .map_err(|e| Error::Connection(format!("socket create failed: {e}")))?;
        socket
            .set_reuseaddr(true)
            .map_err(|e| Error::Connection(format!("set_reuseaddr failed: {e}")))?;
        socket
            .bind(sockaddr)
            .map_err(|e| Error::Connection(format!("bind failed: {e}")))?;
        let tcp = socket
            .listen(128)
            .map_err(|e| Error::Connection(format!("listen failed: {e}")))?;
        Ok(Self {
            tcp,
            server_id: server_id.into(),
            server_name: server_name.into(),
            path: None,
            clock: Arc::new(DefaultClock::default()),
        })
    }

    /// Restrict accepted connections to a specific HTTP path (the Sendspin
    /// spec fixes this to `/sendspin` for real deployments). Mismatches are
    /// rejected with HTTP 404 during the WebSocket handshake; the listener
    /// stays bound. Defaults to accepting any path.
    pub fn path(mut self, path: impl Into<String>) -> Self {
        let path = path.into();
        self.path = Some(if path.starts_with('/') {
            path
        } else {
            format!("/{path}")
        });
        self
    }

    /// Use a custom clock instead of [`DefaultClock`] — mainly for tests that
    /// need deterministic or synchronized-with-a-peer timestamps.
    pub fn clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Accept the next inbound connection, returning the driven
    /// [`ServerConnection`] and the peer's address.
    ///
    /// Per-peer failures surface as [`Error`] without affecting the
    /// listener; callers typically call `accept()` in a loop.
    ///
    /// Not cancel-safe: dropping the returned future mid-handshake tears
    /// down that connection. A peer that connects but stalls the handshake
    /// will block this future indefinitely, so wrap it in a timeout if
    /// untrusted peers can reach the socket.
    pub async fn accept(&self) -> Result<(ServerConnection, SocketAddr), Error> {
        let (tcp_stream, peer_addr) = self
            .tcp
            .accept()
            .await
            .map_err(|e| Error::Connection(format!("TCP accept failed: {e}")))?;
        log::debug!("Accepted TCP connection from {}", peer_addr);

        match self.handshake_and_drive(tcp_stream).await {
            Ok(conn) => Ok((conn, peer_addr)),
            Err(e) => {
                log::warn!("Inbound handshake from {peer_addr} failed: {e}");
                Err(e)
            }
        }
    }

    async fn handshake_and_drive(&self, tcp_stream: TcpStream) -> Result<ServerConnection, Error> {
        let ws = self.handshake_ws(tcp_stream).await?;
        ServerConnection::drive(
            ws,
            &self.server_id,
            &self.server_name,
            Arc::clone(&self.clock),
        )
        .await
    }

    // `ErrorResponse` is large by Clippy's standard but mandated by
    // tungstenite's `Callback` trait — same tradeoff ProtocolListener makes.
    #[allow(clippy::result_large_err)]
    async fn handshake_ws<S>(&self, stream: S) -> Result<WebSocketStream<S>, Error>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        match &self.path {
            Some(expected_path) => {
                let expected = expected_path.clone();
                let callback = move |request: &Request, response: Response| {
                    if request.uri().path() == expected {
                        Ok(response)
                    } else {
                        log::debug!(
                            "Rejecting inbound connection: path {:?} != expected {:?}",
                            request.uri().path(),
                            expected
                        );
                        Err(http::Response::builder()
                            .status(http::StatusCode::NOT_FOUND)
                            .body(None)
                            .expect("static 404 response is well-formed"))
                            as Result<Response, ErrorResponse>
                    }
                };
                accept_hdr_async(stream, callback)
                    .await
                    .map_err(|e| Error::WebSocket(format!("WebSocket handshake failed: {e}")))
            }
            None => accept_async(stream)
                .await
                .map_err(|e| Error::WebSocket(format!("WebSocket handshake failed: {e}"))),
        }
    }

    /// Local bound address.
    pub fn local_addr(&self) -> Result<SocketAddr, Error> {
        self.tcp
            .local_addr()
            .map_err(|e| Error::Connection(format!("local_addr failed: {e}")))
    }
}
