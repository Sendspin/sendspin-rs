// ABOUTME: Continuous discovery + reconnect-with-backoff supervision for
// ABOUTME: clients that only run their own embedded server — the missing
// ABOUTME: robustness half of dial_client, mirroring aiosendspin's
// ABOUTME: SendspinServer._handle_client_connection reconnect loop.

use crate::protocol::messages::Message;
use crate::server::connection::{ServerConnection, ServerSender};
use crate::server::dial::dial_client;
use crate::server::discovery::ClientBrowser;
use crate::sync::raw_clock::Clock;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use tokio::time::Instant;

/// Reconnect backoff ceiling — matches aiosendspin's
/// `MAX_RECONNECT_BACKOFF_S`.
const MAX_BACKOFF: Duration = Duration::from_secs(300);

/// A connection must last at least this long before backoff resets to the
/// minimum — matches aiosendspin's `STABLE_SERVER_INITIATED_SESSION_S`. A
/// device that connects and immediately drops (crash-looping, mid-OTA,
/// whatever) doesn't get hammered at 1-second intervals; the backoff keeps
/// climbing until a connection actually holds.
const STABLE_SESSION: Duration = Duration::from_secs(10);

/// Events for a client discovered and managed by [`ClientManager`]. The
/// manager owns each connection's message loop internally so it can detect
/// disconnection and drive reconnects — callers get a [`ServerSender`] for
/// control instead of the raw [`ServerConnection`].
#[derive(Debug)]
pub enum ClientEvent {
    /// A client connected — the first time, or after a reconnect. `client_id`
    /// is stable across reconnects (it comes from the client's own
    /// `client/hello`), so callers can use it as the group-membership key.
    Connected {
        /// The connected client's identifier, from its `client/hello`.
        client_id: String,
        /// The mDNS instance fullname this connection was dialed from — stable
        /// discovery identity, lets callers map a connection back to the
        /// discovered service (the `client_id` may be an opaque MAC that does
        /// not match the advertised name).
        fullname: String,
        /// Roles this server granted the client.
        active_roles: Vec<String>,
        /// Sender for pushing stream/audio/command messages to this client.
        sender: ServerSender,
    },
    /// A `client/state`, `client/command`, or `client/goodbye` message from
    /// a connected client (`client/time` is consumed internally and never
    /// forwarded, same convention as [`ServerConnection::recv_message`]).
    Message {
        /// Which client sent this message.
        client_id: String,
        /// The message itself.
        message: Box<Message>,
    },
    /// The client disconnected. If it's still discoverable, a reconnect
    /// attempt is already running in the background — this just tells the
    /// caller to stop treating `client_id` as a live group member for now.
    Disconnected {
        /// The client that disconnected.
        client_id: String,
    },
}

struct ManagedClient {
    handle: JoinHandle<()>,
    url: String,
}

/// Discovers Sendspin clients that only run their own embedded server (see
/// [`crate::server::ClientBrowser`]'s docs for why real hardware routinely
/// needs this) and keeps each one connected: dials on discovery, retries
/// with capped exponential backoff on failure or disconnect, and re-dials
/// promptly if the same device reappears at a new address instead of
/// retrying the stale one forever.
pub struct ClientManager {
    tasks: Arc<Mutex<HashMap<String, ManagedClient>>>,
    browse_handle: JoinHandle<()>,
}

impl ClientManager {
    /// Start discovering and managing every Sendspin client this process can
    /// see on the network. Returns immediately; events arrive on the
    /// returned receiver as they happen. Drop the returned `ClientManager`
    /// to stop discovery and every reconnect loop it's running.
    ///
    /// This is unfiltered — on a LAN with other Sendspin servers already
    /// actively serving some of these clients (a different instance of this
    /// same add-on, a Music Assistant install, anything), you will compete
    /// with them for those devices. Most real deployments know which
    /// clients they're supposed to own (e.g. from their own configured
    /// device list) — see [`Self::start_filtered`] to scope discovery to
    /// just those, both for that reason and because it's what let this
    /// crate's own test suite avoid dialing real hardware sitting on the
    /// same network it was developed against.
    pub fn start(
        server_id: impl Into<String>,
        server_name: impl Into<String>,
        clock: Arc<dyn Clock>,
    ) -> Result<(Self, UnboundedReceiver<ClientEvent>), crate::error::Error> {
        Self::start_filtered(server_id, server_name, clock, |_fullname| true)
    }

    /// Like [`Self::start`], but only discovers and manages clients whose
    /// mDNS instance full name (e.g. `my-device._sendspin._tcp.local.`)
    /// satisfies `allow`.
    pub fn start_filtered(
        server_id: impl Into<String>,
        server_name: impl Into<String>,
        clock: Arc<dyn Clock>,
        allow: impl Fn(&str) -> bool + Send + 'static,
    ) -> Result<(Self, UnboundedReceiver<ClientEvent>), crate::error::Error> {
        let server_id = server_id.into();
        let server_name = server_name.into();
        let (event_tx, event_rx) = unbounded_channel();
        let tasks: Arc<Mutex<HashMap<String, ManagedClient>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let browser = ClientBrowser::new()?;
        let tasks_for_browse = Arc::clone(&tasks);
        let browse_handle = tokio::spawn(async move {
            while let Some((fullname, url)) = browser.next_client().await {
                if !allow(&fullname) {
                    continue;
                }
                let mut tasks = tasks_for_browse.lock().unwrap();
                let should_spawn = match tasks.get(&fullname) {
                    Some(existing) if existing.url == url => false,
                    Some(existing) => {
                        log::info!(
                            "[{fullname}] address changed ({} -> {url}), reconnecting",
                            existing.url
                        );
                        existing.handle.abort();
                        true
                    }
                    None => true,
                };
                if should_spawn {
                    let handle = spawn_client_supervisor(
                        fullname.clone(),
                        url.clone(),
                        server_id.clone(),
                        server_name.clone(),
                        Arc::clone(&clock),
                        event_tx.clone(),
                    );
                    tasks.insert(fullname, ManagedClient { handle, url });
                }
            }
        });

        Ok((
            Self {
                tasks,
                browse_handle,
            },
            event_rx,
        ))
    }
}

impl Drop for ClientManager {
    fn drop(&mut self) {
        self.browse_handle.abort();
        for (_, managed) in self.tasks.lock().unwrap().drain() {
            managed.handle.abort();
        }
    }
}

fn spawn_client_supervisor(
    fullname: String,
    url: String,
    server_id: String,
    server_name: String,
    clock: Arc<dyn Clock>,
    event_tx: UnboundedSender<ClientEvent>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut backoff = Duration::from_secs(1);
        loop {
            match dial_client(&url, &server_id, &server_name, Arc::clone(&clock)).await {
                Ok(conn) => {
                    let started = Instant::now();
                    let client_id = conn.client_id().to_string();
                    log::info!("[{fullname}] connected as {client_id} ({url})");
                    let _ = event_tx.send(ClientEvent::Connected {
                        client_id: client_id.clone(),
                        fullname: fullname.clone(),
                        active_roles: conn.active_roles().to_vec(),
                        sender: conn.sender(),
                    });

                    drain_until_disconnected(conn, &client_id, &event_tx).await;

                    let _ = event_tx.send(ClientEvent::Disconnected {
                        client_id: client_id.clone(),
                    });
                    log::info!("[{fullname}] disconnected, will retry");
                    if started.elapsed() >= STABLE_SESSION {
                        backoff = Duration::from_secs(1);
                    }
                }
                Err(e) => {
                    log::warn!("[{fullname}] dial to {url} failed: {e}");
                }
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(MAX_BACKOFF);
        }
    })
}

async fn drain_until_disconnected(
    mut conn: ServerConnection,
    client_id: &str,
    event_tx: &UnboundedSender<ClientEvent>,
) {
    while let Some(message) = conn.recv_message().await {
        if event_tx
            .send(ClientEvent::Message {
                client_id: client_id.to_string(),
                message: Box::new(message),
            })
            .is_err()
        {
            return; // receiver dropped; nothing left to report to
        }
    }
}
