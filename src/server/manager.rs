// ABOUTME: Continuous discovery + reconnect-with-backoff supervision for clients
// ABOUTME: that only run their own embedded server (the supervised form of dial_client)

use crate::protocol::messages::Message;
use crate::server::connection::{ServerConnection, ServerSender};
use crate::server::dial::dial_client;
use crate::server::discovery::{ClientBrowser, Discovered};
use crate::sync::raw_clock::Clock;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::Instant;

/// Initial reconnect backoff, and the value backoff resets to after a stable
/// session or an address change.
const MIN_BACKOFF: Duration = Duration::from_secs(1);

/// Reconnect backoff ceiling — matches aiosendspin's `MAX_RECONNECT_BACKOFF_S`.
const MAX_BACKOFF: Duration = Duration::from_secs(300);

/// A connection must last at least this long before backoff resets to the
/// minimum (matches aiosendspin's `STABLE_SERVER_INITIATED_SESSION_S`), so a
/// crash-looping device isn't hammered at 1-second intervals.
const STABLE_SESSION: Duration = Duration::from_secs(10);

/// Control signal sent to a client supervisor task via a watch channel.
#[derive(Clone, Debug)]
enum Directive {
    /// (Re)dial the client at this URL. A new URL for an already-connected
    /// client makes the supervisor drop the current connection and redial.
    Dial(String),
    /// Stop supervising — the device's mDNS advertisement was removed.
    Stop,
}

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
    directive_tx: watch::Sender<Directive>,
    url: String,
}

/// Discovers Sendspin clients that only run their own embedded server and keeps
/// each one connected: dials on discovery, retries with capped exponential
/// backoff on failure or disconnect, re-dials promptly if a device reappears at
/// a new address, and stops supervising once a device's mDNS advertisement is
/// removed.
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
    /// This is unfiltered: on a LAN where other servers already serve some of
    /// these clients, you will compete with them for those devices. Use
    /// [`Self::start_filtered`] to scope discovery to a known set of devices.
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
            while let Some(event) = browser.next_event().await {
                match event {
                    Discovered::Found { fullname, url } => {
                        if !allow(&fullname) {
                            continue;
                        }
                        let mut tasks = tasks_for_browse.lock().unwrap();
                        match tasks.get_mut(&fullname) {
                            // Same device, same address — already supervised.
                            Some(existing) if existing.url == url => {}
                            // Same device at a new address: redirect the running
                            // supervisor (it closes the current connection, emits
                            // Disconnected, and redials). Respawn only if the
                            // supervisor has already exited.
                            Some(existing) => {
                                log::info!(
                                    "[{fullname}] address changed ({} -> {url}), reconnecting",
                                    existing.url
                                );
                                existing.url = url.clone();
                                if existing
                                    .directive_tx
                                    .send(Directive::Dial(url.clone()))
                                    .is_err()
                                {
                                    *existing = spawn_supervisor(
                                        fullname.clone(),
                                        url,
                                        server_id.clone(),
                                        server_name.clone(),
                                        Arc::clone(&clock),
                                        event_tx.clone(),
                                    );
                                }
                            }
                            None => {
                                let managed = spawn_supervisor(
                                    fullname.clone(),
                                    url,
                                    server_id.clone(),
                                    server_name.clone(),
                                    Arc::clone(&clock),
                                    event_tx.clone(),
                                );
                                tasks.insert(fullname, managed);
                            }
                        }
                    }
                    // The device's advertisement went away: stop supervising it
                    // (gracefully, so a live connection emits Disconnected)
                    // instead of redialing a gone device forever.
                    Discovered::Removed { fullname } => {
                        if let Some(managed) = tasks_for_browse.lock().unwrap().remove(&fullname) {
                            log::info!("[{fullname}] mDNS service removed, stopping supervision");
                            let _ = managed.directive_tx.send(Directive::Stop);
                        }
                    }
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

fn spawn_supervisor(
    fullname: String,
    url: String,
    server_id: String,
    server_name: String,
    clock: Arc<dyn Clock>,
    event_tx: UnboundedSender<ClientEvent>,
) -> ManagedClient {
    let (directive_tx, directive_rx) = watch::channel(Directive::Dial(url.clone()));
    let handle = tokio::spawn(supervise(
        fullname,
        directive_rx,
        server_id,
        server_name,
        clock,
        event_tx,
    ));
    ManagedClient {
        handle,
        directive_tx,
        url,
    }
}

/// Keep one discovered client connected until told to stop. Dials the current
/// directive URL, reports Connected/Message/Disconnected, and reconnects with
/// capped backoff. A `Directive::Dial` with a new URL closes the current
/// connection and redials it immediately; `Directive::Stop` ends the task.
async fn supervise(
    fullname: String,
    mut directive_rx: watch::Receiver<Directive>,
    server_id: String,
    server_name: String,
    clock: Arc<dyn Clock>,
    event_tx: UnboundedSender<ClientEvent>,
) {
    let mut backoff = MIN_BACKOFF;
    loop {
        let url = match directive_rx.borrow_and_update().clone() {
            Directive::Dial(url) => url,
            Directive::Stop => return,
        };

        match dial_client(&url, &server_id, &server_name, Arc::clone(&clock)).await {
            Ok(conn) => {
                // A Stop that arrived during the dial: don't announce a
                // connection we're about to tear down.
                if matches!(*directive_rx.borrow(), Directive::Stop) {
                    return;
                }
                let started = Instant::now();
                let client_id = conn.client_id().to_string();
                log::info!("[{fullname}] connected as {client_id} ({url})");
                let _ = event_tx.send(ClientEvent::Connected {
                    client_id: client_id.clone(),
                    fullname: fullname.clone(),
                    active_roles: conn.active_roles().to_vec(),
                    sender: conn.sender(),
                });

                // Drain until the client disconnects, or a directive redirects
                // us (dropping `conn` here closes that connection).
                let redirected = tokio::select! {
                    _ = drain_until_disconnected(conn, &client_id, &event_tx) => false,
                    _ = directive_rx.changed() => true,
                };

                let _ = event_tx.send(ClientEvent::Disconnected {
                    client_id: client_id.clone(),
                });

                if matches!(*directive_rx.borrow(), Directive::Stop) {
                    return;
                }
                if redirected {
                    log::info!("[{fullname}] address changed, reconnecting immediately");
                    backoff = MIN_BACKOFF;
                    continue;
                }
                log::info!("[{fullname}] disconnected, will retry");
                if started.elapsed() >= STABLE_SESSION {
                    backoff = MIN_BACKOFF;
                }
            }
            Err(e) => {
                log::warn!("[{fullname}] dial to {url} failed: {e}");
            }
        }

        // Wait out the backoff, but wake early if a directive arrives.
        tokio::select! {
            _ = tokio::time::sleep(backoff) => {}
            _ = directive_rx.changed() => {}
        }
        if matches!(*directive_rx.borrow(), Directive::Stop) {
            return;
        }
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }
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
