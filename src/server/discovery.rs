// ABOUTME: mDNS for the server role: advertising this server
// ABOUTME: (`_sendspin-server._tcp.local.`) for clients that dial in, and
// ABOUTME: discovering clients that only run their own embedded server
// ABOUTME: (`_sendspin._tcp.local.`) and need to be dialed instead.

use crate::error::Error;
use mdns_sd::{Receiver, ResolvedService, ServiceDaemon, ServiceEvent, ServiceInfo};

/// Service type clients browse for to discover a Sendspin server (matches
/// aiosendspin's `server/server.py`; distinct from `_sendspin._tcp.local.`,
/// which is what a *client* advertises for server-initiated connections —
/// see `examples/server_initiated_metadata.rs`).
const SERVER_SERVICE_TYPE: &str = "_sendspin-server._tcp.local.";

/// Service type that clients which only run their own embedded server (e.g.
/// ESPHome's `sendspin:` component) advertise for themselves, to be discovered
/// and dialed by a server.
const CLIENT_SERVICE_TYPE: &str = "_sendspin._tcp.local.";

/// A live mDNS advertisement. Unregisters and shuts down its background
/// daemon on drop — hold this alive for as long as the server should stay
/// discoverable.
pub struct Advertisement {
    daemon: ServiceDaemon,
    fullname: String,
}

impl Advertisement {
    /// Advertise a running server. `server_id` should be the same stable
    /// identifier passed to [`crate::server::ServerListener::bind`] — it
    /// becomes both the mDNS instance name and the advertised hostname.
    /// `path` is the HTTP path clients should connect to (the spec fixes
    /// this to `/sendspin` for real deployments).
    pub fn new(server_id: &str, name: &str, port: u16, path: &str) -> Result<Self, Error> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| Error::Connection(format!("mDNS daemon start failed: {e}")))?;
        let service = ServiceInfo::new(
            SERVER_SERVICE_TYPE,
            server_id,
            &format!("{server_id}.local."),
            "",
            port,
            &[("path", path), ("name", name)][..],
        )
        .map_err(|e| Error::Connection(format!("mDNS service build failed: {e}")))?
        .enable_addr_auto();
        let fullname = service.get_fullname().to_string();
        daemon
            .register(service)
            .map_err(|e| Error::Connection(format!("mDNS register failed: {e}")))?;
        Ok(Self { daemon, fullname })
    }
}

impl Drop for Advertisement {
    fn drop(&mut self) {
        if let Err(e) = self.daemon.unregister(&self.fullname) {
            log::warn!("mDNS unregister failed: {e}");
        }
        if let Err(e) = self.daemon.shutdown() {
            log::warn!("mDNS daemon shutdown failed: {e}");
        }
    }
}

/// Discovers Sendspin clients that advertise themselves over mDNS instead of
/// dialing out (see [`CLIENT_SERVICE_TYPE`]'s docs for why real hardware
/// routinely needs this).
pub struct ClientBrowser {
    daemon: ServiceDaemon,
    receiver: Receiver<ServiceEvent>,
}

impl ClientBrowser {
    /// Start browsing. Keep this alive for as long as discovery should keep
    /// running — dropping it shuts down its background daemon.
    pub fn new() -> Result<Self, Error> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| Error::Connection(format!("mDNS daemon start failed: {e}")))?;
        let receiver = daemon
            .browse(CLIENT_SERVICE_TYPE)
            .map_err(|e| Error::Connection(format!("mDNS browse failed: {e}")))?;
        Ok(Self { daemon, receiver })
    }

    /// Wait for the next discovered client with a usable address and path,
    /// returning its WebSocket URL (e.g. `ws://192.168.1.42:8928/sendspin`,
    /// ready to hand to [`crate::server::dial_client`]). Silently skips
    /// events that aren't a fully-resolved, usable service — callers loop on
    /// this to keep discovering. Returns `None` once the daemon shuts down.
    pub async fn next_client_url(&self) -> Option<String> {
        self.next_client().await.map(|(_fullname, url)| url)
    }

    /// Like [`Self::next_client_url`], but also returns the mDNS instance's
    /// full name (e.g. `my-device._sendspin._tcp.local.`) — stable identity
    /// for *this device* across address changes, unlike the URL itself.
    /// [`crate::server::ClientManager`] uses this to notice "same device,
    /// new address" and reconnect there instead of endlessly retrying a
    /// stale address.
    pub async fn next_client(&self) -> Option<(String, String)> {
        while let Ok(event) = self.receiver.recv_async().await {
            if let ServiceEvent::ServiceResolved(info) = event {
                if let Some(url) = resolve_client_url(&info) {
                    return Some((info.fullname.clone(), url));
                }
            }
        }
        None
    }
}

impl Drop for ClientBrowser {
    fn drop(&mut self) {
        if let Err(e) = self.daemon.shutdown() {
            log::warn!("mDNS daemon shutdown failed: {e}");
        }
    }
}

/// Build a WebSocket URL from a resolved client service, or `None` if it has
/// no usable address or well-formed `path` TXT property. Addresses are sorted
/// before selection: the resolver returns them unordered, and [`ClientBrowser`]
/// callers treat an address change as "device moved, reconnect", so a stable
/// choice avoids reconnect thrashing on multi-homed hosts.
fn resolve_client_url(info: &ResolvedService) -> Option<String> {
    let mut addrs: Vec<_> = info
        .get_addresses_v4()
        .into_iter()
        .filter(|a| !a.is_link_local() && !a.is_unspecified())
        .collect();
    addrs.sort();
    let addr = addrs.into_iter().next()?;
    let path = info.get_property_val_str("path")?;
    if !path.starts_with('/') {
        return None;
    }
    Some(format!("ws://{addr}:{port}{path}", port = info.get_port()))
}
