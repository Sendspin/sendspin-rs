// ABOUTME: mDNS for the server role: advertising this server
// ABOUTME: (`_sendspin-server._tcp.local.`) for clients that dial in, and
// ABOUTME: discovering clients that only run their own embedded server
// ABOUTME: (`_sendspin._tcp.local.`) and need to be dialed instead.

use crate::error::Error;
use mdns_sd::{Receiver, ResolvedService, ScopedIp, ServiceDaemon, ServiceEvent, ServiceInfo};
use std::net::IpAddr;

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

    /// Wait for the next discovery event: a client resolving at a usable URL,
    /// or a previously-advertised client being removed. Skips events that
    /// aren't a fully-resolved, usable service or a removal. Returns `None`
    /// once the daemon shuts down. Callers loop on this to keep discovering.
    pub async fn next_event(&self) -> Option<Discovered> {
        while let Ok(event) = self.receiver.recv_async().await {
            match event {
                ServiceEvent::ServiceResolved(info) => {
                    if let Some(url) = resolve_client_url(&info) {
                        return Some(Discovered::Found {
                            fullname: info.fullname.clone(),
                            url,
                        });
                    }
                }
                ServiceEvent::ServiceRemoved(_service_type, fullname) => {
                    return Some(Discovered::Removed { fullname });
                }
                _ => {}
            }
        }
        None
    }

    /// Wait for the next resolved client, returning its mDNS instance full name
    /// (stable identity across address changes) and WebSocket URL (ready to
    /// hand to [`crate::server::dial_client`]). Removal events are skipped.
    pub async fn next_client(&self) -> Option<(String, String)> {
        loop {
            match self.next_event().await? {
                Discovered::Found { fullname, url } => return Some((fullname, url)),
                Discovered::Removed { .. } => continue,
            }
        }
    }

    /// Like [`Self::next_client`], but returns only the WebSocket URL.
    pub async fn next_client_url(&self) -> Option<String> {
        self.next_client().await.map(|(_fullname, url)| url)
    }
}

/// An event from [`ClientBrowser::next_event`].
#[derive(Debug, Clone)]
pub enum Discovered {
    /// A client resolved at a usable WebSocket URL.
    Found {
        /// mDNS instance full name — stable identity across address changes.
        fullname: String,
        /// WebSocket URL, ready to hand to [`crate::server::dial_client`].
        url: String,
    },
    /// A previously-advertised client's service was removed from mDNS.
    Removed {
        /// mDNS instance full name of the service that went away.
        fullname: String,
    },
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
    // Accept both IPv4 and IPv6; drop loopback/link-local/unspecified so we
    // never dial an address that isn't routable to the device.
    let mut addrs: Vec<IpAddr> = info
        .get_addresses()
        .iter()
        .map(ScopedIp::to_ip_addr)
        .filter(|a| !a.is_loopback() && !is_link_local(a) && !a.is_unspecified())
        .collect();
    addrs.sort();
    let addr = addrs.into_iter().next()?;
    let path = info.get_property_val_str("path")?;
    if !path.starts_with('/') {
        return None;
    }
    // IPv6 literals must be bracketed in a URL authority.
    let host = match addr {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => format!("[{v6}]"),
    };
    Some(format!("ws://{host}:{port}{path}", port = info.get_port()))
}

/// Whether an address is link-local (IPv4 169.254.0.0/16 or IPv6 fe80::/10).
fn is_link_local(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_link_local(),
        IpAddr::V6(v6) => (v6.segments()[0] & 0xffc0) == 0xfe80,
    }
}
