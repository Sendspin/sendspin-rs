// ABOUTME: mDNS advertising for the server role (`_sendspin-server._tcp.local.`)
// ABOUTME: so clients can discover this server without a configured address.

use crate::error::Error;
use mdns_sd::{ServiceDaemon, ServiceInfo};

/// Service type clients browse for to discover a Sendspin server (matches
/// aiosendspin's `server/server.py`; distinct from `_sendspin._tcp.local.`,
/// which is what a *client* advertises for server-initiated connections —
/// see `examples/server_initiated_metadata.rs`).
const SERVICE_TYPE: &str = "_sendspin-server._tcp.local.";

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
            SERVICE_TYPE,
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
