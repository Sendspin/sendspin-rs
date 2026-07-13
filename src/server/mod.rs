// ABOUTME: Server-role implementation of the Sendspin protocol — accepts
// ABOUTME: inbound player connections (or dials out to clients that only run
// ABOUTME: their own embedded server), echoes clock-sync timing, streams
// ABOUTME: audio to synchronized multi-client groups, and advertises via mDNS.
//
// Both connection directions turn out to matter for real hardware: some
// clients dial in (ServerListener::accept), but plenty of real embedded
// clients — notably ESPHome's `sendspin:` component as shipped on Home
// Assistant Voice PE, via the `sendspin-cpp` library — only ever run their
// own embedded WebSocket server and never dial out at all. For those, the
// server has to discover them via mDNS (ClientBrowser) and dial in
// (dial_client) — see aiosendspin's `SendspinServer.connect_to_client`/
// `_start_mdns_discovery` for the reference implementation of the same idea.
//
// This is a v1/prototype scope, not full parity with the reference Python
// `aiosendspin` server. Deliberately deferred for now (tracked as follow-up,
// not silently dropped):
//   - Per-client codec transcoding/resampling (v1 is PCM-only, one format
//     per group — a client that can't take the group's format is out of luck
//     for now).
//   - Late-join catch-up and historical buffer replay (a client that joins
//     mid-stream just gets audio from that point forward).
//   - Non-player roles: color, visualizer, artwork, controller, metadata.
//   - Reconnect-with-backoff for dialed clients (dial_client is one-shot;
//     aiosendspin retries indefinitely with exponential backoff).
//   - External player registration.

mod binary;
mod connection;
mod dial;
mod discovery;
mod group;
mod listener;

pub use binary::encode_audio_frame;
pub use connection::{ServerConnection, ServerConnectionGuard, ServerSender};
pub use dial::dial_client;
pub use discovery::{Advertisement, ClientBrowser};
pub use group::{Group, DEFAULT_SEND_AHEAD_US};
pub use listener::ServerListener;
