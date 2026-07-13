// ABOUTME: Server-role implementation of the Sendspin protocol — accepts
// ABOUTME: inbound player connections, echoes clock-sync timing, streams
// ABOUTME: audio to synchronized multi-client groups, and advertises via mDNS.
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
//   - Server-initiated outbound connections with reconnect backoff (only
//     relevant for clients that don't dial in on their own).
//   - External player registration.

mod binary;
mod connection;
mod discovery;
mod group;
mod listener;

pub use binary::encode_audio_frame;
pub use connection::{ServerConnection, ServerConnectionGuard, ServerSender};
pub use discovery::Advertisement;
pub use group::{Group, DEFAULT_SEND_AHEAD_US};
pub use listener::ServerListener;
