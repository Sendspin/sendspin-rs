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
//   - Non-player roles: color, visualizer, artwork, controller, metadata.
//   - External player registration.
//
// Explicitly NOT planned, by decision rather than by deferral: late-join
// catch-up / historical buffer replay. aiosendspin's version of this exists
// almost entirely to re-encode cached history through a newly-joined role's
// own codec/format when it differs from what's cached — moot here, since v1
// has one shared PCM format per group. What's left without that (a member
// joining mid-stream gets stream/start plus whatever the next push_audio()
// call sends, synchronized with everyone already there, but nothing from
// before it joined — proven by tests/group_sync.rs's
// a_late_joiner_gets_current_stream_start_and_only_subsequent_audio) is
// judged sufficient; a short join gap is acceptable.
//
// dial_client itself is one-shot; ClientManager is the supervised version —
// continuous discovery plus reconnect-with-backoff — and is what real
// deployments should use instead of driving ClientBrowser/dial_client by hand.

mod binary;
mod connection;
mod dial;
mod discovery;
mod group;
mod listener;
mod manager;

pub use binary::encode_audio_frame;
pub use connection::{ServerConnection, ServerConnectionGuard, ServerSender};
pub use dial::dial_client;
pub use discovery::{Advertisement, ClientBrowser};
pub use group::{Group, DEFAULT_SEND_AHEAD_US};
pub use listener::ServerListener;
pub use manager::{ClientEvent, ClientManager};
