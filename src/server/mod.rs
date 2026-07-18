// ABOUTME: Server-role implementation of the Sendspin protocol
// ABOUTME: Accepts or dials player clients, syncs clocks, streams audio to synchronized multi-client groups

// Handles both connection directions: clients that dial in
// (`ServerListener::accept`) and clients that only run their own embedded
// server and must be discovered over mDNS and dialed (`ClientBrowser` +
// `dial_client`, or the supervised `ClientManager`).
//
// Not yet supported: per-client codec transcoding (one PCM format per group),
// the non-player roles (color, visualizer, artwork, controller, metadata),
// external player registration, and late-join history replay — a client that
// joins mid-stream receives the current stream and all subsequent audio,
// synchronized with existing members, but nothing buffered from before it
// joined.

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
pub use discovery::{Advertisement, ClientBrowser, Discovered};
pub use group::{Group, DEFAULT_SEND_AHEAD_US};
pub use listener::ServerListener;
pub use manager::{ClientEvent, ClientManager};
