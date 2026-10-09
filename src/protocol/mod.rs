// ABOUTME: Protocol implementation for Sendspin WebSocket protocol
// ABOUTME: Message types, serialization, and WebSocket client

/// WebSocket client implementation
#[cfg(feature = "std")]
pub mod client;
/// Builder for easy construction of the client
#[cfg(feature = "std")]
pub mod client_builder;
/// Inbound WebSocket acceptor for server-initiated connections
#[cfg(feature = "std")]
pub mod listener;
/// Managed connection lifecycle: multi-server arbitration and auto-goodbye
#[cfg(feature = "std")]
pub mod manager;
/// Protocol message type definitions and serialization
pub mod messages;

#[cfg(feature = "std")]
pub use client::{Connection, ConnectionGuard, Controller, WsSender};
#[cfg(feature = "std")]
pub use listener::ProtocolListener;
#[cfg(feature = "std")]
pub use manager::{should_switch, ConnectionManager, ManagedConnection, ManagerConfig};
pub use messages::Message;
