// ABOUTME: Protocol implementation for Sendspin WebSocket protocol
// ABOUTME: Message types, serialization, and WebSocket client

/// WebSocket client implementation
pub mod client;
/// Builder for easy construction of the client
pub mod client_builder;
/// Protocol message type definitions and serialization
pub mod messages;

pub use client::WsSender;
pub use messages::Message;
