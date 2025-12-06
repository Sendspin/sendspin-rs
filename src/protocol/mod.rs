// ABOUTME: Protocol implementation for Sendspin WebSocket protocol
// ABOUTME: Message types, serialization, and WebSocket client

/// WebSocket client implementation
pub mod client;
/// Protocol message type definitions and serialization
pub mod messages;

pub use client::WsSender;
pub use messages::Message;
