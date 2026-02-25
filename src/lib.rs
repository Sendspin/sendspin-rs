// ABOUTME: Main library entry point for sendspin-rs
// ABOUTME: Exports public API for Sendspin Protocol client and server

//! # sendspin-rs
//!
//! Hyper-efficient Rust implementation of the Sendspin Protocol for synchronized multi-room audio streaming.
//!
//! This library provides zero-copy audio pipelines, lock-free concurrency, and async I/O
//! for building high-performance audio streaming clients and servers.

#![warn(missing_docs)]

/// Audio types and processing
pub mod audio;
/// Protocol implementation for WebSocket communication
pub mod protocol;
/// Audio scheduler for timed playback
pub mod scheduler;
/// Clock synchronization utilities
pub mod sync;

pub use protocol::client::ProtocolClient;
pub use protocol::client_builder::ProtocolClientBuilder;
pub use protocol::messages::{ClientHello, ServerHello};
pub use scheduler::AudioScheduler;

/// Result type for sendspin operations
pub type Result<T> = std::result::Result<T, error::Error>;

/// Error types for sendspin
pub mod error {
    use thiserror::Error;

    /// Error types for sendspin operations
    #[derive(Error, Debug)]
    pub enum Error {
        /// WebSocket-related error
        #[error("WebSocket error: {0}")]
        WebSocket(String),

        /// Protocol violation or parsing error
        #[error("Protocol error: {0}")]
        Protocol(String),

        /// Invalid message format received
        #[error("Invalid message format")]
        InvalidMessage,

        /// Connection-related error
        #[error("Connection error: {0}")]
        Connection(String),

        /// Audio output error
        #[error("Audio output error: {0}")]
        Output(String),
    }
}
