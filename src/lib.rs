// ABOUTME: Main library entry point for sendspin-rs
// ABOUTME: Exports public API for Sendspin Protocol client and server

//! # sendspin-rs
//!
//! Hyper-efficient Rust implementation of the Sendspin Protocol for synchronized multi-room audio streaming.
//!
//! This library provides zero-copy audio pipelines, lock-free concurrency, and async I/O
//! for building high-performance audio streaming clients and servers.

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]
// Some pub(crate) helpers (GainRamp, SyncErrorFilter, EngageGate, log
// sampling) are only consumed by the std-gated player/client modules.
#![cfg_attr(not(feature = "std"), allow(dead_code))]

#[cfg(not(feature = "alloc"))]
compile_error!("sendspin requires the `alloc` feature (enabled by default via `std`)");

// `macro_use` only for unit tests, so `vec!`/`format!` resolve in test
// modules under no_std; non-test code imports them explicitly.
#[cfg(feature = "alloc")]
#[cfg_attr(test, macro_use)]
extern crate alloc;

/// Audio types and processing
pub mod audio;
/// Protocol implementation for WebSocket communication
pub mod protocol;
/// Clock synchronization utilities
pub mod sync;

pub(crate) mod log_sampling;
pub(crate) mod sync_primitives;

pub use audio::GainControl;
#[cfg(feature = "std")]
pub use protocol::client::{Connection, ConnectionGuard, Controller, ProtocolClient, WsSender};
#[cfg(feature = "std")]
pub use protocol::client_builder::ProtocolClientBuilder;
#[cfg(feature = "std")]
pub use protocol::listener::ProtocolListener;
#[cfg(feature = "std")]
pub use protocol::manager::{ConnectionManager, ManagedConnection, ManagerConfig};
pub use protocol::messages::ServerHello;
pub use sync::raw_clock::Clock;
#[cfg(feature = "std")]
pub use sync::raw_clock::DefaultClock;

/// Result type for sendspin operations
pub type Result<T> = core::result::Result<T, error::Error>;

/// Error types for sendspin
pub mod error {
    use alloc::string::String;
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
