// ABOUTME: Clock synchronization for Sendspin protocol
// ABOUTME: NTP-style round-trip time calculation and server timestamp conversion

/// Clock synchronization implementation
pub mod clock;

pub use clock::{ClockSync, SyncQuality};
