// ABOUTME: Clock synchronization implementation
// ABOUTME: Calculates RTT and converts server loop time to local Instant

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Clock synchronization quality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncQuality {
    /// Good synchronization (RTT < 50ms)
    Good,
    /// Degraded synchronization (RTT 50-100ms)
    Degraded,
    /// Lost synchronization (RTT > 100ms or no sync)
    Lost,
}

/// Clock synchronization state
#[derive(Debug)]
pub struct ClockSync {
    /// Last known RTT in microseconds
    rtt_micros: Option<i64>,

    /// When server loop started in Unix time (microseconds)
    server_loop_start_unix: Option<i64>,

    /// When we computed this (for staleness detection)
    last_update: Option<Instant>,

    /// Whether we've successfully synced once
    synced: bool,
}

impl ClockSync {
    /// Create a new clock synchronization instance
    pub fn new() -> Self {
        Self {
            rtt_micros: None,
            server_loop_start_unix: None,
            last_update: None,
            synced: false,
        }
    }

    /// Update clock sync with new measurement
    /// t1 = client_transmitted (Unix µs)
    /// t2 = server_received (server loop µs)
    /// t3 = server_transmitted (server loop µs)
    /// t4 = client_received (Unix µs)
    pub fn update(&mut self, t1: i64, t2: i64, t3: i64, t4: i64) {
        // RTT = (t4 - t1) - (t3 - t2)
        let rtt = (t4 - t1) - (t3 - t2);
        self.rtt_micros = Some(rtt);

        // Discard samples with high RTT (network congestion)
        if rtt > 100_000 {
            // 100ms
            eprintln!("Discarding sync sample: high RTT {}µs", rtt);
            return;
        }

        // On first successful sync, compute when the server loop started in Unix µs
        // Per Go reference: ONLY calculate this once, never update it again!
        // The server loop started at a specific moment in time - that never changes.
        if !self.synced {
            let now_unix = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as i64;

            self.server_loop_start_unix = Some(now_unix - t2);
            self.synced = true;

            eprintln!(
                "Clock sync established: t1={}, t2={}, t3={}, t4={}, rtt={}µs, now_unix={}, serverLoopStart={}",
                t1, t2, t3, t4, rtt, now_unix,
                self.server_loop_start_unix.unwrap()
            );
        }

        self.last_update = Some(Instant::now());
    }

    /// Get current RTT in microseconds
    pub fn rtt_micros(&self) -> Option<i64> {
        self.rtt_micros
    }

    /// Convert server loop microseconds to local Instant
    pub fn server_to_local_instant(&self, server_micros: i64) -> Option<Instant> {
        let server_start = self.server_loop_start_unix?;

        // Convert to Unix microseconds
        let unix_micros = server_start + server_micros;

        // Convert to Instant
        let now_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_micros() as i64;

        let now_instant = Instant::now();

        let delta_micros = unix_micros - now_unix;

        if delta_micros >= 0 {
            Some(now_instant + Duration::from_micros(delta_micros as u64))
        } else {
            now_instant.checked_sub(Duration::from_micros((-delta_micros) as u64))
        }
    }

    /// Get sync quality based on RTT
    pub fn quality(&self) -> SyncQuality {
        match self.rtt_micros {
            Some(rtt) if rtt < 50_000 => SyncQuality::Good,
            Some(rtt) if rtt < 100_000 => SyncQuality::Degraded,
            _ => SyncQuality::Lost,
        }
    }

    /// Check if sync is stale (>5 seconds old)
    pub fn is_stale(&self) -> bool {
        match self.last_update {
            Some(last) => last.elapsed() > Duration::from_secs(5),
            None => true,
        }
    }
}

impl Default for ClockSync {
    fn default() -> Self {
        Self::new()
    }
}
