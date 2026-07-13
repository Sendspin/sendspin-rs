// ABOUTME: Multi-client synchronized playback group — every member receives
// ABOUTME: the identical audio bytes tagged with the identical timestamp, so
// ABOUTME: each client's own independent clock-sync offset is all that's
// ABOUTME: needed for sample-accurate multi-room sync (v1-scoped equivalent
// ABOUTME: of aiosendspin's SendspinGroup + PushStream).

use crate::error::Error;
use crate::protocol::messages::{PlayerCommand, StreamPlayerConfig};
use crate::server::connection::ServerSender;
use crate::sync::raw_clock::Clock;
use futures_util::future::join_all;
use std::collections::HashMap;
use std::sync::Arc;

/// How far ahead of "now" (in this group's clock domain) each audio chunk's
/// timestamp is scheduled, by default. Must comfortably exceed a client's own
/// startup buffering latency, or it'll receive chunks whose intended playback
/// time has already passed. v1 uses one fixed lead time for the whole group
/// rather than negotiating per-client (see [`Group::with_send_ahead_us`]) —
/// aiosendspin negotiates this from each client's `required_lead_time_ms`;
/// deferred here along with per-client format negotiation.
pub const DEFAULT_SEND_AHEAD_US: i64 = 250_000;

/// A synchronized playback group.
///
/// The server-side synchronization trick is simple and doesn't require
/// knowing any client's individual clock: every member is sent the *same*
/// audio bytes tagged with the *same* `server/time`-domain timestamp. Each
/// client independently converts that timestamp into its own clock domain
/// (via the offset/drift it tracks from `client/time`/`server/time`
/// exchanges) and schedules local playback there — so two members with
/// converged clock-sync play the same chunk at the same wall-clock instant
/// without the server ever comparing their clocks to each other.
///
/// v1 scope: one shared PCM format for the whole group — no per-client
/// transcoding, so a member that can't take the group's format is a v1
/// limitation, not silently-wrong audio. No late-join catch-up (a client
/// added mid-stream just gets `stream/start` and audio from that point
/// forward) and no historical buffer replay.
pub struct Group {
    clock: Arc<dyn Clock>,
    send_ahead_us: i64,
    members: HashMap<String, ServerSender>,
    stream_config: Option<StreamPlayerConfig>,
}

impl Group {
    /// Create an empty group using `clock` as the shared timestamp domain —
    /// pass the same clock the [`crate::server::ServerListener`] that
    /// accepted these connections was built with, so timestamps here are in
    /// the same domain as the `server/time` replies members already trust.
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            send_ahead_us: DEFAULT_SEND_AHEAD_US,
            members: HashMap::new(),
            stream_config: None,
        }
    }

    /// Override the default send-ahead lead time.
    pub fn with_send_ahead_us(mut self, send_ahead_us: i64) -> Self {
        self.send_ahead_us = send_ahead_us;
        self
    }

    /// Client IDs of every current member.
    pub fn member_ids(&self) -> impl Iterator<Item = &str> {
        self.members.keys().map(String::as_str)
    }

    /// Number of current members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the group has no members.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Add a member. If a stream is already active for this group, starts
    /// it for the new member too (matching the group's already-negotiated
    /// format) — but does not replay any audio already delivered to
    /// existing members (no late-join catch-up in v1, see the type docs).
    pub async fn add_member(
        &mut self,
        client_id: impl Into<String>,
        sender: ServerSender,
    ) -> Result<(), Error> {
        if let Some(cfg) = self.stream_config.clone() {
            sender.send_stream_start(cfg).await?;
        }
        self.members.insert(client_id.into(), sender);
        Ok(())
    }

    /// Remove a member, if present. The caller is responsible for actually
    /// disconnecting it (e.g. via [`crate::server::ServerConnection::disconnect`]) —
    /// this only stops future broadcasts from reaching it.
    pub fn remove_member(&mut self, client_id: &str) -> Option<ServerSender> {
        self.members.remove(client_id)
    }

    /// Start (or restart, e.g. after a format change) the shared stream for
    /// every current member.
    pub async fn start_stream(&mut self, config: StreamPlayerConfig) {
        self.stream_config = Some(config.clone());
        self.broadcast_and_prune(|sender| {
            let config = config.clone();
            async move { sender.send_stream_start(config).await }
        })
        .await;
    }

    /// Push one PCM chunk to every member with a single, shared timestamp —
    /// `now + send_ahead_us` in this group's clock domain — computed once so
    /// every member gets the identical value. Returns that timestamp.
    pub async fn push_audio(&mut self, pcm: &[u8]) -> i64 {
        let timestamp_us = self.clock.now_micros() + self.send_ahead_us;
        self.broadcast_and_prune(|sender| {
            let pcm = pcm.to_vec();
            async move { sender.send_audio_chunk(timestamp_us, &pcm).await }
        })
        .await;
        timestamp_us
    }

    /// Broadcast a player command (volume, mute, static delay) to every
    /// member.
    pub async fn send_player_command(&mut self, command: PlayerCommand) {
        self.broadcast_and_prune(|sender| {
            let command = command.clone();
            async move { sender.send_player_command(command).await }
        })
        .await;
    }

    /// End the shared stream for every current member.
    pub async fn end_stream(&mut self) {
        self.stream_config = None;
        self.broadcast_and_prune(|sender| async move { sender.send_stream_end().await })
            .await;
    }

    /// Run `f` concurrently against every member — a slow member never
    /// blocks delivery to the others — then drop any member `f` failed
    /// against. A failed send means the connection's writer task is gone
    /// (its channel is closed), which is permanent: there is nothing to
    /// retry, so silently keeping a dead member around would just mean every
    /// future broadcast pays for its failure again.
    async fn broadcast_and_prune<F, Fut>(&mut self, f: F)
    where
        F: Fn(ServerSender) -> Fut,
        Fut: std::future::Future<Output = Result<(), Error>>,
    {
        let futures = self.members.iter().map(|(id, sender)| {
            let fut = f(sender.clone());
            let id = id.clone();
            async move { (id, fut.await) }
        });
        let results = join_all(futures).await;
        for (id, result) in results {
            if let Err(e) = result {
                log::warn!("Dropping group member {id}: {e}");
                self.members.remove(&id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::raw_clock::DefaultClock;

    #[test]
    fn new_group_is_empty() {
        let group = Group::new(Arc::new(DefaultClock::default()));
        assert!(group.is_empty());
        assert_eq!(group.len(), 0);
        assert_eq!(group.member_ids().count(), 0);
    }
}
