// ABOUTME: Multi-client synchronized playback group
// ABOUTME: Every member receives identical audio bytes with identical timestamps, so each client's clock-sync offset alone yields sample-accurate multi-room sync

use crate::error::Error;
use crate::protocol::messages::{PlayerCommand, StreamPlayerConfig};
use crate::server::binary::encode_audio_frame;
use crate::server::connection::{AudioEnqueue, ServerSender};
use crate::sync::raw_clock::Clock;
use futures_util::future::join_all;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio_tungstenite::tungstenite::Bytes;

/// How far ahead of "now" (in this group's clock domain) the audio timeline is
/// anchored, by default. Must comfortably exceed a client's own startup
/// buffering latency, or it'll receive chunks whose intended playback time has
/// already passed. v1 uses one fixed lead time for the whole group rather than
/// negotiating per-client (see [`Group::with_send_ahead_us`]); per-client lead
/// negotiation is deferred along with per-client format negotiation.
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
    state: Mutex<State>,
}

/// Mutable group state behind a single lock, so `Group`'s methods take `&self`
/// and a slow member can never block membership changes or another member.
struct State {
    members: HashMap<String, ServerSender>,
    stream_config: Option<StreamPlayerConfig>,
    /// Timestamp (this group's clock domain) to stamp on the next pushed chunk.
    /// `None` until the first push after a start/clear (re)anchors the timeline.
    next_ts_us: Option<i64>,
    /// Carry for the sub-microsecond part of a chunk's duration (numerator over
    /// the sample rate), so advancing the timeline doesn't accumulate drift.
    residue: i64,
}

impl State {
    fn reset_timeline(&mut self) {
        self.next_ts_us = None;
        self.residue = 0;
    }
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
            state: Mutex::new(State {
                members: HashMap::new(),
                stream_config: None,
                next_ts_us: None,
                residue: 0,
            }),
        }
    }

    /// Override the default send-ahead lead time.
    pub fn with_send_ahead_us(mut self, send_ahead_us: i64) -> Self {
        self.send_ahead_us = send_ahead_us;
        self
    }

    /// Client IDs of every current member.
    pub fn member_ids(&self) -> Vec<String> {
        self.state.lock().unwrap().members.keys().cloned().collect()
    }

    /// Number of current members.
    pub fn len(&self) -> usize {
        self.state.lock().unwrap().members.len()
    }

    /// Whether the group has no members.
    pub fn is_empty(&self) -> bool {
        self.state.lock().unwrap().members.is_empty()
    }

    /// Add a member. If a stream is already active for this group, starts
    /// it for the new member too (matching the group's already-negotiated
    /// format) — but does not replay any audio already delivered to
    /// existing members (no late-join catch-up in v1, see the type docs).
    pub async fn add_member(
        &self,
        client_id: impl Into<String>,
        sender: ServerSender,
    ) -> Result<(), Error> {
        let config = self.state.lock().unwrap().stream_config.clone();
        if let Some(cfg) = config {
            sender.send_stream_start(cfg).await?;
        }
        self.state
            .lock()
            .unwrap()
            .members
            .insert(client_id.into(), sender);
        Ok(())
    }

    /// Remove a member, if present. The caller is responsible for actually
    /// disconnecting it (e.g. via [`crate::server::ServerConnection::disconnect`]) —
    /// this only stops future broadcasts from reaching it.
    pub fn remove_member(&self, client_id: &str) -> Option<ServerSender> {
        self.state.lock().unwrap().members.remove(client_id)
    }

    /// Start (or restart, e.g. after a format change) the shared stream for
    /// every current member, and re-anchor the audio timeline.
    pub async fn start_stream(&self, config: StreamPlayerConfig) {
        {
            let mut state = self.state.lock().unwrap();
            state.stream_config = Some(config.clone());
            state.reset_timeline();
        }
        self.broadcast_control(|sender| {
            let config = config.clone();
            async move { sender.send_stream_start(config).await }
        })
        .await;
    }

    /// Push one PCM chunk to every member, stamped with a single shared
    /// timestamp so every member schedules it at the same instant. Returns
    /// that timestamp.
    ///
    /// The timestamp comes from an anchored timeline rather than
    /// `now + lead` per call, so pushing faster or slower than real time
    /// doesn't shift playback: the first push after a start/clear anchors at
    /// `now + send_ahead_us`, and each push advances the timeline by the
    /// chunk's own duration (derived from the stream format). If pushes fall
    /// behind — the timeline would schedule a chunk too close to now — it
    /// re-anchors forward. Enqueue is non-blocking, so one slow member never
    /// delays the others; a member whose connection has died is pruned.
    pub fn push_audio(&self, pcm: &[u8]) -> i64 {
        let mut state = self.state.lock().unwrap();
        let now = self.clock.now_micros();

        // Anchor on the first push, or re-anchor if the timeline has fallen too
        // close to (or behind) now — otherwise chunks would be scheduled in the
        // past. Half the lead is the low-water mark, giving hysteresis so steady
        // real-time pacing doesn't re-anchor every push.
        let ts = match state.next_ts_us {
            Some(t) if t >= now + self.send_ahead_us / 2 => t,
            _ => now + self.send_ahead_us,
        };

        // Advance the timeline by this chunk's exact duration, carrying the
        // fractional-microsecond remainder so it doesn't drift.
        let advanced = match &state.stream_config {
            Some(cfg) => {
                let bytes_per_sample = (cfg.channels as usize) * (cfg.bit_depth as usize / 8);
                if bytes_per_sample > 0 && cfg.sample_rate > 0 {
                    let samples = (pcm.len() / bytes_per_sample) as i64;
                    let total = samples * 1_000_000 + state.residue;
                    let rate = cfg.sample_rate as i64;
                    state.residue = total % rate;
                    Some(ts + total / rate)
                } else {
                    Some(ts)
                }
            }
            None => Some(ts),
        };
        state.next_ts_us = advanced;

        // Encode once; fan the same frame out to every member as cheap refcount
        // clones. Prune members whose connection has died.
        let frame: Bytes = encode_audio_frame(ts, pcm).into();
        let mut dead = Vec::new();
        for (id, sender) in state.members.iter() {
            match sender.enqueue_audio(frame.clone()) {
                Ok(AudioEnqueue::Sent) => {}
                Ok(AudioEnqueue::Evicted) => {
                    log::trace!("group member {id} audio backlog full, dropping chunk")
                }
                Err(_) => dead.push(id.clone()),
            }
        }
        for id in dead {
            log::warn!("dropping dead group member {id}");
            state.members.remove(&id);
        }
        ts
    }

    /// Broadcast a player command (volume, mute, static delay) to every
    /// member.
    pub async fn send_player_command(&self, command: PlayerCommand) {
        self.broadcast_control(|sender| {
            let command = command.clone();
            async move { sender.send_player_command(command).await }
        })
        .await;
    }

    /// End the shared stream for every current member and reset the timeline.
    pub async fn end_stream(&self) {
        {
            let mut state = self.state.lock().unwrap();
            state.stream_config = None;
            state.reset_timeline();
        }
        self.broadcast_control(|sender| async move { sender.send_stream_end().await })
            .await;
    }

    /// Run an awaiting control send `f` concurrently against every member — a
    /// slow member never blocks delivery to the others — then drop any member
    /// `f` failed against (its writer task is gone; the failure is permanent).
    async fn broadcast_control<F, Fut>(&self, f: F)
    where
        F: Fn(ServerSender) -> Fut,
        Fut: std::future::Future<Output = Result<(), Error>>,
    {
        let members: Vec<(String, ServerSender)> = {
            let state = self.state.lock().unwrap();
            state
                .members
                .iter()
                .map(|(id, sender)| (id.clone(), sender.clone()))
                .collect()
        };
        let results = join_all(members.into_iter().map(|(id, sender)| {
            let fut = f(sender);
            async move { (id, fut.await) }
        }))
        .await;
        let mut state = self.state.lock().unwrap();
        for (id, result) in results {
            if let Err(e) = result {
                log::warn!("dropping group member {id}: {e}");
                state.members.remove(&id);
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
        assert_eq!(group.member_ids().len(), 0);
    }
}
