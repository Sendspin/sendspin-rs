#![allow(dead_code)]

//! Dispatches role messages into typed, generation-stamped events for role sinks.

use crate::generation::StreamGeneration;
use crate::protocol::client::{ArtworkChunk, AudioChunk, VisualizerChunk};
use crate::protocol::messages::{
    Message, PlayerCommandType, StreamArtworkConfig, StreamClear, StreamEnd, StreamPlayerConfig,
    StreamStart, StreamVisualizerConfig,
};

const ROLE_PLAYER: &str = "player";
const ROLE_ARTWORK: &str = "artwork";
const ROLE_VISUALIZER: &str = "visualizer";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    Player,
    Artwork,
    Visualizer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RoleGeneration {
    pub(crate) player: StreamGeneration,
    pub(crate) artwork: StreamGeneration,
    pub(crate) visualizer: StreamGeneration,
}

impl Default for RoleGeneration {
    fn default() -> Self {
        Self {
            player: StreamGeneration::initial(),
            artwork: StreamGeneration::initial(),
            visualizer: StreamGeneration::initial(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CommandedControls {
    pub(crate) volume: Option<u8>,
    pub(crate) muted: Option<bool>,
    pub(crate) static_delay_ms: Option<u16>,
}

#[derive(Debug, Clone)]
pub(crate) enum RoleEvent {
    PlayerStart {
        config: StreamPlayerConfig,
        generation: StreamGeneration,
    },
    ArtworkStart {
        config: StreamArtworkConfig,
        generation: StreamGeneration,
    },
    VisualizerStart {
        config: StreamVisualizerConfig,
        generation: StreamGeneration,
    },
    PlayerAudio {
        chunk: AudioChunk,
        generation: StreamGeneration,
    },
    ArtworkData {
        chunk: ArtworkChunk,
        generation: StreamGeneration,
    },
    VisualizerData {
        chunk: VisualizerChunk,
        generation: StreamGeneration,
    },
    Clear {
        role: Role,
        generation: StreamGeneration,
    },
    End {
        role: Role,
        generation: StreamGeneration,
    },
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RoutingState {
    generations: RoleGeneration,
    /// Raw command fields are retained until the session normalizes them.
    controls: CommandedControls,
    closed: bool,
    exhausted: bool,
}

impl RoutingState {
    #[cfg(test)]
    fn with_max_generation(role: Role) -> Self {
        let mut state = Self::default();
        match role {
            Role::Player => state.generations.player = StreamGeneration::max_for_test(),
            Role::Artwork => state.generations.artwork = StreamGeneration::max_for_test(),
            Role::Visualizer => state.generations.visualizer = StreamGeneration::max_for_test(),
        }
        state
    }

    pub(crate) fn generation(&self, role: Role) -> StreamGeneration {
        match role {
            Role::Player => self.generations.player,
            Role::Artwork => self.generations.artwork,
            Role::Visualizer => self.generations.visualizer,
        }
    }

    fn advance(&mut self, role: Role) -> Result<StreamGeneration, RoutingError> {
        if self.exhausted {
            return Err(RoutingError::Closed);
        }
        let generation = self.generation(role).checked_advance().ok_or_else(|| {
            self.exhausted = true;
            RoutingError::GenerationExhausted
        })?;
        match role {
            Role::Player => self.generations.player = generation,
            Role::Artwork => self.generations.artwork = generation,
            Role::Visualizer => self.generations.visualizer = generation,
        }
        Ok(generation)
    }

    pub(crate) fn stream_start(
        &mut self,
        start: StreamStart,
    ) -> Result<Vec<RoleEvent>, RoutingError> {
        if self.closed || self.exhausted {
            return Err(RoutingError::Closed);
        }
        let mut events = Vec::new();
        if let (Some(config), generation) = (start.player, self.generation(Role::Player)) {
            events.push(RoleEvent::PlayerStart { config, generation });
        }
        if let (Some(config), generation) = (start.artwork, self.generation(Role::Artwork)) {
            events.push(RoleEvent::ArtworkStart { config, generation });
        }
        if let (Some(config), generation) = (start.visualizer, self.generation(Role::Visualizer)) {
            events.push(RoleEvent::VisualizerStart { config, generation });
        }
        Ok(events)
    }

    pub(crate) fn audio(&self, chunk: AudioChunk) -> Result<RoleEvent, RoutingError> {
        if self.closed || self.exhausted {
            return Err(RoutingError::Closed);
        }
        Ok(RoleEvent::PlayerAudio {
            chunk,
            generation: self.generation(Role::Player),
        })
    }

    pub(crate) fn artwork(&self, chunk: ArtworkChunk) -> Result<RoleEvent, RoutingError> {
        if self.closed || self.exhausted {
            return Err(RoutingError::Closed);
        }
        Ok(RoleEvent::ArtworkData {
            chunk,
            generation: self.generation(Role::Artwork),
        })
    }

    pub(crate) fn visualizer(&self, chunk: VisualizerChunk) -> Result<RoleEvent, RoutingError> {
        if self.closed || self.exhausted {
            return Err(RoutingError::Closed);
        }
        Ok(RoleEvent::VisualizerData {
            chunk,
            generation: self.generation(Role::Visualizer),
        })
    }

    pub(crate) fn clear(&mut self, clear: StreamClear) -> Result<Vec<RoleEvent>, RoutingError> {
        if self.closed || self.exhausted {
            return Err(RoutingError::Closed);
        }
        let named_artwork = clear
            .roles
            .as_deref()
            .is_some_and(|roles| roles.iter().any(|role| role == ROLE_ARTWORK));
        if named_artwork {
            log::debug!("Ignoring explicit artwork stream clear");
        }
        // Artwork clears use an empty image payload, not stream invalidation.
        let mut targets = role_targets(clear.roles.as_deref());
        targets.retain(|role| matches!(role, Role::Player | Role::Visualizer));
        self.boundary(targets, false)
    }

    pub(crate) fn end(&mut self, end: StreamEnd) -> Result<Vec<RoleEvent>, RoutingError> {
        if self.closed || self.exhausted {
            return Err(RoutingError::Closed);
        }
        let targets = role_targets(end.roles.as_deref());
        self.boundary(targets, true)
    }

    fn boundary(&mut self, targets: Vec<Role>, end: bool) -> Result<Vec<RoleEvent>, RoutingError> {
        let mut events = Vec::new();
        for role in targets {
            let generation = self.advance(role)?;
            events.push(if end {
                RoleEvent::End { role, generation }
            } else {
                RoleEvent::Clear { role, generation }
            });
        }
        Ok(events)
    }

    fn merge_command(&mut self, command: &crate::protocol::messages::PlayerCommand) {
        match command.command {
            PlayerCommandType::Volume if command.volume.is_some() => {
                self.controls.volume = command.volume;
            }
            PlayerCommandType::Mute if command.mute.is_some() => {
                self.controls.muted = command.mute;
            }
            PlayerCommandType::SetStaticDelay if command.static_delay_ms.is_some() => {
                self.controls.static_delay_ms = command.static_delay_ms;
            }
            _ => {}
        }
    }

    pub(crate) fn close(&mut self) {
        self.closed = true;
    }
}

pub(crate) fn role_targets(roles: Option<&[String]>) -> Vec<Role> {
    roles.map_or_else(
        || vec![Role::Player, Role::Artwork, Role::Visualizer],
        |roles| {
            let mut targets = Vec::new();
            for role in roles.iter().filter_map(|role| match role.as_str() {
                ROLE_PLAYER => Some(Role::Player),
                ROLE_ARTWORK => Some(Role::Artwork),
                ROLE_VISUALIZER => Some(Role::Visualizer),
                _ => None,
            }) {
                if !targets.contains(&role) {
                    targets.push(role);
                }
            }
            targets
        },
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoutingError {
    Closed,
    GenerationExhausted,
}

pub(crate) fn dispatch(
    state: &mut RoutingState,
    message: Message,
) -> Result<Vec<RoleEvent>, RoutingError> {
    match message {
        Message::StreamStart(start) => state.stream_start(start),
        Message::StreamClear(clear) => state.clear(clear),
        Message::StreamEnd(end) => state.end(end),
        Message::ServerCommand(command) => {
            if state.closed || state.exhausted {
                return Err(RoutingError::Closed);
            }
            if let Some(player) = command.player.as_ref() {
                state.merge_command(player);
            }
            Ok(Vec::new())
        }
        _ => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::messages::{PlayerCommand, StreamPlayerConfig};
    use std::sync::Arc;

    fn player_config() -> StreamPlayerConfig {
        StreamPlayerConfig {
            codec: "pcm".into(),
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        }
    }

    #[test]
    fn targets_default_and_filter_unknown_names() {
        assert_eq!(
            role_targets(None),
            vec![Role::Player, Role::Artwork, Role::Visualizer]
        );
        assert_eq!(
            role_targets(Some(&["player".into(), "unknown".into()])),
            vec![Role::Player]
        );
        let mut state = RoutingState::default();
        assert!(dispatch(
            &mut state,
            Message::StreamEnd(StreamEnd {
                roles: Some(vec!["unknown".into()])
            })
        )
        .unwrap()
        .is_empty());
        assert!(role_targets(Some(&[])).is_empty());
    }

    #[test]
    fn clear_and_end_targets_follow_role_rules() {
        let mut state = RoutingState::default();
        assert!(state
            .clear(StreamClear { roles: None })
            .unwrap()
            .iter()
            .all(|event| !matches!(
                event,
                RoleEvent::Clear {
                    role: Role::Artwork,
                    ..
                }
            )));
        assert_eq!(state.end(StreamEnd { roles: None }).unwrap().len(), 3);
        assert!(state
            .clear(StreamClear {
                roles: Some(vec![ROLE_ARTWORK.into()])
            })
            .unwrap()
            .is_empty());
    }

    #[test]
    fn starts_stamp_current_generation_and_boundaries_advance_once() {
        let mut state = RoutingState::default();
        let events = state.stream_start(StreamStart {
            player: Some(player_config()),
            artwork: None,
            visualizer: None,
        });
        let events = events.unwrap();
        assert!(
            matches!(events.as_slice(), [RoleEvent::PlayerStart { generation, .. }] if *generation == StreamGeneration::initial())
        );
        let first = state
            .clear(StreamClear {
                roles: Some(vec!["player".into(), "player".into()]),
            })
            .unwrap();
        assert!(
            matches!(first.as_slice(), [RoleEvent::Clear { generation, .. }] if *generation == StreamGeneration::initial().checked_advance().unwrap())
        );
        assert_eq!(
            state.generation(Role::Player),
            StreamGeneration::initial().checked_advance().unwrap()
        );
    }

    #[test]
    fn valid_commands_merge_only_the_named_wire_field() {
        let mut state = RoutingState::default();
        state.merge_command(&PlayerCommand {
            command: PlayerCommandType::Volume,
            volume: Some(80),
            mute: Some(true),
            static_delay_ms: Some(12),
        });
        assert_eq!(state.controls.volume, Some(80));
        assert_eq!(state.controls.muted, None);
        assert_eq!(state.controls.static_delay_ms, None);
        state.merge_command(&PlayerCommand {
            command: PlayerCommandType::Mute,
            volume: Some(20),
            mute: Some(true),
            static_delay_ms: Some(25),
        });
        assert_eq!(state.controls.muted, Some(true));
        state.merge_command(&PlayerCommand {
            command: PlayerCommandType::SetStaticDelay,
            volume: None,
            mute: None,
            static_delay_ms: Some(25),
        });
        assert_eq!(state.controls.static_delay_ms, Some(25));
        state.merge_command(&PlayerCommand {
            command: PlayerCommandType::SetStaticDelay,
            volume: None,
            mute: None,
            static_delay_ms: None,
        });
        state.merge_command(&PlayerCommand {
            command: PlayerCommandType::Unknown,
            volume: Some(1),
            mute: Some(false),
            static_delay_ms: Some(1),
        });
        assert_eq!(state.controls.static_delay_ms, Some(25));
        assert_eq!(state.controls.muted, Some(true));
    }

    #[test]
    fn empty_unknown_and_other_messages_do_not_change_state() {
        let mut state = RoutingState::default();
        let before = state.clone();
        assert!(dispatch(
            &mut state,
            Message::StreamClear(StreamClear {
                roles: Some(vec![])
            })
        )
        .unwrap()
        .is_empty());
        assert!(dispatch(
            &mut state,
            Message::ServerCommand(crate::protocol::messages::ServerCommand { player: None })
        )
        .unwrap()
        .is_empty());
        assert!(dispatch(
            &mut state,
            Message::ServerState(crate::protocol::messages::ServerState {
                metadata: None,
                controller: None,
                color: None
            })
        )
        .unwrap()
        .is_empty());
        assert_eq!(state.generations, before.generations);
        assert_eq!(state.controls, before.controls);
    }

    #[test]
    fn exhausted_state_rejects_start_boundary_command_and_data() {
        let mut state = RoutingState::with_max_generation(Role::Player);
        assert!(matches!(
            dispatch(
                &mut state,
                Message::StreamClear(StreamClear {
                    roles: Some(vec![ROLE_PLAYER.into()])
                })
            ),
            Err(RoutingError::GenerationExhausted)
        ));
        assert!(matches!(
            dispatch(
                &mut state,
                Message::StreamStart(StreamStart {
                    player: Some(player_config()),
                    artwork: None,
                    visualizer: None
                })
            ),
            Err(RoutingError::Closed)
        ));
        assert!(matches!(
            dispatch(
                &mut state,
                Message::ServerCommand(crate::protocol::messages::ServerCommand { player: None })
            ),
            Err(RoutingError::Closed)
        ));
        assert!(state
            .audio(AudioChunk {
                timestamp: 1,
                data: Arc::from([1u8].as_slice())
            })
            .is_err());
    }

    #[test]
    fn terminal_close_rejects_all_events_without_mutating_generations() {
        let mut state = RoutingState::default();
        state.close();
        let before = state.generations;
        let start = StreamStart {
            player: Some(player_config()),
            artwork: None,
            visualizer: None,
        };
        assert!(matches!(
            dispatch(&mut state, Message::StreamStart(start)),
            Err(RoutingError::Closed)
        ));
        assert!(matches!(
            dispatch(
                &mut state,
                Message::StreamClear(StreamClear { roles: None })
            ),
            Err(RoutingError::Closed)
        ));
        assert!(matches!(
            dispatch(&mut state, Message::StreamEnd(StreamEnd { roles: None })),
            Err(RoutingError::Closed)
        ));
        assert!(matches!(
            dispatch(
                &mut state,
                Message::ServerCommand(crate::protocol::messages::ServerCommand { player: None })
            ),
            Err(RoutingError::Closed)
        ));
        assert!(state
            .audio(AudioChunk {
                timestamp: 1,
                data: Arc::from([1u8].as_slice())
            })
            .is_err());
        assert!(state
            .artwork(ArtworkChunk {
                channel: 0,
                timestamp: 1,
                data: Arc::from([1u8].as_slice())
            })
            .is_err());
        assert!(state
            .visualizer(VisualizerChunk {
                type_id: 0x10,
                timestamp: 1,
                data: Arc::from([1u8].as_slice())
            })
            .is_err());
        assert_eq!(state.generations, before);
    }
}
