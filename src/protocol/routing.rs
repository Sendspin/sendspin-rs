//! Dispatches role messages into typed, generation-stamped events for role sinks.

use crate::audio::player_api::{PlayerProducer, PlayerReceiver};
use crate::audio::AppliedControls;
use crate::protocol::client::BinaryFrame;
use crate::protocol::messages::{Message, PlayerCommandType, ServerCommand};
#[cfg(test)]
use crate::protocol::messages::{StreamClear, StreamStart};
use crate::role_core::ProducerError;
use crate::roles::{ArtworkProducer, ArtworkReceiver, VisualizerProducer, VisualizerReceiver};
use tokio::sync::mpsc::UnboundedSender;

const ROLE_PLAYER: &str = "player";
const ROLE_ARTWORK: &str = "artwork";
const ROLE_VISUALIZER: &str = "visualizer";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    Player,
    Artwork,
    Visualizer,
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
    GenerationExhausted,
}

/// Owns the role and message destinations used by the receiving path.
pub(crate) struct ReceiveSinks {
    controls: AppliedControls,
    player: PlayerProducer,
    artwork: ArtworkProducer,
    visualizer: VisualizerProducer,
    messages: UnboundedSender<Message>,
    player_open: bool,
    artwork_open: bool,
    visualizer_open: bool,
    messages_open: bool,
}

impl ReceiveSinks {
    pub(crate) fn channel(
        initial_controls: AppliedControls,
    ) -> (
        Self,
        PlayerReceiver,
        ArtworkReceiver,
        VisualizerReceiver,
        tokio::sync::mpsc::UnboundedReceiver<Message>,
    ) {
        let controls = initial_controls.clone();
        let (player, player_producer) = PlayerReceiver::channel(initial_controls);
        let (artwork, artwork_producer) = ArtworkReceiver::channel();
        let (visualizer, visualizer_producer) = VisualizerReceiver::channel();
        let (messages, message_rx) = tokio::sync::mpsc::unbounded_channel();
        (
            Self {
                controls,
                player: player_producer,
                artwork: artwork_producer,
                visualizer: visualizer_producer,
                messages,
                player_open: true,
                artwork_open: true,
                visualizer_open: true,
                messages_open: true,
            },
            player,
            artwork,
            visualizer,
            message_rx,
        )
    }

    pub(crate) fn close(&self) {
        self.player.close();
        self.artwork.close();
        self.visualizer.close();
    }

    fn producer_error(error: ProducerError, destination: &str, open: &mut bool) -> bool {
        match error {
            ProducerError::ReceiverClosed => {
                *open = false;
                log::debug!("{} receiver closed; disabling destination", destination);
                false
            }
            ProducerError::GenerationExhausted => {
                log::error!(
                    "{} generation exhausted; stopping message router",
                    destination
                );
                true
            }
        }
    }

    pub(crate) fn dispatch_binary(&mut self, frame: BinaryFrame) -> Result<(), RoutingError> {
        match frame {
            BinaryFrame::Audio(chunk) if self.player_open => {
                if let Err(error) = self.player.audio(chunk) {
                    if Self::producer_error(error, "Player", &mut self.player_open) {
                        self.close();
                        return Err(RoutingError::GenerationExhausted);
                    }
                }
            }
            BinaryFrame::Artwork(chunk) if self.artwork_open => {
                if let Err(error) = self.artwork.data(chunk) {
                    if Self::producer_error(error, "Artwork", &mut self.artwork_open) {
                        self.close();
                        return Err(RoutingError::GenerationExhausted);
                    }
                }
            }
            BinaryFrame::Visualizer(chunk) if self.visualizer_open => {
                if let Err(error) = self.visualizer.data(chunk) {
                    if Self::producer_error(error, "Visualizer", &mut self.visualizer_open) {
                        self.close();
                        return Err(RoutingError::GenerationExhausted);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn forward_message(&mut self, message: Message) {
        if self.messages_open && self.messages.send(message).is_err() {
            self.messages_open = false;
            log::debug!("Message receiver closed; disabling message destination");
        }
    }

    pub(crate) fn dispatch_message(&mut self, message: Message) -> Result<(), RoutingError> {
        match message {
            Message::StreamStart(start) => {
                if let Some(config) = start.player {
                    if self.player_open {
                        if let Err(error) = self.player.start(config) {
                            if Self::producer_error(error, "Player", &mut self.player_open) {
                                self.close();
                                return Err(RoutingError::GenerationExhausted);
                            }
                        }
                    }
                }
                if let Some(config) = start.artwork {
                    if self.artwork_open {
                        if let Err(error) = self.artwork.start(config) {
                            if Self::producer_error(error, "Artwork", &mut self.artwork_open) {
                                self.close();
                                return Err(RoutingError::GenerationExhausted);
                            }
                        }
                    }
                }
                if let Some(config) = start.visualizer {
                    if self.visualizer_open {
                        if let Err(error) = self.visualizer.start(config) {
                            if Self::producer_error(error, "Visualizer", &mut self.visualizer_open)
                            {
                                self.close();
                                return Err(RoutingError::GenerationExhausted);
                            }
                        }
                    }
                }
                Ok(())
            }
            Message::StreamClear(clear) => {
                let explicit_artwork = clear
                    .roles
                    .as_deref()
                    .is_some_and(|roles| roles.iter().any(|role| role == ROLE_ARTWORK));
                if explicit_artwork {
                    log::debug!("Ignoring explicit artwork clear; artwork has no clear operation");
                }
                for role in role_targets(clear.roles.as_deref()) {
                    match role {
                        Role::Player if self.player_open => {
                            if let Err(error) = self.player.clear() {
                                if Self::producer_error(error, "Player", &mut self.player_open) {
                                    self.close();
                                    return Err(RoutingError::GenerationExhausted);
                                }
                            }
                        }
                        Role::Visualizer if self.visualizer_open => {
                            if let Err(error) = self.visualizer.clear() {
                                if Self::producer_error(
                                    error,
                                    "Visualizer",
                                    &mut self.visualizer_open,
                                ) {
                                    self.close();
                                    return Err(RoutingError::GenerationExhausted);
                                }
                            }
                        }
                        Role::Artwork => {}
                        Role::Player | Role::Visualizer => {}
                    }
                }
                Ok(())
            }
            Message::StreamEnd(end) => {
                for role in role_targets(end.roles.as_deref()) {
                    match role {
                        Role::Player if self.player_open => {
                            if let Err(error) = self.player.end() {
                                if Self::producer_error(error, "Player", &mut self.player_open) {
                                    self.close();
                                    return Err(RoutingError::GenerationExhausted);
                                }
                            }
                        }
                        Role::Artwork if self.artwork_open => {
                            if let Err(error) = self.artwork.end() {
                                if Self::producer_error(error, "Artwork", &mut self.artwork_open) {
                                    self.close();
                                    return Err(RoutingError::GenerationExhausted);
                                }
                            }
                        }
                        Role::Visualizer if self.visualizer_open => {
                            if let Err(error) = self.visualizer.end() {
                                if Self::producer_error(
                                    error,
                                    "Visualizer",
                                    &mut self.visualizer_open,
                                ) {
                                    self.close();
                                    return Err(RoutingError::GenerationExhausted);
                                }
                            }
                        }
                        Role::Player | Role::Artwork | Role::Visualizer => {}
                    }
                }
                Ok(())
            }
            Message::ServerCommand(command) => {
                let Some(player) = command.player else {
                    self.forward_message(Message::ServerCommand(command));
                    return Ok(());
                };
                let valid = match player.command {
                    PlayerCommandType::Volume if player.volume.is_some() => {
                        self.controls.volume = player.volume.unwrap();
                        true
                    }
                    PlayerCommandType::Mute if player.mute.is_some() => {
                        self.controls.muted = player.mute.unwrap();
                        true
                    }
                    PlayerCommandType::SetStaticDelay if player.static_delay_ms.is_some() => {
                        self.controls.static_delay_ms = player.static_delay_ms.unwrap();
                        true
                    }
                    _ => false,
                };
                if valid {
                    match player.command {
                        PlayerCommandType::Volume => self.player.set_volume(self.controls.volume),
                        PlayerCommandType::Mute => self.player.set_mute(self.controls.muted),
                        PlayerCommandType::SetStaticDelay => {
                            self.player.set_static_delay(self.controls.static_delay_ms)
                        }
                        _ => unreachable!(),
                    }
                } else {
                    self.forward_message(Message::ServerCommand(ServerCommand {
                        player: Some(player),
                    }));
                }
                Ok(())
            }
            other => {
                self.forward_message(other);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{AudioBuffer, OutputError, PlayerOutput};
    use crate::protocol::client::{ArtworkChunk, AudioChunk, VisualizerChunk};
    use crate::protocol::messages::{
        PlayerCommand, StreamArtworkChannelConfig, StreamArtworkConfig, StreamPlayerConfig,
        StreamVisualizerConfig,
    };
    use crate::roles::{ArtworkOutput, VisualizerOutput};
    use futures_util::FutureExt;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct FakePlayer;
    impl PlayerOutput for FakePlayer {
        fn commit(&self, _: AudioBuffer) -> Result<(), OutputError> {
            Ok(())
        }
        fn clear(&self) -> Result<(), OutputError> {
            Ok(())
        }
        fn set_volume(&self, _: u8) -> Result<(), OutputError> {
            Ok(())
        }
        fn set_mute(&self, _: bool) -> Result<(), OutputError> {
            Ok(())
        }
        fn set_static_delay(&self, _: u16) -> Result<(), OutputError> {
            Ok(())
        }
    }
    #[derive(Clone, Default)]
    struct FakeArtwork;
    impl ArtworkOutput for FakeArtwork {
        type Prepared = Vec<u8>;
        fn commit(&self, _: u8, _: Self::Prepared) -> Result<(), OutputError> {
            Ok(())
        }
        fn end(&self) -> Result<(), OutputError> {
            Ok(())
        }
    }
    #[derive(Clone, Default)]
    struct FakeVisualizer;
    impl VisualizerOutput for FakeVisualizer {
        type Prepared = Vec<u8>;
        fn commit(&self, _: Self::Prepared) -> Result<(), OutputError> {
            Ok(())
        }
        fn clear(&self) -> Result<(), OutputError> {
            Ok(())
        }
        fn end(&self) -> Result<(), OutputError> {
            Ok(())
        }
    }
    #[derive(Clone)]
    struct FailingArtwork;
    impl ArtworkOutput for FailingArtwork {
        type Prepared = Vec<u8>;
        fn commit(&self, _: u8, _: Self::Prepared) -> Result<(), OutputError> {
            Err(OutputError::Failed("artwork commit failed".into()))
        }
        fn end(&self) -> Result<(), OutputError> {
            Ok(())
        }
    }
    #[derive(Clone)]
    struct FailingVisualizer;
    impl VisualizerOutput for FailingVisualizer {
        type Prepared = Vec<u8>;
        fn commit(&self, _: Self::Prepared) -> Result<(), OutputError> {
            Err(OutputError::Failed("visualizer commit failed".into()))
        }
        fn clear(&self) -> Result<(), OutputError> {
            Ok(())
        }
        fn end(&self) -> Result<(), OutputError> {
            Ok(())
        }
    }

    fn controls() -> AppliedControls {
        AppliedControls {
            volume: 50,
            muted: false,
            static_delay_ms: 0,
        }
    }
    fn audio(timestamp: i64, data: &[u8]) -> AudioChunk {
        AudioChunk {
            timestamp,
            data: Arc::from(data),
        }
    }
    fn artwork(channel: u8, timestamp: i64, data: &[u8]) -> ArtworkChunk {
        ArtworkChunk {
            channel,
            timestamp,
            data: Arc::from(data),
        }
    }
    fn visualizer(timestamp: i64, data: &[u8]) -> VisualizerChunk {
        VisualizerChunk {
            type_id: 16,
            timestamp,
            data: Arc::from(data),
        }
    }
    fn session_player_config(codec: &str) -> StreamPlayerConfig {
        StreamPlayerConfig {
            codec: codec.into(),
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        }
    }
    fn artwork_config() -> StreamArtworkConfig {
        StreamArtworkConfig {
            channels: vec![StreamArtworkChannelConfig {
                source: crate::protocol::messages::ArtworkSource::Album,
                format: crate::protocol::messages::ImageFormat::Png,
                width: 10,
                height: 10,
            }],
        }
    }
    fn visualizer_config() -> StreamVisualizerConfig {
        StreamVisualizerConfig {
            types: vec![crate::protocol::messages::VisualizerDataType::Loudness],
            rate_max: 10,
            tracks_downbeats: None,
            spectrum: None,
        }
    }

    #[tokio::test]
    async fn generation_exhaustion_terminates_router_and_closes_all_sinks() {
        let (mut sinks, player_receiver, artwork_receiver, visualizer_receiver, _) =
            ReceiveSinks::channel(controls());
        let mut player = player_receiver.attach(FakePlayer, controls());
        let mut artwork = artwork_receiver.attach(FakeArtwork);
        let mut visualizer = visualizer_receiver.attach(FakeVisualizer);
        sinks.player.set_generation_max_for_test();
        assert_eq!(
            sinks.dispatch_message(Message::StreamClear(StreamClear {
                roles: Some(vec![ROLE_PLAYER.into()]),
            })),
            Err(RoutingError::GenerationExhausted)
        );
        assert!(player.recv().await.is_none());
        assert!(artwork.recv().await.is_none());
        assert!(visualizer.recv().await.is_none());
    }

    #[tokio::test]
    async fn receive_sinks_preserve_encoded_audio_and_config_fifo() {
        let (mut sinks, receiver, _, _, _) = ReceiveSinks::channel(controls());
        let mut session = receiver.attach(FakePlayer, controls());
        sinks
            .dispatch_message(Message::StreamStart(StreamStart {
                player: Some(session_player_config("pcm")),
                artwork: None,
                visualizer: None,
            }))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Audio(audio(1, &[1, 2, 3])))
            .unwrap();
        sinks
            .dispatch_message(Message::StreamStart(StreamStart {
                player: Some(session_player_config("opus")),
                artwork: None,
                visualizer: None,
            }))
            .unwrap();
        assert!(
            matches!(session.recv().await, Some(crate::audio::PlayerEvent::Start(c)) if c.codec == "pcm")
        );
        assert!(
            matches!(session.recv().await, Some(crate::audio::PlayerEvent::Audio(w)) if w.timestamp() == 1 && w.data() == [1, 2, 3])
        );
        assert!(
            matches!(session.recv().await, Some(crate::audio::PlayerEvent::Start(c)) if c.codec == "opus")
        );
    }

    #[tokio::test]
    async fn receive_sinks_clear_is_urgent_and_valid_command_is_not_forwarded() {
        let (mut sinks, receiver, _, _, mut messages) = ReceiveSinks::channel(controls());
        let mut session = receiver.attach(FakePlayer, controls());
        sinks
            .dispatch_binary(BinaryFrame::Audio(audio(1, &[9])))
            .unwrap();
        sinks
            .dispatch_message(Message::StreamClear(StreamClear {
                roles: Some(vec!["player".into()]),
            }))
            .unwrap();
        let mut controls_rx = session.controls();
        sinks
            .dispatch_message(Message::ServerCommand(ServerCommand {
                player: Some(PlayerCommand {
                    command: PlayerCommandType::Volume,
                    volume: Some(80),
                    mute: Some(true),
                    static_delay_ms: Some(99),
                }),
            }))
            .unwrap();
        assert!(messages.try_recv().is_err());
        assert!(matches!(
            session.recv().await,
            Some(crate::audio::PlayerEvent::Clear)
        ));
        assert!(session.recv().now_or_never().is_none());
        controls_rx.changed().await.unwrap();
        assert_eq!(controls_rx.borrow().volume, 80);
    }

    #[tokio::test]
    async fn receive_sinks_release_detached_dispatch_reinstall_accepts_fresh_audio() {
        let (mut sinks, receiver, _, _, _) = ReceiveSinks::channel(controls());
        let mut session = receiver.attach_lazy(controls());
        let format = crate::audio::AudioFormat {
            codec: crate::audio::Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };
        session
            .install(crate::audio::synced_player::SyncedPlayer::control_for_test(
                format.clone(),
            ))
            .unwrap();
        session.release_output().unwrap();
        assert_eq!(
            sinks.dispatch_binary(BinaryFrame::Audio(audio(1, &[1]))),
            Ok(())
        );
        sinks
            .dispatch_message(Message::StreamStart(StreamStart {
                player: Some(session_player_config("pcm")),
                artwork: None,
                visualizer: None,
            }))
            .unwrap();
        session
            .install(crate::audio::synced_player::SyncedPlayer::control_for_test(
                format.clone(),
            ))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Audio(audio(2, &[2])))
            .unwrap();
        assert!(matches!(
            session.recv().await,
            Some(crate::audio::PlayerEvent::Start(_))
        ));
        let crate::audio::PlayerEvent::Audio(work) = session.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(
            work.commit(crate::audio::AudioBuffer {
                timestamp: 2,
                samples: std::sync::Arc::from(vec![0, 0]),
                format
            }),
            Ok(())
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn receive_sinks_admission_snapshot_keeps_fresh_audio_after_reinstall() {
        let (mut sinks, receiver, _, _, _) = ReceiveSinks::channel(controls());
        let mut session = receiver.attach_lazy(controls());
        let format = crate::audio::AudioFormat {
            codec: crate::audio::Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };
        session
            .install(crate::audio::synced_player::SyncedPlayer::control_for_test(
                format.clone(),
            ))
            .unwrap();
        session.release_output().unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Audio(audio(7, &[7])))
            .unwrap();
        session
            .install(crate::audio::synced_player::SyncedPlayer::control_for_test(
                format.clone(),
            ))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Audio(audio(8, &[8])))
            .unwrap();
        let work = loop {
            match session.recv().await {
                Some(crate::audio::PlayerEvent::Audio(work)) => break work,
                _ => continue,
            }
        };
        assert_eq!(work.timestamp(), 8);
    }

    #[tokio::test]
    async fn receive_sinks_keep_artwork_and_visualizer_independent() {
        let (mut sinks, _, artwork_receiver, visualizer_receiver, _) =
            ReceiveSinks::channel(controls());
        let mut artwork_session = artwork_receiver.attach(FakeArtwork);
        let mut visualizer_session = visualizer_receiver.attach(FakeVisualizer);
        sinks
            .dispatch_message(Message::StreamStart(StreamStart {
                player: None,
                artwork: Some(artwork_config()),
                visualizer: Some(visualizer_config()),
            }))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Artwork(artwork(2, 3, &[4, 5])))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Visualizer(visualizer(6, &[7, 8])))
            .unwrap();
        assert!(matches!(
            artwork_session.recv().await,
            Some(crate::roles::ArtworkEvent::Start(_))
        ));
        assert!(
            matches!(artwork_session.recv().await, Some(crate::roles::ArtworkEvent::Data(w)) if w.channel() == 2 && w.data() == [4, 5])
        );
        assert!(matches!(
            visualizer_session.recv().await,
            Some(crate::roles::VisualizerEvent::Start(_))
        ));
        assert!(
            matches!(visualizer_session.recv().await, Some(crate::roles::VisualizerEvent::Data(w)) if w.timestamp() == 6 && w.data() == [7, 8])
        );
        sinks.close();
        assert!(artwork_session.recv().await.is_none());
        assert!(visualizer_session.recv().await.is_none());
    }

    #[tokio::test]
    async fn receive_sinks_disables_only_failed_artwork_role() {
        let (mut sinks, player_receiver, artwork_receiver, visualizer_receiver, mut messages) =
            ReceiveSinks::channel(controls());
        let mut player = player_receiver.attach(FakePlayer, controls());
        let mut artwork = artwork_receiver.attach(FailingArtwork);
        let mut visualizer = visualizer_receiver.attach(FakeVisualizer);
        sinks
            .dispatch_message(Message::StreamStart(StreamStart {
                player: Some(session_player_config("pcm")),
                artwork: Some(artwork_config()),
                visualizer: Some(visualizer_config()),
            }))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Artwork(super::tests::artwork(1, 1, &[1])))
            .unwrap();
        assert!(matches!(
            artwork.recv().await,
            Some(crate::roles::ArtworkEvent::Start(_))
        ));
        let crate::roles::ArtworkEvent::Data(work) = artwork.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(
            work.commit(Vec::new()),
            Err(crate::audio::CommitRejected::Output(OutputError::Failed(
                "artwork commit failed".into()
            )))
        );
        assert!(matches!(
            *artwork.status().borrow(),
            crate::audio::SessionStatus::Failed(OutputError::Failed(ref message)) if message == "artwork commit failed"
        ));

        sinks
            .dispatch_binary(BinaryFrame::Artwork(super::tests::artwork(1, 2, &[2])))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Visualizer(super::tests::visualizer(3, &[3])))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Audio(audio(4, &[4, 5])))
            .unwrap();
        sinks
            .dispatch_message(Message::ServerCommand(ServerCommand { player: None }))
            .unwrap();
        assert!(!sinks.artwork_open);
        assert!(sinks.visualizer_open);
        assert!(sinks.player_open);

        assert!(matches!(
            visualizer.recv().await,
            Some(crate::roles::VisualizerEvent::Start(_))
        ));
        assert!(matches!(
            visualizer.recv().await,
            Some(crate::roles::VisualizerEvent::Data(_))
        ));
        assert!(matches!(
            player.recv().await,
            Some(crate::audio::PlayerEvent::Start(_))
        ));
        let crate::audio::PlayerEvent::Audio(work) = player.recv().await.unwrap() else {
            panic!()
        };
        work.commit(crate::audio::AudioBuffer {
            timestamp: 4,
            samples: Arc::from(vec![0, 0]),
            format: crate::audio::AudioFormat {
                codec: crate::audio::Codec::Pcm,
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            },
        })
        .unwrap();
        assert!(matches!(
            messages.recv().await,
            Some(Message::ServerCommand(ServerCommand { player: None }))
        ));
        artwork.shutdown().await;
        visualizer.shutdown().await;
        player.shutdown().await;
    }

    #[tokio::test]
    async fn receive_sinks_disables_only_failed_visualizer_role() {
        let (mut sinks, player_receiver, artwork_receiver, visualizer_receiver, mut messages) =
            ReceiveSinks::channel(controls());
        let mut player = player_receiver.attach(FakePlayer, controls());
        let mut artwork = artwork_receiver.attach(FakeArtwork);
        let mut visualizer = visualizer_receiver.attach(FailingVisualizer);
        sinks
            .dispatch_message(Message::StreamStart(StreamStart {
                player: Some(session_player_config("pcm")),
                artwork: Some(artwork_config()),
                visualizer: Some(visualizer_config()),
            }))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Visualizer(super::tests::visualizer(1, &[1])))
            .unwrap();
        assert!(matches!(
            visualizer.recv().await,
            Some(crate::roles::VisualizerEvent::Start(_))
        ));
        let crate::roles::VisualizerEvent::Data(work) = visualizer.recv().await.unwrap() else {
            panic!()
        };
        assert_eq!(
            work.commit(Vec::new()),
            Err(crate::audio::CommitRejected::Output(OutputError::Failed(
                "visualizer commit failed".into()
            )))
        );
        assert!(matches!(
            *visualizer.status().borrow(),
            crate::audio::SessionStatus::Failed(OutputError::Failed(ref message)) if message == "visualizer commit failed"
        ));

        sinks
            .dispatch_binary(BinaryFrame::Visualizer(super::tests::visualizer(1, &[2])))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Artwork(super::tests::artwork(2, 3, &[3])))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Audio(audio(4, &[4, 5])))
            .unwrap();
        sinks
            .dispatch_message(Message::ServerCommand(ServerCommand { player: None }))
            .unwrap();
        assert!(!sinks.visualizer_open);
        assert!(sinks.artwork_open);
        assert!(sinks.player_open);

        assert!(matches!(
            artwork.recv().await,
            Some(crate::roles::ArtworkEvent::Start(_))
        ));
        assert!(matches!(
            artwork.recv().await,
            Some(crate::roles::ArtworkEvent::Data(_))
        ));
        assert!(matches!(
            player.recv().await,
            Some(crate::audio::PlayerEvent::Start(_))
        ));
        let crate::audio::PlayerEvent::Audio(work) = player.recv().await.unwrap() else {
            panic!()
        };
        work.commit(crate::audio::AudioBuffer {
            timestamp: 4,
            samples: Arc::from(vec![0, 0]),
            format: crate::audio::AudioFormat {
                codec: crate::audio::Codec::Pcm,
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            },
        })
        .unwrap();
        assert!(matches!(
            messages.recv().await,
            Some(Message::ServerCommand(ServerCommand { player: None }))
        ));
        artwork.shutdown().await;
        visualizer.shutdown().await;
        player.shutdown().await;
    }

    #[tokio::test]
    async fn receive_sinks_forward_fallback_messages_once() {
        let (mut sinks, _, _, _, mut messages) = ReceiveSinks::channel(controls());
        let message = Message::ServerCommand(ServerCommand { player: None });
        sinks.dispatch_message(message).unwrap();
        assert!(matches!(
            messages.recv().await,
            Some(Message::ServerCommand(ServerCommand { player: None }))
        ));
        assert!(messages.try_recv().is_err());
    }

    #[tokio::test]
    async fn receive_sinks_player_backlog_does_not_block_boundaries_controls_or_fallback() {
        let (mut sinks, receiver, _, _, mut messages) = ReceiveSinks::channel(controls());
        let mut session = receiver.attach(FakePlayer, controls());
        let mut controls_rx = session.controls();
        for timestamp in 0..65 {
            sinks
                .dispatch_binary(BinaryFrame::Audio(audio(timestamp, &[timestamp as u8])))
                .unwrap();
        }
        sinks
            .dispatch_message(Message::StreamClear(StreamClear {
                roles: Some(vec!["player".into()]),
            }))
            .unwrap();
        sinks
            .dispatch_message(Message::ServerCommand(ServerCommand {
                player: Some(PlayerCommand {
                    command: PlayerCommandType::Volume,
                    volume: Some(77),
                    mute: None,
                    static_delay_ms: None,
                }),
            }))
            .unwrap();
        controls_rx.changed().await.unwrap();
        sinks
            .dispatch_message(Message::ServerCommand(ServerCommand { player: None }))
            .unwrap();

        assert!(matches!(
            session.recv().await,
            Some(crate::audio::PlayerEvent::Clear)
        ));
        sinks
            .dispatch_binary(BinaryFrame::Audio(audio(65, &[65])))
            .unwrap();
        assert!(matches!(
            session.recv().await,
            Some(crate::audio::PlayerEvent::Audio(work)) if work.timestamp() == 65
        ));
        assert_eq!(session.controls().borrow().volume, 77);
        assert!(matches!(
            messages.recv().await,
            Some(Message::ServerCommand(ServerCommand { player: None }))
        ));
    }

    #[tokio::test]
    async fn dropped_artwork_does_not_suppress_player_or_visualizer_start() {
        let (mut sinks, player_receiver, artwork_receiver, visualizer_receiver, _) =
            ReceiveSinks::channel(controls());
        drop(artwork_receiver);
        let mut player = player_receiver.attach(FakePlayer, controls());
        let mut visualizer = visualizer_receiver.attach(FakeVisualizer);
        sinks
            .dispatch_message(Message::StreamStart(StreamStart {
                player: Some(session_player_config("pcm")),
                artwork: Some(artwork_config()),
                visualizer: Some(visualizer_config()),
            }))
            .unwrap();
        assert!(matches!(
            player.recv().await,
            Some(crate::audio::PlayerEvent::Start(_))
        ));
        assert!(matches!(
            visualizer.recv().await,
            Some(crate::roles::VisualizerEvent::Start(_))
        ));
    }

    #[tokio::test]
    async fn dropped_messages_do_not_suppress_player_audio() {
        let (mut sinks, player_receiver, _, _, messages) = ReceiveSinks::channel(controls());
        drop(messages);
        let mut player = player_receiver.attach(FakePlayer, controls());
        sinks
            .dispatch_message(Message::ServerState(
                crate::protocol::messages::ServerState {
                    metadata: None,
                    controller: None,
                    color: None,
                },
            ))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Audio(audio(9, &[1, 2])))
            .unwrap();
        assert!(matches!(
            player.recv().await,
            Some(crate::audio::PlayerEvent::Audio(work))
                if work.timestamp() == 9 && work.data() == [1, 2]
        ));
    }

    #[tokio::test]
    async fn dropped_player_does_not_suppress_artwork_or_visualizer() {
        let (mut sinks, player_receiver, artwork_receiver, visualizer_receiver, _) =
            ReceiveSinks::channel(controls());
        drop(player_receiver);
        let mut artwork_session = artwork_receiver.attach(FakeArtwork);
        let mut visualizer_session = visualizer_receiver.attach(FakeVisualizer);
        sinks
            .dispatch_message(Message::StreamStart(StreamStart {
                player: Some(session_player_config("pcm")),
                artwork: Some(artwork_config()),
                visualizer: Some(visualizer_config()),
            }))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Artwork(super::tests::artwork(1, 10, &[3, 4])))
            .unwrap();
        sinks
            .dispatch_binary(BinaryFrame::Visualizer(super::tests::visualizer(
                11,
                &[5, 6],
            )))
            .unwrap();
        assert!(matches!(
            artwork_session.recv().await,
            Some(crate::roles::ArtworkEvent::Start(_))
        ));
        assert!(matches!(
            artwork_session.recv().await,
            Some(crate::roles::ArtworkEvent::Data(work))
                if work.channel() == 1 && work.timestamp() == 10 && work.data() == [3, 4]
        ));
        assert!(matches!(
            visualizer_session.recv().await,
            Some(crate::roles::VisualizerEvent::Start(_))
        ));
        assert!(matches!(
            visualizer_session.recv().await,
            Some(crate::roles::VisualizerEvent::Data(work))
                if work.timestamp() == 11 && work.data() == [5, 6]
        ));
    }

    #[test]
    fn targets_default_and_filter_unknown_names() {
        assert_eq!(
            role_targets(None),
            vec![Role::Player, Role::Artwork, Role::Visualizer]
        );
        assert_eq!(
            role_targets(Some(&["player".into(), "unknown".into(), "player".into()])),
            vec![Role::Player]
        );
        assert!(role_targets(Some(&[])).is_empty());
    }
}
