use super::client::WriteCommand;
use crate::audio::AppliedControls;
use crate::error::Error;
use crate::protocol::messages::{ClientState, ClientSyncState, Message, PlayerState};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_tungstenite::tungstenite::Message as WsMessage;

pub(crate) enum Command {
    State {
        state: ClientState,
        boundary_applied_controls: AppliedControls,
        ack: oneshot::Sender<Result<(), Error>>,
    },
    Enter {
        ack: oneshot::Sender<Result<(), Error>>,
    },
    Exit {
        player: Option<PlayerState>,
        controls: AppliedControls,
        ack: oneshot::Sender<Result<(), Error>>,
    },
}

pub(crate) struct Config {
    pub commands: mpsc::UnboundedReceiver<Command>,
    pub controls: watch::Receiver<AppliedControls>,
    pub initial: ClientState,
    pub output: mpsc::UnboundedSender<WriteCommand>,
}

fn apply_controls(base: &mut PlayerState, controls: &AppliedControls) {
    if base.volume.is_some() {
        base.volume = Some(controls.volume);
    }
    if base.muted.is_some() {
        base.muted = Some(controls.muted);
    }
    if base.static_delay_ms.is_some() {
        base.static_delay_ms = Some(controls.static_delay_ms);
    }
}

fn merge(base: &mut PlayerState, patch: PlayerState) {
    if patch.volume.is_some() {
        base.volume = patch.volume;
    }
    if patch.muted.is_some() {
        base.muted = patch.muted;
    }
    if patch.static_delay_ms.is_some() {
        base.static_delay_ms = patch.static_delay_ms;
    }
    if patch.required_lead_time_ms.is_some() {
        base.required_lead_time_ms = patch.required_lead_time_ms;
    }
    if patch.min_buffer_ms.is_some() {
        base.min_buffer_ms = patch.min_buffer_ms;
    }
    if patch.supported_commands.is_some() {
        base.supported_commands = patch.supported_commands;
    }
}

fn synchronize_player(
    player: &mut PlayerState,
    boundary: &AppliedControls,
    patch: Option<PlayerState>,
) {
    apply_controls(player, boundary);
    if let Some(patch) = patch {
        merge(player, patch);
    }
}

async fn send(output: &mpsc::UnboundedSender<WriteCommand>, message: Message) -> Result<(), Error> {
    let json = serde_json::to_string(&message).map_err(|e| Error::Protocol(e.to_string()))?;
    let (ack, result) = oneshot::channel();
    output
        .send(WriteCommand::Send {
            msg: WsMessage::Text(json.into()),
            ack,
        })
        .map_err(|_| Error::WebSocket("connection closed".to_string()))?;
    result
        .await
        .map_err(|_| Error::WebSocket("connection closed".to_string()))?
}

async fn reconcile_post_state(
    output: &mpsc::UnboundedSender<WriteCommand>,
    controls: &watch::Receiver<AppliedControls>,
    player: &mut PlayerState,
    last_controls: &mut AppliedControls,
    boundary: &AppliedControls,
) -> Result<(), Error> {
    let current = controls.borrow().clone();
    if current == *boundary {
        return Ok(());
    }
    *last_controls = current.clone();
    apply_controls(player, &current);
    send(
        output,
        Message::ClientState(ClientState {
            state: None,
            player: Some(player.clone()),
        }),
    )
    .await
}

pub(crate) async fn run(mut config: Config) {
    let mut player = config.initial.player.clone().unwrap_or_default();
    let mut external = config.initial.state == Some(ClientSyncState::ExternalSource);
    let mut last_controls = config.initial.player.as_ref().map_or_else(
        || config.controls.borrow().clone(),
        |initial| AppliedControls {
            volume: initial.volume.unwrap_or(100),
            muted: initial.muted.unwrap_or(false),
            static_delay_ms: initial.static_delay_ms.unwrap_or(0),
        },
    );
    let current = config.controls.borrow().clone();
    if current != last_controls && !external {
        last_controls = current.clone();
        apply_controls(&mut player, &current);
        if send(
            &config.output,
            Message::ClientState(ClientState {
                state: None,
                player: Some(player.clone()),
            }),
        )
        .await
        .is_err()
        {
            return;
        }
    }
    loop {
        tokio::select! {
            changed = config.controls.changed(), if !external => {
                if changed.is_err() { break; }
                let controls = config.controls.borrow().clone();
                if controls == last_controls { continue; }
                last_controls = controls.clone();
                let mut next = player.clone();
                apply_controls(&mut next, &controls);
                player = next.clone();
                if let Err(error) = send(&config.output, Message::ClientState(ClientState { state: None, player: Some(next) })).await {
                    log::debug!("Player state reporter exiting: {error}");
                    break;
                }
            }
            command = config.commands.recv() => {
                let Some(command) = command else { break; };
                match command {
                    Command::State { state, boundary_applied_controls, ack } => {
                        let post_boundary = boundary_applied_controls.clone();
                        let was_external = external;
                        let reported_state = state.state;
                        if let Some(value) = reported_state { external = value == ClientSyncState::ExternalSource; }
                        if !was_external && external {
                            last_controls = config.controls.borrow().clone();
                        }
                        if !was_external && external {
                            if let Some(patch) = state.player.clone() {
                                merge(&mut player, patch);
                            }
                        } else {
                            // Reconcile the retained player before publishing any state update.
                            synchronize_player(
                                &mut player,
                                &boundary_applied_controls,
                                state.player.clone(),
                            );
                            last_controls = boundary_applied_controls;
                        }
                        let entering_external = !was_external && external;
                        let result = send(&config.output, Message::ClientState(ClientState {
                            state: reported_state,
                            player: if entering_external {
                                state.player.map(|_| player.clone())
                            } else {
                                Some(player.clone())
                            },
                        })).await;
                        if result.is_ok()
                            && !external
                            && !was_external
                            && reconcile_post_state(
                                &config.output,
                                &config.controls,
                                &mut player,
                                &mut last_controls,
                                &post_boundary,
                            )
                            .await
                            .is_err()
                        {
                            break;
                        }
                        let _ = ack.send(result);
                    }
                    Command::Enter { ack } => {
                        external = true;
                        let result = send(&config.output, Message::ClientState(ClientState { state: Some(ClientSyncState::ExternalSource), player: None })).await;
                        let _ = ack.send(result);
                    }
                    Command::Exit { player: patch, controls: boundary, ack } => {
                        external = false;
                        synchronize_player(&mut player, &boundary, patch);
                        last_controls = boundary;
                        let result = send(&config.output, Message::ClientState(ClientState { state: Some(ClientSyncState::Synchronized), player: Some(player.clone()) })).await;
                        if result.is_ok() {
                            let current = config.controls.borrow().clone();
                            if current != last_controls {
                                last_controls = current.clone();
                                apply_controls(&mut player, &current);
                                let followup = send(&config.output, Message::ClientState(ClientState { state: None, player: Some(player.clone()) })).await;
                                if followup.is_err() { break; }
                            }
                        }
                        let _ = ack.send(result);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    fn controls(volume: u8) -> AppliedControls {
        AppliedControls {
            volume,
            muted: false,
            static_delay_ms: 0,
        }
    }

    fn initial(volume: u8) -> ClientState {
        ClientState {
            state: Some(ClientSyncState::Synchronized),
            player: Some(PlayerState {
                volume: Some(volume),
                muted: Some(false),
                static_delay_ms: Some(0),
                required_lead_time_ms: None,
                min_buffer_ms: None,
                supported_commands: None,
            }),
        }
    }

    async fn read_messages(
        mut rx: mpsc::UnboundedReceiver<WriteCommand>,
        count: usize,
    ) -> Vec<String> {
        let mut result = Vec::new();
        for _ in 0..count {
            let Some(WriteCommand::Send {
                msg: WsMessage::Text(text),
                ack,
            }) = rx.recv().await
            else {
                break;
            };
            result.push(text.to_string());
            let _ = ack.send(Ok(()));
        }
        result
    }

    #[tokio::test]
    async fn startup_reconciles_applied_change_before_first_poll() {
        let (_commands, command_rx) = mpsc::unbounded_channel();
        let (_tx, controls_rx) = watch::channel(controls(80));
        let (output, output_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(Config {
            commands: command_rx,
            controls: controls_rx,
            initial: initial(50),
            output,
        }));
        let messages = timeout(Duration::from_secs(1), read_messages(output_rx, 1))
            .await
            .unwrap();
        assert!(messages[0].contains("80"));
        task.abort();
    }

    #[tokio::test]
    async fn startup_deduplicates_equal_initial_controls() {
        let (_commands, command_rx) = mpsc::unbounded_channel();
        let (_tx, controls_rx) = watch::channel(controls(50));
        let (output, mut output_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(Config {
            commands: command_rx,
            controls: controls_rx,
            initial: initial(50),
            output,
        }));
        assert!(timeout(Duration::from_millis(20), output_rx.recv())
            .await
            .is_err());
        task.abort();
    }

    #[tokio::test]
    async fn raw_sync_state_uses_enqueue_boundary_after_external_source() {
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (tx, controls_rx) = watch::channel(controls(50));
        let (output, output_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(Config {
            commands: command_rx,
            controls: controls_rx,
            initial: initial(50),
            output,
        }));
        let (ack, result) = oneshot::channel();
        commands
            .send(Command::State {
                state: ClientState {
                    state: Some(ClientSyncState::ExternalSource),
                    player: None,
                },
                boundary_applied_controls: controls(50),
                ack,
            })
            .unwrap();
        let mut output_rx = output_rx;
        let Some(WriteCommand::Send {
            msg: WsMessage::Text(text),
            ack: writer_ack,
        }) = timeout(Duration::from_secs(1), output_rx.recv())
            .await
            .unwrap()
        else {
            panic!("missing external-source state")
        };
        let message: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(message["payload"]["state"], "external_source");
        assert!(!text.contains("\"player\""));
        let _ = writer_ack.send(Ok(()));
        let _ = result.await;
        tx.send(controls(30)).unwrap();

        let (ack, result) = oneshot::channel();
        commands
            .send(Command::Exit {
                player: None,
                controls: controls(70),
                ack,
            })
            .unwrap();
        let Some(WriteCommand::Send {
            msg: WsMessage::Text(text),
            ack,
        }) = timeout(Duration::from_secs(1), output_rx.recv())
            .await
            .unwrap()
        else {
            panic!("missing synchronized exit state")
        };
        let _ = ack.send(Ok(()));
        let Some(WriteCommand::Send {
            msg: WsMessage::Text(followup),
            ack,
        }) = timeout(Duration::from_secs(1), output_rx.recv())
            .await
            .unwrap()
        else {
            panic!("missing post-exit controls")
        };
        let _ = ack.send(Ok(()));
        let _ = result.await;
        assert!(text.contains("synchronized"));
        assert!(text.contains("70"));
        assert!(followup.contains("30"));
        task.abort();
    }

    #[tokio::test]
    async fn raw_sync_state_replays_change_after_boundary_before_processing() {
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (tx, controls_rx) = watch::channel(controls(50));
        let (output, output_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(Config {
            commands: command_rx,
            controls: controls_rx,
            initial: ClientState {
                state: Some(ClientSyncState::ExternalSource),
                player: Some(PlayerState {
                    volume: Some(50),
                    ..PlayerState::default()
                }),
            },
            output,
        }));
        let (ack, result) = oneshot::channel();
        commands
            .send(Command::State {
                state: ClientState {
                    state: Some(ClientSyncState::Synchronized),
                    player: Some(PlayerState {
                        volume: Some(70),
                        ..PlayerState::default()
                    }),
                },
                boundary_applied_controls: controls(70),
                ack,
            })
            .unwrap();
        tx.send(controls(33)).unwrap();
        let messages = timeout(Duration::from_secs(1), read_messages(output_rx, 2))
            .await
            .unwrap();
        let _ = result.await;
        assert!(messages[0].contains("70"));
        assert!(messages[1].contains("33"));
        task.abort();
    }

    #[tokio::test]
    async fn manual_state_boundary_reconciles_controls_before_metadata_patch() {
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (_tx, controls_rx) = watch::channel(controls(50));
        let (output, output_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(Config {
            commands: command_rx,
            controls: controls_rx,
            initial: initial(50),
            output,
        }));
        let (ack, result) = oneshot::channel();
        commands
            .send(Command::State {
                state: ClientState {
                    state: Some(ClientSyncState::Synchronized),
                    player: Some(PlayerState {
                        min_buffer_ms: Some(25),
                        ..PlayerState::default()
                    }),
                },
                boundary_applied_controls: controls(30),
                ack,
            })
            .unwrap();
        let messages = timeout(Duration::from_secs(1), read_messages(output_rx, 1))
            .await
            .unwrap();
        let _ = result.await;
        let message: serde_json::Value = serde_json::from_str(&messages[0]).unwrap();
        let player = &message["payload"]["player"];
        assert_eq!(player["volume"], 30);
        assert_eq!(player["min_buffer_ms"], 25);
        task.abort();
    }

    #[tokio::test]
    async fn synchronized_state_with_newer_controls_reports_boundary_then_followup() {
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (tx, controls_rx) = watch::channel(controls(50));
        let (output, output_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(Config {
            commands: command_rx,
            controls: controls_rx,
            initial: initial(50),
            output,
        }));
        // Consume the watch update first, then enqueue a command with an older boundary.
        tx.send(controls(80)).unwrap();
        let mut output_rx = output_rx;
        let Some(WriteCommand::Send {
            msg: WsMessage::Text(automatic),
            ack: writer_ack,
        }) = timeout(Duration::from_secs(1), output_rx.recv())
            .await
            .unwrap()
        else {
            panic!("missing automatic state")
        };
        assert!(automatic.contains("80"));

        let (ack_state, result) = oneshot::channel();
        commands
            .send(Command::State {
                state: ClientState {
                    state: Some(ClientSyncState::Synchronized),
                    player: Some(PlayerState {
                        min_buffer_ms: Some(40),
                        ..PlayerState::default()
                    }),
                },
                boundary_applied_controls: controls(50),
                ack: ack_state,
            })
            .unwrap();
        let _ = writer_ack.send(Ok(()));
        let messages = timeout(Duration::from_secs(1), read_messages(output_rx, 2))
            .await
            .unwrap();
        let _ = result.await;
        assert_eq!(messages.len(), 2);
        let boundary: serde_json::Value = serde_json::from_str(&messages[0]).unwrap();
        let final_state: serde_json::Value = serde_json::from_str(&messages[1]).unwrap();
        assert_eq!(boundary["payload"]["state"], "synchronized");
        assert_eq!(boundary["payload"]["player"]["volume"], 50);
        assert_eq!(boundary["payload"]["player"]["min_buffer_ms"], 40);
        assert_eq!(final_state["payload"]["state"], serde_json::Value::Null);
        assert_eq!(final_state["payload"]["player"]["volume"], 80);
        assert_eq!(final_state["payload"]["player"]["min_buffer_ms"], 40);
        task.abort();
    }

    #[tokio::test]
    async fn explicit_exit_applies_latest_enqueue_controls_without_patch() {
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (_tx, controls_rx) = watch::channel(controls(33));
        let (output, output_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(Config {
            commands: command_rx,
            controls: controls_rx,
            initial: ClientState {
                state: Some(ClientSyncState::ExternalSource),
                player: Some(PlayerState {
                    volume: Some(50),
                    muted: Some(false),
                    static_delay_ms: Some(0),
                    required_lead_time_ms: None,
                    min_buffer_ms: Some(40),
                    supported_commands: None,
                }),
            },
            output,
        }));
        let (ack, result) = oneshot::channel();
        commands
            .send(Command::Exit {
                player: None,
                controls: controls(33),
                ack,
            })
            .unwrap();
        let messages = timeout(Duration::from_secs(5), read_messages(output_rx, 1))
            .await
            .unwrap();
        result.await.unwrap().unwrap();
        let state: serde_json::Value = serde_json::from_str(&messages[0]).unwrap();
        assert_eq!(state["payload"]["state"], "synchronized");
        assert_eq!(state["payload"]["player"]["volume"], 33);
        assert_eq!(state["payload"]["player"]["min_buffer_ms"], 40);
        task.abort();
    }

    #[tokio::test]
    async fn generic_sync_applies_latest_controls_without_boundary_or_patch() {
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (tx, controls_rx) = watch::channel(controls(50));
        let (output, mut output_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(Config {
            commands: command_rx,
            controls: controls_rx,
            initial: initial(50),
            output,
        }));
        let (ack, result) = oneshot::channel();
        commands
            .send(Command::State {
                state: ClientState {
                    state: Some(ClientSyncState::ExternalSource),
                    player: None,
                },
                boundary_applied_controls: controls(50),
                ack,
            })
            .unwrap();
        let Some(WriteCommand::Send {
            ack: writer_ack, ..
        }) = timeout(Duration::from_secs(1), output_rx.recv())
            .await
            .unwrap()
        else {
            panic!("missing external state")
        };
        let _ = writer_ack.send(Ok(()));
        let _ = result.await;
        tx.send(controls(33)).unwrap();
        let (ack, result) = oneshot::channel();
        commands
            .send(Command::State {
                state: ClientState {
                    state: Some(ClientSyncState::Synchronized),
                    player: None,
                },
                boundary_applied_controls: controls(33),
                ack,
            })
            .unwrap();
        let messages = timeout(Duration::from_secs(1), read_messages(output_rx, 1))
            .await
            .unwrap();
        let _ = result.await;
        assert!(messages[0].contains("33"));
        task.abort();
    }

    #[tokio::test]
    async fn exit_reconciles_change_after_enqueue_boundary() {
        let (commands, command_rx) = mpsc::unbounded_channel();
        let (tx, controls_rx) = watch::channel(controls(50));
        let (output, output_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(run(Config {
            commands: command_rx,
            controls: controls_rx,
            initial: initial(50),
            output,
        }));
        let (ack, result) = oneshot::channel();
        commands
            .send(Command::Exit {
                player: None,
                controls: controls(50),
                ack,
            })
            .unwrap();
        tx.send(controls(33)).unwrap();
        let messages = timeout(Duration::from_secs(1), read_messages(output_rx, 2))
            .await
            .unwrap();
        let _ = result.await;
        assert!(messages
            .iter()
            .any(|message| message.contains("synchronized")));
        assert!(messages.iter().any(|message| message.contains("33")));
        task.abort();
    }
}
