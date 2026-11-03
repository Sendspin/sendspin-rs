// ABOUTME: Minimal Sendspin player using the public player session API

use sendspin::audio::decode::{Decoder, StreamDecoder};
use sendspin::audio::{
    AudioBuffer, AudioFormat, CommitRejected, InstallRejected, PlayerEvent, PlayerSession,
    SessionStatus, SyncedPlayer, SyncedPlayerConfig,
};
use sendspin::protocol::messages::{
    AudioFormatSpec, PlayerState, PlayerStateCommand, PlayerV1Support,
};
use sendspin::sync::ClockSync;
use sendspin::ProtocolClientBuilder;
use std::error::Error;

type SharedClock = std::sync::Arc<parking_lot::Mutex<ClockSync>>;

fn install_output(
    session: &mut PlayerSession,
    format: &AudioFormat,
    clock_sync: &SharedClock,
    output: &mut Option<SyncedPlayer>,
) -> Result<(), Box<dyn Error>> {
    if let Err(error) = session.release_output() {
        log::warn!("Failed to clear the previous audio output: {error}");
    }
    drop(output.take());
    let player = SyncedPlayer::new(
        format.clone(),
        std::sync::Arc::clone(clock_sync),
        SyncedPlayerConfig::new(),
    )?;
    match session
        .begin_output_install()
        .and_then(|permit| permit.install(player.control()))
    {
        Ok(()) => *output = Some(player),
        // The next `recv` returns `None` and ends the loop.
        Err(InstallRejected::Closed) => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

async fn receive_audio(
    session: &mut PlayerSession,
    clock_sync: SharedClock,
    output: &mut Option<SyncedPlayer>,
) -> Result<(), Box<dyn Error>> {
    let mut decoder: Option<StreamDecoder> = None;
    while let Some(event) = session.recv().await {
        match event {
            PlayerEvent::OutputLost(error) => {
                log::warn!("Audio output lost, reopening: {error}");
                drop(output.take());
                if let Some(state) = decoder.as_ref() {
                    install_output(session, state.format(), &clock_sync, output)?;
                }
            }
            PlayerEvent::Start(config) => {
                let state = StreamDecoder::new(&config)?;
                // Keep buffered audio playing when the new format fits the open device.
                let reusable = output.as_ref().is_some_and(|player| {
                    player.format().sample_rate == state.format().sample_rate
                        && player.format().channels == state.format().channels
                });
                if !reusable {
                    install_output(session, state.format(), &clock_sync, output)?;
                }
                decoder = Some(state);
            }
            PlayerEvent::Audio(work) => {
                let Some(state) = decoder.as_ref() else {
                    continue;
                };
                let samples = state.decode(work.data())?;
                let buffer = AudioBuffer {
                    timestamp: work.timestamp(),
                    samples,
                    format: state.format().clone(),
                };
                // A failed device is reported as the next `OutputLost` event.
                if let Err(CommitRejected::Output(error)) = work.commit(buffer) {
                    log::warn!("Audio output rejected a buffer: {error}");
                }
            }
            // The library already clears the output queue before publishing this event.
            PlayerEvent::Clear => {}
            PlayerEvent::End => {
                session.release_output()?;
                drop(output.take());
                decoder = None;
            }
        }
    }
    let status = session.status().borrow().clone();
    match status {
        SessionStatus::Closed => Ok(()),
        SessionStatus::Failed(error) => Err(format!("player session failed: {error}").into()),
        SessionStatus::Open | SessionStatus::OutputLost(_) => {
            Err("player event stream ended while still open".into())
        }
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let server = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "ws://localhost:8927/sendspin".into());
    let name = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "Sendspin-RS Player".into());
    let id = std::env::args()
        .nth(3)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let client = ProtocolClientBuilder::builder()
        .client_id(id)
        .name(name)
        .player_v1_support(PlayerV1Support {
            supported_formats: vec![
                AudioFormatSpec {
                    codec: "pcm".into(),
                    channels: 2,
                    sample_rate: 48_000,
                    bit_depth: 16,
                },
                AudioFormatSpec {
                    codec: "pcm".into(),
                    channels: 2,
                    sample_rate: 48_000,
                    bit_depth: 24,
                },
                AudioFormatSpec {
                    codec: "opus".into(),
                    channels: 2,
                    sample_rate: 48_000,
                    bit_depth: 16,
                },
                AudioFormatSpec {
                    codec: "flac".into(),
                    channels: 2,
                    sample_rate: 48_000,
                    bit_depth: 16,
                },
                AudioFormatSpec {
                    codec: "flac".into(),
                    channels: 2,
                    sample_rate: 48_000,
                    bit_depth: 24,
                },
            ],
            buffer_capacity: 50 * 1024 * 1024,
            supported_commands: vec!["volume".into(), "mute".into()],
        })
        .initial_player_state(PlayerState {
            volume: Some(100),
            muted: Some(false),
            static_delay_ms: Some(0),
            required_lead_time_ms: Some(500),
            min_buffer_ms: Some(500),
            supported_commands: Some(vec![PlayerStateCommand::SetStaticDelay]),
        })
        .build()
        .connect(&server)
        .await?;
    let conn = client.split();
    drop(conn.artwork);
    drop(conn.visualizer);
    drop(conn.messages);
    let mut session: PlayerSession = conn.player.attach_empty();
    let mut output = None;
    let result = receive_audio(&mut session, conn.clock_sync.clone(), &mut output).await;
    session.shutdown().await;
    drop(output);
    drop(conn.guard);
    result
}
