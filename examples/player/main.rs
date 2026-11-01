// ABOUTME: Minimal Sendspin player using the public lazy-session API

use sendspin::audio::decode::{Decoder, StreamDecoder};
use sendspin::audio::{
    AppliedControls, AudioBuffer, CommitRejected, PlayerEvent, SessionStatus, SyncedPlayer,
    SyncedPlayerConfig,
};
use sendspin::protocol::messages::{
    AudioFormatSpec, PlayerState, PlayerStateCommand, PlayerV1Support,
};
use sendspin::ProtocolClientBuilder;
use std::error::Error;

fn classify_status(status: SessionStatus) -> Result<bool, Box<dyn Error>> {
    match status {
        SessionStatus::Open => Ok(false),
        SessionStatus::Closed => Ok(true),
        SessionStatus::OutputLost(error) => Err(format!("audio output lost: {error}").into()),
        SessionStatus::Failed(error) => Err(format!("player session failed: {error}").into()),
    }
}

async fn receive_audio(
    session: &mut sendspin::audio::PlayerSession,
    clock_sync: std::sync::Arc<parking_lot::Mutex<sendspin::sync::ClockSync>>,
    output: &mut Option<SyncedPlayer>,
) -> Result<(), Box<dyn Error>> {
    let mut status = session.status();
    let mut decoder: Option<StreamDecoder> = None;
    loop {
        let event = tokio::select! {
            biased;
            changed = status.changed() => {
                if changed.is_err() {
                    classify_status(status.borrow().clone()).map(|_| None)
                } else if classify_status(status.borrow().clone())? {
                    Ok(None)
                } else {
                    continue;
                }
            }
            event = session.recv() => Ok(event),
        }?;
        let Some(event) = event else {
            if classify_status(status.borrow().clone())? {
                break;
            }
            return Err("player event stream ended while still open".into());
        };
        match event {
            PlayerEvent::Start(config) => {
                session.release_output()?;
                drop(output.take());
                let state = StreamDecoder::new(&config)?;
                let player = SyncedPlayer::new(
                    state.format().clone(),
                    std::sync::Arc::clone(&clock_sync),
                    SyncedPlayerConfig::new(),
                )?;
                session.begin_output_install()?.install(player.control())?;
                decoder = Some(state);
                *output = Some(player);
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
                match work.commit(buffer) {
                    Ok(()) => {}
                    Err(CommitRejected::Obsolete | CommitRejected::Closed) => continue,
                    Err(error) => return Err(error.into()),
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
    Ok(())
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
    let mut session = conn.player.attach_lazy(AppliedControls {
        volume: 100,
        muted: false,
        static_delay_ms: 0,
    });
    let mut output = None;
    let result = receive_audio(&mut session, conn.clock_sync.clone(), &mut output).await;
    session.shutdown().await;
    drop(output);
    drop(conn.guard);
    result
}
