// ABOUTME: End-to-end player example
// ABOUTME: Connects to a server and delegates audio preparation to an owning thread

mod worker;

use clap::{Parser, ValueEnum};
use sendspin::audio::{AppliedControls, PlayerEvent};
use sendspin::protocol::messages::{
    AudioFormatSpec, Message, PlayerFormatRequest, PlayerState, PlayerStateCommand, PlayerV1Support,
};
use sendspin::ProtocolClientBuilder;
use std::error::Error;
use std::future::Future;
use tokio::sync::{watch, Semaphore};
use worker::{Command, Worker};

use cpal::traits::{DeviceTrait, HostTrait};

fn print_devices() -> Result<(), Box<dyn Error>> {
    let mut index = 0;
    let mut example = None;
    for host_id in cpal::available_hosts() {
        let host = cpal::host_from_id(host_id)?;
        for device in host.devices()? {
            let id = device
                .id()
                .map_or_else(|_| "Unknown ID".to_string(), |id| id.to_string());
            if device
                .supported_output_configs()
                .is_ok_and(|mut configs| configs.next().is_some())
            {
                example.get_or_insert_with(|| id.clone());
                println!("[{index}] {id}");
                index += 1;
            }
        }
    }
    if index == 0 {
        println!("No devices found");
    } else if let Some(id) = example {
        println!("Select with --audio-device 0 or --audio-device \"{id}\"");
    }
    Ok(())
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CodecChoice {
    Pcm,
    Opus,
    Flac,
}

#[derive(Parser, Debug)]
#[command(name = "player", about = "Connect to Sendspin server and play audio")]
struct Args {
    #[arg(short, long, default_value = "ws://localhost:8927/sendspin")]
    server: String,
    #[arg(short, long, default_value = "Sendspin-RS Player")]
    name: String,
    #[arg(short = 'i', long = "id")]
    id: Option<String>,
    #[arg(long, value_enum, help = "Preferred stream codec")]
    codec: Option<CodecChoice>,
    #[arg(
        long = "list-audio-devices",
        help = "List usable output devices with their index and id"
    )]
    list_audio_devices: bool,
    #[arg(
        long = "audio-device",
        help = "Output device index from --list-audio-devices or exact device id"
    )]
    audio_device: Option<String>,
}

async fn report_player_state<E, F, Fut>(
    mut controls: watch::Receiver<AppliedControls>,
    mut state: PlayerState,
    mut stop: watch::Receiver<bool>,
    mut send: F,
) where
    F: FnMut(Message) -> Fut,
    Fut: Future<Output = Result<(), E>>,
    E: std::fmt::Debug,
{
    loop {
        let value = controls.borrow_and_update().clone();
        state.volume = Some(value.volume);
        state.muted = Some(value.muted);
        state.static_delay_ms = Some(value.static_delay_ms);
        let sending = send(Message::ClientState(
            sendspin::protocol::messages::ClientState {
                state: None,
                player: Some(state.clone()),
            },
        ));
        tokio::pin!(sending);
        tokio::select! {
            result = &mut sending => {
                if let Err(error) = result { log::debug!("Player state report failed: {:?}", error); break; }
            }
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow() { break; }
            }
        }
        tokio::select! {
            changed = controls.changed() => if changed.is_err() { break; },
            changed = stop.changed() => if changed.is_err() || *stop.borrow() { break; },
        }
    }
}

async fn request_format_after_first_start<F, Fut, E>(
    mut started: watch::Receiver<bool>,
    mut stop: watch::Receiver<bool>,
    request: F,
) where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<(), E>>,
    E: std::fmt::Debug,
{
    loop {
        tokio::select! {
            changed = started.changed() => {
                if changed.is_err() { return; }
                if *started.borrow() {
                    let mut request = std::pin::pin!(request());
                    tokio::select! {
                        result = &mut request => {
                            if let Err(error) = result { log::warn!("Player format request failed: {:?}", error); }
                        }
                        changed = stop.changed() => {
                            if changed.is_err() || *stop.borrow() { return; }
                        }
                    }
                    return;
                }
            }
            changed = stop.changed() => if changed.is_err() || *stop.borrow() { return; },
        }
    }
}

async fn cleanup_forwarder(
    forwarder_finished: bool,
    forwarder: tokio::task::JoinHandle<()>,
    stop_tx: watch::Sender<bool>,
    reporter: tokio::task::JoinHandle<()>,
    format_task: Option<tokio::task::JoinHandle<()>>,
) {
    let _ = stop_tx.send(true);
    let _ = reporter.await;
    if let Some(task) = format_task {
        let _ = task.await;
    }
    // The flag records whether the select branch already consumed the handle.
    if !forwarder_finished {
        let _ = forwarder.await;
    }
}

async fn admit_command(
    stop: &mut watch::Receiver<bool>,
    admission: &std::sync::Arc<Semaphore>,
) -> Option<tokio::sync::OwnedSemaphorePermit> {
    loop {
        tokio::select! {
            biased;
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow() { return None; }
            }
            permit = admission.clone().acquire_owned() => return permit.ok(),
        }
    }
}

fn request_formats_enabled(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true")
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let args = Args::parse();
    if args.list_audio_devices {
        print_devices()?;
        return Ok(());
    }
    if let Some(query) = args.audio_device.as_deref() {
        let query = query.to_string();
        tokio::task::spawn_blocking(move || Worker::validate_device(Some(&query))).await??;
    }
    let formats: &[(&str, u8)] = match args.codec {
        Some(CodecChoice::Pcm) => &[("pcm", 24), ("pcm", 16)],
        Some(CodecChoice::Opus) => &[("opus", 16)],
        Some(CodecChoice::Flac) => &[("flac", 24), ("flac", 16)],
        None => &[
            ("flac", 24),
            ("flac", 16),
            ("opus", 16),
            ("pcm", 24),
            ("pcm", 16),
        ],
    };
    let supported_formats = formats
        .iter()
        .map(|&(codec, bit_depth)| AudioFormatSpec {
            codec: codec.to_string(),
            channels: 2,
            sample_rate: 48_000,
            bit_depth,
        })
        .collect();
    let initial_player_state = PlayerState {
        volume: Some(100),
        muted: Some(false),
        static_delay_ms: Some(0),
        required_lead_time_ms: Some(500),
        min_buffer_ms: Some(500),
        supported_commands: Some(vec![PlayerStateCommand::SetStaticDelay]),
    };
    let client = ProtocolClientBuilder::builder()
        .client_id(args.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()))
        .name(args.name)
        .player_v1_support(PlayerV1Support {
            supported_formats,
            buffer_capacity: 50 * 1024 * 1024,
            supported_commands: vec!["volume".into(), "mute".into()],
        })
        .initial_player_state(initial_player_state.clone())
        .build()
        .connect(&args.server)
        .await?;
    let conn = client.split();
    drop(conn.artwork);
    drop(conn.visualizer);
    let mut messages = conn.messages;
    let session = conn.player.attach_lazy(AppliedControls {
        volume: 100,
        muted: false,
        static_delay_ms: 0,
    });
    let controls = session.controls();
    let mut status = session.status();
    let worker = Worker::start(args.audio_device, conn.clock_sync.clone());
    let admission = std::sync::Arc::new(Semaphore::new(1));
    let sender = conn.sender;
    let _guard = conn.guard;
    let (stop_tx, stop_rx) = watch::channel(false);

    let report_sender = sender.clone();
    let reporter = tokio::spawn(report_player_state(
        controls.clone(),
        initial_player_state.clone(),
        stop_rx,
        move |message| {
            let sender = report_sender.clone();
            async move { sender.send_message(message).await }
        },
    ));

    let request_formats =
        request_formats_enabled(std::env::var("SS_REQUEST_FORMAT").ok().as_deref());
    let (started_tx, started_rx) = watch::channel(false);
    let mut format_task = None;
    if request_formats {
        let format_sender = sender.clone();
        let format_stop = stop_tx.subscribe();
        format_task = Some(tokio::spawn(request_format_after_first_start(
            started_rx,
            format_stop,
            move || {
                let sender = format_sender.clone();
                async move {
                    sender
                        .request_player_format(PlayerFormatRequest {
                            codec: Some("pcm".into()),
                            channels: Some(2),
                            sample_rate: Some(48_000),
                            bit_depth: Some(16),
                        })
                        .await
                }
            },
        )));
    }
    let forward_stop = stop_tx.subscribe();
    let mut forwarder = tokio::spawn(async move {
        let mut session = session;
        let mut stop = forward_stop;
        let worker = worker;
        loop {
            tokio::select! {
                biased;
                changed = stop.changed() => {
                    if changed.is_err() || *stop.borrow() { break; }
                }
                event = session.recv() => {
                    let Some(event) = event else { break; };
                    let permit = match admit_command(&mut stop, &admission).await {
                        Some(permit) => permit,
                        None => break,
                    };
                    let command = match event {
                        PlayerEvent::Start(config) => {
                            let _ = started_tx.send(true);
                            Command::Start(config, permit)
                        }
                        PlayerEvent::Clear => Command::Clear(permit),
                        PlayerEvent::End => Command::End(permit),
                        PlayerEvent::Audio(work) => {
                            let install = match session.begin_output_install() {
                                Ok(permit) => Some(permit),
                                Err(sendspin::audio::InstallRejected::AlreadyInstalled) => None,
                                Err(error) => { log::debug!("Audio output installation unavailable: {error:?}"); None }
                            };
                            Command::Audio(work, permit, install)
                        }
                    };
                    if worker.send(command).is_err() { break; }
                }
            }
        }
        session.shutdown().await;
        worker.shutdown().await;
    });

    let mut forwarder_finished = false;
    loop {
        tokio::select! {
            Some(message) = messages.recv() => {
                log::debug!("Received message: {:?}", message);
            }
            changed = status.changed() => {
                if changed.is_err() || !matches!(*status.borrow(), sendspin::audio::SessionStatus::Open) { break; }
            }
            result = &mut forwarder => {
                forwarder_finished = true;
                if let Err(error) = result { log::error!("Player forwarding task failed: {error}"); }
                break;
            }
            else => break,
        }
    }
    cleanup_forwarder(
        forwarder_finished,
        forwarder,
        stop_tx,
        reporter,
        format_task,
    )
    .await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::time::{timeout, Duration};

    fn controls(volume: u8) -> AppliedControls {
        AppliedControls {
            volume,
            muted: false,
            static_delay_ms: 7,
        }
    }

    fn state() -> PlayerState {
        PlayerState {
            volume: None,
            muted: None,
            static_delay_ms: Some(9),
            required_lead_time_ms: Some(500),
            min_buffer_ms: Some(600),
            supported_commands: Some(vec![PlayerStateCommand::SetStaticDelay]),
        }
    }

    #[tokio::test]
    async fn reporter_cancels_stalled_send_and_joins() {
        let (_controls_tx, controls_rx) = watch::channel(controls(1));
        let (stop_tx, stop_rx) = watch::channel(false);
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let entered_tx = Arc::new(Mutex::new(Some(entered_tx)));
        let task = tokio::spawn(report_player_state(
            controls_rx,
            state(),
            stop_rx,
            move |_| {
                if let Some(sender) = entered_tx.lock().unwrap().take() {
                    let _ = sender.send(());
                }
                std::future::pending::<Result<(), ()>>()
            },
        ));
        timeout(Duration::from_secs(5), entered_rx)
            .await
            .expect("reporter did not start within 5 seconds")
            .expect("reporter start signal dropped");
        stop_tx.send(true).unwrap();
        timeout(Duration::from_secs(5), task)
            .await
            .expect("reporter did not stop within 5 seconds")
            .expect("reporter task panicked");
    }

    #[tokio::test]
    async fn reporter_serializes_full_player_state() {
        let (_controls_tx, controls_rx) = watch::channel(controls(42));
        let (stop_tx, stop_rx) = watch::channel(false);
        let sent = Arc::new(Mutex::new(Vec::<Message>::new()));
        let seen = sent.clone();
        let (sent_tx, sent_rx) = tokio::sync::oneshot::channel();
        let sent_tx = Arc::new(Mutex::new(Some(sent_tx)));
        let task = tokio::spawn(report_player_state(
            controls_rx,
            state(),
            stop_rx,
            move |message| {
                let seen = seen.clone();
                let sent_tx = sent_tx.clone();
                async move {
                    seen.lock().unwrap().push(message);
                    if let Some(sender) = sent_tx.lock().unwrap().take() {
                        let _ = sender.send(());
                    }
                    Ok::<(), ()>(())
                }
            },
        ));
        timeout(Duration::from_secs(5), sent_rx)
            .await
            .expect("reporter did not send state within 5 seconds")
            .expect("reporter send signal dropped");
        stop_tx.send(true).unwrap();
        timeout(Duration::from_secs(5), task)
            .await
            .expect("reporter did not stop within 5 seconds")
            .expect("reporter task panicked");
        let messages = sent.lock().unwrap();
        let Message::ClientState(client) = &messages[0] else {
            panic!("wrong message")
        };
        let player = client.player.as_ref().unwrap();
        assert_eq!(player.volume, Some(42));
        assert_eq!(player.required_lead_time_ms, Some(500));
        assert_eq!(serde_json::to_value(player).unwrap()["min_buffer_ms"], 600);
    }

    #[tokio::test]
    async fn format_request_cancels_inflight_request_on_stop() {
        let (started_tx, started_rx) = watch::channel(false);
        let (stop_tx, stop_rx) = watch::channel(false);
        let entered = Arc::new(tokio::sync::Notify::new());
        let seen = entered.clone();
        let task = tokio::spawn(request_format_after_first_start(
            started_rx,
            stop_rx,
            move || {
                seen.notify_one();
                std::future::pending::<Result<(), ()>>()
            },
        ));
        started_tx.send(true).unwrap();
        timeout(Duration::from_secs(5), entered.notified())
            .await
            .unwrap();
        stop_tx.send(true).unwrap();
        timeout(Duration::from_secs(5), task)
            .await
            .expect("format task did not stop within 5 seconds")
            .expect("format task panicked");
    }

    #[tokio::test]
    async fn cleanup_forwarder_handles_completed_forwarder() {
        let forwarder = tokio::spawn(async {});
        let reporter = tokio::spawn(async {});
        let mut forwarder = forwarder;
        timeout(Duration::from_secs(5), &mut forwarder)
            .await
            .expect("forwarder did not become ready within 5 seconds")
            .expect("forwarder task panicked");
        timeout(
            Duration::from_secs(5),
            cleanup_forwarder(true, forwarder, watch::channel(false).0, reporter, None),
        )
        .await
        .expect("cleanup did not complete within 5 seconds");
    }

    #[test]
    fn format_env_values_follow_explicit_boolean_semantics() {
        assert!(!request_formats_enabled(None));
        assert!(request_formats_enabled(Some("1")));
        assert!(request_formats_enabled(Some("true")));
        assert!(request_formats_enabled(Some(" TRUE ")));
        assert!(!request_formats_enabled(Some("0")));
        assert!(!request_formats_enabled(Some("false")));
        assert!(!request_formats_enabled(Some("yes")));
        assert!(!request_formats_enabled(Some("on")));
        assert!(!request_formats_enabled(Some("")));
    }

    #[tokio::test]
    async fn format_request_waits_for_first_start_and_runs_once() {
        let (started_tx, started_rx) = watch::channel(false);
        let (stop_tx, stop_rx) = watch::channel(false);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = calls.clone();
        let mut task = Box::pin(request_format_after_first_start(
            started_rx,
            stop_rx,
            move || {
                seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async { Ok::<(), ()>(()) }
            },
        ));
        let waker = futures_util::task::noop_waker();
        let mut context = std::task::Context::from_waker(&waker);
        assert!(matches!(
            task.as_mut().poll(&mut context),
            std::task::Poll::Pending
        ));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        started_tx.send(true).unwrap();
        timeout(Duration::from_secs(5), task.as_mut())
            .await
            .expect("format request did not finish within 5 seconds");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        stop_tx.send(true).ok();
    }

    #[tokio::test]
    async fn admission_blocks_second_command_until_first_is_released() {
        let admission = Arc::new(Semaphore::new(1));
        let first = admission.clone().acquire_owned().await.unwrap();
        let (_stop_tx, mut stop_rx) = watch::channel(false);
        let mut blocked = Box::pin(admit_command(&mut stop_rx, &admission));
        let waker = futures_util::task::noop_waker();
        let mut context = std::task::Context::from_waker(&waker);
        assert!(matches!(
            blocked.as_mut().poll(&mut context),
            std::task::Poll::Pending
        ));
        drop(first);
        assert!(timeout(Duration::from_secs(5), blocked)
            .await
            .expect("admission did not unblock within 5 seconds")
            .is_some());
    }
}
