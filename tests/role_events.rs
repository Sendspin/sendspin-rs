use futures_util::{SinkExt, StreamExt};
use sendspin::audio::decode::{Decoder, OpusDecoder, PcmDecoder};
use sendspin::audio::{
    AppliedControls, AudioBuffer, AudioFormat, Codec, OutputError, PlayerEvent, PlayerOutput,
    SessionStatus,
};
use sendspin::protocol::client::binary_types;
use sendspin::protocol::messages::{
    ArtworkSource, ConnectionReason, ImageFormat, Message, PlayerCommand, PlayerCommandType,
    ServerCommand, ServerHello, StreamArtworkChannelConfig, StreamArtworkConfig, StreamClear,
    StreamEnd, StreamPlayerConfig, StreamStart, StreamVisualizerConfig, VisualizerDataType,
};
use sendspin::roles::{ArtworkEvent, ArtworkOutput, VisualizerEvent, VisualizerOutput};
use sendspin::{Clock, ProtocolClientBuilder};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::{tungstenite::Message as WsMessage, WebSocketStream};

#[derive(Clone, Default)]
struct Events(Arc<Mutex<Vec<String>>>);

#[derive(Clone)]
struct RecordedOutput {
    commits: Arc<Mutex<Vec<AudioBufferRecord>>>,
    clears: Arc<Mutex<usize>>,
    volumes: Arc<Mutex<Vec<u8>>>,
    effects: watch::Sender<OutputState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OutputState {
    clear_count: usize,
    volume: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AudioBufferRecord {
    timestamp: i64,
    samples: Arc<[i32]>,
    format: AudioFormat,
}

impl RecordedOutput {
    fn new() -> (Self, watch::Receiver<OutputState>) {
        let (effects, receiver) = watch::channel(OutputState {
            clear_count: 0,
            volume: 0,
        });
        (
            Self {
                commits: Arc::new(Mutex::new(Vec::new())),
                clears: Arc::new(Mutex::new(0)),
                volumes: Arc::new(Mutex::new(Vec::new())),
                effects,
            },
            receiver,
        )
    }
}

impl Default for RecordedOutput {
    fn default() -> Self {
        Self::new().0
    }
}

impl PlayerOutput for RecordedOutput {
    fn commit(&self, buffer: AudioBuffer) -> Result<(), OutputError> {
        self.commits.lock().unwrap().push(AudioBufferRecord {
            timestamp: buffer.timestamp,
            samples: buffer.samples,
            format: buffer.format,
        });
        Ok(())
    }
    fn clear(&self) -> Result<(), OutputError> {
        let count = {
            let mut clears = self.clears.lock().unwrap();
            *clears += 1;
            *clears
        };
        let volume = *self.volumes.lock().unwrap().last().unwrap_or(&0);
        self.effects.send_replace(OutputState {
            clear_count: count,
            volume,
        });
        Ok(())
    }
    fn set_volume(&self, volume: u8) -> Result<(), OutputError> {
        self.volumes.lock().unwrap().push(volume);
        let clear_count = *self.clears.lock().unwrap();
        self.effects.send_replace(OutputState {
            clear_count,
            volume,
        });
        Ok(())
    }
    fn set_mute(&self, _: bool) -> Result<(), OutputError> {
        Ok(())
    }
    fn set_static_delay(&self, _: u16) -> Result<(), OutputError> {
        Ok(())
    }
}
impl PlayerOutput for Events {
    fn commit(&self, _: AudioBuffer) -> Result<(), OutputError> {
        self.0.lock().unwrap().push("audio".into());
        Ok(())
    }
    fn clear(&self) -> Result<(), OutputError> {
        self.0.lock().unwrap().push("clear".into());
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
impl ArtworkOutput for Events {
    type Prepared = Vec<u8>;
    fn commit(&self, _: u8, _: Vec<u8>) -> Result<(), OutputError> {
        self.0.lock().unwrap().push("artwork".into());
        Ok(())
    }
    fn end(&self) -> Result<(), OutputError> {
        self.0.lock().unwrap().push("art-end".into());
        Ok(())
    }
}
impl VisualizerOutput for Events {
    type Prepared = Vec<u8>;
    fn commit(&self, _: Vec<u8>) -> Result<(), OutputError> {
        self.0.lock().unwrap().push("visualizer".into());
        Ok(())
    }
    fn clear(&self) -> Result<(), OutputError> {
        self.0.lock().unwrap().push("viz-clear".into());
        Ok(())
    }
    fn end(&self) -> Result<(), OutputError> {
        self.0.lock().unwrap().push("viz-end".into());
        Ok(())
    }
}

struct FakeClock {
    current: std::sync::atomic::AtomicI64,
}
impl FakeClock {
    fn new() -> Self {
        Self {
            current: std::sync::atomic::AtomicI64::new(1_000),
        }
    }
    fn set(&self, value: i64) {
        self.current
            .store(value, std::sync::atomic::Ordering::Release);
    }
}
impl Clock for FakeClock {
    fn now_micros(&self) -> i64 {
        self.current.load(std::sync::atomic::Ordering::Acquire)
    }
}

struct Fixture {
    client: sendspin::ProtocolClient,
    server_tx: mpsc::UnboundedSender<WsMessage>,
    clock: Arc<FakeClock>,
    observed: mpsc::UnboundedReceiver<Message>,
}

async fn fixture() -> Fixture {
    let (client_io, server_io) = tokio::io::duplex(64 * 1024);
    let client_ws = WebSocketStream::from_raw_socket(client_io, Role::Client, None).await;
    let mut server_ws = WebSocketStream::from_raw_socket(server_io, Role::Server, None).await;
    let (server_tx, mut commands) = mpsc::unbounded_channel();
    let (observed_tx, mut observed) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let hello = server_ws
            .next()
            .await
            .expect("client hello")
            .expect("hello frame");
        let hello_text = match hello {
            WsMessage::Text(text) => text,
            other => panic!("expected client/hello, got {other:?}"),
        };
        assert!(matches!(
            serde_json::from_str::<Message>(&hello_text).unwrap(),
            Message::ClientHello(_)
        ));
        let hello = Message::ServerHello(ServerHello {
            server_id: "role-events".into(),
            name: "Role events".into(),
            version: 1,
            active_roles: vec![
                "player@v1".into(),
                "artwork@v1".into(),
                "visualizer@v1".into(),
            ],
            connection_reason: ConnectionReason::Playback,
        });
        server_ws
            .send(WsMessage::Text(
                serde_json::to_string(&hello).unwrap().into(),
            ))
            .await
            .unwrap();
        loop {
            tokio::select! {
                command = commands.recv() => match command { Some(frame) => { if server_ws.send(frame).await.is_err() { break; } }, None => break },
                frame = server_ws.next() => match frame { Some(Ok(WsMessage::Text(text))) => { if let Ok(message) = serde_json::from_str::<Message>(&text) { let _ = observed_tx.send(message); } }, Some(Ok(WsMessage::Close(_))) | None => break, Some(Ok(_)) => {}, Some(Err(_)) => break }
            }
        }
    });
    let clock = Arc::new(FakeClock::new());
    let clock_trait: Arc<dyn Clock> = clock.clone();
    let builder = ProtocolClientBuilder::builder()
        .client_id("role-events".into())
        .name("Role events".into())
        .build()
        .clock(clock_trait);
    let client = timeout(Duration::from_secs(5), builder.accept(client_ws))
        .await
        .expect("accept timeout")
        .expect("accept");
    let observed_message = timeout(Duration::from_secs(5), observed.recv())
        .await
        .expect("client/state timeout")
        .expect("client/state frame");
    assert!(matches!(observed_message, Message::ClientState(_)));
    Fixture {
        client,
        server_tx,
        clock,
        observed,
    }
}

fn player_start(codec: &str) -> Message {
    Message::StreamStart(StreamStart {
        player: Some(StreamPlayerConfig {
            codec: codec.into(),
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        }),
        artwork: None,
        visualizer: None,
    })
}
fn audio(timestamp: i64, data: &[u8]) -> WsMessage {
    let mut frame = vec![binary_types::PLAYER_AUDIO];
    frame.extend_from_slice(&timestamp.to_be_bytes());
    frame.extend_from_slice(data);
    WsMessage::Binary(frame.into())
}
fn event_buffer(timestamp: i64, codec: Codec) -> AudioBuffer {
    AudioBuffer {
        timestamp,
        samples: Arc::from([0i32, 0i32]),
        format: AudioFormat {
            codec,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        },
    }
}

async fn queued_codec_transition(first: Codec) {
    let f = fixture().await;
    let mut split = f.client.split();
    let output = RecordedOutput::default();
    let mut session = split.player.attach(
        output.clone(),
        AppliedControls {
            volume: 100,
            muted: false,
            static_delay_ms: 0,
        },
    );
    let pcm_data = [0, 1, 2, 3];
    let opus_data = [0xff, 0x41, 0x03, 0xff, 0xfe, 0, 0, 0];
    let (first_codec, second_codec) = match first {
        Codec::Pcm => ("pcm", "opus"),
        Codec::Opus => ("opus", "pcm"),
        _ => unreachable!(),
    };
    f.server_tx
        .send(
            serde_json::to_string(&player_start(first_codec))
                .unwrap()
                .into(),
        )
        .unwrap();
    f.server_tx
        .send(audio(
            10,
            if first == Codec::Pcm {
                &pcm_data
            } else {
                &opus_data
            },
        ))
        .unwrap();
    f.server_tx
        .send(
            serde_json::to_string(&player_start(second_codec))
                .unwrap()
                .into(),
        )
        .unwrap();
    f.server_tx
        .send(audio(
            20,
            if first == Codec::Pcm {
                &opus_data
            } else {
                &pcm_data
            },
        ))
        .unwrap();
    f.server_tx
        .send(
            serde_json::to_string(&Message::ServerCommand(ServerCommand { player: None }))
                .unwrap()
                .into(),
        )
        .unwrap();
    match timeout(Duration::from_secs(5), split.messages.recv())
        .await
        .unwrap()
    {
        Some(Message::ServerCommand(ServerCommand { player: None })) => {}
        other => panic!("expected queue marker, got {other:?}"),
    }

    let expected_codecs = if first == Codec::Pcm {
        [(10, Codec::Pcm), (20, Codec::Opus)]
    } else {
        [(10, Codec::Opus), (20, Codec::Pcm)]
    };
    let mut works = Vec::new();
    for (expected_timestamp, expected_codec) in expected_codecs {
        let config = match timeout(Duration::from_secs(5), session.recv())
            .await
            .unwrap()
            .unwrap()
        {
            PlayerEvent::Start(config) => config,
            other => panic!("expected start before timestamp {expected_timestamp}, got {other:?}"),
        };
        assert_eq!(
            config.codec,
            if expected_codec == Codec::Pcm {
                "pcm"
            } else {
                "opus"
            }
        );
        assert_eq!(config.sample_rate, 48_000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bit_depth, 16);
        let codec = match config.codec.as_str() {
            "pcm" => Codec::Pcm,
            "opus" => Codec::Opus,
            other => panic!("unexpected codec {other}"),
        };
        let format = AudioFormat {
            codec,
            sample_rate: config.sample_rate,
            channels: config.channels,
            bit_depth: config.bit_depth,
            codec_header: None,
        };
        let decoder: Box<dyn Decoder> = match codec {
            Codec::Pcm => Box::new(PcmDecoder::new(config.bit_depth)),
            Codec::Opus => Box::new(OpusDecoder::new(config.sample_rate, config.channels).unwrap()),
            _ => unreachable!(),
        };
        let work = match timeout(Duration::from_secs(5), session.recv())
            .await
            .unwrap()
            .unwrap()
        {
            PlayerEvent::Audio(work) => work,
            other => panic!("expected audio at timestamp {expected_timestamp}, got {other:?}"),
        };
        assert_eq!(work.timestamp(), expected_timestamp);
        let expected_data = match codec {
            Codec::Pcm => &pcm_data[..],
            Codec::Opus => &opus_data[..],
            _ => unreachable!(),
        };
        assert_eq!(work.data(), expected_data);
        let decoded_actual = decoder.decode(work.data()).unwrap();
        works.push((work, decoded_actual, format));
    }
    let pcm_reference = PcmDecoder::new(16).decode(&pcm_data).unwrap();
    let opus_reference = OpusDecoder::new(48_000, 2)
        .unwrap()
        .decode(&opus_data)
        .unwrap();
    for (work, decoded_actual, format) in works {
        let timestamp = work.timestamp();
        assert!(work
            .commit(AudioBuffer {
                timestamp,
                samples: decoded_actual,
                format,
            })
            .is_ok());
    }
    let records = output.commits.lock().unwrap().clone();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].timestamp, 10);
    assert_eq!(records[1].timestamp, 20);
    assert_eq!(
        records[0].format.codec,
        if first == Codec::Pcm {
            Codec::Pcm
        } else {
            Codec::Opus
        }
    );
    assert_eq!(
        records[1].format.codec,
        if first == Codec::Pcm {
            Codec::Opus
        } else {
            Codec::Pcm
        }
    );
    assert_eq!(
        records[0].samples.as_ref(),
        if first == Codec::Pcm {
            pcm_reference.as_ref()
        } else {
            opus_reference.as_ref()
        }
    );
    assert_eq!(
        records[1].samples.as_ref(),
        if first == Codec::Pcm {
            opus_reference.as_ref()
        } else {
            pcm_reference.as_ref()
        }
    );
    assert_ne!(opus_reference.as_ref(), pcm_reference.as_ref());
}

#[tokio::test]
async fn queued_pcm_to_opus_decodes_in_order() {
    queued_codec_transition(Codec::Pcm).await;
}

#[tokio::test]
async fn queued_opus_to_pcm_decodes_in_order() {
    queued_codec_transition(Codec::Opus).await;
}

#[tokio::test]
async fn artwork_and_visualizer_roles_stay_independent_without_player_stream() {
    let f = fixture().await;
    let split = f.client.split();
    let mut artwork = split.artwork.attach(Events::default());
    let mut visualizer = split.visualizer.attach(Events::default());
    let start = Message::StreamStart(StreamStart {
        player: None,
        artwork: Some(StreamArtworkConfig {
            channels: vec![StreamArtworkChannelConfig {
                source: ArtworkSource::Album,
                format: ImageFormat::Png,
                width: 2,
                height: 2,
            }],
        }),
        visualizer: Some(StreamVisualizerConfig {
            types: vec![VisualizerDataType::Loudness],
            rate_max: 10,
            tracks_downbeats: None,
            spectrum: None,
        }),
    });
    f.server_tx
        .send(serde_json::to_string(&start).unwrap().into())
        .unwrap();
    f.server_tx
        .send(WsMessage::Binary(
            vec![binary_types::ARTWORK_CHANNEL_0, 0, 0, 0, 0, 0, 0, 0, 1, 7].into(),
        ))
        .unwrap();
    f.server_tx
        .send(WsMessage::Binary(
            vec![binary_types::VISUALIZER_LOUDNESS, 0, 0, 0, 0, 0, 0, 0, 2, 9].into(),
        ))
        .unwrap();
    assert!(matches!(
        timeout(Duration::from_secs(5), artwork.recv())
            .await
            .unwrap(),
        Some(ArtworkEvent::Start(_))
    ));
    assert!(matches!(
        timeout(Duration::from_secs(5), visualizer.recv())
            .await
            .unwrap(),
        Some(VisualizerEvent::Start(_))
    ));
    match timeout(Duration::from_secs(5), artwork.recv())
        .await
        .unwrap()
        .unwrap()
    {
        ArtworkEvent::Data(work) => {
            assert_eq!(work.channel(), 0);
            assert_eq!(work.timestamp(), 1);
            assert!(!work.is_clear());
            assert_eq!(work.data(), &[7]);
            assert!(work.commit(vec![7]).is_ok());
        }
        _ => panic!("expected artwork data"),
    }
    match timeout(Duration::from_secs(5), visualizer.recv())
        .await
        .unwrap()
        .unwrap()
    {
        VisualizerEvent::Data(work) => {
            assert_eq!(work.type_id(), 16);
            assert_eq!(work.timestamp(), 2);
            assert_eq!(work.data(), &[9]);
            assert!(work.commit(vec![9]).is_ok());
        }
        _ => panic!("expected visualizer data"),
    }
}

#[tokio::test]
async fn urgent_clear_volume_end_and_malformed_input_preserve_new_media() {
    let f = fixture().await;
    let split = f.client.split();
    let (output, mut effects) = RecordedOutput::new();
    let mut session = split.player.attach(
        output.clone(),
        AppliedControls {
            volume: 100,
            muted: false,
            static_delay_ms: 0,
        },
    );
    f.server_tx
        .send(serde_json::to_string(&player_start("pcm")).unwrap().into())
        .unwrap();
    f.server_tx.send(audio(1, &[1, 2])).unwrap();
    match timeout(Duration::from_secs(5), session.recv())
        .await
        .unwrap()
    {
        Some(PlayerEvent::Start(config)) => assert_eq!(config.codec, "pcm"),
        other => panic!("expected initial start, got {other:?}"),
    }
    let held = match timeout(Duration::from_secs(5), session.recv())
        .await
        .unwrap()
    {
        Some(PlayerEvent::Audio(work)) => work,
        other => panic!("expected held pre-boundary audio, got {other:?}"),
    };
    f.server_tx
        .send(WsMessage::Binary(
            vec![binary_types::PLAYER_AUDIO, 1, 2].into(),
        ))
        .unwrap();
    f.server_tx
        .send(
            serde_json::to_string(&Message::StreamClear(StreamClear { roles: None }))
                .unwrap()
                .into(),
        )
        .unwrap();
    f.server_tx
        .send(
            serde_json::to_string(&Message::ServerCommand(ServerCommand {
                player: Some(PlayerCommand {
                    command: PlayerCommandType::Volume,
                    volume: Some(42),
                    mute: None,
                    static_delay_ms: None,
                }),
            }))
            .unwrap()
            .into(),
        )
        .unwrap();
    f.server_tx
        .send(
            serde_json::to_string(&Message::StreamEnd(StreamEnd {
                roles: Some(vec!["player".into()]),
            }))
            .unwrap()
            .into(),
        )
        .unwrap();
    f.server_tx
        .send(serde_json::to_string(&player_start("pcm")).unwrap().into())
        .unwrap();
    f.server_tx.send(audio(2, &[3, 4])).unwrap();
    timeout(Duration::from_secs(5), async {
        loop {
            effects.changed().await.expect("effects watch closed");
            let state = effects.borrow().clone();
            if state.clear_count >= 1 && state.volume == 42 {
                break;
            }
        }
    })
    .await
    .expect("effects did not reach clear+volume state");
    let mut events = Vec::new();
    let mut new_work = None;
    for _ in 0..4 {
        match timeout(Duration::from_secs(5), session.recv())
            .await
            .unwrap()
        {
            Some(PlayerEvent::Clear) => events.push("clear"),
            Some(PlayerEvent::End) => events.push("end"),
            Some(PlayerEvent::Start(config)) => {
                assert_eq!(config.codec, "pcm");
                events.push("start");
            }
            Some(PlayerEvent::Audio(work)) => {
                assert_eq!(work.timestamp(), 2);
                assert_eq!(work.data(), &[3, 4]);
                events.push("audio");
                new_work = Some(work);
            }
            None => panic!("session closed after malformed/boundary input"),
        }
    }
    assert_eq!(events, ["clear", "end", "start", "audio"]);
    assert_eq!(
        held.commit(event_buffer(1, Codec::Pcm)),
        Err(sendspin::audio::CommitRejected::Obsolete)
    );
    let new_work = new_work.expect("new audio");
    let samples = PcmDecoder::new(16).decode(new_work.data()).unwrap();
    assert_eq!(
        new_work.commit(AudioBuffer {
            timestamp: 2,
            samples,
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None
            },
        }),
        Ok(())
    );
    assert_eq!(output.volumes.lock().unwrap().as_slice(), &[100, 42]);
    assert_eq!(*output.clears.lock().unwrap(), 1);
    assert_eq!(output.commits.lock().unwrap().len(), 1);
    assert_eq!(output.commits.lock().unwrap()[0].timestamp, 2);
}

#[tokio::test]
async fn preparation_worker_blocked_before_decode_does_not_block_urgent_controls() {
    let f = fixture().await;
    let split = f.client.split();
    let (output, mut effects) = RecordedOutput::new();
    let mut session = split.player.attach(
        output.clone(),
        AppliedControls {
            volume: 100,
            muted: false,
            static_delay_ms: 0,
        },
    );
    let mut controls = session.controls();
    f.server_tx
        .send(serde_json::to_string(&player_start("pcm")).unwrap().into())
        .unwrap();
    f.server_tx.send(audio(11, &[1, 2])).unwrap();
    assert!(matches!(
        timeout(Duration::from_secs(5), session.recv())
            .await
            .unwrap(),
        Some(PlayerEvent::Start(_))
    ));
    let work = match timeout(Duration::from_secs(5), session.recv())
        .await
        .unwrap()
    {
        Some(PlayerEvent::Audio(work)) => work,
        other => panic!("expected audio work, got {other:?}"),
    };

    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        entered_tx.send(()).unwrap();
        release_rx.recv().unwrap();
        let samples = PcmDecoder::new(16).decode(work.data()).unwrap();
        work.commit(AudioBuffer {
            timestamp: 11,
            samples,
            format: AudioFormat {
                codec: Codec::Pcm,
                sample_rate: 48_000,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            },
        })
    });
    entered_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("preparation worker did not enter");

    f.server_tx
        .send(
            serde_json::to_string(&Message::StreamClear(StreamClear { roles: None }))
                .unwrap()
                .into(),
        )
        .unwrap();
    f.server_tx
        .send(
            serde_json::to_string(&Message::ServerCommand(ServerCommand {
                player: Some(PlayerCommand {
                    command: PlayerCommandType::Volume,
                    volume: Some(42),
                    mute: None,
                    static_delay_ms: None,
                }),
            }))
            .unwrap()
            .into(),
        )
        .unwrap();
    timeout(Duration::from_secs(5), async {
        loop {
            if effects.borrow().clear_count >= 1 && effects.borrow().volume == 42 {
                break;
            }
            effects.changed().await.unwrap();
        }
    })
    .await
    .expect("urgent controls did not apply while preparation was blocked");
    timeout(Duration::from_secs(5), async {
        loop {
            if controls.borrow().volume == 42 {
                break;
            }
            controls.changed().await.unwrap();
        }
    })
    .await
    .expect("controls watch did not publish volume 42");

    release_tx.send(()).unwrap();
    assert_eq!(
        worker.join().unwrap(),
        Err(sendspin::audio::CommitRejected::Obsolete)
    );
    assert!(output.commits.lock().unwrap().is_empty());
}

#[tokio::test]
async fn clock_reply_uses_observed_client_time_and_graceful_peer_close_rejects_held_work() {
    let mut f = fixture().await;
    let t1 = match timeout(Duration::from_secs(5), f.observed.recv())
        .await
        .unwrap()
    {
        Some(Message::ClientTime(time)) => time.client_transmitted,
        other => panic!("expected first observed ClientTime before reply, got {other:?}"),
    };
    let mut split = f.client.split();
    assert_eq!(split.clock_sync.lock().rtt_micros(), None);
    f.clock.set(t1 + 1_000);
    let mut session = split.player.attach(
        Events::default(),
        AppliedControls {
            volume: 100,
            muted: false,
            static_delay_ms: 0,
        },
    );
    f.server_tx
        .send(serde_json::to_string(&player_start("pcm")).unwrap().into())
        .unwrap();
    f.server_tx.send(audio(3, &[1, 2])).unwrap();
    assert!(matches!(
        timeout(Duration::from_secs(5), session.recv())
            .await
            .unwrap(),
        Some(PlayerEvent::Start(_))
    ));
    let work = match timeout(Duration::from_secs(5), session.recv())
        .await
        .unwrap()
    {
        Some(PlayerEvent::Audio(work)) => work,
        Some(other) => panic!("expected audio before close, got {other:?}"),
        None => panic!("session terminated before audio"),
    };
    f.server_tx
        .send(
            serde_json::to_string(&Message::ServerTime(
                sendspin::protocol::messages::ServerTime {
                    client_transmitted: t1,
                    server_received: t1,
                    server_transmitted: t1,
                },
            ))
            .unwrap()
            .into(),
        )
        .unwrap();
    f.server_tx
        .send(
            serde_json::to_string(&Message::ServerCommand(ServerCommand { player: None }))
                .unwrap()
                .into(),
        )
        .unwrap();
    match timeout(Duration::from_secs(5), split.messages.recv())
        .await
        .unwrap()
    {
        Some(Message::ServerCommand(_)) => {}
        other => panic!("expected router-processed general marker, got {other:?}"),
    }
    f.server_tx.send(WsMessage::Close(None)).unwrap();
    assert!(timeout(Duration::from_secs(5), session.recv())
        .await
        .unwrap()
        .is_none());
    assert!(matches!(
        work.commit(event_buffer(3, Codec::Pcm)),
        Err(sendspin::audio::CommitRejected::Closed)
    ));
    timeout(Duration::from_secs(5), session.shutdown())
        .await
        .expect("session shutdown timed out");
    assert_eq!(split.clock_sync.lock().rtt_micros(), Some(1_000));
}

#[tokio::test]
async fn dropped_connection_guard_closes_session_and_rejects_held_work() {
    let f = fixture().await;
    let split = f.client.split();

    let mut session = split.player.attach(
        Events::default(),
        AppliedControls {
            volume: 100,
            muted: false,
            static_delay_ms: 0,
        },
    );
    let mut watch = session.status();
    f.server_tx
        .send(serde_json::to_string(&player_start("pcm")).unwrap().into())
        .unwrap();
    f.server_tx.send(audio(4, &[1, 2])).unwrap();
    assert!(matches!(
        timeout(Duration::from_secs(5), session.recv())
            .await
            .unwrap(),
        Some(PlayerEvent::Start(_))
    ));
    let work = match timeout(Duration::from_secs(5), session.recv())
        .await
        .unwrap()
    {
        Some(PlayerEvent::Audio(work)) => work,
        Some(other) => panic!("expected audio before guard drop, got {other:?}"),
        None => panic!("session terminated before audio"),
    };
    drop(split.guard);
    assert!(timeout(Duration::from_secs(5), watch.changed())
        .await
        .unwrap()
        .is_ok());
    let status = watch.borrow().clone();
    assert_eq!(status, SessionStatus::Closed);
    assert!(matches!(
        work.commit(event_buffer(4, Codec::Pcm)),
        Err(sendspin::audio::CommitRejected::Closed)
    ));
    assert!(timeout(Duration::from_secs(5), session.recv())
        .await
        .unwrap()
        .is_none());
}
