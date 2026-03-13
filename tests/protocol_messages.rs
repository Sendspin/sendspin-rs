use sendspin::protocol::messages::{
    ArtworkChannel, ArtworkFormatRequest, ArtworkSource, ArtworkV1Support, AudioFormatSpec,
    ClientCommand, ClientGoodbye, ClientHello, ClientState, ClientSyncState, ClientTime,
    ConnectionReason, ControllerCommand, DeviceInfo, GoodbyeReason, ImageFormat, Message,
    PlaybackState, PlayerState, PlayerV1Support, RepeatMode, ServerTime,
    StreamArtworkChannelConfig,
};

// =============================================================================
// Handshake Tests
// =============================================================================

#[test]
fn test_client_hello_serialization() {
    let hello = ClientHello {
        client_id: "test-client-123".to_string(),
        name: "Test Player".to_string(),
        version: 1,
        supported_roles: vec!["player@v1".to_string()],
        device_info: Some(DeviceInfo {
            product_name: Some("Sendspin-RS Player".to_string()),
            manufacturer: Some("Sendspin".to_string()),
            software_version: Some("0.1.0".to_string()),
        }),
        player_v1_support: Some(PlayerV1Support {
            supported_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 50 * 1024 * 1024, // 50 MB
            supported_commands: vec!["volume".to_string(), "mute".to_string()],
        }),
        artwork_v1_support: None,
        visualizer_v1_support: None,
    };

    let message = Message::ClientHello(hello);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"type\":\"client/hello\""));
    assert!(json.contains("\"client_id\":\"test-client-123\""));
    assert!(json.contains("\"player@v1_support\""));
    assert!(json.contains("\"player@v1\""));
}

#[test]
fn test_server_hello_deserialization() {
    let json = r#"{
        "type": "server/hello",
        "payload": {
            "server_id": "server-456",
            "name": "Test Server",
            "version": 1,
            "active_roles": ["player@v1"],
            "connection_reason": "playback"
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::ServerHello(hello) => {
            assert_eq!(hello.server_id, "server-456");
            assert_eq!(hello.name, "Test Server");
            assert_eq!(hello.version, 1);
            assert_eq!(hello.active_roles, vec!["player@v1"]);
            assert_eq!(hello.connection_reason, ConnectionReason::Playback);
        }
        _ => panic!("Expected ServerHello"),
    }
}

// =============================================================================
// State Tests
// =============================================================================

#[test]
fn test_client_state_serialization() {
    let state = ClientState {
        state: Some(ClientSyncState::Synchronized),
        player: Some(PlayerState {
            volume: Some(100),
            muted: Some(false),
            static_delay_ms: Some(0),
            supported_commands: None,
        }),
    };

    let message = Message::ClientState(state);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"type\":\"client/state\""));
    assert!(json.contains("\"state\":\"synchronized\""));
    assert!(json.contains("\"volume\":100"));
    assert!(json.contains("\"static_delay_ms\":0"));
}

#[test]
fn test_client_sync_state_error() {
    let state = ClientState {
        state: Some(ClientSyncState::Error),
        player: None,
    };

    let message = Message::ClientState(state);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"state\":\"error\""));
}

#[test]
fn test_client_sync_state_external_source() {
    let state = ClientState {
        state: Some(ClientSyncState::ExternalSource),
        player: None,
    };

    let message = Message::ClientState(state);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"state\":\"external_source\""));
}

#[test]
fn test_server_state_metadata_deserialization() {
    let json = r#"{
        "type": "server/state",
        "payload": {
            "metadata": {
                "timestamp": 1234567890,
                "title": "Test Song",
                "artist": "Test Artist",
                "album": "Test Album",
                "year": 2024,
                "track": 3,
                "progress": {
                    "track_progress": 60000,
                    "track_duration": 180000,
                    "playback_speed": 1000
                },
                "repeat": "off",
                "shuffle": false
            }
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::ServerState(state) => {
            let metadata = state.metadata.expect("Expected metadata");
            assert_eq!(metadata.timestamp, 1234567890);
            assert_eq!(metadata.title, Some("Test Song".to_string()));
            assert_eq!(metadata.artist, Some("Test Artist".to_string()));
            assert_eq!(metadata.album, Some("Test Album".to_string()));
            assert_eq!(metadata.year, Some(2024));
            assert_eq!(metadata.track, Some(3));

            let progress = metadata.progress.expect("Expected progress");
            assert_eq!(progress.track_progress, 60000);
            assert_eq!(progress.track_duration, 180000);
            assert_eq!(progress.playback_speed, 1000);

            assert_eq!(metadata.repeat, Some(RepeatMode::Off));
            assert_eq!(metadata.shuffle, Some(false));
        }
        _ => panic!("Expected ServerState"),
    }
}

#[test]
fn test_server_state_controller_deserialization() {
    let json = r#"{
        "type": "server/state",
        "payload": {
            "controller": {
                "supported_commands": ["play", "next", "previous", "volume", "mute"],
                "volume": 75,
                "muted": false
            }
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::ServerState(state) => {
            let controller = state.controller.expect("Expected controller");
            assert_eq!(controller.volume, 75);
            assert!(!controller.muted);
            assert!(controller.supported_commands.contains(&"play".to_string()));
            assert!(controller
                .supported_commands
                .contains(&"volume".to_string()));
        }
        _ => panic!("Expected ServerState"),
    }
}

// =============================================================================
// Command Tests
// =============================================================================

#[test]
fn test_client_command_serialization() {
    let command = ClientCommand {
        controller: Some(ControllerCommand {
            command: "play".to_string(),
            volume: None,
            mute: None,
        }),
    };

    let message = Message::ClientCommand(command);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"type\":\"client/command\""));
    assert!(json.contains("\"command\":\"play\""));
}

#[test]
fn test_server_command_deserialization() {
    let json = r#"{
        "type": "server/command",
        "payload": {
            "player": {
                "command": "play",
                "volume": 80
            }
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::ServerCommand(cmd) => {
            let player = cmd.player.expect("Expected player command");
            assert_eq!(player.command, "play");
            assert_eq!(player.volume, Some(80));
            assert!(player.mute.is_none());
            assert!(player.static_delay_ms.is_none());
        }
        _ => panic!("Expected ServerCommand"),
    }
}

#[test]
fn test_server_command_set_static_delay() {
    let json = r#"{
        "type": "server/command",
        "payload": {
            "player": {
                "command": "set_static_delay",
                "static_delay_ms": 250
            }
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::ServerCommand(cmd) => {
            let player = cmd.player.expect("Expected player command");
            assert_eq!(player.command, "set_static_delay");
            assert_eq!(player.static_delay_ms, Some(250));
            assert!(player.volume.is_none());
        }
        _ => panic!("Expected ServerCommand"),
    }
}

#[test]
fn test_client_command_volume() {
    let command = ClientCommand {
        controller: Some(ControllerCommand {
            command: "volume".to_string(),
            volume: Some(50),
            mute: None,
        }),
    };

    let message = Message::ClientCommand(command);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"volume\":50"));
}

// =============================================================================
// Stream Control Tests
// =============================================================================

#[test]
fn test_stream_start_deserialization() {
    let json = r#"{
        "type": "stream/start",
        "payload": {
            "player": {
                "codec": "pcm",
                "sample_rate": 48000,
                "channels": 2,
                "bit_depth": 24
            }
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::StreamStart(start) => {
            let player = start.player.expect("Expected player config");
            assert_eq!(player.codec, "pcm");
            assert_eq!(player.sample_rate, 48000);
            assert_eq!(player.channels, 2);
            assert_eq!(player.bit_depth, 24);
            assert!(start.artwork.is_none());
            assert!(start.visualizer.is_none());
        }
        _ => panic!("Expected StreamStart"),
    }
}

#[test]
fn test_stream_end_deserialization() {
    let json = r#"{
        "type": "stream/end",
        "payload": {
            "roles": ["player@v1"]
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::StreamEnd(end) => {
            assert_eq!(end.roles, Some(vec!["player@v1".to_string()]));
        }
        _ => panic!("Expected StreamEnd"),
    }
}

#[test]
fn test_stream_clear_deserialization() {
    let json = r#"{
        "type": "stream/clear",
        "payload": {}
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::StreamClear(clear) => {
            assert!(clear.roles.is_none());
        }
        _ => panic!("Expected StreamClear"),
    }
}

// =============================================================================
// Group Tests
// =============================================================================

#[test]
fn test_group_update_deserialization() {
    let json = r#"{
        "type": "group/update",
        "payload": {
            "playback_state": "playing",
            "group_id": "living-room",
            "group_name": "Living Room"
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::GroupUpdate(update) => {
            assert_eq!(update.playback_state, Some(PlaybackState::Playing));
            assert_eq!(update.group_id, Some("living-room".to_string()));
            assert_eq!(update.group_name, Some("Living Room".to_string()));
        }
        _ => panic!("Expected GroupUpdate"),
    }
}

// Test all playback state variants (per spec: only 'playing' and 'stopped')
#[test]
fn test_playback_state_variants() {
    let states = [
        (r#""playing""#, PlaybackState::Playing),
        (r#""stopped""#, PlaybackState::Stopped),
    ];

    for (json_val, expected) in states {
        let parsed: PlaybackState = serde_json::from_str(json_val).unwrap();
        assert_eq!(parsed, expected);
    }
}

// =============================================================================
// Goodbye Tests
// =============================================================================

#[test]
fn test_client_goodbye_serialization() {
    let goodbye = ClientGoodbye {
        reason: GoodbyeReason::AnotherServer,
    };

    let message = Message::ClientGoodbye(goodbye);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"type\":\"client/goodbye\""));
    assert!(json.contains("\"reason\":\"another_server\""));
}

#[test]
fn test_goodbye_reason_variants() {
    let reasons = [
        (r#""another_server""#, GoodbyeReason::AnotherServer),
        (r#""shutdown""#, GoodbyeReason::Shutdown),
        (r#""restart""#, GoodbyeReason::Restart),
        (r#""user_request""#, GoodbyeReason::UserRequest),
    ];

    for (json_val, expected) in reasons {
        let parsed: GoodbyeReason = serde_json::from_str(json_val).unwrap();
        assert_eq!(parsed, expected);
    }
}

// =============================================================================
// Repeat Mode Tests
// =============================================================================

#[test]
fn test_repeat_mode_variants() {
    let modes = [
        (r#""off""#, RepeatMode::Off),
        (r#""one""#, RepeatMode::One),
        (r#""all""#, RepeatMode::All),
    ];

    for (json_val, expected) in modes {
        let parsed: RepeatMode = serde_json::from_str(json_val).unwrap();
        assert_eq!(parsed, expected);
    }
}

// =============================================================================
// Artwork Tests
// =============================================================================

#[test]
fn test_artwork_v1_support_serialization() {
    let support = ArtworkV1Support {
        channels: vec![
            ArtworkChannel {
                source: ArtworkSource::Album,
                format: ImageFormat::Jpeg,
                media_width: 800,
                media_height: 800,
            },
            ArtworkChannel {
                source: ArtworkSource::Artist,
                format: ImageFormat::Png,
                media_width: 400,
                media_height: 400,
            },
        ],
    };

    let json = serde_json::to_string(&support).unwrap();

    assert!(json.contains("\"source\":\"album\""));
    assert!(json.contains("\"format\":\"jpeg\""));
    assert!(json.contains("\"media_width\":800"));
    assert!(json.contains("\"source\":\"artist\""));
    assert!(json.contains("\"format\":\"png\""));
}

#[test]
fn test_artwork_source_variants() {
    let sources = [
        (r#""album""#, ArtworkSource::Album),
        (r#""artist""#, ArtworkSource::Artist),
        (r#""none""#, ArtworkSource::None),
    ];

    for (json_val, expected) in sources {
        let parsed: ArtworkSource = serde_json::from_str(json_val).unwrap();
        assert_eq!(parsed, expected);
    }
}

#[test]
fn test_image_format_variants() {
    let formats = [
        (r#""jpeg""#, ImageFormat::Jpeg),
        (r#""png""#, ImageFormat::Png),
        (r#""bmp""#, ImageFormat::Bmp),
    ];

    for (json_val, expected) in formats {
        let parsed: ImageFormat = serde_json::from_str(json_val).unwrap();
        assert_eq!(parsed, expected);
    }
}

// =============================================================================
// Time Sync Tests
// =============================================================================

#[test]
fn test_client_time_serialization() {
    let msg = Message::ClientTime(ClientTime {
        client_transmitted: 1_700_000_000_000_000,
    });
    let json = serde_json::to_string(&msg).unwrap();

    assert!(json.contains("\"type\":\"client/time\""));
    assert!(json.contains("\"client_transmitted\":1700000000000000"));

    // Round-trip
    let parsed: Message = serde_json::from_str(&json).unwrap();
    match parsed {
        Message::ClientTime(ct) => {
            assert_eq!(ct.client_transmitted, 1_700_000_000_000_000);
        }
        _ => panic!("Expected ClientTime"),
    }
}

#[test]
fn test_server_time_deserialization() {
    let json = r#"{
        "type": "server/time",
        "payload": {
            "client_transmitted": 1000000,
            "server_received": 1005100,
            "server_transmitted": 1005110
        }
    }"#;

    let msg: Message = serde_json::from_str(json).unwrap();
    match msg {
        Message::ServerTime(st) => {
            assert_eq!(st.client_transmitted, 1_000_000);
            assert_eq!(st.server_received, 1_005_100);
            assert_eq!(st.server_transmitted, 1_005_110);
        }
        _ => panic!("Expected ServerTime"),
    }
}

#[test]
fn test_server_time_fields_feed_clock_sync() {
    use sendspin::sync::ClockSync;

    // Simulate two sync rounds using the same field mapping
    // that message_router uses: st.client_transmitted = t1,
    // st.server_received = t2, st.server_transmitted = t3.
    let mut sync = ClockSync::new();
    assert!(!sync.is_synchronized());

    let st1 = ServerTime {
        client_transmitted: 1_000_000,
        server_received: 1_005_100,
        server_transmitted: 1_005_100,
    };
    let t4_1: i64 = 1_000_200;
    sync.update(
        st1.client_transmitted,
        st1.server_received,
        st1.server_transmitted,
        t4_1,
    );

    let st2 = ServerTime {
        client_transmitted: 2_000_000,
        server_received: 2_005_100,
        server_transmitted: 2_005_100,
    };
    let t4_2: i64 = 2_000_200;
    sync.update(
        st2.client_transmitted,
        st2.server_received,
        st2.server_transmitted,
        t4_2,
    );

    // After two rounds, the filter should be synchronized
    assert!(sync.is_synchronized());

    // Verify the offset is reasonable: server time ≈ client time + 5000µs
    let client = sync.server_to_client_micros(3_005_100);
    assert!(client.is_some());
    let diff = (client.unwrap() - 3_000_100).abs();
    assert!(diff < 50, "offset drift too large: {}", diff);
}

// =============================================================================
// Stream Artwork Config Tests
// =============================================================================

#[test]
fn test_stream_start_artwork_deserialization() {
    let json = r#"{
        "type": "stream/start",
        "payload": {
            "artwork": {
                "channels": [
                    {
                        "source": "album",
                        "format": "jpeg",
                        "width": 800,
                        "height": 800
                    }
                ]
            }
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::StreamStart(start) => {
            assert!(start.player.is_none());
            let artwork = start.artwork.expect("Expected artwork config");
            assert_eq!(artwork.channels.len(), 1);
            let ch = &artwork.channels[0];
            assert_eq!(ch.source, ArtworkSource::Album);
            assert_eq!(ch.format, ImageFormat::Jpeg);
            assert_eq!(ch.width, 800);
            assert_eq!(ch.height, 800);
        }
        _ => panic!("Expected StreamStart"),
    }
}

#[test]
fn test_stream_start_artwork_multi_channel() {
    let json = r#"{
        "type": "stream/start",
        "payload": {
            "artwork": {
                "channels": [
                    { "source": "album", "format": "jpeg", "width": 800, "height": 800 },
                    { "source": "artist", "format": "png", "width": 400, "height": 400 }
                ]
            }
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::StreamStart(start) => {
            let artwork = start.artwork.expect("Expected artwork config");
            assert_eq!(artwork.channels.len(), 2);
            assert_eq!(artwork.channels[0].source, ArtworkSource::Album);
            assert_eq!(artwork.channels[1].source, ArtworkSource::Artist);
            assert_eq!(artwork.channels[1].format, ImageFormat::Png);
        }
        _ => panic!("Expected StreamStart"),
    }
}

#[test]
fn test_stream_artwork_channel_config_roundtrip() {
    let config = StreamArtworkChannelConfig {
        source: ArtworkSource::Album,
        format: ImageFormat::Jpeg,
        width: 256,
        height: 256,
    };

    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains("\"source\":\"album\""));
    assert!(json.contains("\"format\":\"jpeg\""));
    assert!(json.contains("\"width\":256"));

    let parsed: StreamArtworkChannelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.source, ArtworkSource::Album);
    assert_eq!(parsed.width, 256);
}

// =============================================================================
// Deserialization Error Path Tests
// =============================================================================

#[test]
fn test_malformed_json_rejected() {
    let result = serde_json::from_str::<Message>("not valid json at all");
    assert!(result.is_err());
}

#[test]
fn test_empty_json_object_rejected() {
    let result = serde_json::from_str::<Message>("{}");
    assert!(result.is_err());
}

#[test]
fn test_unknown_message_type_rejected() {
    let json = r#"{"type": "invalid/type", "payload": {}}"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_err());
}

#[test]
fn test_missing_payload_rejected() {
    let json = r#"{"type": "client/hello"}"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_err());
}

#[test]
fn test_client_hello_missing_required_fields() {
    // Missing client_id, name, version, supported_roles
    let json = r#"{"type": "client/hello", "payload": {}}"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_err());
}

#[test]
fn test_server_hello_missing_required_fields() {
    let json = r#"{"type": "server/hello", "payload": {"server_id": "s1"}}"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_err());
}

#[test]
fn test_invalid_connection_reason_rejected() {
    let json = r#"{
        "type": "server/hello",
        "payload": {
            "server_id": "s1",
            "name": "Test",
            "version": 1,
            "active_roles": [],
            "connection_reason": "invalid_reason"
        }
    }"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_err());
}

#[test]
fn test_invalid_playback_state_rejected() {
    let result = serde_json::from_str::<PlaybackState>(r#""paused""#);
    assert!(
        result.is_err(),
        "only 'playing' and 'stopped' are valid per spec"
    );
}

#[test]
fn test_invalid_repeat_mode_rejected() {
    let result = serde_json::from_str::<RepeatMode>(r#""shuffle""#);
    assert!(result.is_err());
}

#[test]
fn test_invalid_goodbye_reason_rejected() {
    let result = serde_json::from_str::<GoodbyeReason>(r#""timeout""#);
    assert!(result.is_err());
}

#[test]
fn test_wrong_type_in_payload_field_rejected() {
    // version should be u32, not string
    let json = r#"{
        "type": "client/hello",
        "payload": {
            "client_id": "c1",
            "name": "Test",
            "version": "not_a_number",
            "supported_roles": []
        }
    }"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_err());
}

#[test]
fn test_null_payload_rejected() {
    let json = r#"{"type": "client/hello", "payload": null}"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_err());
}

#[test]
fn test_client_state_empty_payload_accepted() {
    // ClientState with player: None should be valid
    let json = r#"{"type": "client/state", "payload": {}}"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_ok());
    match result.unwrap() {
        Message::ClientState(state) => assert!(state.player.is_none()),
        _ => panic!("Expected ClientState"),
    }
}

#[test]
fn test_server_state_empty_payload_accepted() {
    // ServerState with no metadata or controller should be valid
    let json = r#"{"type": "server/state", "payload": {}}"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_ok());
    match result.unwrap() {
        Message::ServerState(state) => {
            assert!(state.metadata.is_none());
            assert!(state.controller.is_none());
        }
        _ => panic!("Expected ServerState"),
    }
}

#[test]
fn test_stream_end_empty_roles_accepted() {
    let json = r#"{"type": "stream/end", "payload": {}}"#;
    let result = serde_json::from_str::<Message>(json);
    assert!(result.is_ok());
    match result.unwrap() {
        Message::StreamEnd(end) => assert!(end.roles.is_none()),
        _ => panic!("Expected StreamEnd"),
    }
}

// =============================================================================
// Deprecation & Wire Format Tests
// =============================================================================

/// Old JSON that includes `state` inside the player object (from pre-0.2.0
/// clients/servers) should still deserialize without error. The unknown field
/// is ignored since PlayerState no longer has a `state` field.
#[test]
fn test_player_state_ignores_legacy_state_field() {
    let json = r#"{
        "type": "client/state",
        "payload": {
            "player": {
                "state": "error",
                "volume": 50,
                "muted": true
            }
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::ClientState(cs) => {
            assert!(cs.state.is_none());

            let player = cs.player.expect("Expected player");
            assert_eq!(player.volume, Some(50));
            assert_eq!(player.muted, Some(true));
            assert!(player.static_delay_ms.is_none());
            assert!(player.supported_commands.is_none());
        }
        _ => panic!("Expected ClientState"),
    }
}

/// supported_commands roundtrip: serialize with commands, deserialize, verify.
#[test]
fn test_player_state_supported_commands_roundtrip() {
    let state = ClientState {
        state: Some(ClientSyncState::Synchronized),
        player: Some(PlayerState {
            volume: Some(100),
            muted: Some(false),
            static_delay_ms: Some(0),
            supported_commands: Some(vec![
                "set_static_delay".to_string(),
                "volume".to_string(),
            ]),
        }),
    };

    let message = Message::ClientState(state);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"supported_commands\":[\"set_static_delay\",\"volume\"]"));

    // Roundtrip
    let parsed: Message = serde_json::from_str(&json).unwrap();
    match parsed {
        Message::ClientState(cs) => {
            let player = cs.player.expect("Expected player");
            let cmds = player.supported_commands.expect("Expected supported_commands");
            assert_eq!(cmds, vec!["set_static_delay", "volume"]);
        }
        _ => panic!("Expected ClientState"),
    }
}

/// ArtworkFormatRequest uses typed enums for source and format,
/// verify they serialize to the correct wire strings.
#[test]
fn test_artwork_format_request_typed_enums() {
    let request = ArtworkFormatRequest {
        channel: 0,
        source: Some(ArtworkSource::Album),
        format: Some(ImageFormat::Jpeg),
        media_width: Some(800),
        media_height: Some(800),
    };

    let json = serde_json::to_string(&request).unwrap();

    // Enums should serialize as lowercase strings, not struct representations
    assert!(json.contains("\"source\":\"album\""), "source not serialized as string: {}", json);
    assert!(json.contains("\"format\":\"jpeg\""), "format not serialized as string: {}", json);

    // Roundtrip
    let parsed: ArtworkFormatRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.source, Some(ArtworkSource::Album));
    assert_eq!(parsed.format, Some(ImageFormat::Jpeg));
    assert_eq!(parsed.channel, 0);
}

/// ArtworkFormatRequest should deserialize from wire JSON with string values
/// (this is what a real server would send).
#[test]
fn test_artwork_format_request_from_wire_json() {
    let json = r#"{
        "channel": 1,
        "source": "artist",
        "format": "png",
        "media_width": 400,
        "media_height": 400
    }"#;

    let parsed: ArtworkFormatRequest = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.channel, 1);
    assert_eq!(parsed.source, Some(ArtworkSource::Artist));
    assert_eq!(parsed.format, Some(ImageFormat::Png));
    assert_eq!(parsed.media_width, Some(400));
    assert_eq!(parsed.media_height, Some(400));
}
