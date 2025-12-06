use sendspin::protocol::messages::{
    AudioFormatSpec, ClientCommand, ClientGoodbye, ClientHello, ClientState, ConnectionReason,
    ControllerCommand, ControllerState, DeviceInfo, GoodbyeReason, GroupUpdate, Message,
    MetadataState, PlaybackState, PlayerState, PlayerSyncState, PlayerV1Support, RepeatMode,
    ServerState, StreamClear, StreamEnd, TrackProgress,
};
use serde_json;

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
        player: Some(PlayerState {
            state: PlayerSyncState::Synchronized,
            volume: Some(100),
            muted: Some(false),
        }),
    };

    let message = Message::ClientState(state);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"type\":\"client/state\""));
    assert!(json.contains("\"state\":\"synchronized\""));
    assert!(json.contains("\"volume\":100"));
}

#[test]
fn test_player_sync_state_error() {
    let state = ClientState {
        player: Some(PlayerState {
            state: PlayerSyncState::Error,
            volume: None,
            muted: None,
        }),
    };

    let message = Message::ClientState(state);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"state\":\"error\""));
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
                "progress": {
                    "position": 60000000,
                    "duration": 180000000,
                    "playback_speed": 1.0
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

            let progress = metadata.progress.expect("Expected progress");
            assert_eq!(progress.position, 60000000);
            assert_eq!(progress.duration, 180000000);
            assert_eq!(progress.playback_speed, Some(1.0));

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
                "supported_commands": ["play", "pause", "next", "previous", "volume", "mute"],
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
            assert!(controller.supported_commands.contains(&"volume".to_string()));
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

#[test]
fn test_playback_state_variants() {
    // Test all playback state variants (per spec: only 'playing' and 'stopped')
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
