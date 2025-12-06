use sendspin::protocol::messages::{
    AudioFormatSpec, ClientHello, DeviceInfo, Message, PlayerSupport, ServerHello,
};
use serde_json;

#[test]
fn test_client_hello_serialization() {
    let hello = ClientHello {
        client_id: "test-client-123".to_string(),
        name: "Test Player".to_string(),
        version: 1,
        supported_roles: vec!["player".to_string()],
        device_info: DeviceInfo {
            product_name: "Sendspin-RS Player".to_string(),
            manufacturer: "Sendspin".to_string(),
            software_version: "0.1.0".to_string(),
        },
        player_support: Some(PlayerSupport {
            support_codecs: vec!["pcm".to_string()],
            support_channels: vec![2],
            support_sample_rates: vec![48000],
            support_bit_depth: vec![16, 24],
            support_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 100,
            supported_commands: vec!["play".to_string(), "pause".to_string()],
        }),
        metadata_support: None,
    };

    let message = Message::ClientHello(hello);
    let json = serde_json::to_string(&message).unwrap();

    assert!(json.contains("\"type\":\"client/hello\""));
    assert!(json.contains("\"client_id\":\"test-client-123\""));
}

#[test]
fn test_server_hello_deserialization() {
    let json = r#"{
        "type": "server/hello",
        "payload": {
            "server_id": "server-456",
            "name": "Test Server",
            "version": 1
        }
    }"#;

    let message: Message = serde_json::from_str(json).unwrap();

    match message {
        Message::ServerHello(hello) => {
            assert_eq!(hello.server_id, "server-456");
            assert_eq!(hello.name, "Test Server");
            assert_eq!(hello.version, 1);
        }
        _ => panic!("Expected ServerHello"),
    }
}
