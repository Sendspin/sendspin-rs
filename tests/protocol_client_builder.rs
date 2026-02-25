use sendspin::protocol::client::ProtocolClientBuilder;
use sendspin::protocol::messages::{
    ArtworkChannel, ArtworkSource, ArtworkV1Support, AudioFormatSpec, ImageFormat, PlayerV1Support,
    VisualizerV1Support,
};

#[test]
fn test_supported_roles_with_player_v1_support() {
    let builder = ProtocolClientBuilder::builder()
        .client_id("test".to_string())
        .name("Test Client".to_string())
        .player_v1_support(PlayerV1Support {
            supported_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 1024,
            supported_commands: vec![],
        })
        .build();

    assert_eq!(builder.supported_roles(), &["player@v1"]);
}

#[test]
fn test_supported_roles_with_artwork_v1_support() {
    let builder = ProtocolClientBuilder::builder()
        .client_id("test".to_string())
        .name("Test Client".to_string())
        .artwork_v1_support(ArtworkV1Support {
            channels: vec![ArtworkChannel {
                source: ArtworkSource::Album,
                format: ImageFormat::Jpeg,
                media_width: 300,
                media_height: 300,
            }],
        })
        .build();

    // Default player@v1 is always present, plus the artwork@v1 we added
    assert_eq!(builder.supported_roles(), &["player@v1", "artwork@v1"]);
}

#[test]
fn test_supported_roles_with_visualizer_v1_support() {
    let builder = ProtocolClientBuilder::builder()
        .client_id("test".to_string())
        .name("Test Client".to_string())
        .visualizer_v1_support(VisualizerV1Support {
            buffer_capacity: 1024,
        })
        .build();

    // Default player@v1 is always present, plus the visualizer@v1 we added
    assert_eq!(builder.supported_roles(), &["player@v1", "visualizer@v1"]);
}

#[test]
fn test_supported_roles_with_multiple_supports() {
    let builder = ProtocolClientBuilder::builder()
        .client_id("test".to_string())
        .name("Test Client".to_string())
        .player_v1_support(PlayerV1Support {
            supported_formats: vec![AudioFormatSpec {
                codec: "pcm".to_string(),
                channels: 2,
                sample_rate: 48000,
                bit_depth: 24,
            }],
            buffer_capacity: 1024,
            supported_commands: vec![],
        })
        .artwork_v1_support(ArtworkV1Support {
            channels: vec![ArtworkChannel {
                source: ArtworkSource::Album,
                format: ImageFormat::Jpeg,
                media_width: 300,
                media_height: 300,
            }],
        })
        .visualizer_v1_support(VisualizerV1Support {
            buffer_capacity: 1024,
        })
        .build();

    assert_eq!(
        builder.supported_roles(),
        &["player@v1", "artwork@v1", "visualizer@v1"]
    );
}

#[test]
fn test_default_player_v1_support_applied_at_build_time() {
    // When no player support is explicitly set, the builder has default player@v1 support
    let builder = ProtocolClientBuilder::builder()
        .client_id("test".to_string())
        .name("Test Client".to_string())
        .build();

    // Default player@v1 role is present
    assert_eq!(builder.supported_roles(), &["player@v1"]);

    // Default player support is applied
    let support = builder
        .player_v1_support()
        .expect("should have default player support");
    assert_eq!(support.supported_formats.len(), 1);
    assert_eq!(support.supported_formats[0].codec, "pcm");
    assert_eq!(support.supported_formats[0].channels, 2);
    assert_eq!(support.supported_formats[0].sample_rate, 48000);
    assert_eq!(support.supported_formats[0].bit_depth, 24);
    assert_eq!(support.buffer_capacity, 50 * 1024 * 1024); // 50 MB
    assert_eq!(
        support.supported_commands,
        vec!["volume".to_string(), "mute".to_string()]
    );
}

#[test]
fn test_explicit_player_support_is_preserved() {
    let custom_support = PlayerV1Support {
        supported_formats: vec![AudioFormatSpec {
            codec: "opus".to_string(),
            channels: 1,
            sample_rate: 44100,
            bit_depth: 16,
        }],
        buffer_capacity: 1024,
        supported_commands: vec!["pause".to_string()],
    };

    let builder = ProtocolClientBuilder::builder()
        .client_id("test".to_string())
        .name("Test Client".to_string())
        .player_v1_support(custom_support.clone())
        .build();

    let support = builder
        .player_v1_support()
        .expect("should have player support");
    assert_eq!(support.supported_formats[0].codec, "opus");
    assert_eq!(support.buffer_capacity, 1024);
}
