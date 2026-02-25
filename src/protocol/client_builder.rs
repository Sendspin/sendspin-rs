// ABOUTME: Builder exposed for public usage of the library

use crate::error::Error;
use crate::protocol::messages::{
    ArtworkV1Support, AudioFormatSpec, ClientHello, DeviceInfo, PlayerV1Support,
    VisualizerV1Support,
};
use crate::ProtocolClient;
use typed_builder::TypedBuilder;

/// Intermediate builder struct before finalization
#[derive(Clone)]
pub struct ProtocolClientBuilderRaw {
    client_id: String,
    name: String,
    product_name: Option<String>,
    manufacturer: Option<String>,
    software_version: Option<String>,
    player_v1_support: Option<PlayerV1Support>,
    artwork_v1_support: Option<ArtworkV1Support>,
    visualizer_v1_support: Option<VisualizerV1Support>,
}

impl From<ProtocolClientBuilderRaw> for ProtocolClientBuilder {
    fn from(raw: ProtocolClientBuilderRaw) -> Self {
        // Build supported_roles based on which supports are configured
        let mut supported_roles = Vec::new();

        // Default player support if not explicitly set
        let player_v1_support = raw.player_v1_support.or_else(|| {
            Some(PlayerV1Support {
                supported_formats: vec![AudioFormatSpec {
                    codec: "pcm".to_string(),
                    channels: 2,
                    sample_rate: 48000,
                    bit_depth: 24,
                }],
                buffer_capacity: 50 * 1024 * 1024,
                supported_commands: vec!["volume".to_string(), "mute".to_string()],
            })
        });

        // Player role is always present (since we set a default)
        supported_roles.push("player@v1".to_string());

        if raw.artwork_v1_support.is_some() {
            supported_roles.push("artwork@v1".to_string());
        }
        if raw.visualizer_v1_support.is_some() {
            supported_roles.push("visualizer@v1".to_string());
        }

        ProtocolClientBuilder {
            client_id: raw.client_id,
            name: raw.name,
            product_name: raw.product_name,
            manufacturer: raw.manufacturer,
            software_version: raw.software_version,
            supported_roles,
            player_v1_support,
            artwork_v1_support: raw.artwork_v1_support,
            visualizer_v1_support: raw.visualizer_v1_support,
        }
    }
}

#[derive(TypedBuilder, Clone)]
#[builder(build_method(into = ProtocolClientBuilder))]
/// Builder Class for ProtocolClient
pub struct ProtocolClientBuilderFields {
    client_id: String,
    name: String,
    #[builder(default = None)]
    product_name: Option<String>,
    #[builder(default = None)]
    manufacturer: Option<String>,
    #[builder(default = None)]
    software_version: Option<String>,
    #[builder(default = None, setter(transform = |x: PlayerV1Support| Some(x)))]
    player_v1_support: Option<PlayerV1Support>,
    #[builder(default = None, setter(transform = |x: ArtworkV1Support| Some(x)))]
    artwork_v1_support: Option<ArtworkV1Support>,
    #[builder(default = None, setter(transform = |x: VisualizerV1Support| Some(x)))]
    visualizer_v1_support: Option<VisualizerV1Support>,
}

impl From<ProtocolClientBuilderFields> for ProtocolClientBuilder {
    fn from(fields: ProtocolClientBuilderFields) -> Self {
        let raw = ProtocolClientBuilderRaw {
            client_id: fields.client_id,
            name: fields.name,
            product_name: fields.product_name,
            manufacturer: fields.manufacturer,
            software_version: fields.software_version,
            player_v1_support: fields.player_v1_support,
            artwork_v1_support: fields.artwork_v1_support,
            visualizer_v1_support: fields.visualizer_v1_support,
        };
        raw.into()
    }
}

/// Builder Class for ProtocolClient
#[derive(Clone)]
pub struct ProtocolClientBuilder {
    client_id: String,
    name: String,
    product_name: Option<String>,
    manufacturer: Option<String>,
    software_version: Option<String>,
    supported_roles: Vec<String>,
    player_v1_support: Option<PlayerV1Support>,
    artwork_v1_support: Option<ArtworkV1Support>,
    visualizer_v1_support: Option<VisualizerV1Support>,
}

impl ProtocolClientBuilder {
    /// Create a new builder
    pub fn builder() -> ProtocolClientBuilderFieldsBuilder {
        ProtocolClientBuilderFields::builder()
    }

    /// Get the supported roles that will be sent in the client hello
    pub fn supported_roles(&self) -> &[String] {
        &self.supported_roles
    }

    /// Get the player v1 support configuration
    pub fn player_v1_support(&self) -> Option<&PlayerV1Support> {
        self.player_v1_support.as_ref()
    }

    /// Connect to Sendspin server
    pub async fn connect(self, url: &str) -> Result<ProtocolClient, Error> {
        let hello = ClientHello {
            client_id: self.client_id.clone(),
            name: self.name.clone(),
            version: 1,
            supported_roles: self.supported_roles.clone(),
            device_info: Some(DeviceInfo {
                product_name: self.product_name.clone(),
                manufacturer: Some(self.manufacturer.unwrap_or("Sendspin".to_string())),
                software_version: self.software_version.clone(),
            }),
            player_v1_support: self.player_v1_support.clone(),
            artwork_v1_support: self.artwork_v1_support.clone(),
            visualizer_v1_support: self.visualizer_v1_support.clone(),
        };
        ProtocolClient::connect(url, hello).await
    }
}
