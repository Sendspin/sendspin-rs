// ABOUTME: Builder exposed for public usage of the library

use crate::error::Error;
use crate::protocol::client::ConnectParams;
use crate::protocol::listener::ProtocolListener;
use crate::protocol::messages::{
    ArtworkV1Support, ClientHello, ClientState, ClientSyncState, DeviceInfo, PlayerState,
    PlayerV1Support, UnpairedAccess, VisualizerV1Support,
};
use crate::protocol::noise::Identity;
use crate::protocol::persistence::PersistenceProvider;
use crate::sync::raw_clock::{Clock, DefaultClock};
use crate::ProtocolClient;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::{TcpListener, ToSocketAddrs};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::WebSocketStream;
use typed_builder::TypedBuilder;

/// Intermediate builder struct before finalization
#[derive(Clone)]
pub(crate) struct ProtocolClientBuilderRaw {
    persistence: Arc<dyn PersistenceProvider>,
    name: String,
    product_name: Option<String>,
    manufacturer: Option<String>,
    software_version: Option<String>,
    mac_address: Option<String>,
    player_v1_support: Option<PlayerV1Support>,
    artwork_v1_support: Option<ArtworkV1Support>,
    visualizer_v1_support: Option<VisualizerV1Support>,
    initial_sync_state: ClientSyncState,
    initial_player_state: Option<PlayerState>,
    automatic_player_state: bool,
    unpaired_access: bool,
    metadata: bool,
    controller: bool,
    color: bool,
}

impl From<ProtocolClientBuilderRaw> for ProtocolClientBuilder {
    fn from(raw: ProtocolClientBuilderRaw) -> Self {
        // Build supported_roles based on which supports are configured
        let mut supported_roles = Vec::new();
        if raw.player_v1_support.is_some() {
            supported_roles.push("player@v1".to_string());
        }
        if raw.artwork_v1_support.is_some() {
            supported_roles.push("artwork@v1".to_string());
        }
        if raw.visualizer_v1_support.is_some() {
            supported_roles.push("visualizer@v1".to_string());
        }
        if raw.metadata {
            supported_roles.push("metadata@v1".to_string());
        }
        if raw.controller {
            supported_roles.push("controller@v1".to_string());
        }
        if raw.color {
            supported_roles.push("color@v1".to_string());
        }

        ProtocolClientBuilder {
            persistence: raw.persistence,
            identity: Arc::new(Mutex::new(None)),
            name: raw.name,
            product_name: raw.product_name,
            manufacturer: raw.manufacturer,
            software_version: raw.software_version,
            mac_address: raw.mac_address,
            supported_roles,
            player_v1_support: raw.player_v1_support,
            clock: Arc::new(DefaultClock::new()),
            artwork_v1_support: raw.artwork_v1_support,
            visualizer_v1_support: raw.visualizer_v1_support,
            initial_sync_state: raw.initial_sync_state,
            initial_player_state: raw.initial_player_state,
            automatic_player_state: raw.automatic_player_state,
            unpaired_access: raw.unpaired_access,
        }
    }
}

#[derive(TypedBuilder, Clone)]
#[builder(build_method(into = ProtocolClientBuilder))]
/// Builder Class for ProtocolClient
pub struct ProtocolClientBuilderFields {
    /// Storage for the client's identity key, which must survive restarts.
    persistence: Arc<dyn PersistenceProvider>,
    name: String,
    #[builder(default = None)]
    product_name: Option<String>,
    #[builder(default = None)]
    manufacturer: Option<String>,
    #[builder(default = None)]
    software_version: Option<String>,
    #[builder(default = None)]
    mac_address: Option<String>,
    #[builder(default = None, setter(transform = |x: PlayerV1Support| Some(x)))]
    player_v1_support: Option<PlayerV1Support>,
    #[builder(default = None, setter(transform = |x: ArtworkV1Support| Some(x)))]
    artwork_v1_support: Option<ArtworkV1Support>,
    #[builder(default = None, setter(transform = |x: VisualizerV1Support| Some(x)))]
    visualizer_v1_support: Option<VisualizerV1Support>,
    /// Initial top-level operational state sent in the first `client/state`.
    /// [`ClientSyncState::ExternalSource`] when another source owns playback.
    #[builder(default = ClientSyncState::Synchronized)]
    initial_sync_state: ClientSyncState,
    /// Must set `volume` or `muted` when the player advertises the matching command.
    #[builder(default = None, setter(transform = |x: PlayerState| Some(x)))]
    initial_player_state: Option<PlayerState>,
    /// Automatically report successful player control changes as client/state updates.
    #[builder(default = true)]
    automatic_player_state: bool,
    /// Admit servers without a pairing record to activate roles and play.
    /// Without pairing these connections are open to man-in-the-middle attacks.
    #[builder(default = false)]
    unpaired_access: bool,
    #[builder(default = false, setter(transform = || true))]
    metadata: bool,
    #[builder(default = false, setter(transform = || true))]
    controller: bool,
    #[builder(default = false, setter(transform = || true))]
    color: bool,
}

impl From<ProtocolClientBuilderFields> for ProtocolClientBuilder {
    fn from(fields: ProtocolClientBuilderFields) -> Self {
        let raw = ProtocolClientBuilderRaw {
            persistence: fields.persistence,
            name: fields.name,
            product_name: fields.product_name,
            manufacturer: fields.manufacturer,
            software_version: fields.software_version,
            mac_address: fields.mac_address,
            player_v1_support: fields.player_v1_support,
            artwork_v1_support: fields.artwork_v1_support,
            visualizer_v1_support: fields.visualizer_v1_support,
            initial_sync_state: fields.initial_sync_state,
            initial_player_state: fields.initial_player_state,
            automatic_player_state: fields.automatic_player_state,
            unpaired_access: fields.unpaired_access,
            metadata: fields.metadata,
            controller: fields.controller,
            color: fields.color,
        };
        raw.into()
    }
}

/// Builder Class for ProtocolClient
#[derive(Clone)]
pub struct ProtocolClientBuilder {
    persistence: Arc<dyn PersistenceProvider>,
    /// Loaded on first use and shared by clones, so concurrent connections
    /// never generate competing keys.
    identity: Arc<Mutex<Option<Identity>>>,
    name: String,
    product_name: Option<String>,
    manufacturer: Option<String>,
    software_version: Option<String>,
    mac_address: Option<String>,
    supported_roles: Vec<String>,
    player_v1_support: Option<PlayerV1Support>,
    artwork_v1_support: Option<ArtworkV1Support>,
    visualizer_v1_support: Option<VisualizerV1Support>,
    initial_sync_state: ClientSyncState,
    initial_player_state: Option<PlayerState>,
    automatic_player_state: bool,
    unpaired_access: bool,
    clock: Arc<dyn Clock>,
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

    /// Override the default clock with a custom implementation.
    ///
    /// By default, the builder uses [`DefaultClock`] which reads
    /// `CLOCK_MONOTONIC_RAW` on Linux (immune to NTP slew) and the
    /// platform's native raw monotonic source elsewhere. Override this
    /// for testing or for platforms with alternative high-precision clocks.
    pub fn clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Connect to Sendspin server.
    ///
    /// Accepts anything that implements [`IntoClientRequest`], such as a URL string
    /// for simple connections. For custom headers (for example, auth cookies), callers
    /// will typically build an `http::Request<()>` — see the [`IntoClientRequest`] docs
    /// for the full set of supported request types.
    pub async fn connect<R: IntoClientRequest + Unpin>(
        self,
        request: R,
    ) -> Result<ProtocolClient, Error> {
        ProtocolClient::connect(request, self.into_params()?).await
    }

    /// Adopt an already-upgraded WebSocket stream and drive the protocol
    /// from `client/init` onwards.
    ///
    /// Use this when you're terminating TLS, routing by HTTP path, or
    /// otherwise need to own the WebSocket layer yourself. For the common
    /// "bind a TCP socket and accept inbound peers" case, use
    /// [`Self::listen`].
    pub async fn accept<S>(self, ws_stream: WebSocketStream<S>) -> Result<ProtocolClient, Error>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        ProtocolClient::drive(ws_stream, self.into_params()?).await
    }

    /// Bind a TCP listener and produce a [`ProtocolListener`] that accepts
    /// inbound WebSocket peers. The builder is cloned per accepted peer.
    pub async fn listen<A: ToSocketAddrs>(self, addr: A) -> Result<ProtocolListener, Error> {
        self.validate()?;
        self.identity()?;
        let tcp = TcpListener::bind(addr)
            .await
            .map_err(|e| Error::Connection(format!("TCP bind failed: {e}")))?;
        let local = tcp
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        log::info!("ProtocolListener bound on {local}");
        Ok(ProtocolListener::new(tcp, self))
    }

    fn validate(&self) -> Result<(), Error> {
        if self.supported_roles.is_empty() {
            return Err(Error::Protocol(
                "the client builder has no roles enabled".into(),
            ));
        }
        if let Some(support) = &self.player_v1_support {
            let player = self.initial_player_state.as_ref();
            let advertises = |command| support.supported_commands.iter().any(|c| c == command);
            if advertises("volume") && player.and_then(|p| p.volume).is_none() {
                return Err(Error::Protocol(
                    "initial_player_state must set volume when the player supports the volume command"
                        .into(),
                ));
            }
            if advertises("mute") && player.and_then(|p| p.muted).is_none() {
                return Err(Error::Protocol(
                    "initial_player_state must set muted when the player supports the mute command"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    fn identity(&self) -> Result<Identity, Error> {
        let mut identity = self.identity.lock();
        if identity.is_none() {
            *identity = Some(Identity::load_or_create(self.persistence.as_ref())?);
        }
        Ok(identity.clone().expect("loaded above"))
    }

    fn into_params(self) -> Result<ConnectParams, Error> {
        self.validate()?;
        let identity = self.identity()?;
        let hello = ClientHello {
            name: self.name,
            supported_roles: self.supported_roles,
            device_info: Some(DeviceInfo {
                product_name: self.product_name,
                manufacturer: Some(self.manufacturer.unwrap_or_else(|| "Sendspin".to_string())),
                software_version: self.software_version,
                mac_address: self.mac_address,
            }),
            player_v1_support: self.player_v1_support,
            artwork_v1_support: self.artwork_v1_support,
            visualizer_v1_support: self.visualizer_v1_support,
            unpaired_access: UnpairedAccess {
                enabled: self.unpaired_access,
            },
        };

        let initial_state = ClientState {
            state: Some(self.initial_sync_state),
            player: self.initial_player_state,
        };

        Ok(ConnectParams {
            identity,
            hello,
            initial_state,
            clock: self.clock,
            automatic_player_state: self.automatic_player_state,
        })
    }
}
