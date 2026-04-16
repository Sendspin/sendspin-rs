// ABOUTME: Protocol message type definitions and serialization
// ABOUTME: Supports all Sendspin protocol messages per spec

use serde::{Deserialize, Serialize};

/// Top-level protocol message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum Message {
    // === Handshake messages ===
    /// Client hello handshake message
    #[serde(rename = "client/hello")]
    ClientHello(ClientHello),

    /// Server hello handshake response
    #[serde(rename = "server/hello")]
    ServerHello(ServerHello),

    // === Time synchronization ===
    /// Client time synchronization request
    #[serde(rename = "client/time")]
    ClientTime(ClientTime),

    /// Server time synchronization response
    #[serde(rename = "server/time")]
    ServerTime(ServerTime),

    // === State messages ===
    /// Client state update
    #[serde(rename = "client/state")]
    ClientState(ClientState),

    /// Server state update (metadata, controller info)
    #[serde(rename = "server/state")]
    ServerState(ServerState),

    // === Command messages ===
    /// Server command to client (player commands)
    #[serde(rename = "server/command")]
    ServerCommand(ServerCommand),

    /// Client command to server (controller commands)
    #[serde(rename = "client/command")]
    ClientCommand(ClientCommand),

    // === Stream control messages ===
    /// Stream start notification
    #[serde(rename = "stream/start")]
    StreamStart(StreamStart),

    /// Stream end notification
    #[serde(rename = "stream/end")]
    StreamEnd(StreamEnd),

    /// Stream clear notification
    #[serde(rename = "stream/clear")]
    StreamClear(StreamClear),

    /// Client request for specific stream format
    #[serde(rename = "stream/request-format")]
    StreamRequestFormat(StreamRequestFormat),

    // === Group messages ===
    /// Group update notification
    #[serde(rename = "group/update")]
    GroupUpdate(GroupUpdate),

    // === Connection lifecycle ===
    /// Client goodbye message
    #[serde(rename = "client/goodbye")]
    ClientGoodbye(ClientGoodbye),
}

// =============================================================================
// Handshake Messages
// =============================================================================

/// Client hello message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientHello {
    /// Unique client identifier
    pub client_id: String,
    /// Human-readable client name
    pub name: String,
    /// Protocol version number
    pub version: u32,
    /// List of supported roles with versions (e.g., "player@v1", "controller@v1")
    pub supported_roles: Vec<String>,
    /// Device information (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_info: Option<DeviceInfo>,
    /// Player capabilities (if client supports player@v1 role)
    #[serde(rename = "player@v1_support", skip_serializing_if = "Option::is_none")]
    pub player_v1_support: Option<PlayerV1Support>,
    /// Artwork capabilities (if client supports artwork@v1 role)
    #[serde(rename = "artwork@v1_support", skip_serializing_if = "Option::is_none")]
    pub artwork_v1_support: Option<ArtworkV1Support>,
    /// Visualizer capabilities (if client supports visualizer@v1 role)
    #[serde(
        rename = "visualizer@v1_support",
        skip_serializing_if = "Option::is_none"
    )]
    pub visualizer_v1_support: Option<VisualizerV1Support>,
}

/// Device information (all fields optional per spec)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Product name (e.g., "Sendspin-RS Player")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub product_name: Option<String>,
    /// Manufacturer name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manufacturer: Option<String>,
    /// Software version string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software_version: Option<String>,
}

/// Player@v1 capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerV1Support {
    /// List of supported audio formats
    pub supported_formats: Vec<AudioFormatSpec>,
    /// Buffer capacity in chunks
    pub buffer_capacity: u32,
    /// List of supported playback commands
    pub supported_commands: Vec<String>,
}

/// Audio format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormatSpec {
    /// Codec name (e.g., "pcm", "opus", "flac")
    pub codec: String,
    /// Number of audio channels
    pub channels: u8,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Bit depth per sample
    pub bit_depth: u8,
}

/// Artwork@v1 capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtworkV1Support {
    /// Supported artwork channels (1-4 channels, array index is channel number)
    pub channels: Vec<ArtworkChannel>,
}

/// Artwork channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtworkChannel {
    /// Artwork source type
    pub source: ArtworkSource,
    /// Image format
    pub format: ImageFormat,
    /// Max width in pixels
    pub media_width: u32,
    /// Max height in pixels
    pub media_height: u32,
}

/// Artwork source type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ArtworkSource {
    /// Album artwork
    Album,
    /// Artist image
    Artist,
    /// No artwork (channel disabled)
    None,
}

/// Image format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    /// JPEG format
    Jpeg,
    /// PNG format
    Png,
    /// BMP format
    Bmp,
}

/// Visualizer@v1 capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizerV1Support {
    /// Buffer capacity for visualization data
    pub buffer_capacity: u32,
}

/// Server hello message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerHello {
    /// Unique server identifier
    pub server_id: String,
    /// Human-readable server name
    pub name: String,
    /// Protocol version number
    pub version: u32,
    /// List of roles activated by server for this client
    pub active_roles: Vec<String>,
    /// Reason for connection: 'discovery' or 'playback'
    pub connection_reason: ConnectionReason,
}

/// Connection reason enum
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConnectionReason {
    /// Server connected for discovery/announcement
    Discovery,
    /// Server connected for active playback
    Playback,
}

// =============================================================================
// Time Synchronization
// =============================================================================

/// Client time sync message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientTime {
    /// Client transmission timestamp (raw monotonic microseconds)
    pub client_transmitted: i64,
}

/// Server time sync response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerTime {
    /// Original client transmission timestamp
    pub client_transmitted: i64,
    /// Server reception timestamp (server loop microseconds)
    pub server_received: i64,
    /// Server transmission timestamp (server loop microseconds)
    pub server_transmitted: i64,
}

// =============================================================================
// State Messages
// =============================================================================

/// Client state update message (wraps role-specific state)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClientState {
    /// Client operational state
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<ClientSyncState>,
    /// Player state (if player role active)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub player: Option<PlayerState>,
}

/// Player state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlayerState {
    /// Current volume level (0-100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<u8>,
    /// Whether audio is muted
    #[serde(skip_serializing_if = "Option::is_none")]
    pub muted: Option<bool>,
    /// Static delay in milliseconds (0-5000) to compensate for external speaker/amplifier latency
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub static_delay_ms: Option<u16>,
    /// Supported player state commands
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub supported_commands: Option<Vec<PlayerStateCommand>>,
}

/// Commands that can appear in PlayerState.supported_commands
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlayerStateCommand {
    /// Client supports set_static_delay command
    SetStaticDelay,
}

/// Client synchronization state (top-level in client/state)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClientSyncState {
    /// Client is operational and synchronized with server timestamps
    Synchronized,
    /// Client has a problem preventing normal operation
    Error,
    /// Client is in use by an external system (e.g., different audio source, HDMI input)
    ExternalSource,
}

/// Server state update message (metadata and controller info)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerState {
    /// Metadata state (track info, progress, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<MetadataState>,
    /// Controller state (supported commands, volume, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub controller: Option<ControllerState>,
}

/// Metadata state from server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataState {
    /// Server timestamp for progress calculation (microseconds)
    pub timestamp: i64,
    /// Track title
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Artist name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artist: Option<String>,
    /// Album artist
    #[serde(skip_serializing_if = "Option::is_none")]
    pub album_artist: Option<String>,
    /// Album name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub album: Option<String>,
    /// Artwork URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artwork_url: Option<String>,
    /// Release year
    #[serde(skip_serializing_if = "Option::is_none")]
    pub year: Option<u32>,
    /// Track number (1-indexed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track: Option<u32>,
    /// Current track progress
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<TrackProgress>,
    /// Repeat mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat: Option<RepeatMode>,
    /// Shuffle state
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shuffle: Option<bool>,
}

/// Track progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackProgress {
    /// Current position in milliseconds
    pub track_progress: i64,
    /// Total duration in milliseconds (0 for unknown/live streams)
    pub track_duration: i64,
    /// Playback speed multiplier * 1000 (1000 = normal, 0 = paused)
    pub playback_speed: i32,
}

/// Repeat mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RepeatMode {
    /// No repeat
    Off,
    /// Repeat current track
    One,
    /// Repeat all tracks
    All,
}

/// Controller state from server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerState {
    /// List of supported commands
    pub supported_commands: Vec<String>,
    /// Current volume level (0-100)
    pub volume: u8,
    /// Whether audio is muted
    pub muted: bool,
}

// =============================================================================
// Command Messages
// =============================================================================

/// Server command message (wraps role-specific commands)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCommand {
    /// Player command (if targeting player role)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub player: Option<PlayerCommand>,
}

/// Player-specific command from server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerCommand {
    /// Command to execute
    pub command: PlayerCommandType,
    /// Optional volume level (0-100)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<u8>,
    /// Optional mute state
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mute: Option<bool>,
    /// Optional static delay in milliseconds (0-5000)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_delay_ms: Option<u16>,
}

/// Player command type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlayerCommandType {
    /// Set volume level
    Volume,
    /// Set mute state
    Mute,
    /// Set static delay
    SetStaticDelay,
    /// Unknown command (forward compatibility)
    #[serde(other)]
    Unknown,
}

/// Client command message (controller commands to server)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCommand {
    /// Controller command
    #[serde(skip_serializing_if = "Option::is_none")]
    pub controller: Option<ControllerCommand>,
}

/// Controller command from client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerCommand {
    /// Command to execute
    pub command: ControllerCommandType,
    /// Optional volume level (0-100) for volume command
    #[serde(skip_serializing_if = "Option::is_none")]
    pub volume: Option<u8>,
    /// Optional mute state for mute command
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mute: Option<bool>,
}

/// Controller command type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControllerCommandType {
    /// Resume playback
    Play,
    /// Pause playback
    Pause,
    /// Stop playback and reset position
    Stop,
    /// Skip to next track
    Next,
    /// Skip to previous track
    Previous,
    /// Set group volume
    Volume,
    /// Set group mute state
    Mute,
    /// Disable repeat
    RepeatOff,
    /// Repeat current track
    RepeatOne,
    /// Repeat all tracks
    RepeatAll,
    /// Randomize playback order
    Shuffle,
    /// Restore original playback order
    Unshuffle,
    /// Switch to next group
    Switch,
}

// =============================================================================
// Stream Control Messages
// =============================================================================

/// Stream start message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStart {
    /// Player stream configuration (optional - only if player role active)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub player: Option<StreamPlayerConfig>,
    /// Artwork stream configuration (optional - only if artwork role active)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artwork: Option<StreamArtworkConfig>,
    /// Visualizer stream configuration (optional - only if visualizer role active)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visualizer: Option<StreamVisualizerConfig>,
}

/// Stream player configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamPlayerConfig {
    /// Audio codec name
    pub codec: String,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u8,
    /// Bit depth per sample
    pub bit_depth: u8,
    /// Optional codec-specific header (base64 encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec_header: Option<String>,
}

/// Stream artwork configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamArtworkConfig {
    /// Configuration for each active artwork channel, array index is the channel number
    pub channels: Vec<StreamArtworkChannelConfig>,
}

/// Configuration for a single artwork channel in stream/start
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamArtworkChannelConfig {
    /// Artwork source type
    pub source: ArtworkSource,
    /// Format of the encoded image
    pub format: ImageFormat,
    /// Width in pixels of the encoded image
    pub width: u32,
    /// Height in pixels of the encoded image
    pub height: u32,
}

/// Stream visualizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamVisualizerConfig {
    // FFT details TBD per spec
}

/// Stream end message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEnd {
    /// Roles for which streaming has ended (optional, all if not specified)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roles: Option<Vec<String>>,
}

/// Stream clear message (clear buffers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamClear {
    /// Roles for which buffers should be cleared (optional, all if not specified)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roles: Option<Vec<String>>,
}

/// Stream format request from client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamRequestFormat {
    /// Requested player format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub player: Option<PlayerFormatRequest>,
    /// Requested artwork format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artwork: Option<ArtworkFormatRequest>,
}

/// Player format request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerFormatRequest {
    /// Preferred codec
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec: Option<String>,
    /// Preferred channel count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<u8>,
    /// Preferred sample rate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    /// Preferred bit depth
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bit_depth: Option<u8>,
}

/// Artwork format request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtworkFormatRequest {
    /// Artwork channel to request
    pub channel: u8,
    /// Preferred image source
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<ArtworkSource>,
    /// Preferred image format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ImageFormat>,
    /// Display width in pixels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_width: Option<u32>,
    /// Display height in pixels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_height: Option<u32>,
}

// =============================================================================
// Group Messages
// =============================================================================

/// Group update notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupUpdate {
    /// Current playback state of the group
    #[serde(skip_serializing_if = "Option::is_none")]
    pub playback_state: Option<PlaybackState>,
    /// Group identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    /// Human-readable group name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_name: Option<String>,
}

/// Group playback state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PlaybackState {
    /// Audio is playing
    Playing,
    /// Playback is stopped
    Stopped,
}

// =============================================================================
// Connection Lifecycle
// =============================================================================

/// Client goodbye message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientGoodbye {
    /// Reason for disconnection
    pub reason: GoodbyeReason,
}

/// Goodbye reason
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GoodbyeReason {
    /// Switching to another server
    AnotherServer,
    /// Client is shutting down
    Shutdown,
    /// Client is restarting
    Restart,
    /// User requested disconnect
    UserRequest,
}

// =============================================================================
// Legacy Aliases (deprecated)
// =============================================================================

/// Legacy type alias for backwards compatibility
#[deprecated(note = "Use PlayerV1Support instead")]
pub type PlayerSupport = PlayerV1Support;
