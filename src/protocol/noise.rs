//! Noise `KKpsk2` handshake (client as responder) and the encrypted transport that follows it.

use crate::error::Error;
use crate::protocol::messages::{CipherSuite, ClientInit, Message, NoiseHandshake};
use crate::protocol::persistence::PersistenceProvider;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use curve25519_dalek::montgomery::MontgomeryPoint;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use snow::StatelessTransportState;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::WebSocketStream;

const NOISE_PARAMS: &str = "Noise_KKpsk2_25519_ChaChaPoly_SHA256";
const PROTOCOL_VERSION: u32 = 1;
const IDENTITY_KEY: &str = "keypair";
const PSK_ID_LABEL: &[u8] = b"sendspin-psk-id-v1";
const NOISE_MAX_MESSAGE: usize = 65535;
const AEAD_TAG: usize = 16;
/// Largest plaintext of one transport message, message type byte included.
const MAX_PLAINTEXT: usize = NOISE_MAX_MESSAGE - AEAD_TAG;

/// `SHA-256("sendspin-sentinel-psk-v1")`.
const SENTINEL_PSK: [u8; 32] = [
    0x1b, 0x5e, 0x24, 0xdb, 0xc1, 0xae, 0xd9, 0x5f, 0xc2, 0xa5, 0xa3, 0x38, 0xa9, 0x0c, 0x05, 0xdf,
    0x44, 0xbd, 0x10, 0xf5, 0xec, 0x1f, 0x4c, 0xd6, 0x6c, 0xbf, 0x86, 0x27, 0x27, 0x67, 0xb9, 0xd3,
];

/// Time each side has to send the next message while the connection is being set up.
pub(crate) const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) const MESSAGE_TYPE_JSON: u8 = 0;
const MESSAGE_TYPE_FRAGMENT: u8 = 1;
const FRAGMENT_LAST: u8 = 0x01;
const FRAGMENT_FIRST: u8 = 0x02;
const FRAGMENT_RESERVED: u8 = 0xFC;

fn noise_error(error: snow::Error) -> Error {
    Error::Protocol(format!("Noise: {error}"))
}

fn noise_builder() -> snow::Builder<'static> {
    snow::Builder::new(
        NOISE_PARAMS
            .parse()
            .expect("the Sendspin Noise protocol name is valid"),
    )
}

/// The client's static Curve25519 keypair, whose public key is its `client_id`.
#[derive(Clone)]
pub(crate) struct Identity {
    private: [u8; 32],
    public: [u8; 32],
}

impl Identity {
    /// Load the stored private key, generating and storing one on first use.
    pub(crate) fn load_or_create(store: &dyn PersistenceProvider) -> Result<Self, Error> {
        let stored = store
            .load(IDENTITY_KEY)
            .map_err(|e| Error::Persistence(format!("loading the identity key failed: {e}")))?;
        let private: [u8; 32] = match stored {
            Some(bytes) => bytes.try_into().map_err(|_| {
                Error::Persistence("the stored identity key is not 32 bytes".to_string())
            })?,
            None => {
                let keypair = noise_builder().generate_keypair().map_err(noise_error)?;
                let private: [u8; 32] = keypair
                    .private
                    .try_into()
                    .expect("Curve25519 private keys are 32 bytes");
                store.save(IDENTITY_KEY, &private).map_err(|e| {
                    Error::Persistence(format!("saving the identity key failed: {e}"))
                })?;
                private
            }
        };
        Ok(Self {
            private,
            public: MontgomeryPoint::mul_base_clamped(private).to_bytes(),
        })
    }

    pub(crate) fn client_id(&self) -> String {
        URL_SAFE_NO_PAD.encode(self.public)
    }
}

/// The category the server declares for the PSK that keys the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub(crate) enum PskCategory {
    #[serde(rename = "lt")]
    LongTerm,
    #[serde(rename = "pr")]
    Pairing,
    #[serde(rename = "sn")]
    Sentinel,
}

#[derive(Deserialize)]
struct Message1Payload {
    psk_id: String,
    psk_category: PskCategory,
}

fn psk_id(psk: &[u8; 32]) -> String {
    let digest = Sha256::new()
        .chain_update(PSK_ID_LABEL)
        .chain_update(psk)
        .finalize();
    URL_SAFE_NO_PAD.encode(digest)
}

/// Session state produced by a completed handshake.
pub(crate) struct Session {
    pub(crate) server_id: String,
    pub(crate) psk_category: PskCategory,
    pub(crate) encryptor: Encryptor,
    pub(crate) decryptor: Decryptor,
}

/// Next complete WebSocket message within [`HANDSHAKE_TIMEOUT`], skipping control frames.
pub(crate) async fn next_handshake_frame<S>(
    stream: &mut SplitStream<WebSocketStream<S>>,
) -> Result<WsMessage, Error>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    loop {
        let frame = tokio::time::timeout(HANDSHAKE_TIMEOUT, stream.next())
            .await
            .map_err(|_| Error::Connection("timed out waiting for the server".to_string()))?;
        match frame {
            Some(Ok(WsMessage::Ping(_) | WsMessage::Pong(_))) => continue,
            Some(Ok(WsMessage::Close(_))) | None => {
                return Err(Error::Connection(
                    "server closed the connection".to_string(),
                ))
            }
            Some(Ok(message)) => return Ok(message),
            Some(Err(e)) => return Err(Error::WebSocket(e.to_string())),
        }
    }
}

async fn expect_text<S>(stream: &mut SplitStream<WebSocketStream<S>>) -> Result<String, Error>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    match next_handshake_frame(stream).await? {
        WsMessage::Text(text) => Ok(text.to_string()),
        other => Err(Error::Protocol(format!(
            "expected a text handshake message, got {other:?}"
        ))),
    }
}

/// Run `client/init`, `server/init` and the two Noise handshake messages.
///
/// Every failure after `client/init` is silent: the caller drops the
/// connection without sending anything further.
pub(crate) async fn handshake<S>(
    sink: &mut SplitSink<WebSocketStream<S>, WsMessage>,
    stream: &mut SplitStream<WebSocketStream<S>>,
    identity: &Identity,
) -> Result<Session, Error>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let client_init = serde_json::to_string(&Message::ClientInit(ClientInit {
        client_id: identity.client_id(),
        version: PROTOCOL_VERSION,
        suite: CipherSuite::ChaChaPolySha256,
    }))
    .map_err(|e| Error::Protocol(e.to_string()))?;
    log::debug!("Sending client/init: {client_init}");
    sink.send(WsMessage::Text(client_init.clone().into()))
        .await
        .map_err(|e| Error::WebSocket(e.to_string()))?;

    let server_init = expect_text(stream).await?;
    log::debug!("Received {server_init}");
    let server_id = match serde_json::from_str::<Message>(&server_init) {
        Ok(Message::ServerInit(init)) if init.version == PROTOCOL_VERSION => init.server_id,
        Ok(Message::ServerError(error)) => {
            return Err(Error::Connection(format!(
                "server rejected client/init: {:?}",
                error.reason
            )))
        }
        _ => {
            return Err(Error::Protocol(format!(
                "unusable server/init: {server_init}"
            )))
        }
    };
    let server_key: [u8; 32] = URL_SAFE_NO_PAD
        .decode(&server_id)
        .ok()
        .and_then(|key| key.try_into().ok())
        .ok_or_else(|| {
            Error::Protocol(format!("server_id is not a Curve25519 key: {server_id}"))
        })?;

    let prologue = [client_init.as_bytes(), server_init.as_bytes()].concat();
    let mut noise = noise_builder()
        .local_private_key(&identity.private)
        .and_then(|builder| builder.remote_public_key(&server_key))
        .and_then(|builder| builder.prologue(&prologue))
        .and_then(snow::Builder::build_responder)
        .map_err(noise_error)?;

    let message1 = match serde_json::from_str::<Message>(&expect_text(stream).await?) {
        Ok(Message::NoiseHandshake(handshake)) => URL_SAFE_NO_PAD
            .decode(&handshake.data)
            .map_err(|e| Error::Protocol(format!("Noise message 1 is not base64url: {e}")))?,
        _ => return Err(Error::Protocol("expected Noise message 1".to_string())),
    };
    let mut buffer = vec![0_u8; NOISE_MAX_MESSAGE];
    let len = noise
        .read_message(&message1, &mut buffer)
        .map_err(noise_error)?;
    let payload: Message1Payload = serde_json::from_slice(&buffer[..len])
        .map_err(|e| Error::Protocol(format!("malformed Noise message 1 payload: {e}")))?;
    // The Sentinel is the only PSK the client holds, so a miss falls back to it too.
    if payload.psk_category != PskCategory::Sentinel || payload.psk_id != psk_id(&SENTINEL_PSK) {
        log::info!(
            "No PSK matches the server's {:?} psk_id; falling back to the Sentinel PSK",
            payload.psk_category
        );
    }
    let psk_category = PskCategory::Sentinel;
    noise.set_psk(2, &SENTINEL_PSK).map_err(noise_error)?;

    let len = noise
        .write_message(b"{}", &mut buffer)
        .map_err(noise_error)?;
    let message2 = serde_json::to_string(&Message::NoiseHandshake(NoiseHandshake {
        data: URL_SAFE_NO_PAD.encode(&buffer[..len]),
    }))
    .map_err(|e| Error::Protocol(e.to_string()))?;
    sink.send(WsMessage::Text(message2.into()))
        .await
        .map_err(|e| Error::WebSocket(e.to_string()))?;

    let transport = Arc::new(noise.into_stateless_transport_mode().map_err(noise_error)?);
    Ok(Session {
        server_id,
        psk_category,
        encryptor: Encryptor {
            transport: Arc::clone(&transport),
            nonce: 0,
        },
        decryptor: Decryptor {
            transport,
            nonce: 0,
            reassembly: None,
        },
    })
}

/// Send `message` encrypted as JSON.
pub(crate) async fn send_encrypted<S>(
    sink: &mut SplitSink<WebSocketStream<S>, WsMessage>,
    encryptor: &mut Encryptor,
    message: &Message,
) -> Result<(), Error>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    // Time pings go out at 1Hz for as long as the connection lives; keep
    // that housekeeping at trace so debug shows only meaningful traffic.
    let level = if matches!(message, Message::ClientTime(_)) {
        log::Level::Trace
    } else {
        log::Level::Debug
    };
    log::log!(level, "Sending message: {message:?}");
    for frame in encryptor.seal_json(message)? {
        sink.send(frame)
            .await
            .map_err(|e| Error::WebSocket(e.to_string()))?;
    }
    Ok(())
}

/// Receive the next encrypted JSON message while the connection is being set up.
pub(crate) async fn recv_encrypted<S>(
    stream: &mut SplitStream<WebSocketStream<S>>,
    decryptor: &mut Decryptor,
) -> Result<Message, Error>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    loop {
        let WsMessage::Binary(data) = next_handshake_frame(stream).await? else {
            return Err(Error::Protocol(
                "cleartext message after the handshake".to_string(),
            ));
        };
        let Some(plaintext) = decryptor.open(&data)? else {
            continue;
        };
        if plaintext[0] != MESSAGE_TYPE_JSON {
            return Err(Error::Protocol(format!(
                "expected a JSON message, got binary type {}",
                plaintext[0]
            )));
        }
        let message = serde_json::from_slice(&plaintext[1..])
            .map_err(|e| Error::Protocol(format!("unparseable message: {e}")))?;
        log::debug!("Received message: {message:?}");
        return Ok(message);
    }
}

/// Encrypts outgoing messages, fragmenting those too large for one transport message.
pub(crate) struct Encryptor {
    transport: Arc<StatelessTransportState>,
    nonce: u64,
}

impl Encryptor {
    /// The WebSocket binary messages that carry one message of `message_type`.
    pub(crate) fn seal(
        &mut self,
        message_type: u8,
        payload: &[u8],
    ) -> Result<Vec<WsMessage>, Error> {
        fragment(message_type, payload)
            .into_iter()
            .map(|plaintext| {
                let mut ciphertext = vec![0_u8; plaintext.len() + AEAD_TAG];
                let len = self
                    .transport
                    .write_message(self.nonce, &plaintext, &mut ciphertext)
                    .map_err(noise_error)?;
                self.nonce += 1;
                ciphertext.truncate(len);
                Ok(WsMessage::Binary(ciphertext.into()))
            })
            .collect()
    }

    /// The WebSocket binary messages that carry `message` as JSON.
    pub(crate) fn seal_json(&mut self, message: &Message) -> Result<Vec<WsMessage>, Error> {
        let json = serde_json::to_vec(message).map_err(|e| Error::Protocol(e.to_string()))?;
        self.seal(MESSAGE_TYPE_JSON, &json)
    }
}

fn fragment(message_type: u8, payload: &[u8]) -> Vec<Vec<u8>> {
    if payload.len() < MAX_PLAINTEXT {
        return vec![[&[message_type], payload].concat()];
    }
    let (first, mut rest) = payload.split_at(MAX_PLAINTEXT - 3);
    let mut fragments = vec![[
        &[MESSAGE_TYPE_FRAGMENT, FRAGMENT_FIRST, message_type],
        first,
    ]
    .concat()];
    while !rest.is_empty() {
        let (data, tail) = rest.split_at(rest.len().min(MAX_PLAINTEXT - 2));
        fragments.push([&[MESSAGE_TYPE_FRAGMENT, 0], data].concat());
        rest = tail;
    }
    if let Some(last) = fragments.last_mut() {
        last[1] |= FRAGMENT_LAST;
    }
    fragments
}

/// Decrypts incoming binary messages and reassembles fragmented ones.
pub(crate) struct Decryptor {
    transport: Arc<StatelessTransportState>,
    nonce: u64,
    /// The in-flight fragmented message, its original type in byte 0.
    reassembly: Option<Vec<u8>>,
}

impl Decryptor {
    /// Decrypt one binary message. Returns the complete plaintext (message
    /// type in byte 0), or `None` while a fragmented message is in flight.
    ///
    /// An error means the connection must close: the message failed
    /// authentication or broke the fragmentation rules.
    pub(crate) fn open(&mut self, ciphertext: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let mut plaintext = vec![0_u8; ciphertext.len()];
        let len = self
            .transport
            .read_message(self.nonce, ciphertext, &mut plaintext)
            .map_err(noise_error)?;
        self.nonce += 1;
        plaintext.truncate(len);
        match plaintext.first() {
            None => Err(Error::Protocol("empty binary message".to_string())),
            Some(&MESSAGE_TYPE_FRAGMENT) => self.reassemble(&plaintext),
            Some(_) if self.reassembly.is_some() => Err(Error::Protocol(
                "binary message interleaved with a fragmented message".to_string(),
            )),
            Some(_) => Ok(Some(plaintext)),
        }
    }

    fn reassemble(&mut self, fragment: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let flags = *fragment
            .get(1)
            .ok_or_else(|| Error::Protocol("fragment without flags".to_string()))?;
        if flags & FRAGMENT_RESERVED != 0 {
            return Err(Error::Protocol(format!("fragment flags {flags:#04x}")));
        }
        if flags & FRAGMENT_FIRST != 0 {
            let orig_type = *fragment
                .get(2)
                .ok_or_else(|| Error::Protocol("first fragment without a type".to_string()))?;
            if orig_type == MESSAGE_TYPE_FRAGMENT || self.reassembly.is_some() {
                return Err(Error::Protocol("malformed fragment sequence".to_string()));
            }
            self.reassembly = Some([&[orig_type], &fragment[3..]].concat());
        } else {
            self.reassembly
                .as_mut()
                .ok_or_else(|| Error::Protocol("fragment without a first fragment".to_string()))?
                .extend_from_slice(&fragment[2..]);
        }
        Ok(if flags & FRAGMENT_LAST != 0 {
            self.reassembly.take()
        } else {
            None
        })
    }
}

/// Both sides of a completed handshake: the server's encryptor and the client's decryptor.
#[cfg(test)]
pub(crate) fn test_session_pair() -> (Encryptor, Decryptor) {
    let server = noise_builder().generate_keypair().unwrap();
    let client = noise_builder().generate_keypair().unwrap();
    let mut initiator = noise_builder()
        .local_private_key(&server.private)
        .unwrap()
        .remote_public_key(&client.public)
        .unwrap()
        .psk(2, &SENTINEL_PSK)
        .unwrap()
        .build_initiator()
        .unwrap();
    let mut responder = noise_builder()
        .local_private_key(&client.private)
        .unwrap()
        .remote_public_key(&server.public)
        .unwrap()
        .psk(2, &SENTINEL_PSK)
        .unwrap()
        .build_responder()
        .unwrap();
    let mut message = vec![0_u8; NOISE_MAX_MESSAGE];
    let mut payload = vec![0_u8; NOISE_MAX_MESSAGE];
    let len = initiator.write_message(b"{}", &mut message).unwrap();
    responder
        .read_message(&message[..len], &mut payload)
        .unwrap();
    let len = responder.write_message(b"{}", &mut message).unwrap();
    initiator
        .read_message(&message[..len], &mut payload)
        .unwrap();
    (
        Encryptor {
            transport: Arc::new(initiator.into_stateless_transport_mode().unwrap()),
            nonce: 0,
        },
        Decryptor {
            transport: Arc::new(responder.into_stateless_transport_mode().unwrap()),
            nonce: 0,
            reassembly: None,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::persistence::MemoryStore;

    #[test]
    fn sentinel_constants_match_the_spec() {
        assert_eq!(
            Sha256::digest(b"sendspin-sentinel-psk-v1").as_slice(),
            SENTINEL_PSK
        );
        assert_eq!(
            psk_id(&SENTINEL_PSK),
            "GFsV9tLaSQm9HcFWpKsgYQOr7wFTvNUtkmFwuVz3zoo"
        );
    }

    #[test]
    fn identity_is_generated_once_and_reloaded() {
        let store = MemoryStore::new();
        let first = Identity::load_or_create(&store).unwrap();
        let second = Identity::load_or_create(&store).unwrap();
        assert_eq!(first.client_id(), second.client_id());
        assert_eq!(first.client_id().len(), 43);
    }

    #[test]
    fn identity_derives_the_public_key_of_the_stored_private_key() {
        let keypair = noise_builder().generate_keypair().unwrap();
        let store = MemoryStore::new();
        store.save(IDENTITY_KEY, &keypair.private).unwrap();
        let identity = Identity::load_or_create(&store).unwrap();
        assert_eq!(identity.public.to_vec(), keypair.public);
    }

    fn binary(message: &WsMessage) -> &[u8] {
        match message {
            WsMessage::Binary(data) => data,
            other => panic!("expected binary, got {other:?}"),
        }
    }

    #[test]
    fn large_messages_round_trip_through_fragments() {
        let (mut encryptor, mut decryptor) = test_session_pair();
        let payload: Vec<u8> = (0..150_000_u32).map(|i| i as u8).collect();
        let sealed = encryptor.seal(4, &payload).unwrap();
        assert_eq!(sealed.len(), 3);
        assert!(sealed
            .iter()
            .all(|message| binary(message).len() <= NOISE_MAX_MESSAGE));
        let mut opened: Vec<_> = sealed
            .iter()
            .map(|message| decryptor.open(binary(message)).unwrap())
            .collect();
        assert!(opened[0].is_none() && opened[1].is_none());
        let message = opened.pop().unwrap().unwrap();
        assert_eq!(message[0], 4);
        assert_eq!(&message[1..], payload);
    }

    #[test]
    fn a_message_that_fits_is_not_fragmented() {
        let (mut encryptor, mut decryptor) = test_session_pair();
        let payload = vec![7_u8; MAX_PLAINTEXT - 1];
        let sealed = encryptor.seal(MESSAGE_TYPE_JSON, &payload).unwrap();
        assert_eq!(sealed.len(), 1);
        let message = decryptor.open(binary(&sealed[0])).unwrap().unwrap();
        assert_eq!(message[0], MESSAGE_TYPE_JSON);
        assert_eq!(message.len(), MAX_PLAINTEXT);
    }

    #[test]
    fn a_replayed_message_fails_authentication() {
        let (mut encryptor, mut decryptor) = test_session_pair();
        let sealed = encryptor.seal(MESSAGE_TYPE_JSON, b"{}").unwrap();
        decryptor.open(binary(&sealed[0])).unwrap();
        assert!(decryptor.open(binary(&sealed[0])).is_err());
    }

    #[test]
    fn a_message_interleaved_with_fragments_is_rejected() {
        let (mut encryptor, mut decryptor) = test_session_pair();
        let fragments = encryptor.seal(4, &vec![0_u8; MAX_PLAINTEXT * 2]).unwrap();
        let single = encryptor.seal(4, b"chunk").unwrap();
        decryptor.open(binary(&fragments[0])).unwrap();
        assert!(decryptor.open(binary(&single[0])).is_err());
    }
}
