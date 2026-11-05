// ABOUTME: Server side of the Sendspin rc1 handshake and encrypted transport,
// ABOUTME: shared by the integration tests that drive a real client.

// Each test crate uses a different subset of these helpers.
#![allow(dead_code)]

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use curve25519_dalek::MontgomeryPoint;
use futures_util::{SinkExt, StreamExt};
use sendspin::protocol::messages::{
    Activity, ClientHello, ClientTime, Message, NoiseHandshake, ServerActivate, ServerHello,
    ServerInit, ServerTime,
};
use sha2::{Digest, Sha256};
use snow::StatelessTransportState;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::WebSocketStream;

const NOISE_PARAMS: &str = "Noise_KKpsk2_25519_ChaChaPoly_SHA256";

fn sentinel_psk() -> [u8; 32] {
    Sha256::digest(b"sendspin-sentinel-psk-v1").into()
}

fn psk_id(psk: &[u8; 32]) -> String {
    URL_SAFE_NO_PAD.encode(
        Sha256::new()
            .chain_update(b"sendspin-psk-id-v1")
            .chain_update(psk)
            .finalize(),
    )
}

/// The static private key of the test server called `name`.
fn server_key(name: &str) -> [u8; 32] {
    Sha256::digest(name.as_bytes()).into()
}

/// The `server_id` of the test server called `name`.
pub fn server_id(name: &str) -> String {
    URL_SAFE_NO_PAD.encode(MontgomeryPoint::mul_base_clamped(server_key(name)).to_bytes())
}

/// The server end of one connection after the Noise handshake.
pub struct TestServer<S> {
    ws: WebSocketStream<S>,
    transport: StatelessTransportState,
    send_nonce: u64,
    recv_nonce: u64,
    /// This server's `server_id`, its static public key.
    pub server_id: String,
}

impl<S: AsyncRead + AsyncWrite + Unpin> TestServer<S> {
    /// Run `server/init`, a Sentinel-keyed Noise handshake and the hello
    /// exchange, returning the client's `client/hello`.
    pub async fn handshake(ws: WebSocketStream<S>) -> (Self, ClientHello) {
        Self::handshake_as(ws, "test-server").await
    }

    /// [`Self::handshake`] as the test server called `name`.
    pub async fn handshake_as(ws: WebSocketStream<S>, name: &str) -> (Self, ClientHello) {
        let mut server = Self::noise_handshake(ws, name).await;
        server
            .send(&Message::ServerHello(ServerHello {
                name: "Test Server".to_string(),
                languages: None,
            }))
            .await;
        match server.recv().await {
            Some(Message::ClientHello(hello)) => (server, hello),
            other => panic!("expected client/hello, got {other:?}"),
        }
    }

    /// Run `server/init` and a Sentinel-keyed Noise handshake as the test
    /// server called `name`.
    async fn noise_handshake(mut ws: WebSocketStream<S>, name: &str) -> Self {
        let client_init = match ws.next().await {
            Some(Ok(WsMessage::Text(text))) => text.to_string(),
            other => panic!("expected client/init, got {other:?}"),
        };
        let client_id = match serde_json::from_str::<Message>(&client_init).unwrap() {
            Message::ClientInit(init) => init.client_id,
            other => panic!("expected client/init, got {other:?}"),
        };
        let client_key = URL_SAFE_NO_PAD.decode(&client_id).unwrap();
        let server_id = server_id(name);
        let server_init = serde_json::to_string(&Message::ServerInit(ServerInit {
            server_id: server_id.clone(),
            version: 1,
        }))
        .unwrap();
        let prologue = [client_init.as_bytes(), server_init.as_bytes()].concat();
        let psk = sentinel_psk();
        let mut noise = snow::Builder::new(NOISE_PARAMS.parse().unwrap())
            .local_private_key(&server_key(name))
            .unwrap()
            .remote_public_key(&client_key)
            .unwrap()
            .prologue(&prologue)
            .unwrap()
            .psk(2, &psk)
            .unwrap()
            .build_initiator()
            .unwrap();
        ws.send(WsMessage::Text(server_init.into())).await.unwrap();

        let mut buffer = vec![0_u8; 65535];
        let payload = serde_json::json!({ "psk_id": psk_id(&psk), "psk_category": "sn" });
        let len = noise
            .write_message(payload.to_string().as_bytes(), &mut buffer)
            .unwrap();
        let message1 = serde_json::to_string(&Message::NoiseHandshake(NoiseHandshake {
            data: URL_SAFE_NO_PAD.encode(&buffer[..len]),
        }))
        .unwrap();
        ws.send(WsMessage::Text(message1.into())).await.unwrap();

        let message2 = match ws.next().await {
            Some(Ok(WsMessage::Text(text))) => {
                match serde_json::from_str::<Message>(&text).unwrap() {
                    Message::NoiseHandshake(handshake) => {
                        URL_SAFE_NO_PAD.decode(handshake.data).unwrap()
                    }
                    other => panic!("expected Noise message 2, got {other:?}"),
                }
            }
            other => panic!("expected Noise message 2, got {other:?}"),
        };
        let len = noise.read_message(&message2, &mut buffer).unwrap();
        assert_eq!(&buffer[..len], b"{}", "Noise message 2 payload");
        Self {
            ws,
            transport: noise.into_stateless_transport_mode().unwrap(),
            send_nonce: 0,
            recv_nonce: 0,
            server_id,
        }
    }

    /// Send a decrypted binary message (message type in byte 0) encrypted.
    pub async fn send_plaintext(&mut self, plaintext: &[u8]) {
        let mut ciphertext = vec![0_u8; plaintext.len() + 16];
        let len = self
            .transport
            .write_message(self.send_nonce, plaintext, &mut ciphertext)
            .unwrap();
        self.send_nonce += 1;
        ciphertext.truncate(len);
        // The client may already have closed after an earlier message.
        let _ = self.ws.send(WsMessage::Binary(ciphertext.into())).await;
    }

    pub async fn send(&mut self, message: &Message) {
        let json = serde_json::to_vec(message).unwrap();
        self.send_plaintext(&[&[0], json.as_slice()].concat()).await;
    }

    /// Send a test frame: text is a JSON message to encrypt, binary is a
    /// decrypted binary message, anything else goes out as is.
    pub async fn send_frame(&mut self, frame: WsMessage) {
        match frame {
            WsMessage::Text(json) => self.send_plaintext(&[&[0], json.as_bytes()].concat()).await,
            WsMessage::Binary(plaintext) => self.send_plaintext(&plaintext).await,
            other => {
                let _ = self.ws.send(other).await;
            }
        }
    }

    /// Send a frame without encrypting it.
    pub async fn send_raw(&mut self, frame: WsMessage) {
        let _ = self.ws.send(frame).await;
    }

    /// The next JSON message from the client as text, or `None` once it closed.
    pub async fn recv_json(&mut self) -> Option<String> {
        loop {
            match self.ws.next().await? {
                Ok(WsMessage::Binary(ciphertext)) => {
                    let mut plaintext = vec![0_u8; ciphertext.len()];
                    let len = self
                        .transport
                        .read_message(self.recv_nonce, &ciphertext, &mut plaintext)
                        .unwrap();
                    self.recv_nonce += 1;
                    assert_eq!(plaintext[0], 0, "client sent a non-JSON binary message");
                    return Some(String::from_utf8(plaintext[1..len].to_vec()).unwrap());
                }
                Ok(WsMessage::Close(_)) | Err(_) => return None,
                Ok(WsMessage::Text(text)) => panic!("client sent cleartext {text}"),
                Ok(_) => continue,
            }
        }
    }

    pub async fn recv(&mut self) -> Option<Message> {
        let json = self.recv_json().await?;
        Some(serde_json::from_str(&json).unwrap())
    }

    /// Activate `roles` with the `playback` activity.
    pub async fn activate(&mut self, roles: Vec<String>) {
        self.activate_with(vec![Activity::Playback], roles).await;
    }

    pub async fn activate_with(&mut self, activities: Vec<Activity>, roles: Vec<String>) {
        self.send(&Message::ServerActivate(ServerActivate {
            activities,
            active_roles: Some(roles),
        }))
        .await;
    }

    /// Reply to `time` as a server whose clock equals the client's.
    pub async fn answer_time(&mut self, time: &ClientTime) {
        let t = time.client_transmitted;
        self.send(&Message::ServerTime(ServerTime {
            client_transmitted: t,
            server_received: t,
            server_transmitted: t,
        }))
        .await;
    }

    /// Answer `client/time`, forward every other client message as JSON to
    /// `to_test`, and send the frames from `from_test`, until the client
    /// closes the connection or `to_test` is dropped.
    pub async fn serve(
        mut self,
        to_test: UnboundedSender<String>,
        mut from_test: UnboundedReceiver<WsMessage>,
    ) {
        let mut sending = true;
        loop {
            tokio::select! {
                json = self.recv_json() => {
                    let Some(json) = json else { break };
                    if let Ok(Message::ClientTime(time)) = serde_json::from_str(&json) {
                        self.answer_time(&time).await;
                    } else if to_test.send(json).is_err() {
                        break;
                    }
                }
                frame = from_test.recv(), if sending => match frame {
                    Some(frame) => self.send_frame(frame).await,
                    None => sending = false,
                },
                () = to_test.closed() => break,
            }
        }
    }
}
