# sendspin-rs
=======

> [!WARNING]  
> THIS IS A WIP. Please help!

Hyper-efficient Rust implementation of the [Sendspin Protocol](https://github.com/Sendspin/spec) for synchronized multi-room audio streaming.

## Features

- **Zero-copy audio pipeline** - Minimal allocations, maximum performance
- **Responsive concurrency** - Urgent controls are handled independently of audio preparation
- **Async I/O** - Efficient WebSocket handling with Tokio
- **Type-safe protocol** - Leverage Rust's type system for correctness

## Performance Notes

Latency depends on callback work, scheduling, queueing, device buffering, and audio
already submitted to the hardware. The library does not promise a fixed hardware
latency or a 10 ms end-to-end guarantee.

## Current Status

**Phase 1: Foundation** ✅ (Complete)
- Core audio types (Sample, AudioFormat, AudioBuffer)
- Protocol message types with serde serialization
- WebSocket client with handshake
- PCM decoder (16-bit and 24-bit)
- Clock synchronization (NTP-style)

**Current capabilities**
- Ordered role streams with responsive playback controls
- Dedicated player worker example with cpal output
- Async protocol I/O
- Typed player, artwork, and visualizer sessions


## Installing build dependencies

### Debian/Ubuntu/Mint

```
sudo apt install libasound2-dev
```

### Fedora/Centos

```
dnf install alsa-lib-devel
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
sendspin = "0.1"
```

### Basic Client Example

```rust
use sendspin::ProtocolClientBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ProtocolClientBuilder::builder()
        .client_id(uuid::Uuid::new_v4().to_string())
        .name("My Player".to_string())
        .build()
        .connect("ws://localhost:8080/sendspin")
        .await?;

    // Client is now connected and ready to receive audio

    Ok(())
}
```

See `examples/` directory for more examples.

### Receiving a stream

After connecting, split the client and attach the role receiver to the output that owns that role. Eager attachment starts the player session and keeps ordered events with the output owner:

```rust,no_run
use sendspin::audio::{AppliedControls, PlayerOutput, PlayerReceiver, PlayerSession};

fn attach_player<O: PlayerOutput>(
    receiver: PlayerReceiver,
    output: O,
) -> PlayerSession<O> {
    receiver.attach(
        output,
        AppliedControls {
            volume: 100,
            muted: false,
            static_delay_ms: 0,
        },
    )
}
```

- On the async side, await `session.recv()` and match `PlayerEvent::Audio(work)` before handing `work` to the output-preparation thread. Other event variants remain available to the owning worker.
- On that thread, prepare the audio and call `AudioWork::commit` with the prepared data. The library rejects work that has been superseded or closed; applications do not need sequence bookkeeping.
- For lazy playback, use `attach_lazy`; `begin_output_install` returns an `OutputInstallPermit` that installs `PlaybackControl`. See [`examples/player/main.rs`](examples/player/main.rs) for the lazy worker lifecycle.
- Eager and lazy attachment require a Tokio runtime. Keep output callbacks short, synchronous, nonblocking, and nonreentrant.
- Drop unused receivers to disable their delivery, and retain the connection guard until shutdown.

## Receiving API migration

| Previous approach | Current approach |
| --- | --- |
| Read audio chunks from `connection.audio` (`UnboundedReceiver`), with role configuration and controls in `messages` | Attach `connection.player` to a `PlayerOutput` or use `attach_lazy` for a worker-created output |
| Role controls and boundaries handled in the message loop | `PlayerSession` applies urgent controls independently and yields ordered `Start`, audio work, `Clear`, and `End` events |
| Read raw artwork/visualizer chunks from their receivers | Attach `connection.artwork` and `connection.visualizer` to their respective output traits |
| Separate messages, raw-data bookkeeping, and decoder lifetime managed in application code | Attach the role endpoint, prepare each item on an owning worker, and commit the returned `AudioWork`; this avoids stale enqueue work during session transitions |
| Unused role receivers retained | Drop unused endpoints; retain the connection guard until shutdown |

Callbacks are synchronous, short, nonblocking, serialized, and nonreentrant. `Clear` continues a stream; player `End` calls the player output's `clear()` and does not close the session. Artwork and visualizer `End` call their outputs' custom `end()` behavior. Format-only starts preserve buffered media, while unsupported sample rates or channel counts are rejected rather than resampled. `AudioWork::commit` reports `FormatMismatch` for one item and keeps the player session open; all other role output failures are terminal and require a new connection. There is no fixed hardware-latency or 10 ms guarantee.

For the complete migration, lazy output release/reinstall workflow, output error policy, and verification map, see [docs/role-sessions.md](docs/role-sessions.md).

## Architecture

See [docs/rust-thoughts.md](docs/rust-thoughts.md) for detailed architecture and implementation notes.

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Run examples
cargo run --example basic_client

# Build with optimizations
cargo build --release
```

### Verification (cargo-make)

```bash
# Install cargo-make (one time)
cargo install cargo-make

# Run the full verification suite
cargo make verify
```

`cargo make verify` runs: tests, Clippy, doc build, doctests, and formatting checks.

## Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_sample_from_i16

# Run with logging
RUST_LOG=debug cargo test
```

## License

MIT OR Apache-2.0

## Contributing

Contributions welcome! Please open an issue or PR.
