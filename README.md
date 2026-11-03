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

## Current Status

**Phase 1: Foundation** ✅ (Complete)
- Core audio types (Sample, AudioFormat, AudioBuffer)
- Protocol message types with serde serialization
- WebSocket client with handshake
- PCM decoder (16-bit and 24-bit)
- Clock synchronization (NTP-style)

**Current capabilities**
- Ordered role streams with responsive playback controls
- Minimal player example with cpal output
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

See the [role-session guide](docs/role-sessions.md) for the current receiving API and the
`examples/` directory for complete clients. A minimal player session starts by attaching an
output and returns a `PlayerSession`:

```rust,no_run
fn attach<O: sendspin::audio::PlayerOutput>(
    receiver: sendspin::audio::PlayerReceiver,
    output: O,
) -> sendspin::audio::PlayerSession<O> {
    receiver.attach(output)
}
```

The guide covers output recovery, controls, and prepared audio work.

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
