# Conformance suite: new server role vs. real aiosendspin

Proves the new server role (`src/server/`) is observably equivalent to the
real, unmodified Python `aiosendspin[server]` reference implementation — not
just "compiles and passes unit tests," but "a real Sendspin protocol client
sees the same thing from either server."

## How it works

1. **`oracle_server.py`** drives a real `aiosendspin.server.SendspinServer`
   through a fixed, canned stimulus: accept one client, stream
   `fixtures/stimulus.pcm` chunk by chunk, send a `volume` command mid-stream
   and a `mute` command later, end the stream, close.
2. **`../../examples/conformance_server.rs`** drives our new Rust
   `ServerListener`/`ServerConnection` through the *identical* stimulus — same
   fixture file, same chunk size, same timing, same command values at the same
   chunk indices (the constants at the top of each file must stay in sync).
3. **`run_against_client.py`** connects a real, unmodified aiosendspin
   *protocol client* (the same library Music Assistant and other Sendspin
   consumers speak against) to whichever server it's pointed at, and records
   every event it observes — `server/hello`, `stream/start`, audio bytes
   (hashed), `server/command`, `stream/end`, disconnect — as JSON.
4. **`compare.py`** diffs the two recordings for semantic equivalence: exact
   audio byte match (sha256), identical event ordering, identical field
   values per event. Deliberately *not* asserted: wire-frame chunk count —
   aiosendspin's `PushStream` re-packetizes each commit into its own
   granularity (20 commits arrive as 80 wire frames); that's an
   implementation detail, not part of what v1 needs to match.

## Running

```
python3 -m venv tests/conformance/.venv
tests/conformance/.venv/bin/pip install 'aiosendspin[server]==6.1.1'
bash tests/conformance/run.sh
```

Set `CONFORMANCE_DEBUG=1` when running the scripts directly (not through
`run.sh`) for verbose `aiosendspin` logging.

## A real bug this caught

The first version of `oracle_server.py` registered its `ClientAddedEvent`
listener *after* `await server.start_server(...)`. `start_server()` opens the
TCP listener partway through its own execution — mDNS advertising setup
afterward is slow enough (hundreds of ms) that a real external client process
routinely connected and fired the event *before `start_server()` had even
returned*, with zero listeners registered. The event was fired and silently
dropped; the handshake looked completely successful from every log line, and
`push_stream_ready.set()` just never happened. Register listeners before
`start_server()`, not after — a good instance of the general "the server is
doing meaningful async I/O internally before returning" trap.
