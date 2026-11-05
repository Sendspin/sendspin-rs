# Role sessions

This guide describes the current typed receiving API. A connection is split into role
sessions; each session owns delivery, ordering, output admission, and lifecycle status for
one role.

## Player session

After `client.split()`, `conn.player` is a `PlayerReceiver`. Eager attachment supplies an
output immediately and starts from the controls in the builder's initial player state:

```rust,no_run
use sendspin::audio::{PlayerOutput, PlayerReceiver, PlayerSession};

fn attach<O: PlayerOutput>(receiver: PlayerReceiver, output: O) -> PlayerSession<O> {
    receiver.attach(output)
}
```

`attach_empty()` creates a generic `PlayerSession<O>` without an output, and a
plain `PlayerSession` uses the built-in `PlaybackControl`. Call `begin_output_install`, then
install the output with the returned permit. The builder's initial player state seeds the
snapshot; commands received before attachment overlay it per field. Controls remain observable
and updatable while detached; installation replays retained controls before admitting work.
Audio work received while no output is installed can be committed after the next installation,
and committing it earlier returns `Obsolete`. `release_output` only invalidates work already
returned by `recv`.

## Decoding

Receive `PlayerEvent`s from the session and build a `StreamDecoder` for each start configuration.
It validates PCM, Opus, and FLAC formats and decodes each audio work item:
```rust
let decoder = StreamDecoder::new(&config)?;
let samples = decoder.decode(work.data())?;
```
Call `AudioWork::commit` with the decoded samples.
Work is one-shot: obsolete, closed, or output-rejected work returns the standard typed error and
does not require application sequence counters. Keep output callbacks synchronous, short,
nonblocking, serialized, and nonreentrant.

`recv` returns `PlayerEvent::OutputLost(error)` once when the player output was detached after
a recoverable output failure. Install a fresh output to recover. `Closed` means the session has
ended. A player `OutputError::Failed` or callback-related `CommitRejected::Output` failure
detaches the output and rejects admitted work until recovery; `FormatMismatch` rejects only that
item. An `InstallRejected::Failed` result consumes its permit and requires a fresh reservation.
Artwork and visualizer sessions treat output failures as terminal and report
`RoleStatus::Failed`. A retry is a new output installation, not a new connection, and uses the
retained stream configuration.

## Controls and state

`PlayerSession::update_controls(ControlUpdate)` is the local control authority for both
installed and detached output. It merges only the supplied fields; `set_controls(AppliedControls)`
replaces the complete local snapshot. The caller owns OS or hardware readback and reports it
through `update_controls`.

Automatic player-state reporting requires the negotiated player role and an initial
player-state object; otherwise state reporting is explicit. Fields set to `None` are omitted
from serialized messages, and external-source exit uses the caller’s explicit patch before
later control changes are reported.

During output loss, control changes are still published and reported, and they are applied
when an output is installed.

## Migration

| Older receiving shape | Current shape |
| --- | --- |
| `connection.audio` and separate unbounded chunks | Typed `player`, `artwork`, and `visualizer` role sessions |
| Configuration and controls in a general `messages` stream | Typed role events plus `PlayerSession` controls |
| Application-maintained ordering counters | One-shot `AudioWork`/role work and typed receipts |

The typed sessions own admission and lifecycle ordering; applications own preparation,
output-device ownership, and hardware readback.
