# Role sessions and receiving migration

This guide describes the receiving API introduced by the role-session migration. It is intended for applications that previously consumed raw audio, artwork, or visualizer messages.

## API migration

| Before | Now |
| --- | --- |
| Read audio chunks from `connection.audio` (`UnboundedReceiver`), with role configuration and controls in `messages` | Take `connection.player` (or the equivalent split/managed endpoint), then use `PlayerReceiver::attach` or `attach_lazy`; prepare each item on an owning worker and commit the returned `AudioWork` |
| Read raw artwork and visualizer chunks from their receivers | Attach `connection.artwork` and `connection.visualizer` to their respective `ArtworkOutput` and `VisualizerOutput` implementations |
| Filter stream controls and boundaries in the message loop | Receive ordered `Start`, data work, `Clear` where defined, and `End` events from the role session; urgent controls are applied by the session driver |
| Coordinate decoded audio with stream configuration and buffered output in application code | Commit the returned `AudioWork` for that item; the session rejects work made obsolete by a boundary, release, or newer receipt |
| Use `messages` for role-owned envelopes | Reserve `messages` for other parsed envelopes and diagnostics; role stream data and controls are delivered through role endpoints |

The receiver is ownership-bearing and is attached once. A session requires a Tokio runtime. Output callbacks are synchronous, short, nonblocking, serialized, and nonreentrant. They must not call back into the session.

## Player workflow

`PlayerSession::recv` yields configuration, audio work, and boundary events. Preparation may happen on a dedicated worker, but receiving does not wait for decoding, output creation, or network acknowledgements. Commit the prepared buffer through the `AudioWork` returned by `recv`; do not reproduce invalidation or locking logic in the application.

`Clear` invalidates already-issued work and continues the stream. Player `End` calls the player output's `clear()`; artwork and visualizer `End` call their outputs' custom `end()` behavior. `End` is not session shutdown, and a later `Start` can reactivate the stream. `session.shutdown().await`, dropping the session, transport loss, or an explicit terminal output failure closes the session. A closed session's `recv` returns `None`; it does not drain trailing playback events.

A codec-only `Start` preserves buffered media, while each item is prepared using the format in effect when it was received. The built-in player advertises one sample rate and channel count. There is no resampling or seamless device-format transition: unsupported rate/channel input is rejected rather than silently queued in the wrong format.

`AudioWork::commit` reports `OutputError::FormatMismatch` for one item and leaves the player session open. Callback, player-boundary, setter, and installation errors are terminal; the first error is retained, cleanup is attempted once, and later work is rejected. Artwork and visualizer output errors are terminal as well. Recovery requires a new connection or endpoint; this API does not promise in-connection recovery of a failed device.

The library does not promise a fixed hardware latency or a 10 ms end-to-end guarantee. Callback duration, scheduling, queueing, device buffering, and already-submitted hardware samples affect latency.

## Lazy output and external-source handoff

Use `PlayerReceiver::attach_lazy` when the output is created by a worker after the first usable configuration or audio. Controls are available before installation. The worker reserves and installs a `PlaybackControl` through `begin_output_install`; installation replays the retained controls and current boundary state before admitting new work.

To release a locally owned device without closing the role endpoint:

1. Call `LazyPlayerSession::release_output()`.
2. Wait for the output worker to acknowledge and drop the old device/`SyncedPlayer`.
3. Enter the external-source state only after the old worker has dropped its output.
4. When returning, resume with the current state: `exit_external_source` may carry a full `PlayerState`; read the OS/device state as needed, while the session retains controls and the boundary/configuration needed for reinstall. The worker retains the latest start/configuration information for the replacement output.
5. Reserve a new install slot and install the replacement output from the retained configuration.
6. Resume receiving after the replacement is installed.

Release fences old work and install permits. Audio received while detached is dropped; configuration and boundaries remain ordered, and unbounded media queues should explicitly drop unused or detached audio rather than drain indefinitely. Releasing a failed session is rejected; this is a handoff operation, not device-failure recovery. Keep the connection guard alive until shutdown so the transport can close the session cleanly.

## Artwork and visualizer outputs

Implement the explicit output traits and attach the corresponding receiver. Prepared payloads are application-defined and committed once through the work item. Artwork channels and visualizer data are independent of player preparation. A role can be dropped when unused; dropping one endpoint does not disable the other roles or the connection.

## Controls and ownership

Volume, mute, and static delay are reported through the player session's retained `controls()` watch receiver and applied independently of media preparation. Read or clone the snapshot before an `.await` when constructing a handoff state; `controls()` returns a `watch::Receiver<AppliedControls>`, not an `AppliedControls` value. Values are normalized by the library (volume at most 100 and static delay at most 5000 ms). Hardware or OS mixer changes must be read through the platform's APIs; this library tracks only its own software controls. A connection or manager owns the transport and its guard; a session owns its output coordination. Do not treat manager displacement or a dropped role receiver as permission to keep using a session after it closes.

## Verification map

The repository's production tests cover the migration paths below:

- Role event ordering and format association: `queued_pcm_to_opus_decodes_in_order` and `queued_opus_to_pcm_decodes_in_order` in `tests/role_events.rs`.
- Public client routing: `test_message_router_forwards_binary_audio`, `test_message_router_forwards_binary_artwork`, and `test_message_router_forwards_binary_visualizer` in `tests/protocol_client.rs`.
- External-source lifecycle: `test_enter_external_source_sends_external_source_state`, `test_exit_external_source_sends_synchronized_state`, and `test_exit_external_source_sends_full_player_state` in `tests/protocol_client.rs`.
- Listener acceptance and handoff: `test_listen_role_session_delivers_player_events`, `test_listen_accepts_and_drives_handshake`, and `test_listen_path_mismatch_rejects_but_listener_survives` in `tests/protocol_listener.rs`; `manager_role_session_handoff_and_displacement_fences_audio`, `test_playback_displaces_discovery_incumbent`, and `test_playback_displaces_playback` in `tests/connection_manager.rs`.

Manual status remains user-relevant: the tests use fake outputs and do not read back a physical device or OS mixer. A hardware test must separately verify platform mixer readback and the release/worker-drop/external-source handoff on the target device.
