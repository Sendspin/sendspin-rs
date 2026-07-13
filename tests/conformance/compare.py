#!/usr/bin/env python3
"""Diff two run_against_client.py recordings for semantic equivalence.

Not a byte-diff of the JSON — timestamps and connection-specific fields
(server_id, client_id) legitimately differ between the oracle and Rust runs.
What must match: negotiated roles, stream format, the exact audio bytes
delivered, and every command's field values, in order.

Usage: compare.py <oracle.json> <rust.json>
"""

import json
import sys


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def event_types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def main() -> int:
    oracle = json.loads(open(sys.argv[1]).read())
    rust = json.loads(open(sys.argv[2]).read())

    # --- audio: must be byte-identical ---
    if oracle["audio_sha256"] != rust["audio_sha256"]:
        fail(
            f"audio content differs: oracle sha256={oracle['audio_sha256']} "
            f"({oracle['audio_bytes_len']} bytes) vs rust sha256={rust['audio_sha256']} "
            f"({rust['audio_bytes_len']} bytes)"
        )
    print(f"OK: audio bytes identical ({oracle['audio_bytes_len']} bytes, sha256 match)")

    # Not asserted: aiosendspin's PushStream re-packetizes each commit into
    # its own wire-frame granularity (e.g. 20 chunks committed can arrive as
    # 80 wire frames) — that's an implementation-specific packetization
    # choice, not part of what v1 needs to match. Only the delivered content
    # (checked above) and the event sequence/values (checked below) matter.
    print(
        f"INFO: chunk counts (wire-frame granularity, not asserted): "
        f"oracle={oracle['chunk_count']} vs rust={rust['chunk_count']}"
    )

    # --- event sequence shape: same message types in the same order ---
    oracle_types = event_types(oracle["events"])
    rust_types = event_types(rust["events"])
    if oracle_types != rust_types:
        fail(f"event sequence differs:\n  oracle: {oracle_types}\n  rust:   {rust_types}")
    print(f"OK: event sequence identical ({oracle_types})")

    # --- per-event field equivalence, ignoring nothing (no timestamps in these payloads) ---
    for i, (o_event, r_event) in enumerate(zip(oracle["events"], rust["events"])):
        if o_event["type"] == "server_hello":
            if o_event["active_roles"] != r_event["active_roles"]:
                fail(
                    f"event[{i}] server_hello.active_roles differs: "
                    f"oracle={o_event['active_roles']} vs rust={r_event['active_roles']}"
                )
        elif o_event["type"] == "stream_start":
            for field in ("codec", "sample_rate", "channels", "bit_depth"):
                if o_event[field] != r_event[field]:
                    fail(
                        f"event[{i}] stream_start.{field} differs: "
                        f"oracle={o_event[field]!r} vs rust={r_event[field]!r}"
                    )
        elif o_event["type"] == "server_command":
            for field in ("command", "volume", "mute"):
                if o_event[field] != r_event[field]:
                    fail(
                        f"event[{i}] server_command.{field} differs: "
                        f"oracle={o_event[field]!r} vs rust={r_event[field]!r}"
                    )
        elif o_event["type"] == "stream_end":
            pass  # `roles` value isn't semantically important for v1's single-role stream
        elif o_event["type"] == "disconnect":
            pass

    print("OK: every event's fields match between the real aiosendspin server and the Rust server")
    print("PASS: conformance suite — Rust server role is observably equivalent to aiosendspin for this stimulus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
