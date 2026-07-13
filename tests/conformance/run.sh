#!/bin/bash
# Conformance suite: proves the new server role (src/server/) is observably
# equivalent to the real, unmodified aiosendspin[server] reference
# implementation for a fixed stimulus — same audio bytes delivered, same
# stream format, same volume/mute command values, same event ordering.
#
# Both servers are driven through the *identical* canned stimulus
# (fixtures/stimulus.pcm, chunked and timed the same way) by oracle_server.py
# and examples/conformance_server.rs respectively, then the exact same real
# aiosendspin protocol client (run_against_client.py) is pointed at each in
# turn to record what it actually observed. compare.py diffs the two
# recordings for semantic equivalence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV="$SCRIPT_DIR/.venv"
ORACLE_PORT="${ORACLE_PORT:-18927}"
RUST_PORT="${RUST_PORT:-18928}"
OUT_DIR="$(mktemp -d)"

ORACLE_PID=""
RUST_PID=""

cleanup() {
  [ -n "$ORACLE_PID" ] && kill "$ORACLE_PID" 2>/dev/null || true
  [ -n "$RUST_PID" ] && kill "$RUST_PID" 2>/dev/null || true
  rm -rf "$OUT_DIR"
}
trap cleanup EXIT

if [ ! -x "$VENV/bin/python3" ]; then
  echo "FAIL: $VENV not set up — run: python3 -m venv $VENV && $VENV/bin/pip install 'aiosendspin[server]==6.1.1'" >&2
  exit 1
fi

echo "--- building conformance_server example ---"
(cd "$CRATE_DIR" && cargo build --example conformance_server --quiet)

echo "--- run 1: real aiosendspin server (oracle) ---"
"$VENV/bin/python3" "$SCRIPT_DIR/oracle_server.py" "$ORACLE_PORT" &
ORACLE_PID=$!
sleep 1
"$VENV/bin/python3" "$SCRIPT_DIR/run_against_client.py" "$ORACLE_PORT" "$OUT_DIR/oracle.json" 6
wait "$ORACLE_PID" || true
ORACLE_PID=""

echo "--- run 2: new Rust server role ---"
"$CRATE_DIR/target/debug/examples/conformance_server" "$RUST_PORT" &
RUST_PID=$!
sleep 1
"$VENV/bin/python3" "$SCRIPT_DIR/run_against_client.py" "$RUST_PORT" "$OUT_DIR/rust.json" 6
wait "$RUST_PID" || true
RUST_PID=""

echo "--- comparing recordings ---"
"$VENV/bin/python3" "$SCRIPT_DIR/compare.py" "$OUT_DIR/oracle.json" "$OUT_DIR/rust.json"
