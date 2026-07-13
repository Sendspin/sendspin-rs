#!/usr/bin/env python3
"""Conformance-suite observer: connects a real, unmodified aiosendspin
protocol client (the same library Music Assistant and real ESP32-adjacent
devices speak against) to a Sendspin server and records every event it
receives as JSON.

Run once against oracle_server.py and once against examples/conformance_server.rs
(same canned stimulus on both) and diff the two recordings with compare.py.

Usage: run_against_client.py <port> <output.json> [settle_seconds]
"""

import asyncio
import hashlib
import json
import logging
import os
import sys

from aiosendspin.client.client import SendspinClient
from aiosendspin.models.player import ClientHelloPlayerSupport, SupportedAudioFormat
from aiosendspin.models.types import AudioCodec, PlayerCommand, Roles

if os.environ.get("CONFORMANCE_DEBUG"):
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s.%(msecs)03d %(name)s %(message)s", datefmt="%H:%M:%S")

SAMPLE_RATE = 48000
CHANNELS = 2
BIT_DEPTH = 16


async def main() -> int:
    port = int(sys.argv[1])
    out_path = sys.argv[2]
    settle_s = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0

    events: list[dict] = []
    audio_bytes = bytearray()
    chunk_count = 0

    player_support = ClientHelloPlayerSupport(
        supported_formats=[
            SupportedAudioFormat(
                codec=AudioCodec.PCM, channels=CHANNELS, sample_rate=SAMPLE_RATE, bit_depth=BIT_DEPTH
            )
        ],
        # Bytes, not chunks — must comfortably exceed one commit's worth of
        # audio or the server blocks waiting for the buffer to drain (see
        # spikes/03-sendspin-pushstream.md).
        buffer_capacity=SAMPLE_RATE * CHANNELS * (BIT_DEPTH // 8) * 2,
        supported_commands=[PlayerCommand.VOLUME, PlayerCommand.MUTE],
    )
    client = SendspinClient(
        client_id="conformance-observer",
        client_name="Conformance Observer",
        roles=[Roles.PLAYER],
        player_support=player_support,
    )

    def on_server_hello(hello: object) -> None:
        events.append({"type": "server_hello", "active_roles": list(hello.active_roles)})

    def on_stream_start(msg: object) -> None:
        cfg = msg.payload.player
        events.append(
            {
                "type": "stream_start",
                "codec": cfg.codec.value if cfg is not None else None,
                "sample_rate": cfg.sample_rate if cfg is not None else None,
                "channels": cfg.channels if cfg is not None else None,
                "bit_depth": cfg.bit_depth if cfg is not None else None,
            }
        )

    def on_audio_chunk(_server_timestamp_us: int, data: bytes, _fmt: object) -> None:
        nonlocal chunk_count
        chunk_count += 1
        audio_bytes.extend(data)

    def on_stream_end(roles: list[str] | None) -> None:
        events.append({"type": "stream_end", "roles": roles})

    def on_server_command(payload: object) -> None:
        player = payload.player
        events.append(
            {
                "type": "server_command",
                "command": player.command.value if player is not None else None,
                "volume": player.volume if player is not None else None,
                "mute": player.mute if player is not None else None,
            }
        )

    def on_disconnect() -> None:
        events.append({"type": "disconnect"})

    client.add_server_hello_listener(on_server_hello)
    client.add_stream_start_listener(on_stream_start)
    client.add_audio_chunk_listener(on_audio_chunk)
    client.add_stream_end_listener(on_stream_end)
    client.add_server_command_listener(on_server_command)
    client.add_disconnect_listener(on_disconnect)

    await client.connect(f"ws://127.0.0.1:{port}/sendspin")
    await asyncio.sleep(settle_s)
    await client.disconnect()

    result = {
        "events": events,
        "chunk_count": chunk_count,
        "audio_bytes_len": len(audio_bytes),
        "audio_sha256": hashlib.sha256(bytes(audio_bytes)).hexdigest(),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}: {chunk_count} chunks, {len(audio_bytes)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
