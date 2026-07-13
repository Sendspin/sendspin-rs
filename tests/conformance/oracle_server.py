#!/usr/bin/env python3
"""Conformance-suite oracle: drives a real, unmodified aiosendspin server
through a fixed, canned stimulus — accept one client, stream
fixtures/stimulus.pcm chunk by chunk, send a volume command mid-stream and a
mute command later, then end the stream and close.

examples/conformance_server.rs drives our new Rust server role through the
*identical* stimulus (same fixture file, same chunk size, same timing, same
command values at the same chunk indices). run_against_client.py is pointed
at each server in turn to record what a real protocol client actually
observes; compare.py diffs the two recordings. See tests/conformance/README.md.

Usage: oracle_server.py <port>
"""

import asyncio
import sys
from pathlib import Path

from aiohttp import ClientSession
from aiosendspin.server import ClientAddedEvent, SendspinServer
from aiosendspin.server.audio import AudioFormat as ServerAudioFormat

SAMPLE_RATE = 48000
CHANNELS = 2
BIT_DEPTH = 16
CHUNK_MS = 100
CHUNK_BYTES = SAMPLE_RATE * CHUNK_MS // 1000 * CHANNELS * (BIT_DEPTH // 8)
VOLUME_AT_CHUNK = 5
VOLUME_VALUE = 42
MUTE_AT_CHUNK = 12
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "stimulus.pcm"


async def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8927
    stimulus = FIXTURE_PATH.read_bytes()
    chunks = [stimulus[i : i + CHUNK_BYTES] for i in range(0, len(stimulus), CHUNK_BYTES)]

    ready = asyncio.Event()
    holder: dict[str, object] = {}

    async with ClientSession() as session:
        loop = asyncio.get_running_loop()
        server = SendspinServer(loop, "conformance-oracle", "Conformance Oracle", session)

        def on_event(_server: SendspinServer, event: object) -> None:
            if isinstance(event, ClientAddedEvent):
                loop.create_task(setup(event.client_id))

        async def setup(client_id: str) -> None:
            client = server.get_client(client_id)
            if client is None:
                return
            # Hello handshake completes asynchronously after ClientAddedEvent
            # fires; poll for it (same pattern as spikes/03-sendspin-pushstream).
            for _ in range(50):
                try:
                    client.info  # noqa: B018 - raises AssertionError until hello is processed
                    break
                except AssertionError:
                    await asyncio.sleep(0.1)
            else:
                print("FAIL: client hello never completed", file=sys.stderr)
                return
            holder["stream"] = client.group.start_stream()
            holder["player"] = client.role("player@v1")
            ready.set()

        # Must be registered before start_server(): start_server() opens the
        # TCP listener partway through its own execution (mDNS advertising
        # setup afterward is slow enough, ~hundreds of ms, that a real
        # external client process can connect and fire client/added before
        # start_server() even returns). An event fired with no listener
        # registered yet is simply lost — this bit us during development as
        # `ready` never getting set despite the handshake completing fine.
        server.add_event_listener(on_event)
        await server.start_server(
            port=port, host="127.0.0.1", advertise_addresses=["127.0.0.1"], discover_clients=False
        )

        try:
            await asyncio.wait_for(ready.wait(), timeout=30)
        except asyncio.TimeoutError:
            print("FAIL: no client connected within 30s", file=sys.stderr)
            return 1

        stream = holder["stream"]
        player = holder["player"]
        fmt = ServerAudioFormat(
            sample_rate=SAMPLE_RATE, bit_depth=BIT_DEPTH, channels=CHANNELS, sample_type="int"
        )

        for i, chunk in enumerate(chunks):
            if i == VOLUME_AT_CHUNK and player is not None:
                player.set_volume(VOLUME_VALUE)
            if i == MUTE_AT_CHUNK and player is not None:
                player.set_mute(True)
            stream.prepare_audio(chunk, fmt)
            await stream.commit_audio()
            await asyncio.sleep(CHUNK_MS / 1000)

        # Give the last chunks time to arrive before tearing down.
        await asyncio.sleep(1.0)
        stream.stop()  # sends stream/end — mirrors conformance_server.rs's explicit send_stream_end()
        await asyncio.sleep(0.2)
        await server.close()

    print(f"oracle done: sent {len(chunks)} chunks, {len(stimulus)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
