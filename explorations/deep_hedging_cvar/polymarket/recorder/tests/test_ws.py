"""Shard reconnect (1013 close), resubscribe, dynamic subscribe, PING/PONG and stale dead-man, against a
local websocket server that mimics the CLOB market channel."""
import asyncio
import json
import sys
import time
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from polyrec.config import WSConfig  # noqa: E402
from polyrec.ws import Shard  # noqa: E402


class Sink:
    def __init__(self):
        self.connects, self.disconnects, self.frames, self.controls = [], [], [], []

    def on_connect(self, shard, assets):
        self.connects.append(sorted(assets))

    def on_disconnect(self, shard, reason, was):
        self.disconnects.append((reason, was))

    def on_frame(self, shard, raw, now):
        self.frames.append(raw)

    def on_control(self, shard, kind, assets, **kw):
        self.controls.append((kind, list(assets)))


def book(a):
    return {"event_type": "book", "asset_id": a, "market": "0xm", "timestamp": "1", "hash": "h",
            "bids": [{"price": "0.4", "size": "1"}], "asks": [{"price": "0.6", "size": "1"}]}


def test_reconnect_resubscribe_dynamic_and_stale():
    received = []          # (conn_no, message)
    state = {"n": 0}

    async def handler(ws):
        state["n"] += 1
        k = state["n"]
        sub = json.loads(await ws.recv())
        received.append((k, sub))
        await ws.send(json.dumps([book(a) for a in sub["assets_ids"]]))
        if k == 1:
            await ws.send(json.dumps({"event_type": "price_change", "market": "0xm", "timestamp": "2",
                                      "price_changes": []}))
            await ws.close(1013, "slow consumer: send buffer full")
            return
        if k == 2:
            async for m in ws:
                received.append((k, m))
                if m == "PING":
                    await ws.send("PONG")
                    if sum(1 for c, x in received if c == 2 and x == "PING") >= 2:
                        await asyncio.sleep(5)        # go silent: no PONG, no data -> client dead-man
            return
        async for m in ws:                            # conn 3: healthy
            received.append((k, m))
            if m == "PING":
                await ws.send("PONG")

    async def main():
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            cfg = WSConfig(url=f"ws://127.0.0.1:{port}", ping_interval_s=0.2, stale_timeout_s=0.7,
                           backoff_min_s=0.05, backoff_max_s=0.2)
            sink = Sink()
            sh = Shard(1, cfg, sink)
            sh.add("m1", ["A", "B"])
            task = asyncio.create_task(sh.run())
            t0 = time.time()
            while state["n"] < 2 and time.time() - t0 < 5:
                await asyncio.sleep(0.02)
            while not sh.connected and time.time() - t0 < 5:
                await asyncio.sleep(0.02)
            sh.add("m2", ["C", "D"])                  # dynamic subscribe on the live connection
            while state["n"] < 3 and time.time() - t0 < 10:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.3)
            sh.close()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return sink

    sink = asyncio.run(main())
    subs = [(k, m) for k, m in received if isinstance(m, dict)]
    assert subs[0][1]["assets_ids"] == ["A", "B"] and subs[0][1]["type"] == "market"
    assert subs[0][1]["custom_feature_enabled"] is True
    assert subs[1][1]["assets_ids"] == ["A", "B"]                     # resubscribed after the 1013 close
    dyn = [json.loads(m) for k, m in received if k == 2 and isinstance(m, str) and m.startswith("{")]
    assert dyn and dyn[0]["operation"] == "subscribe" and dyn[0]["assets_ids"] == ["C", "D"]
    assert subs[2][1]["assets_ids"] == ["A", "B", "C", "D"]           # third connection: full set
    assert "1013" in sink.disconnects[0][0] and sink.disconnects[0][1]
    assert len(sink.connects) >= 3
    assert any(m == "PING" for k, m in received if k == 2)            # heartbeat sent
    assert b"PONG" in sink.frames
    assert ("_subscribe", ["C", "D"]) in sink.controls
