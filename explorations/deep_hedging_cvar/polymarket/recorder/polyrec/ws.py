"""CLOB market-channel websocket shards.

Protocol (docs.polymarket.com, market channel, checked 2026-10-15):
  * URL wss://ws-subscriptions-clob.polymarket.com/ws/market
  * first frame {"assets_ids": [...], "type": "market", "custom_feature_enabled": true}
    (initial_dump defaults to true: one `book` per asset; level defaults to 2)
  * later frames {"operation": "subscribe"|"unsubscribe", "assets_ids": [...]} change the set in place
  * application heartbeat: send text "PING" every 10 s, server answers "PONG"
  * events: book, price_change, last_trade_price, tick_size_change (+ best_bid_ask, new_market,
    market_resolved with custom features); a frame is one event object or a JSON array of them.
Measured here: the server closes a connection with 1013 "slow consumer: send buffer full" when one
connection carries many busy tokens, so markets are spread over small shards (default 2 markets).
"""
from __future__ import annotations

import asyncio
import logging
import random
import time

import orjson
import websockets

from .config import UA, WSConfig

log = logging.getLogger("polyrec.ws")


class Shard:
    def __init__(self, sid: int, cfg: WSConfig, sink):
        self.id, self.cfg, self.sink = sid, cfg, sink
        self.assets: set[str] = set()
        self.markets: set[str] = set()
        self.ws = None
        self.connected = False
        self.closing = False
        self.wake = asyncio.Event()
        self.last_rx = 0.0
        self.last_server_ts = 0
        self.connected_at = 0.0
        self.n_connects = 0
        self.n_disconnects = 0
        self.last_error = ""
        self.task: asyncio.Task | None = None

    # ------------------------------------------------------------------ membership
    def add(self, slug: str, assets: list[str]):
        new = [a for a in assets if a not in self.assets]
        self.markets.add(slug)
        self.assets.update(new)
        if new and self.ws is not None and self.connected:
            self._send_soon({"operation": "subscribe", "assets_ids": new,
                             "custom_feature_enabled": self.cfg.custom_features})
            self.sink.on_control(self, "_subscribe", new)
        self.wake.set()

    def remove(self, slug: str, assets: list[str]):
        gone = [a for a in assets if a in self.assets]
        self.markets.discard(slug)
        self.assets.difference_update(gone)
        if gone and self.ws is not None and self.connected:
            self._send_soon({"operation": "unsubscribe", "assets_ids": gone})
            self.sink.on_control(self, "_unsubscribe", gone)
        self.wake.set()

    def _send_soon(self, msg: dict):
        ws = self.ws

        async def _send():
            try:
                await ws.send(orjson.dumps(msg).decode())
            except Exception as e:  # noqa: BLE001  (reconnect path resubscribes everything)
                log.warning("shard %d send failed: %s", self.id, e)
        asyncio.get_running_loop().create_task(_send())

    def close(self):
        self.closing = True
        self.wake.set()
        if self.ws is not None:
            asyncio.get_running_loop().create_task(self.ws.close())

    # ------------------------------------------------------------------ connection loop
    async def run(self):
        backoff = self.cfg.backoff_min_s
        while not self.closing:
            if not self.assets:
                self.wake.clear()
                await self.wake.wait()
                continue
            t0 = time.time()
            reason = "?"
            try:
                async with websockets.connect(
                        self.cfg.url, ping_interval=None, max_queue=self.cfg.max_queue,
                        max_size=self.cfg.max_frame_bytes, open_timeout=self.cfg.open_timeout_s,
                        close_timeout=3, compression=None, user_agent_header=UA) as ws:
                    self.ws = ws
                    assets = sorted(self.assets)
                    await ws.send(orjson.dumps({"assets_ids": assets, "type": "market",
                                                "custom_feature_enabled": self.cfg.custom_features}).decode())
                    self.connected = True
                    self.connected_at = self.last_rx = time.time()
                    self.n_connects += 1
                    self.sink.on_connect(self, assets)
                    pinger = asyncio.create_task(self._pinger(ws))
                    try:
                        while True:
                            raw = await ws.recv(decode=False)
                            now = time.time()
                            self.last_rx = now
                            self.sink.on_frame(self, raw, now)
                            if backoff > self.cfg.backoff_min_s and now - self.connected_at > 60:
                                backoff = self.cfg.backoff_min_s
                    finally:
                        pinger.cancel()
            except asyncio.CancelledError:
                reason = "cancelled"
                raise
            except websockets.ConnectionClosed as e:
                rc = e.rcvd
                reason = f"closed {rc.code} {rc.reason}" if rc else f"closed ({e})"
            except Exception as e:  # noqa: BLE001  (DNS, TLS, timeouts, proxy, ...)
                reason = f"{type(e).__name__}: {e}"
            finally:
                was = self.connected
                self.connected = False
                self.ws = None
                if was:
                    self.n_disconnects += 1
                self.last_error = reason
                self.sink.on_disconnect(self, reason, was)
            if self.closing:
                break
            up = time.time() - t0
            if "1013" in reason and up > 5:
                backoff = self.cfg.backoff_min_s     # "try again later": reconnect promptly
            log.warning("shard %d (%d assets) disconnected after %.0fs: %s; retry in %.1fs",
                        self.id, len(self.assets), up, reason, backoff)
            await asyncio.sleep(backoff * (0.5 + random.random()))
            backoff = min(backoff * 2, self.cfg.backoff_max_s)

    async def _pinger(self, ws):
        """Heartbeat + dead-man: PING every interval; no frame for stale_timeout_s -> drop the socket."""
        try:
            while True:
                await asyncio.sleep(self.cfg.ping_interval_s)
                if time.time() - self.last_rx > self.cfg.stale_timeout_s:
                    log.warning("shard %d stale (%.0fs without frames); reconnecting", self.id,
                                time.time() - self.last_rx)
                    ws.transport.abort() if ws.transport is not None else None
                    return
                await ws.send("PING")
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            log.debug("pinger %d: %s", self.id, e)
