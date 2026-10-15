"""The recorder: discovery -> websocket shards -> in-memory books -> 1 Hz on-change parquet rows,
with 60 s REST reconciles, a raw event log, incremental trades and per-market finalization.

Everything runs on one asyncio loop in one process; parquet writes and hourly compaction run in a
worker thread. Snapshot timing is causal: a row stamped with second t contains exactly the events
received before t (a frame received at or after t first closes tick t, then is applied).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import resource
import shutil
import signal
import time
import uuid
from collections import Counter

import orjson

from . import __version__
from .book import EV_LEVEL, EV_SNAP, OrderBook, diff_books, mirror, parse_best, parse_levels, replay
from .config import CLOB, Config
from .discovery import MarketInfo, discover, fetch_market_final, resolution
from .http import Http, HttpError
from .metadata import market_row
from .schema import MARKET_QC_SCHEMA, MARKETS_SCHEMA, RECONCILE_SCHEMA, TRADES_SCHEMA
from .sinks import BookRowBuffer, PartSink, RawLog, TableSink, dataset_files, hour_key
from .trades import LegSet, build_trade_rows, fetch_all, fetch_since, trade_rows_to_table
from .watchdog import EXIT_NO_DATA, EXIT_RSS, CoverageLog, LoopWatchdog, sd_notify
from .ws import Shard

log = logging.getLogger("polyrec")
REC_BUF_MAX = 200_000


class Token:
    __slots__ = ("id", "market", "idx", "outcome", "book", "meta", "shard", "next_rec", "rec_inflight",
                 "rec_trigger", "last_rec_start", "n_rec", "n_mismatch", "n_bba", "tick", "dead")

    def __init__(self, tid: str, market: "Market", idx: int, outcome: str):
        self.id, self.market, self.idx, self.outcome = tid, market, idx, outcome
        self.book = OrderBook()
        i = market.info
        self.meta = (i.slug, i.coin, i.contract_length, tid, outcome)
        self.shard: Shard | None = None
        self.next_rec = 0.0
        self.rec_inflight = False
        self.rec_trigger = "periodic"
        self.last_rec_start = 0.0
        self.n_rec = self.n_mismatch = self.n_bba = 0
        self.tick = None
        self.dead = False


class Market:
    def __init__(self, info: MarketInfo, src, now: float):
        self.info = info
        start = info.open if info.open is not None else now
        self.sub_at = start - src.pre_open_s
        self.write_until = info.end + src.post_close_s
        self.unsub_at = info.end + src.linger_s
        self.finalize_at = None
        self.state = "pending"
        self.tokens = [Token(t, self, k, o) for k, (t, o) in enumerate(zip(info.tokens, info.outcomes))]
        self.live = False
        self.first_live = None
        self.n_rows = 0
        self.n_disconnects = 0
        self.taker, self.allf = LegSet(), LegSet()
        self.next_trade_poll = 0.0
        self.trade_inflight = False
        self.resolved_asset = None
        self.finalize_tries = 0


class Stats(Counter):
    pass


class Recorder:
    def __init__(self, cfg: Config, run_id: str | None = None, duration_s: float | None = None):
        self.cfg = cfg
        self.run_id = run_id or time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
        self.duration_s = duration_s
        out = cfg.out
        out.mkdir(parents=True, exist_ok=True)
        lvl = cfg.output.compression_level
        self.books = BookRowBuffer(out, self.run_id, lvl)
        self.recon_sink = TableSink(out, "reconcile", RECONCILE_SCHEMA, self.run_id, lvl)
        self.trades_sink = TableSink(out, "trades", TRADES_SCHEMA, self.run_id, lvl, dictionary_cols=["slug"])
        self.markets_sink = TableSink(out, "markets", MARKETS_SCHEMA, self.run_id, lvl)
        self.qc_sink = TableSink(out, "market_qc", MARKET_QC_SCHEMA, self.run_id, lvl)
        self.sinks: list[PartSink] = [self.books, self.recon_sink, self.trades_sink, self.markets_sink, self.qc_sink]
        self.raw = RawLog(out, self.run_id, cfg.output.raw_zstd_level, cfg.output.raw_enabled)
        self.cov = CoverageLog(out, self.run_id)
        self.markets: dict[str, Market] = {}
        self.tokens: dict[str, Token] = {}
        self.by_condition: dict[str, Market] = {}
        self.finalized: dict[str, float] = {}       # slug -> time (pruned after 2 days)
        self.recorded_slugs: set[str] = set()       # slugs with book rows from earlier runs (recovery)
        self.shards: list[Shard] = []
        self.dirty: set[Token] = set()
        self.next_tick = int(time.time()) + 1
        self.st = Stats()
        self.started = time.time()
        self.last_data = time.time()
        self.stop_event: asyncio.Event | None = None
        self.exit_code = 0
        self.watchdog = LoopWatchdog(cfg.health.loop_stall_exit_s)
        self._compact_lock = asyncio.Lock()
        self._rate_prev = (time.time(), Stats())
        self.rates: dict = {}
        self.loop_lag_ms = 0.0
        self.last_discovery_ok = 0.0
        self.discovery_errors = 0
        self.last_error = ""
        self.http: Http | None = None
        self._shard_seq = 0
        self.keep_hashes = cfg.output.raw_keep_order_hashes
        self._last_flush = time.time()
        self._last_raw_flush = time.time()

    # ================================================================== frame handling (hot path)
    def on_frame(self, shard: Shard, raw: bytes, now: float):
        st = self.st
        st["frames"] += 1
        st["bytes"] += len(raw)
        if now >= self.next_tick:
            self._flush_tick(now)
        if raw == b"PONG":
            st["pong"] += 1
            return
        try:
            d = orjson.loads(raw)
        except orjson.JSONDecodeError:
            st["parse_errors"] += 1
            if st["parse_errors"] <= 5:
                log.warning("unparseable frame on shard %d: %r", shard.id, raw[:200])
            return
        self.last_data = now
        r_ms = int(now * 1000)
        if type(d) is list:
            for e in d:
                if type(e) is dict:
                    self._on_event(shard, e, now, r_ms)
        elif type(d) is dict:
            self._on_event(shard, d, now, r_ms)

    def _on_event(self, shard: Shard, e: dict, now: float, r_ms: int):
        st = self.st
        et = e.get("event_type")
        ts = e.get("timestamp")
        ts = int(ts) if ts else 0
        if ts > shard.last_server_ts:
            shard.last_server_ts = ts
        if et == "price_change":
            st["price_change"] += 1
            tokens = self.tokens
            for it in e.get("price_changes") or ():
                tok = tokens.get(it.get("asset_id"))
                if not self.keep_hashes:
                    it.pop("hash", None)
                if tok is None:
                    st["pc_untracked"] += 1
                    continue
                b = tok.book
                is_bid = it.get("side") == "BUY"
                p = float(it["price"])
                s = float(it["size"])
                st["pc_items"] += 1
                if b.apply_level(is_bid, p, s, ts):
                    self.dirty.add(tok)
                if b.rec_buf is not None and len(b.rec_buf) < REC_BUF_MAX:
                    b.rec_buf.append((EV_LEVEL, ts, is_bid, p, s))
                if b.initialized:
                    ebb = it.get("best_bid")
                    if ebb is not None and (parse_best(ebb, True) != b.best_bid
                                            or parse_best(it.get("best_ask"), False) != b.best_ask):
                        tok.n_bba += 1
                        st["bba_mismatch"] += 1
                        if (self.cfg.reconcile.on_bba_mismatch and not tok.rec_inflight
                                and now - tok.last_rec_start >= self.cfg.reconcile.min_gap_s):
                            tok.next_rec = 0.0
                            tok.rec_trigger = "bba"
                    st["bba_checks"] += 1
            e.pop("market", None)
        elif et == "book":
            st["book"] += 1
            tok = self.tokens.get(e.get("asset_id"))
            if tok is not None:
                b = tok.book
                bids, asks = parse_levels(e.get("bids")), parse_levels(e.get("asks"))
                if b.rec_buf is not None:
                    b.rec_buf.append((EV_SNAP, ts, dict(bids), dict(asks), None))
                was = b.initialized
                if b.set_snapshot(bids, asks, ts):
                    self.dirty.add(tok)
                if not was:
                    self._token_live(tok, now)
        elif et == "best_bid_ask":
            st["best_bid_ask"] += 1
            e.pop("market", None)
        elif et == "last_trade_price":
            st["last_trade_price"] += 1
        elif et == "tick_size_change":
            st["tick_size_change"] += 1
            tok = self.tokens.get(e.get("asset_id"))
            if tok is not None:
                tok.tick = e.get("new_tick_size")
        elif et == "market_resolved":
            st["market_resolved"] += 1
            m = self.by_condition.get(e.get("market"))
            if m is not None:
                m.resolved_asset = e.get("winning_asset_id")
        elif et == "new_market":
            st["new_market"] += 1
        else:
            st["other_events"] += 1
        if self.raw.enabled:
            e["r"] = r_ms
            e["c"] = shard.id
            self.raw.write(orjson.dumps(e), now)

    def _flush_tick(self, now: float):
        """Close tick self.next_tick: one row per changed, live book, stamped with that second."""
        t = self.next_tick
        if self.dirty:
            mx = 0
            touched = set()
            for tok in self.dirty:
                b = tok.book
                if not b.initialized or not b.dirty:
                    continue
                m = tok.market
                if t > m.write_until or m.state != "subscribed":
                    b.dirty = False
                    continue
                bp, bs, ap, asz = b.ladders(mx)
                self.books.append(tok.meta, m.info.end, t, bp, bs, ap, asz, b.best_bid, b.best_ask,
                                  b.crossed, b.reconciled)
                b.dirty = False
                b.reconciled = False
                m.n_rows += 1
                self.st["rows"] += 1
                if b.crossed:
                    self.st["rows_crossed"] += 1
                touched.add(m)
            self.dirty.clear()
            for m in touched:                      # Up/Down mirror consistency (quality metric)
                a, z = m.tokens[0].book, m.tokens[1].book
                if a.initialized and z.initialized:
                    mb, ma = mirror(z.bids, z.asks)
                    self.st["mirror_checks"] += 1
                    if diff_books(a.bids, a.asks, mb, ma, 1e-6) is not None:
                        self.st["mirror_mismatch"] += 1
        self.next_tick = int(now) + 1

    # ================================================================== liveness / coverage
    def _token_live(self, tok: Token, now: float):
        m = tok.market
        if not m.live and all(t.book.initialized for t in m.tokens) and m.state == "subscribed":
            m.live = True
            if m.first_live is None:
                m.first_live = now
            self.cov.write("live", now, slug=m.info.slug)

    def _market_down(self, m: Market, now: float, why: str):
        if m.live:
            m.live = False
            self.cov.write("down", now, slug=m.info.slug, why=why)

    def on_connect(self, shard: Shard, assets):
        self.st["connects"] += 1
        self.on_control(shard, "_connect", assets)

    def on_disconnect(self, shard: Shard, reason: str, was_connected: bool):
        now = time.time()
        self.st["disconnects"] += int(was_connected)
        for a in list(shard.assets):
            tok = self.tokens.get(a)
            if tok is None:
                continue
            tok.book.initialized = False
            if was_connected:
                m = tok.market
                if m.live:
                    m.n_disconnects += 1
                self._market_down(m, now, reason[:80])
        self.on_control(shard, "_disconnect", [], reason=reason[:200])

    def on_control(self, shard: Shard, kind: str, assets, **kw):
        if self.raw.enabled:
            now = time.time()
            self.raw.write(orjson.dumps({"event_type": kind, "r": int(now * 1000), "c": shard.id,
                                         "assets_ids": list(assets), **kw}), now)

    # ================================================================== market lifecycle
    def _register(self, info: MarketInfo, src, now: float):
        if info.slug in self.markets or info.slug in self.finalized:
            return
        m = Market(info, src, now)
        if now >= m.unsub_at:
            # already over: only finalize it if an earlier run recorded it (crash/restart recovery)
            if info.slug not in self.recorded_slugs or now - info.end > self.cfg.trades.finalize_lookback_s:
                return
            m.state = "ended"
            m.finalize_at = now + 5
        self.markets[info.slug] = m
        self.by_condition[info.condition_id] = m
        log.debug("registered %s (open %s, end %s, state %s)", info.slug, info.open, info.end, m.state)

    def _subscribe(self, m: Market, now: float):
        per = max(1, self.cfg.ws.markets_per_conn)
        shard = next((s for s in self.shards if not s.closing and len(s.markets) < per), None)
        if shard is None:
            self._shard_seq += 1
            shard = Shard(self._shard_seq, self.cfg.ws, self)
            shard.task = asyncio.get_running_loop().create_task(shard.run())
            self.shards.append(shard)
        for k, t in enumerate(m.tokens):
            t.shard = shard
            t.next_rec = now + 20 + 40 * ((hash(t.id) & 0xFFFF) / 0xFFFF)  # stagger the 60 s cycle
            self.tokens[t.id] = t
        m.state = "subscribed"
        shard.add(m.info.slug, [t.id for t in m.tokens])
        self.st["subscribed_markets"] += 1

    def _unsubscribe(self, m: Market, now: float, why="end"):
        shard = m.tokens[0].shard
        self._market_down(m, now, why)
        for t in m.tokens:
            self.tokens.pop(t.id, None)
            self.dirty.discard(t)
            t.book.rec_buf = None
        if shard is not None:
            shard.remove(m.info.slug, [t.id for t in m.tokens])
            if not shard.markets:
                shard.close()
                self.shards.remove(shard)
        m.state = "ended"
        m.finalize_at = m.info.end + self.cfg.trades.settle_delay_s

    def _schedule(self, now: float):
        for m in list(self.markets.values()):
            if m.state == "pending":
                if now >= m.unsub_at:
                    m.state, m.finalize_at = "ended", m.info.end + self.cfg.trades.settle_delay_s
                elif now >= m.sub_at:
                    self._subscribe(m, now)
            elif m.state == "subscribed":
                if now >= m.unsub_at:
                    self._unsubscribe(m, now)
            elif m.state == "ended":
                if now >= m.finalize_at:
                    m.state = "finalizing"
                    asyncio.get_running_loop().create_task(self._finalize(m))
            elif m.state == "done":
                self.markets.pop(m.info.slug, None)
                self.by_condition.pop(m.info.condition_id, None)
                self.finalized[m.info.slug] = now
        if len(self.finalized) > 20000 or int(now) % 3600 == 0:
            cut = now - 2 * 86400
            self.finalized = {s: t for s, t in self.finalized.items() if t > cut}

    async def _finalize(self, m: Market):
        """Full trade sweep + final gamma metadata -> trades / markets / market_qc rows."""
        i = m.info
        now = time.time()
        try:
            g = await fetch_market_final(self.http, i.slug) or dict(i.gamma, eventMetadata=i.event_meta)
            win, _ = resolution(g)
            if win is None and m.resolved_asset is None and now - i.end < 1800 and m.finalize_tries < 12:
                m.finalize_tries += 1                       # not resolved yet: try again shortly
                m.state, m.finalize_at = "ended", now + 120
                return
            if win is None and m.resolved_asset in i.tokens:   # WS market_resolved, gamma lagging
                k = i.tokens.index(m.resolved_asset)
                g = dict(g, closed=True, outcomePrices=orjson.dumps(["1", "0"] if k == 0 else ["0", "1"]).decode())
            trunc = False
            if self.cfg.trades.enabled:
                tk, t1 = await fetch_all(self.http, i.condition_id, True)
                al, t2 = await fetch_all(self.http, i.condition_id, False)
                n_inc = len(m.allf)
                m.taker.merge(tk)
                m.allf.merge(al)
                trunc = t1 or t2
                tbl = trade_rows_to_table(build_trade_rows(i.slug, m.taker, m.allf))
                self.trades_sink.append_table(tbl, time.time())
                self.st["trade_legs"] += tbl.num_rows
            else:
                n_inc = 0
            row = market_row(i.slug, i.coin, i.contract_length, g, len(m.taker), len(m.allf), trunc)
            self.markets_sink.append(row, time.time())
            toks = m.tokens
            self.qc_sink.append(dict(
                slug=i.slug, run_id=self.run_id, finalized_at=int(time.time() * 1e6),
                first_live=None if m.first_live is None else int(m.first_live * 1e6),
                n_rows=m.n_rows, n_reconcile=sum(t.n_rec for t in toks), n_mismatch=sum(t.n_mismatch for t in toks),
                n_bba_mismatch=sum(t.n_bba for t in toks), n_disconnects=m.n_disconnects,
                n_trades_incremental=n_inc, trades_truncated=trunc, resolved=row["winner_up"] is not None),
                time.time())
            self.st["finalized"] += 1
            m.state = "done"
            log.info("finalized %s: rows=%d legs=%d winner_up=%s", i.slug, m.n_rows, len(m.allf), row["winner_up"])
        except Exception as e:  # noqa: BLE001
            m.finalize_tries += 1
            self.last_error = f"finalize {i.slug}: {e}"
            log.warning("finalize %s failed (%d): %s", i.slug, m.finalize_tries, e)
            if m.finalize_tries < 20:
                m.state, m.finalize_at = "ended", time.time() + 60 * min(m.finalize_tries, 10)
            else:
                m.state = "done"
                self.st["finalize_failed"] += 1

    # ================================================================== background loops
    async def _ticker(self):
        """1 Hz: close ticks when no frames arrive, run the market schedule, kick the watchdogs."""
        while True:
            now = time.time()
            delay = max(0.0, self.next_tick - now) + 0.003
            t_sleep = time.monotonic()
            await asyncio.sleep(delay)
            self.loop_lag_ms = max(0.0, (time.monotonic() - t_sleep - delay) * 1000)
            now = time.time()
            if now >= self.next_tick:
                self._flush_tick(now)
            self.watchdog.kick()
            try:
                self._schedule(now)
            except Exception as e:  # noqa: BLE001
                log.exception("schedule error: %s", e)
            if int(now) % 10 == 0:
                sd_notify("WATCHDOG=1")

    async def _discovery_loop(self):
        while True:
            now = time.time()
            n_new = 0
            for src in self.cfg.sources:
                try:
                    found = await discover(self.http, src, now)
                except Exception as e:  # noqa: BLE001
                    self.discovery_errors += 1
                    self.last_error = f"discovery {src.name}: {e}"
                    log.warning("discovery %s failed: %s", src.name, e)
                    continue
                self.last_discovery_ok = time.time()
                for info in found:
                    if info.slug not in self.markets:
                        self._register(info, src, now)
                        n_new += info.slug in self.markets
            if n_new:
                log.info("discovery: %d new markets (%d tracked)", n_new, len(self.markets))
            await asyncio.sleep(self.cfg.discovery_interval_s)

    async def _reconcile_loop(self):
        rc = self.cfg.reconcile
        sem = asyncio.Semaphore(rc.concurrency)
        while True:
            await asyncio.sleep(0.5)
            now = time.time()
            due = [t for t in self.tokens.values()
                   if t.next_rec <= now and not t.rec_inflight and not t.dead
                   and t.shard is not None and t.shard.connected and t.market.state == "subscribed"]
            if not due:
                continue
            for t in due:
                t.rec_inflight = True
                t.last_rec_start = now
                t.book.rec_buf = []
            if rc.mode == "books":
                for k in range(0, len(due), rc.batch):
                    asyncio.get_running_loop().create_task(self._reconcile_batch(due[k:k + rc.batch], sem))
            else:
                for t in due:
                    asyncio.get_running_loop().create_task(self._reconcile_batch([t], sem))

    async def _reconcile_batch(self, toks: list[Token], sem: asyncio.Semaphore):
        rc = self.cfg.reconcile
        t_req = time.time()
        err = None
        books = {}
        try:
            async with sem:
                if len(toks) == 1 and rc.mode != "books":
                    d = await self.http.get(f"{CLOB}/book", params={"token_id": toks[0].id}, tries=2)
                    books = {str(d.get("asset_id")): d}
                else:
                    ds = await self.http.post(f"{CLOB}/books", json=[{"token_id": t.id} for t in toks], tries=2)
                    books = {str(d.get("asset_id")): d for d in ds}
        except HttpError as e:
            err = e
            if e.status == 404 and len(toks) == 1:
                toks[0].dead = True                         # no order book any more (market closed)
        except Exception as e:  # noqa: BLE001
            err = e
        rtt = (time.time() - t_req) * 1000
        now = time.time()
        if self.raw.enabled:
            for d in books.values():
                self.raw.write(orjson.dumps({"event_type": "rest_book", "r": int(now * 1000), **d}), now)
        if not err:
            await asyncio.sleep(rc.grace_s)
        for t in toks:
            try:
                d = books.get(t.id)
                if err is not None or d is None:
                    self._recon_row(t, "error", None, rtt, 0, None, None, error=str(err or "missing in /books")[:300])
                    self.st["reconcile_error"] += 1
                else:
                    self._compare(t, d, rtt)
            except Exception as e:  # noqa: BLE001
                log.exception("reconcile compare %s: %s", t.id, e)
            finally:
                t.book.rec_buf = None
                t.rec_inflight = False
                t.next_rec = time.time() + (rc.interval_s if err is None else 15)
                t.rec_trigger = "periodic"

    def _compare(self, tok: Token, d: dict, rtt: float):
        b = tok.book
        if tok.market.state != "subscribed" or tok.id not in self.tokens:
            return
        rest_ts = int(d.get("timestamp") or 0)
        rb, ra = parse_levels(d.get("bids")), parse_levels(d.get("asks"))
        buf = b.rec_buf or []
        cb, ca = replay(rb, ra, buf, rest_ts)
        n_rep = sum(1 for ev in buf if ev[1] >= rest_ts)
        if not b.initialized:
            # no stream snapshot yet (e.g. a subscribe without initial dump): seed from REST + replay
            if tok.shard is None or not tok.shard.connected:
                self._recon_row(tok, "skipped", rest_ts, rtt, n_rep, None, (cb, ca))
                return
            if b.set_snapshot(cb, ca, max(rest_ts, b.last_ts)):
                self.dirty.add(tok)
            self.st["reconcile_seeded"] += 1
            self._recon_row(tok, "seeded", rest_ts, rtt, n_rep, None, (cb, ca))
            self._token_live(tok, time.time())
            return
        tok.n_rec += 1
        self.st["reconcile"] += 1
        diff = diff_books(b.bids, b.asks, cb, ca, self.cfg.reconcile.size_tol)
        if diff is None:
            self._recon_row(tok, "ok", rest_ts, rtt, n_rep, None, (cb, ca))
            return
        tok.n_mismatch += 1
        self.st["reconcile_mismatch"] += 1
        if diff.top_differs:
            self.st["reconcile_mismatch_top"] += 1
        self._recon_row(tok, "mismatch", rest_ts, rtt, n_rep, diff, (cb, ca))
        if b.set_snapshot(cb, ca, b.last_ts):
            b.reconciled = True
            self.dirty.add(tok)

    def _recon_row(self, tok: Token, status: str, rest_ts, rtt, n_rep, diff, cand, error=None):
        b = tok.book
        cb, ca = cand if cand else ({}, {})
        self.recon_sink.append(dict(
            checked_at=int(time.time() * 1e6), market_slug=tok.market.info.slug, token_id=tok.id,
            outcome=tok.outcome, trigger=tok.rec_trigger, status=status, rest_ts=rest_ts,
            stream_ts=b.last_ts or None, rtt_ms=rtt, n_replayed=n_rep,
            n_bid_diff=diff.n_bid_diff if diff else 0, n_ask_diff=diff.n_ask_diff if diff else 0,
            max_size_diff=diff.max_size_diff if diff else 0.0, top_differs=diff.top_differs if diff else False,
            stream_best_bid=b.best_bid, stream_best_ask=b.best_ask,
            rest_best_bid=max(cb) if cb else None, rest_best_ask=min(ca) if ca else None, error=error),
            time.time())

    async def _trades_loop(self):
        tc = self.cfg.trades
        sem = asyncio.Semaphore(tc.concurrency)

        async def poll(m: Market):
            try:
                async with sem:
                    for legs, taker_only in ((m.taker, True), (m.allf, False)):
                        since = legs.max_ts - tc.overlap_s if legs.max_ts else 0
                        rows = await fetch_since(self.http, m.info.condition_id, taker_only, since, tc.page)
                        self.st["trade_rows_polled"] += len(rows)
                        legs.merge(rows)
            except Exception as e:  # noqa: BLE001
                self.st["trade_poll_errors"] += 1
                log.debug("trade poll %s: %s", m.info.slug, e)
            finally:
                m.trade_inflight = False
                m.next_trade_poll = time.time() + tc.poll_interval_s

        while True:
            await asyncio.sleep(1.0)
            now = time.time()
            for m in list(self.markets.values()):
                if m.state == "subscribed" and now >= m.next_trade_poll and not m.trade_inflight and now >= (m.info.open or 0):
                    m.trade_inflight = True
                    asyncio.get_running_loop().create_task(poll(m))

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(1.0)
            now = time.time()
            if now - self._last_raw_flush >= self.cfg.output.raw_flush_interval_s:
                self._last_raw_flush = now
                try:
                    self.raw.flush()
                except Exception as e:  # noqa: BLE001
                    log.error("raw log flush: %s", e)
            big = self.books.n >= self.cfg.output.max_buffer_rows
            if big or now - self._last_flush >= self.cfg.output.flush_interval_s \
                    or self.books.pending:
                self._last_flush = now
                await self.flush_all()
                await self.compact_finished_hours()

    async def flush_all(self):
        books = self.books.take_all()
        tables = [(s, s.take()) for s in self.sinks if isinstance(s, TableSink)]

        def work():
            for cols in books:
                self.books.write_cols(cols)
            for s, items in tables:
                s.write_items(items)
        try:
            await asyncio.to_thread(work)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"flush: {e}"
            log.exception("flush failed: %s", e)

    async def compact_finished_hours(self, include_current: bool = False):
        cur = hour_key(time.time())
        async with self._compact_lock:
            for s in self.sinks:
                for h in s.part_hours():
                    if h < cur or include_current:
                        try:
                            await asyncio.to_thread(s.compact, h)
                        except Exception as e:  # noqa: BLE001
                            log.exception("compaction %s %s failed: %s", s.name, h, e)

    async def _status_loop(self):
        hc = self.cfg.health
        last_summary = time.time()
        while True:
            await asyncio.sleep(hc.status_interval_s)
            now = time.time()
            self.cov.write("hb", now)
            try:
                self._update_rates(now)
                status = self.status(now)
                tmp = self.cfg.out / ".status.json.tmp"
                tmp.write_bytes(orjson.dumps(status, option=orjson.OPT_INDENT_2))
                os.replace(tmp, self.cfg.out / "status.json")
            except Exception as e:  # noqa: BLE001
                log.error("status write: %s", e)
                status = {}
            if now - last_summary >= hc.summary_interval_s and status:
                last_summary = now
                r = status["rates_per_s"]
                q = status["quality"]
                log.info("conns %d/%d markets live %d/%d | frames %.0f/s %.2f MB/s rows %.1f/s | reconcile %d "
                         "mismatch %.2f%% | bba mismatch %.3f%% | rss %d MB lag %.0f ms",
                         status["ws"]["connected"], status["ws"]["shards"], status["markets"]["live"],
                         status["markets"]["subscribed"], r["frames"], r["bytes"] / 1e6, r["rows"],
                         q["reconcile_checks"], 100 * q["reconcile_mismatch_rate"], 100 * q["bba_mismatch_rate"],
                         status["process"]["rss_mb"], self.loop_lag_ms)
            # guards
            free_mb = status.get("disk", {}).get("free_mb", 1e9) if status else 1e9
            if free_mb < hc.min_free_disk_mb and not self.raw.paused:
                log.error("low disk (%d MB free): pausing the raw event log", free_mb)
                self.raw.paused = True
            elif free_mb > 1.5 * hc.min_free_disk_mb and self.raw.paused:
                self.raw.paused = False
            rss = status.get("process", {}).get("rss_mb", 0) if status else 0
            if rss > hc.max_rss_mb:
                log.critical("RSS %d MB above limit %d MB: restarting", rss, hc.max_rss_mb)
                self.request_stop(EXIT_RSS)
            if self.tokens and now - self.last_data > hc.no_data_exit_s:
                log.critical("no market data for %.0fs with %d subscribed tokens: restarting",
                             now - self.last_data, len(self.tokens))
                self.request_stop(EXIT_NO_DATA)

    def _update_rates(self, now):
        t0, prev = self._rate_prev
        dt = max(now - t0, 1e-6)
        keys = ("frames", "bytes", "price_change", "pc_items", "book", "best_bid_ask", "last_trade_price",
                "rows", "reconcile", "reconcile_mismatch", "bba_mismatch")
        self.rates = {k: (self.st[k] - prev[k]) / dt for k in keys}
        self._rate_prev = (now, Stats(self.st))

    def status(self, now: float) -> dict:
        st = self.st
        try:
            with open("/proc/self/statm") as f:
                rss_mb = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        except OSError:
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        ru = resource.getrusage(resource.RUSAGE_SELF)
        du = shutil.disk_usage(self.cfg.out)
        live = [m for m in self.markets.values() if m.live]
        n_rec = st["reconcile"]
        return {
            "version": __version__, "run_id": self.run_id, "pid": os.getpid(),
            "time": now, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "uptime_s": round(now - self.started, 1),
            "process": {"rss_mb": round(rss_mb, 1), "cpu_s": round(ru.ru_utime + ru.ru_stime, 1),
                        "cpu_pct_avg": round(100 * (ru.ru_utime + ru.ru_stime) / max(now - self.started, 1), 1),
                        "loop_lag_ms": round(self.loop_lag_ms, 1)},
            "ws": {"shards": len(self.shards), "connected": sum(s.connected for s in self.shards),
                   "connects": st["connects"], "disconnects": st["disconnects"],
                   "last_frame_age_s": round(now - self.last_data, 1),
                   "max_server_lag_s": round(max((now - s.last_server_ts / 1000 for s in self.shards
                                                  if s.connected and s.last_server_ts), default=0), 3)},
            "markets": {"tracked": len(self.markets), "subscribed": sum(m.state == "subscribed" for m in self.markets.values()),
                        "live": len(live), "tokens": len(self.tokens),
                        "pending": sum(m.state == "pending" for m in self.markets.values()),
                        "awaiting_finalize": sum(m.state in ("ended", "finalizing") for m in self.markets.values()),
                        "finalized": st["finalized"], "finalize_failed": st["finalize_failed"]},
            "rates_per_s": {k: round(v, 2) for k, v in self.rates.items()},
            "totals": dict(st),
            "quality": {
                "reconcile_checks": n_rec, "reconcile_mismatch": st["reconcile_mismatch"],
                "reconcile_mismatch_rate": st["reconcile_mismatch"] / n_rec if n_rec else 0.0,
                "reconcile_mismatch_top_of_book": st["reconcile_mismatch_top"],
                "reconcile_errors": st["reconcile_error"],
                "bba_checks": st["bba_checks"], "bba_mismatch": st["bba_mismatch"],
                "bba_mismatch_rate": st["bba_mismatch"] / st["bba_checks"] if st["bba_checks"] else 0.0,
                "mirror_checks": st["mirror_checks"],
                "mirror_mismatch_rate": st["mirror_mismatch"] / st["mirror_checks"] if st["mirror_checks"] else 0.0,
                "rows": st["rows"], "rows_crossed_rate": st["rows_crossed"] / st["rows"] if st["rows"] else 0.0,
                "parse_errors": st["parse_errors"]},
            "output": {"book_rows_written": self.books.rows_written, "book_bytes_parts": self.books.bytes_written,
                       "raw_lines": self.raw.lines, "raw_bytes_in": self.raw.bytes_in,
                       "raw_bytes_out": self.raw.bytes_out(), "raw_paused": self.raw.paused,
                       "buffered_rows": self.books.n},
            "disk": {"free_mb": du.free // 2**20, "used_pct": round(100 * du.used / du.total, 1)},
            "discovery": {"last_ok_age_s": round(now - self.last_discovery_ok, 1) if self.last_discovery_ok else None,
                          "errors": self.discovery_errors},
            "http": {"requests": self.http.n_requests if self.http else 0, "errors": self.http.n_errors if self.http else 0},
            "last_error": self.last_error,
        }

    # ================================================================== run / shutdown
    def request_stop(self, code: int = 0):
        if self.exit_code == 0:
            self.exit_code = code
        if self.stop_event is not None:
            self.stop_event.set()

    def _load_recorded_slugs(self):
        """Slugs with book rows in the last two days of output (for post-restart finalization)."""
        days = [time.strftime("%Y-%m-%d", time.gmtime(time.time() - k * 86400)) for k in (0, 1)]
        import pyarrow.parquet as pq
        for f in dataset_files(self.cfg.out, "books", days):
            try:
                self.recorded_slugs.update(pq.read_table(f, columns=["market_slug"]).column(0).unique().to_pylist())
            except Exception:  # noqa: BLE001
                pass
        for f in dataset_files(self.cfg.out, "markets", days):
            try:
                for s in pq.read_table(f, columns=["slug"]).column(0).to_pylist():
                    self.finalized[s] = time.time()
            except Exception:  # noqa: BLE001
                pass

    async def run(self) -> int:
        loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.request_stop, 0)
        self.watchdog.start()
        self._load_recorded_slugs()
        await self.compact_finished_hours()                    # leftovers of earlier runs
        self.cov.write("start")
        log.info("polyrec %s run %s -> %s (sources: %s)", __version__, self.run_id, self.cfg.out,
                 ", ".join(f"{s.name}:{s.slug_regex}" for s in self.cfg.sources))
        async with Http(self.cfg.http_timeout_s) as http:
            self.http = http
            tasks = [loop.create_task(c) for c in (self._ticker(), self._discovery_loop(), self._reconcile_loop(),
                                                   self._flush_loop(), self._status_loop())]
            if self.cfg.trades.enabled:
                tasks.append(loop.create_task(self._trades_loop()))
            sd_notify("READY=1")
            try:
                if self.duration_s:
                    try:
                        await asyncio.wait_for(self.stop_event.wait(), self.duration_s)
                    except asyncio.TimeoutError:
                        log.info("duration %.0fs reached", self.duration_s)
                else:
                    await self.stop_event.wait()
            finally:
                sd_notify("STOPPING=1")
                log.info("shutting down (exit code %d)", self.exit_code)
                await self._shutdown(tasks)
        return self.exit_code

    async def _shutdown(self, tasks):
        for t in tasks:
            t.cancel()
        now = time.time()
        for m in self.markets.values():
            if m.state == "subscribed":
                self._market_down(m, now, "shutdown")
        for s in list(self.shards):
            s.close()
            if s.task:
                s.task.cancel()
        await asyncio.gather(*tasks, *(s.task for s in self.shards if s.task), return_exceptions=True)
        self._flush_tick(time.time() if time.time() >= self.next_tick else self.next_tick)
        await self.flush_all()
        await self.compact_finished_hours()
        self.raw.close()
        self._update_rates(time.time())
        try:
            st = self.status(time.time())
            st["stopped"] = True
            (self.cfg.out / "status.json").write_bytes(orjson.dumps(st, option=orjson.OPT_INDENT_2))
        except Exception:  # noqa: BLE001
            pass
        self.cov.write("stop")
        self.cov.close()
        self.watchdog.stopped.set()
        log.info("stopped: rows=%d frames=%d reconcile=%d mismatch=%d", self.st["rows"], self.st["frames"],
                 self.st["reconcile"], self.st["reconcile_mismatch"])
