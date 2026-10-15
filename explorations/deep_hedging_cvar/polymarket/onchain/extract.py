#!/usr/bin/env python3
"""Pull decoded OrderFilled / OrdersMatched / FeeCharged events of both CTF Exchange V2 contracts from keyless
archive RPCs into per-day parquet partitions.

  python extract.py --start 2026-08-21 --end 2026-08-21 --market-set crypto-updown --max-event-start 2026-08-23
  python extract.py --start 2026-08-21T01:50 --end 2026-08-21T02:15 --slugs bnb-updown-5m-1787277900 --out data/val_bnb
  python extract.py --from-block 92381000 --to-block 92382000 --all

How it works
  * Block window: dates (inclusive, UTC) or datetimes (end exclusive) -> blocks via a cached block->time map
    (rpc.BlockClock, interpolation search over headers); or explicit --from-block/--to-block.
  * Work units: segments of --seg blocks on an absolute grid, clipped to day boundaries, so each segment belongs to
    exactly one UTC day. Workers (--workers) each take a segment and walk it with eth_getLogs (address = both
    exchanges, topic0 in the selected events). `tokenId` is not indexed, so every exchange log is downloaded
    (~190 logs/block) and filtered locally: a tx is kept if any of its OrderFilled/OrdersMatched logs touches a
    token in the market set; all its logs (incl. FeeCharged) are then decoded.
  * RPC robustness (rpc.py): endpoint pool with per-endpoint concurrency caps, failover to another endpoint on
    429/5xx/timeouts/truncated bodies with exponential cooldown, adaptive block span per endpoint (shrinks on
    result-limit errors, uses tenderly's suggested range, grows back slowly under a learned cap), empty-result
    and out-of-range sanity checks, and a random cross-endpoint recount (--verify-frac).
  * Checkpoint/resume: each finished segment is written atomically as data/<out>/<table>/date=D/part-LO-HI.parquet
    and appended to <out>/_ledger.jsonl with a fingerprint of the filter. Re-running skips finished segments.
    When a whole UTC day is done the parts are compacted into <table>/date=D/day.parquet and the day's stats
    (incl. the block-time interpolation check) are added to the ledger.
  * block_time is the `blockTimestamp` the RPC returns with each log (exact; all three default endpoints send it).
    Endpoints that omit it fall back to the interpolated clock. Each completed day records the error of
    interpolating from 2000-block header anchors against the exact times.

Outputs (<out> = data/<set>)
  fills/date=D/      block, block_time, tx_hash, log_index, exchange, order_hash, maker, taker, side, token_id,
                     maker_amount, taker_amount, fee, price, size, role       (amounts in 1e-6 USDC / shares)
  matches/date=D/    OrdersMatched: block, block_time, tx_hash, log_index, exchange, order_hash (taker order),
                     maker (taker order's maker), side, token_id, maker_amount, taker_amount, price, size
  fees/date=D/       FeeCharged: block, block_time, tx_hash, log_index, exchange, receiver, amount
  markets/date=D.parquet  token -> market map used as the filter for that day (market-set / slugs modes)
  orders/date=D/     (--calldata) original order size, limit price, creation time per matched order
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import common as C
import rpc

OF, OM, FC = C.TOPIC_OF, C.TOPIC_OM, C.TOPIC_FC
I64 = (1 << 63) - 1

SCHEMAS = {
    "fills": pa.schema([("block", pa.int64()), ("block_time", pa.int64()), ("tx_hash", pa.string()),
                        ("log_index", pa.int32()), ("exchange", pa.string()), ("order_hash", pa.string()),
                        ("maker", pa.string()), ("taker", pa.string()), ("side", pa.string()),
                        ("token_id", pa.string()), ("maker_amount", pa.int64()), ("taker_amount", pa.int64()),
                        ("fee", pa.int64()), ("price", pa.float64()), ("size", pa.float64()), ("role", pa.string())]),
    "matches": pa.schema([("block", pa.int64()), ("block_time", pa.int64()), ("tx_hash", pa.string()),
                          ("log_index", pa.int32()), ("exchange", pa.string()), ("order_hash", pa.string()),
                          ("maker", pa.string()), ("side", pa.string()), ("token_id", pa.string()),
                          ("maker_amount", pa.int64()), ("taker_amount", pa.int64()), ("price", pa.float64()),
                          ("size", pa.float64())]),
    "fees": pa.schema([("block", pa.int64()), ("block_time", pa.int64()), ("tx_hash", pa.string()),
                       ("log_index", pa.int32()), ("exchange", pa.string()), ("receiver", pa.string()),
                       ("amount", pa.int64())]),
}
RAW_COLS = {
    "fills": ["block", "block_time", "tx_hash", "log_index", "exchange", "order_hash", "maker", "taker", "side",
              "token_id", "maker_amount", "taker_amount", "fee", "role"],
    "matches": ["block", "block_time", "tx_hash", "log_index", "exchange", "order_hash", "maker", "side", "token_id",
                "maker_amount", "taker_amount"],
    "fees": ["block", "block_time", "tx_hash", "log_index", "exchange", "receiver", "amount"],
}


# ============================================================================================ decoding
class Decoder:
    """Filters raw logs to txs touching `tokens` (None = keep all) and decodes them into column lists."""

    def __init__(self, tokens: set[str] | None):
        self.keep = None if tokens is None else {format(int(t), "064x") for t in tokens}
        self.tokdec: dict[str, str] = {}

    def tok(self, h: str) -> str:
        s = self.tokdec.get(h)
        if s is None:
            s = self.tokdec[h] = str(int(h, 16))
        return s

    @staticmethod
    def new_cols():
        return {t: {c: [] for c in cs} for t, cs in RAW_COLS.items()}

    def run(self, logs: list[dict], cols: dict) -> int:
        if self.keep is not None:
            keep, ktx = self.keep, set()
            for L in logs:
                t0 = L["topics"][0]
                if (t0 == OF or t0 == OM) and L["data"][66:130] in keep:
                    ktx.add(L["transactionHash"])
            logs = [L for L in logs if L["transactionHash"] in ktx]
        F, M, X = cols["fills"], cols["matches"], cols["fees"]
        bc: dict[str, tuple[int, int]] = {}
        for L in logs:
            tp = L["topics"]
            t0 = tp[0].lower()
            bn = L["blockNumber"]
            b = bc.get(bn)
            if b is None:
                bts = L.get("blockTimestamp")
                b = bc[bn] = (int(bn, 16), int(bts, 16) if bts else -1)
            d = L["data"]
            addr = L["address"].lower()
            if t0 == OF:
                taker = "0x" + tp[3][-40:]
                F["block"].append(b[0]); F["block_time"].append(b[1])
                F["tx_hash"].append(L["transactionHash"]); F["log_index"].append(int(L["logIndex"], 16))
                F["exchange"].append(C.EXCHANGES.get(addr, addr)); F["order_hash"].append(tp[1])
                F["maker"].append("0x" + tp[2][-40:]); F["taker"].append(taker)
                F["side"].append("BUY" if int(d[2:66], 16) == 0 else "SELL")
                F["token_id"].append(self.tok(d[66:130]))
                F["maker_amount"].append(min(int(d[130:194], 16), I64))
                F["taker_amount"].append(min(int(d[194:258], 16), I64))
                F["fee"].append(min(int(d[258:322], 16), I64))
                F["role"].append("taker" if taker == addr else "maker")
            elif t0 == OM:
                M["block"].append(b[0]); M["block_time"].append(b[1])
                M["tx_hash"].append(L["transactionHash"]); M["log_index"].append(int(L["logIndex"], 16))
                M["exchange"].append(C.EXCHANGES.get(addr, addr)); M["order_hash"].append(tp[1])
                M["maker"].append("0x" + tp[2][-40:])
                M["side"].append("BUY" if int(d[2:66], 16) == 0 else "SELL")
                M["token_id"].append(self.tok(d[66:130]))
                M["maker_amount"].append(min(int(d[130:194], 16), I64))
                M["taker_amount"].append(min(int(d[194:258], 16), I64))
            elif t0 == FC:
                X["block"].append(b[0]); X["block_time"].append(b[1])
                X["tx_hash"].append(L["transactionHash"]); X["log_index"].append(int(L["logIndex"], 16))
                X["exchange"].append(C.EXCHANGES.get(addr, addr)); X["receiver"].append("0x" + tp[1][-40:])
                X["amount"].append(min(int(d[2:66], 16), I64))
        return len(logs)


def to_table(name: str, cols: dict) -> pa.Table:
    d = dict(cols)
    if name in ("fills", "matches"):
        ma = np.asarray(d["maker_amount"], dtype=np.float64)
        ta = np.asarray(d["taker_amount"], dtype=np.float64)
        buy = np.asarray(d["side"], dtype=object) == "BUY"
        sh = np.where(buy, ta, ma)
        usdc = np.where(buy, ma, ta)
        with np.errstate(divide="ignore", invalid="ignore"):
            d["price"] = np.where(sh > 0, usdc / sh, np.nan)
        d["size"] = sh / 1e6
    sch = SCHEMAS[name]
    return pa.table({f.name: pa.array(d[f.name], type=f.type) for f in sch}, schema=sch)


# ============================================================================================ ledger
class Ledger:
    def __init__(self, out: Path):
        self.path = out / "_ledger.jsonl"
        self.lock = threading.Lock()
        self.segs: dict[tuple, dict] = {}
        self.days: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue            # torn last line after a crash
                if r.get("type") == "seg":
                    self.segs[(r["date"], r["lo"], r["hi"], r["fp"])] = r
                elif r.get("type") == "day":
                    self.days[r["date"]] = r

    def append(self, rec: dict):
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if rec["type"] == "seg":
                self.segs[(rec["date"], rec["lo"], rec["hi"], rec["fp"])] = rec
            else:
                self.days[rec["date"]] = rec


def part_path(out: Path, table: str, date: str, lo: int, hi: int) -> Path:
    return out / table / f"date={date}" / f"part-{lo:010d}-{hi:010d}.parquet"


# ============================================================================================ fetching
class Extractor:
    def __init__(self, pool: rpc.Pool, clock: rpc.BlockClock, out: Path, events: list[str],
                 verify_frac: float = 0.02, max_tries: int = 20):
        self.pool, self.clock, self.out = pool, clock, out
        self.topics = [C.TOPICS[e] for e in events]
        self.verify_frac, self.max_tries = verify_frac, max_tries
        self.ledger = Ledger(out)
        self.lock = threading.Lock()
        self.tot = dict(blocks=0, logs=0, kept=0, segs=0, verify=0, verify_mismatch=0)

    def _getlogs(self, ep: rpc.Endpoint, a: int, b: int) -> list:
        logs = ep.call("eth_getLogs", [{"fromBlock": hex(a), "toBlock": hex(b), "address": C.EXCHANGE_ADDRS,
                                        "topics": [self.topics]}], timeout=120)
        if not isinstance(logs, list):
            raise rpc.TransientError(f"{ep.name}: non-list result")
        if logs:
            b0, b1 = int(logs[0]["blockNumber"], 16), int(logs[-1]["blockNumber"], 16)
            if b0 < a or b1 > b or any(L.get("removed") for L in logs):
                raise rpc.TransientError(f"{ep.name}: logs outside [{a},{b}] or removed")
        return logs

    def fetch_segment(self, lo: int, hi: int, dec: Decoder):
        cols = Decoder.new_cols()
        pos, n_logs, n_kept, tries, empties = lo, 0, 0, 0, 0
        used = set()
        ep = self.pool.acquire()
        try:
            while pos <= hi:
                span = min(int(ep.span), hi - pos + 1)
                try:
                    logs = self._getlogs(ep, pos, pos + span - 1)
                    if not logs and span >= 20 and empties < 2:
                        empties += 1      # ~190 logs/block: an empty multi-block answer is suspicious
                        raise rpc.TransientError(f"{ep.name}: empty result for {span} blocks")
                except rpc.RangeError as e:
                    if span <= ep.span_min:
                        e = rpc.TransientError(f"{ep.name}: range error at min span: {e}")
                    else:
                        sug = (e.suggest_hi - pos + 1) if e.suggest_hi and e.suggest_hi >= pos else None
                        ep.on_range_error(span, sug)
                        continue
                    logs = None
                    err = e
                except rpc.TransientError as e:
                    logs, err = None, e
                if logs is None:
                    tries += 1
                    if tries > self.max_tries:
                        raise RuntimeError(f"segment {lo}-{hi}: giving up at block {pos}: {err}")
                    self.pool.release(ep, ok=False)
                    prev, ep = ep, None
                    time.sleep(min(20.0, 0.5 * 2 ** min(tries, 6)) * random.random())
                    ep = self.pool.acquire(avoid=prev)
                    continue
                ep.on_ok(span)
                ep._stat(logs=len(logs))
                used.add(ep.name)
                tries = 0
                n_logs += len(logs)
                n_kept += dec.run(logs, cols)
                del logs
                pos += span
        finally:
            if ep is not None:
                self.pool.release(ep, ok=True)
        if self.verify_frac and random.random() < self.verify_frac:
            self._verify(lo, hi, cols, dec)
        return cols, n_logs, n_kept, sorted(used)

    def _verify(self, lo, hi, cols, dec):
        """Recount a random 10-block window on another endpoint and compare decoded rows."""
        a = random.randint(lo, max(lo, hi - 9))
        b = min(hi, a + 9)
        c2 = Decoder.new_cols()
        dec.run(self.pool.request(lambda e: self._getlogs(e, a, b)), c2)
        n1 = sum(1 for x in cols["fills"]["block"] if a <= x <= b)
        n2 = len(c2["fills"]["block"])
        with self.lock:
            self.tot["verify"] += 1
        if n1 != n2:
            with self.lock:
                self.tot["verify_mismatch"] += 1
            raise RuntimeError(f"verify mismatch in [{a},{b}]: {n1} vs {n2} fills")

    def run_segment(self, date: str, lo: int, hi: int, fp: str, dec: Decoder) -> dict:
        t0 = time.time()
        cols, n_logs, n_kept, used = self.fetch_segment(lo, hi, dec)
        counts = {}
        for name in SCHEMAS:
            tbl = to_table(name, cols[name])
            counts[name] = tbl.num_rows
            if tbl.num_rows:
                C.write_parquet_atomic(tbl, part_path(self.out, name, date, lo, hi), compression_level=3)
        rec = dict(type="seg", date=date, lo=lo, hi=hi, fp=fp, logs=n_logs, kept_logs=n_kept, **counts,
                   secs=round(time.time() - t0, 2), eps=used, t=int(time.time()))
        self.ledger.append(rec)
        with self.lock:
            self.tot["blocks"] += hi - lo + 1
            self.tot["logs"] += n_logs
            self.tot["kept"] += n_kept
            self.tot["segs"] += 1
        return rec

    def seg_done(self, date, lo, hi, fp) -> bool:
        r = self.ledger.segs.get((date, lo, hi, fp))
        if r is None:
            return False
        return all(r.get(n, 0) == 0 or part_path(self.out, n, date, lo, hi).exists() for n in SCHEMAS)

    # ---------------------------------------------------------------------------------- compaction
    def compact_day(self, date: str, lo: int, hi: int, fp: str, wall: float) -> dict:
        stats = dict(type="day", date=date, lo=lo, hi=hi, fp=fp, blocks=hi - lo + 1, wall_s=round(wall, 1))
        segs = [r for (d, a, b, f), r in self.ledger.segs.items() if d == date and f == fp]
        stats["logs"] = int(sum(r["logs"] for r in segs))
        exact = []
        for name in SCHEMAS:
            ddir = self.out / name / f"date={date}"
            parts = sorted(ddir.glob("part-*.parquet")) if ddir.exists() else []
            if parts:
                tbl = pa.concat_tables([pq.read_table(p) for p in parts])
                tbl = tbl.sort_by([("block", "ascending"), ("log_index", "ascending")])
            else:
                tbl = SCHEMAS[name].empty_table()
            bt = tbl.column("block_time").to_numpy()
            blk = tbl.column("block").to_numpy()
            miss = bt < 0
            if miss.any():          # endpoint did not send blockTimestamp: interpolate from headers
                self.clock.ensure_anchors(lo, hi)
                bt = bt.copy()
                bt[miss] = np.round(self.clock.interp(blk[miss])).astype(np.int64)
                tbl = tbl.set_column(1, "block_time", pa.array(bt, pa.int64()))
            stats[f"{name}_interp_time"] = int(miss.sum())
            exact.append(pd.DataFrame({"b": blk[~miss], "t": bt[~miss]}))
            stats[name] = tbl.num_rows
            C.write_parquet_atomic(tbl, ddir / "day.parquet", compression_level=6)
            for p in parts:
                p.unlink()
        stats["clock"] = self.clock_check(pd.concat(exact).drop_duplicates("b").sort_values("b"), lo, hi)
        self.clock.save()
        self.ledger.append(stats)
        return stats

    def clock_check(self, ex: pd.DataFrame, lo: int, hi: int, stride: int = 2000) -> dict:
        """Error of interpolating block times from headers every `stride` blocks vs the exact log timestamps."""
        if ex.empty:
            return {}
        self.clock.ensure_anchors(lo, hi, stride)
        err = self.clock.interp(ex.b.values, lo, hi, stride) - ex.t.values
        ae = np.abs(err)
        return dict(stride=stride, n_blocks=int(len(ae)), mean_abs_s=round(float(ae.mean()), 3),
                    p50_abs_s=round(float(np.median(ae)), 3), p99_abs_s=round(float(np.quantile(ae, 0.99)), 3),
                    max_abs_s=round(float(ae.max()), 3), frac_round_exact=round(float((np.round(err) == 0).mean()), 4),
                    mean_block_s=round(float(np.diff(ex.t.values).sum() / max(1, np.diff(ex.b.values).sum())), 4))


# ============================================================================================ planning
def plan(clock: rpc.BlockClock, args) -> list[tuple[str, int, int, bool]]:
    """-> [(date, lo, hi, full_day)] block windows, one per UTC day touched."""
    if args.from_block is not None:
        lo, hi = args.from_block, args.to_block
        d0, d1 = C.ts_date(clock.ts(lo)), C.ts_date(clock.ts(hi))
    else:
        t0 = C.utc_ts(args.start)
        t1 = C.utc_ts(args.end) + (86400 if len(args.end) == 10 else 0)
        head = clock.latest()
        lo = clock.block_at_or_after(t0, head)
        hi = clock.block_at_or_after(t1, head) - 1
        d0, d1 = C.ts_date(t0), C.ts_date(t1 - 1)
    head = clock.latest()
    out = []
    for d in C.date_range(d0, d1):
        a = C.utc_ts(d)
        da = clock.block_at_or_after(a, head)
        db = clock.block_at_or_after(a + 86400, head) - 1
        x, y = max(lo, da), min(hi, db)
        if x <= y:
            out.append((d, x, y, (x, y) == (da, db)))
    clock.save()
    return out


def segments(lo: int, hi: int, seg: int) -> list[tuple[int, int]]:
    out, a = [], lo
    while a <= hi:
        b = min(hi, (a // seg + 1) * seg - 1)
        out.append((a, b))
        a = b + 1
    return out


def day_tokens(args, date: str):
    """-> (token map DataFrame or None, fingerprint)."""
    import markets as M
    ev = ",".join(args.events)
    if args.all:
        return None, hashlib.sha1(f"all|{ev}".encode()).hexdigest()[:12]
    if args.tokens:
        p = Path(args.tokens)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, dtype=str)
        df["token_id"] = df.token_id.astype(str)
    elif args.slugs:
        df = M.Gamma().markets(args.slugs.split(","))
    else:
        t0 = C.utc_ts(date)
        mes = C.utc_ts(args.max_event_start) if args.max_event_start else None
        df = M.crypto_updown(t0, t0 + 86400, max_event_start=mes)
    toks = sorted(df.token_id.astype(str).unique())
    return df, hashlib.sha1(("|".join(toks) + "|" + ev).encode()).hexdigest()[:12]


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="YYYY-MM-DD (inclusive) or YYYY-MM-DDTHH:MM[:SS] UTC")
    ap.add_argument("--end", help="YYYY-MM-DD (inclusive) or datetime (exclusive)")
    ap.add_argument("--from-block", type=int)
    ap.add_argument("--to-block", type=int)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--market-set", choices=["crypto-updown"], default=None)
    g.add_argument("--slugs", help="comma-separated market slugs")
    g.add_argument("--tokens", help="parquet/csv with a token_id column")
    g.add_argument("--all", action="store_true", help="no market filter (~5M fills/day, ~10 GB JSON/day)")
    ap.add_argument("--max-event-start", help="crypto-updown: drop markets whose event starts at/after this date")
    ap.add_argument("--events", default="OrderFilled,OrdersMatched,FeeCharged")
    ap.add_argument("--out", help="output dir (default data/<set>)")
    ap.add_argument("--rpc", action="append", help="url[,conc[,span[,span_max[,rps[,max_batch]]]]]; repeatable (default: 3 keyless)")
    ap.add_argument("--workers", type=int, default=4, help="segments in flight (also capped by endpoint conc)")
    ap.add_argument("--seg", type=int, default=1000, help="checkpoint segment size in blocks")
    ap.add_argument("--verify-frac", type=float, default=0.02, help="fraction of segments recounted elsewhere")
    ap.add_argument("--calldata", action="store_true", help="also decode matchOrders calldata (orders/)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.from_block is None and not (args.start and args.end):
        ap.error("give --start/--end or --from-block/--to-block")
    if not (args.market_set or args.slugs or args.tokens or args.all):
        args.market_set = "crypto-updown"
    args.events = args.events.split(",")
    name = "all" if args.all else (args.market_set or ("slugs" if args.slugs else Path(args.tokens).stem))
    out = Path(args.out) if args.out else C.DATA / name
    out.mkdir(parents=True, exist_ok=True)

    pool = rpc.Pool.from_specs(args.rpc)
    clock = rpc.BlockClock(pool)
    ex = Extractor(pool, clock, out, args.events, args.verify_frac)
    days = plan(clock, args)
    cpu0, t_start = time.process_time(), time.time()

    work, meta = [], {}
    for date, lo, hi, full in days:
        tokmap, fp = day_tokens(args, date)
        if tokmap is not None:
            C.write_parquet_atomic(pa.Table.from_pandas(tokmap, preserve_index=False), out / "markets" / f"date={date}.parquet")
        dec = Decoder(None if tokmap is None else set(tokmap.token_id.astype(str)))
        dr = ex.ledger.days.get(date)
        if dr and dr["fp"] == fp and (dr["lo"], dr["hi"]) == (lo, hi):
            print(f"{date}: already complete ({dr.get('fills')} fills)")
            continue
        segs = segments(lo, hi, args.seg)
        todo = [(a, b) for a, b in segs if not ex.seg_done(date, a, b, fp)]
        meta[date] = dict(lo=lo, hi=hi, full=full, fp=fp, left=len(todo), t0=None)
        ntok = "all" if tokmap is None else f"{tokmap.token_id.nunique()} tokens"
        print(f"{date}: blocks {lo}-{hi} ({hi-lo+1}), {len(segs)} segs, {len(todo)} to do, filter {ntok}, fp {fp}")
        work += [(date, a, b, fp, dec) for a, b in todo]
        if not todo and full:
            meta[date]["left"] = 0
    if args.dry_run:
        return 0

    failed = []

    def finish_day(date):
        m = meta[date]
        if m["full"]:
            s = ex.compact_day(date, m["lo"], m["hi"], m["fp"], time.time() - (m["t0"] or t_start))
            print(f"{date}: compacted fills={s['fills']} matches={s['matches']} fees={s['fees']} logs={s['logs']} "
                  f"clock={s['clock']}", flush=True)
        if args.calldata:
            import calldata
            calldata.run_day(out, date, pool, clock)

    for date, m in meta.items():
        if m["left"] == 0:
            finish_day(date)
    last = time.time()
    with ThreadPoolExecutor(args.workers) as tp:
        futs = {}
        for w in work:
            futs[tp.submit(ex.run_segment, *w)] = w
            if meta[w[0]]["t0"] is None:
                meta[w[0]]["t0"] = time.time()
        for f in as_completed(futs):
            date, a, b = futs[f][:3]
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                failed.append((date, a, b, str(e)[:300]))
                print(f"FAILED {date} {a}-{b}: {e}", file=sys.stderr, flush=True)
                continue
            meta[date]["left"] -= 1
            if meta[date]["left"] == 0:
                finish_day(date)
            if time.time() - last > 30:
                last = time.time()
                el = last - t_start
                t = ex.tot
                print(f"  {t['segs']}/{len(work)} segs  {t['blocks']/el:.0f} blocks/s  {t['logs']/el:.0f} logs/s  "
                      f"cpu {100*(time.process_time()-cpu0)/el:.0f}%  "
                      f"MB {sum(e.stats['bytes'] for e in pool.eps)/1e6:.0f}  "
                      f"spans {[int(e.span) for e in pool.eps]}", flush=True)
    clock.save()
    el = time.time() - t_start
    t = ex.tot
    summary = dict(wall_s=round(el, 1), cpu_s=round(time.process_time() - cpu0, 1), blocks=t["blocks"],
                   logs=t["logs"], kept_logs=t["kept"], blocks_per_s=round(t["blocks"] / max(el, 1e-9), 1),
                   logs_per_s=round(t["logs"] / max(el, 1e-9)), verify=t["verify"],
                   verify_mismatch=t["verify_mismatch"], failed_segments=len(failed), endpoints=pool.stats())
    print(json.dumps(summary, indent=1))
    with open(out / "_runs.jsonl", "a") as f:
        f.write(json.dumps(dict(summary, argv=sys.argv[1:], t=int(time.time()))) + "\n")
    return 1 if failed else 0


def load(out: Path | str, table: str = "fills", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Read a table back (all days, or inclusive date range)."""
    root = Path(out) / table
    dirs = sorted(p for p in root.glob("date=*") if p.is_dir())
    if start:
        dirs = [p for p in dirs if start <= p.name[5:] <= (end or start)]
    frames = [pd.read_parquet(p).assign(date=p.name[5:]) for p in dirs if any(p.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else SCHEMAS[table].empty_table().to_pandas()


if __name__ == "__main__":
    sys.exit(main())
