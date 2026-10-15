"""Trades from data-api GET /trades, collected incrementally while a market is recorded and swept in
full after it ends; output rows match data/raw/trades.parquet (see data/fetch_trades.py).

API facts (verified in data/README.md): newest first; `limit` capped at 10000 and `offset` > 10000
rejected, so one query reaches 20000 rows (split by side=BUY/SELL beyond that). takerOnly=true gives
one row per matched tx (the taker leg); takerOnly=false gives every leg (taker + makers).

There is no trade id, and a tx can legitimately contain identical legs, so legs are kept as a
multiset keyed by (tx, wallet, side, outcomeIndex, price, size, timestamp, asset); merging two
listings takes the max count per key, never the sum, so overlapping polls do not double count.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pyarrow as pa

from .config import DATA_API
from .http import HttpError
from .schema import TRADES_SCHEMA

log = logging.getLogger("polyrec.trades")
PAGE_CAP = 10000
MAX_OFFSET = 10000


def leg_key(r: dict) -> tuple:
    return (r["transactionHash"], r.get("proxyWallet") or "", r["side"], int(r["outcomeIndex"]),
            float(r["price"]), float(r["size"]), int(r["timestamp"]), str(r.get("asset") or ""))


class LegSet:
    """Multiset of trade legs for one market and one takerOnly mode."""

    def __init__(self):
        self.counts: Counter = Counter()
        self.max_ts = 0

    def merge(self, rows: list) -> int:
        c = Counter(leg_key(r) for r in rows)
        added = 0
        for k, n in c.items():
            if n > self.counts.get(k, 0):
                added += n - self.counts.get(k, 0)
                self.counts[k] = n
            if k[6] > self.max_ts:
                self.max_ts = k[6]
        return added

    def merge_set(self, other: "LegSet"):
        for k, n in other.counts.items():
            if n > self.counts.get(k, 0):
                self.counts[k] = n
        self.max_ts = max(self.max_ts, other.max_ts)

    def __len__(self):
        return sum(self.counts.values())

    def legs(self):
        for k, n in self.counts.items():
            for _ in range(n):
                yield k


async def fetch_since(http, cond: str, taker_only: bool, since_ts: int, page: int = 500) -> list:
    """Newest-first pages until a page reaches trades older than since_ts (incremental poll)."""
    rows, offset = [], 0
    while True:
        q = {"market": cond, "limit": page, "offset": offset, "takerOnly": "true" if taker_only else "false"}
        pg = await http.get(f"{DATA_API}/trades", params=q)
        rows.extend(pg)
        if len(pg) < page or not pg or min(int(r["timestamp"]) for r in pg) < since_ts:
            return rows
        if offset + page > MAX_OFFSET:
            return rows
        offset += page


async def _paged(http, cond: str, taker_only: bool, side: str | None):
    rows, offset, truncated = [], 0, False
    while True:
        q = {"market": cond, "limit": PAGE_CAP, "offset": offset, "takerOnly": "true" if taker_only else "false"}
        if side:
            q["side"] = side
        try:
            pg = await http.get(f"{DATA_API}/trades", params=q)
        except HttpError as e:
            if e.status == 400 and b"offset" in e.body:
                return rows, True
            raise
        rows.extend(pg)
        if len(pg) < PAGE_CAP:
            return rows, truncated
        if offset + PAGE_CAP > MAX_OFFSET:
            return rows, True
        offset += PAGE_CAP


async def fetch_all(http, cond: str, taker_only: bool):
    """Full history (same strategy as data/fetch_trades.py). Returns (rows, truncated)."""
    rows, trunc = await _paged(http, cond, taker_only, None)
    if not trunc:
        return rows, False
    rows, trunc_any = [], False
    for side in ("BUY", "SELL"):
        r, t = await _paged(http, cond, taker_only, side)
        rows.extend(r)
        trunc_any |= t
    return rows, trunc_any


def build_trade_rows(slug: str, taker: LegSet, allf: LegSet) -> list[tuple]:
    """Rows (slug, tx, timestamp, outcome_idx, side, price, size, is_taker) like fetch_trades.build_outputs."""
    taker_keys = Counter()
    for k in taker.legs():
        taker_keys[(k[0], k[1], k[2], k[3], k[4], k[5])] += 1
    rows = []
    for k in allf.legs():
        kk = (k[0], k[1], k[2], k[3], k[4], k[5])
        is_t = taker_keys.get(kk, 0) > 0
        if is_t:
            taker_keys[kk] -= 1
        rows.append((slug, k[0], k[6], k[3], 1 if k[2] == "BUY" else -1, k[4], k[5], is_t))
    # taker legs missing from the takerOnly=false listing (should be none)
    ts_of = {}
    for k in taker.legs():
        ts_of.setdefault((k[0], k[1], k[2], k[3], k[4], k[5]), k[6])
    for kk, n in taker_keys.items():
        for _ in range(n):
            rows.append((slug, kk[0], ts_of[kk], kk[3], 1 if kk[2] == "BUY" else -1, kk[4], kk[5], True))
    return rows


def rows_consistency(rows: list[tuple]) -> dict:
    """The checks data/features.py asserts, on build_trade_rows output for ONE market: one taker leg per
    tx, every taker tx has maker legs, maker sizes sum to the taker size. data-api indexes the
    takerOnly=false view a few minutes after the taker view, so a fresh market can fail this."""
    tk, mk, dup = {}, {}, 0
    for _, tx, _, _, _, _, sz, ist in rows:
        if ist:
            dup += tx in tk
            tk[tx] = sz
        else:
            mk[tx] = mk.get(tx, 0.0) + sz
    missing = sum(1 for tx in tk if tx not in mk)
    dev = max((abs(mk[tx] - tk[tx]) for tx in tk if tx in mk), default=0.0)
    return {"ok": dup == 0 and missing == 0 and dev < 0.01, "taker_dups": dup,
            "txs_without_maker_legs": missing, "max_abs_maker_sum_minus_taker": dev}


def trade_rows_to_table(rows: list[tuple]) -> pa.Table:
    """rows from build_trade_rows (any number of markets) -> table in the trades.parquet schema."""
    if not rows:
        return TRADES_SCHEMA.empty_table()
    slug, tx, ts, oi, side, px, sz, ist = (list(x) for x in zip(*rows))
    ts = np.asarray(ts, np.int64)
    oi = np.asarray(oi, np.int8)
    side = np.asarray(side, np.int8)
    ist = np.asarray(ist, bool)
    # aggressor side in the Up (outcome 0) frame, from the taker leg of each tx
    up_side = np.where(oi == 0, side, -side)
    agg = {}
    for t_, u, it in zip(tx, up_side, ist):
        if it and t_ not in agg:
            agg[t_] = int(u)
    aggressor = np.array([agg.get(t_, 0) for t_ in tx], np.int8)
    order = sorted(range(len(rows)), key=lambda i: (slug[i], ts[i], tx[i], not ist[i]))
    o = np.asarray(order, np.int64)
    slugs_sorted = [slug[i] for i in order]
    cats = sorted(set(slugs_sorted))
    code = {s: i for i, s in enumerate(cats)}
    txbuf = b"".join(bytes.fromhex(tx[i][2:] if tx[i].startswith("0x") else tx[i]) for i in order)
    return pa.table({
        "slug": pa.DictionaryArray.from_arrays(pa.array([code[s] for s in slugs_sorted], pa.int32()),
                                               pa.array(cats, pa.string())),
        "timestamp": pa.array(ts[o]),
        "outcome_idx": pa.array(oi[o]),
        "side": pa.array(side[o]),
        "price": pa.array(np.asarray(px, np.float64)[o].astype(np.float32)),
        "size": pa.array(np.asarray(sz, np.float64)[o].astype(np.float32)),
        "is_taker": pa.array(ist[o]),
        "aggressor_up": pa.array(aggressor[o]),
        "tx": pa.FixedSizeBinaryArray.from_buffers(pa.binary(32), len(order), [None, pa.py_buffer(txbuf)]),
    }, schema=TRADES_SCHEMA)


def trade_sanity(t) -> dict:
    """Per-market checks that data/features.py asserts globally (pandas DataFrame of trades rows)."""
    out = {}
    for slug, g in t.groupby("slug", observed=True):
        tk = g[g.is_taker]
        mk = g[~g.is_taker]
        uniq = bool(tk.tx.is_unique)
        no_maker = len(set(tk.tx) - set(mk.tx))
        dev = 0.0
        if uniq and len(mk):
            msum = mk.groupby("tx")["size"].sum().astype(np.float64)
            d = (msum - tk.set_index("tx")["size"].astype(np.float64).reindex(msum.index)).abs()
            dev = float(d.max()) if d.notna().any() else 0.0      # NaN (maker tx without taker) skipped, as in features.py
        out[str(slug)] = {"taker_unique": uniq, "txs_without_maker_legs": no_maker,
                          "max_abs_maker_sum_minus_taker": dev,
                          "ok": uniq and no_maker == 0 and dev < 0.01}
    return out
