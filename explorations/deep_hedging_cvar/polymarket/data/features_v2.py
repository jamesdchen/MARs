#!/usr/bin/env python3
"""Sub-second Binance order-flow features from aggTrades (protocol v2, DECISIONS.md D7).

Library part (used by analysis/orderflow/ and by the writer below):
  * aggtrades(path)          parse one data.binance.vision aggTrades zip (spot or USD-M)
  * buckets(coin, day)       250 ms bucket aggregates of one UTC day (cached as .npz)
  * CoinGrid(coin, t0, t1)   per-second causal order-flow quantities on the unix-second grid

Causality: the value at row t (unix second t) uses only trades with transact_time < t (strictly
before the second boundary), the same convention as the 1 s kline with open_time t - 1 that
feat.bin's Binance columns use. A 250 ms bucket k of a day covers [day0 + k/4, day0 + (k+1)/4);
row t reads buckets that end at or before t.

Inputs are cached zips in data/raw/aggtrades/ (data/fetch_aggtrades.py, git-ignored).
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AGG = HERE / "raw" / "aggtrades"
BCACHE = AGG / "buckets250"
PERP = {"hype"}
BPS = 4                     # buckets per second (250 ms)
BUCKET_US = 1_000_000 // BPS


def zip_path(coin: str, day: str) -> Path:
    sym = coin.upper() + "USDT"
    return AGG / f"{sym}-aggTrades-{day}.zip"


def aggtrades(path: Path):
    """(ts_us int64, price float64, qty float64, taker_buy bool) of one daily zip, time-sorted.
    Spot dumps carry microseconds and no header; USD-M dumps milliseconds and a header row."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == 1 and names[0].endswith(".csv"), names
        raw = z.read(names[0])
    header = 0 if raw[:12] == b"agg_trade_id" else None
    df = pd.read_csv(io.BytesIO(raw), header=header, usecols=[1, 2, 5, 6],
                     names=None if header == 0 else ["a", "price", "qty", "f", "l", "ts", "bm", "best"],
                     dtype={1: np.float64, 2: np.float64} if header is None else None)
    df.columns = ["price", "qty", "ts", "bm"]
    ts = df.ts.to_numpy(np.int64)
    if ts.max() < 10**14:                 # milliseconds (USD-M) -> microseconds
        ts = ts * 1000
    bm = df.bm.to_numpy()
    if bm.dtype != bool:
        bm = np.asarray(pd.Series(bm).astype(str).str.lower() == "true")
    px = df.price.to_numpy(np.float64)
    q = df.qty.to_numpy(np.float64)
    o = np.argsort(ts, kind="stable")
    return ts[o], px[o], q[o], ~bm[o]     # is_buyer_maker True => the taker sold


def buckets(coin: str, day: str) -> dict:
    """250 ms aggregates of one UTC day: count, buy/sell notional (USD), last/high/low price
    (NaN where a bucket has no trade). Cached in data/raw/aggtrades/buckets250/."""
    BCACHE.mkdir(parents=True, exist_ok=True)
    cpath = BCACHE / f"{coin}-{day}.npz"
    if cpath.exists():
        with np.load(cpath, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    ts, px, q, buy = aggtrades(zip_path(coin, day))
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp()) * 1_000_000
    k = (ts - day0) // BUCKET_US
    assert k.min() >= 0 and k.max() < 86400 * BPS, (coin, day, k.min(), k.max())
    nb = 86400 * BPS
    usd = px * q
    out = {
        "cnt": np.bincount(k, minlength=nb).astype(np.int32),
        "buy": np.bincount(k, weights=np.where(buy, usd, 0.0), minlength=nb),
        "sell": np.bincount(k, weights=np.where(buy, 0.0, usd), minlength=nb),
        "buy_qty": np.bincount(k, weights=np.where(buy, q, 0.0), minlength=nb),
        "sell_qty": np.bincount(k, weights=np.where(buy, 0.0, q), minlength=nb),
    }
    last = np.full(nb, np.nan)
    hi = np.full(nb, np.nan)
    lo = np.full(nb, np.nan)
    last[k] = px                                  # trades are time-sorted: the last write wins
    np.fmax.at(hi, k, px)
    np.fmin.at(lo, k, px)
    out.update(last=last, hi=hi, lo=lo)
    tmp = cpath.with_name(f".{cpath.stem}.tmp{os.getpid()}.npz")
    np.savez(tmp, **out)
    os.replace(tmp, cpath)
    return out


def _ffill(x: np.ndarray) -> np.ndarray:
    idx = np.where(np.isnan(x), 0, np.arange(len(x)))
    np.maximum.accumulate(idx, out=idx)
    y = x[idx]
    if np.isnan(y[0]):                            # leading gap: back-fill from the first trade
        first = np.flatnonzero(~np.isnan(x))[0]
        y[:first] = x[first]
    return y


def _ewma(x: np.ndarray, halflife: float) -> np.ndarray:
    from scipy.signal import lfilter
    a = 1.0 - 0.5 ** (1.0 / halflife)
    y, _ = lfilter([a], [1.0, -(1.0 - a)], x, zi=[(1.0 - a) * x[: min(len(x), 60)].mean()])
    return y


class CoinGrid:
    """Per-second causal order-flow quantities of one coin for unix seconds t in [t0, t1].

    Built from the 250 ms buckets of every UTC day touching [t0 - warm, t1]. Attribute arrays
    are indexed by t - t0. `px(t)` is the last trade price strictly before t (= the close of the
    1 s kline with open_time t - 1)."""

    def __init__(self, coin: str, t0: int, t1: int, warm_s: int = 3600):
        self.coin, self.t0, self.t1 = coin, t0, t1
        d0 = pd.Timestamp(t0 - warm_s, unit="s", tz="UTC").floor("D")
        d1 = pd.Timestamp(t1, unit="s", tz="UTC").floor("D")
        days = [str(d.date()) for d in pd.date_range(d0, d1, freq="D")]
        parts = [buckets(coin, d) for d in days]
        B = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.days = days
        base = int(d0.timestamp())                  # unix second of bucket 0
        self.base = base
        last = _ffill(B["last"])
        lnp = np.log(last)
        nbk = len(last)
        # row t reads buckets < BPS * (t - base); index of the last bucket that ends by t:
        sec = np.arange(t0, t1 + 1, dtype=np.int64)
        bt = BPS * (sec - base) - 1
        assert bt.min() >= BPS * 120 and bt.max() < nbk, "aggTrades do not cover the requested range"
        self.sec = sec
        cs = lambda x: np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
        net = B["buy"] - B["sell"]
        tot = B["buy"] + B["sell"]
        c_net, c_tot, c_cnt = cs(net), cs(tot), cs(B["cnt"].astype(np.float64))
        dl = np.diff(lnp, prepend=lnp[0])
        c_dl2 = cs(dl * dl)

        def win(c, k):                              # sum over the k buckets ending at bt
            return c[bt + 1] - c[bt + 1 - k]

        self.lnp = lnp[bt]                          # log last price before t
        self.ret = {k: lnp[bt] - lnp[bt - k] for k in (1, 2, 4, 8, 20, 40)}   # k buckets back
        self.net = {k: win(c_net, k) for k in (1, 2, 4, 20, 40)}             # signed taker USD
        self.tot = {k: win(c_tot, k) for k in (1, 2, 4, 20, 40)}
        self.cnt = {k: win(c_cnt, k) for k in (1, 4, 20)}
        self.rv = {k: np.sqrt(np.maximum(win(c_dl2, k), 0.0)) for k in (40,)}  # 10 s realised vol
        # intra-second high-low over the last second (4 buckets), log range; 0 without trades
        hi = np.where(np.isnan(B["hi"]), -np.inf, B["hi"])
        lo = np.where(np.isnan(B["lo"]), np.inf, B["lo"])
        H = np.max(np.stack([hi[bt - j] for j in range(BPS)]), 0)
        L = np.min(np.stack([lo[bt - j] for j in range(BPS)]), 0)
        self.hl1 = np.where(np.isfinite(H) & np.isfinite(L), np.log(H / np.where(np.isfinite(L), L, 1.0)), 0.0)
        # typical activity: EWMA (half-life 300 s) of USD volume and trade count per second,
        # over complete seconds up to t (causal); seconds start at bucket 0 of base
        nsec = nbk // BPS
        vol_s = tot[: nsec * BPS].reshape(nsec, BPS).sum(1)
        cnt_s = B["cnt"][: nsec * BPS].reshape(nsec, BPS).sum(1).astype(np.float64)
        si = sec - base - 1                         # second [t-1, t) has index t - base - 1
        self.vol_ewma = _ewma(vol_s, 300.0)[si]
        self.cnt_ewma = _ewma(cnt_s, 300.0)[si]
        # future log returns (targets only; never written into feat): ln p(t + h) - ln p(t)
        self._lnp_full, self._bt = lnp, bt

    def fwd_ret(self, h_s: float) -> np.ndarray:
        """ln px(t + h) - ln px(t): FUTURE information, for fitting/evaluating only."""
        k = int(round(h_s * BPS))
        j = np.minimum(self._bt + k, len(self._lnp_full) - 1)
        return self._lnp_full[j] - self._lnp_full[self._bt]

    def px(self) -> np.ndarray:
        return np.exp(self.lnp)
