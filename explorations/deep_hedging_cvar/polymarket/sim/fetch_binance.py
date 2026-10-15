#!/usr/bin/env python3
"""Cache months of Binance 1 s spot closes for the generator's underlying paths.

Source: data.binance.vision daily 1 s klines (api.binance.com is blocked), downloaded and
checksum-verified with the same code as ../data/fetch_underlying.py (imported, cache redirected).

LEAKAGE GUARD: the evaluation days (val 2026-08-23, test 2026-08-24) plus a 1-day embargo either side
(2026-08-22 .. 2026-08-25) are never fetched or loaded (SIMULATOR.md §1). Requests for those days are
dropped with a message, and `load_closes` asserts none is present.

Output: sim/cache/binance/<coin>/<YYYY-MM-DD>.npy   float64[86400], index i <-> kline open_time day0+i
(close first knowable at day0+i+1). The zip is deleted after parsing.

    python sim/fetch_binance.py --coins btc eth sol xrp --start 2026-07-23 --end 2026-08-21
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "binance"
EXCLUDED_DAYS = {"2026-08-22", "2026-08-23", "2026-08-24", "2026-08-25"}
SPOT_COINS = {"btc", "eth", "sol", "xrp", "bnb", "doge", "zec"}   # hype has no spot 1s klines (see data/README)


def _fu():
    sys.path.insert(0, str(HERE.parent / "data"))
    import fetch_underlying as fu  # noqa: E402
    fu.CACHE = CACHE / "_zips"
    fu.CACHE.mkdir(parents=True, exist_ok=True)
    return fu


def fetch_day(fu, coin: str, day: str) -> str:
    out = CACHE / coin / f"{day}.npy"
    if out.exists():
        return "cached"
    p = fu.get_zip(coin, day)
    if p is None:
        return "missing"
    df = fu.parse(coin, day, p)
    c = df.close.values.astype(np.float64)
    if np.isnan(c[0]):                      # leading gap: back-fill from the first trade of the day
        c = pd.Series(c).bfill().values
    assert len(c) == 86400 and np.isfinite(c).all() and (c > 0).all(), (coin, day)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, c)
    p.unlink()
    return "ok"


def days_in(start: str, end: str) -> list[str]:
    days = [str(d.date()) for d in pd.date_range(start, end, freq="D")]
    bad = sorted(set(days) & EXCLUDED_DAYS)
    if bad:
        print("LEAKAGE GUARD: skipping excluded days", bad)
    return [d for d in days if d not in EXCLUDED_DAYS]


def available_days(coin: str) -> list[str]:
    d = CACHE / coin
    days = sorted(p.stem for p in d.glob("*.npy")) if d.exists() else []
    assert not set(days) & EXCLUDED_DAYS, f"LEAKAGE: excluded day cached for {coin}"
    return days


def load_closes(coin: str, start: str | None = None, end: str | None = None):
    """Contiguous segments of cached closes: list of (t0_unix, close float64[n]).
    A segment breaks wherever a day is missing (e.g. around the excluded window)."""
    days = [d for d in available_days(coin) if (start is None or d >= start) and (end is None or d <= end)]
    segs, cur, t_prev = [], [], None
    for d in days:
        t0 = int(pd.Timestamp(d, tz="UTC").timestamp())
        if cur and t0 != t_prev + 86400:
            segs.append((cur[0][0], np.concatenate([a for _, a in cur])))
            cur = []
        cur.append((t0, np.load(CACHE / coin / f"{d}.npy")))
        t_prev = t0
    if cur:
        segs.append((cur[0][0], np.concatenate([a for _, a in cur])))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=["btc", "eth", "sol", "xrp"])
    ap.add_argument("--start", default="2026-07-23")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    fu = _fu()
    coins = [c for c in a.coins if c in SPOT_COINS]
    if set(a.coins) - SPOT_COINS:
        print("skipping (no spot 1s klines):", sorted(set(a.coins) - SPOT_COINS))
    jobs = [(c, d) for c in coins for d in days_in(a.start, a.end)]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda cd: fetch_day(fu, *cd), jobs))
    for (c, d), r in zip(jobs, res):
        if r != "cached" and r != "ok":
            print(c, d, r)
    print({c: len(available_days(c)) for c in coins}, "days cached")


if __name__ == "__main__":
    main()
