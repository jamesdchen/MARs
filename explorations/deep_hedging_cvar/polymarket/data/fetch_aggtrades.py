#!/usr/bin/env python3
"""Fetch Binance aggTrades (tick-level taker trades) for the sub-second order-flow study
(analysis/orderflow/; parsed by analysis/orderflow/binance_flow.py).

Source: https://data.binance.vision daily dumps, sha256-checked against the .CHECKSUM files.
  * btc eth sol xrp bnb doge zec: spot  data/spot/daily/aggTrades/<SYM>USDT/<SYM>USDT-aggTrades-<day>.zip
  * hype (not on Binance spot):   USD-M data/futures/um/daily/aggTrades/HYPEUSDT/HYPEUSDT-aggTrades-<day>.zip
    (the same perp proxy as fetch_underlying.py uses for HYPE's 1 s closes)

Cache: data/raw/aggtrades/ (git-ignored). Existing files are reused, so reruns are incremental.

    python3 data/fetch_aggtrades.py                                # Aug 20-24 2026, all 8 coins
    python3 data/fetch_aggtrades.py --coins btc --start 2026-06-14 --end 2026-06-15   # v2 test build

Columns of every CSV (spot has no header, futures has one):
  agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker[, is_best_match]
is_buyer_maker = true means the taker SOLD. Spot transact_time is in microseconds (since 2025),
futures in milliseconds; binance_flow.py normalises both.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "raw" / "aggtrades"
OLD_CACHE = HERE / "raw" / "cache" / "binance"     # fetch_underlying.py already holds HYPE's perp aggTrades
BASE = "https://data.binance.vision/data"
COINS = ["btc", "eth", "sol", "xrp", "bnb", "doge", "hype", "zec"]
PERP = {"hype"}


def url_for(coin: str, day: str) -> str:
    sym = coin.upper() + "USDT"
    kind = "futures/um" if coin in PERP else "spot"
    return f"{BASE}/{kind}/daily/aggTrades/{sym}/{sym}-aggTrades-{day}.zip"


def download(url: str, tries: int = 6) -> bytes | None:
    delay = 1.0
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == tries - 1:
                raise
        time.sleep(delay + random.random())
        delay *= 2
    return None


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get(coin: str, day: str) -> tuple[str, str]:
    url = url_for(coin, day)
    name = url.rsplit("/", 1)[1]
    path = OUT / name
    if path.exists():
        return name, "cached"
    chk = download(url + ".CHECKSUM")
    want = chk.split()[0].decode() if chk is not None else None
    old = OLD_CACHE / name
    if old.exists() and want is not None and sha256(old.read_bytes()) == want:
        os.link(old, path)
        return name, "linked"
    data = download(url)
    if data is None:
        return name, "MISSING (404)"
    if want is not None and sha256(data) != want:
        raise RuntimeError(f"checksum mismatch {url}")
    tmp = path.with_name(f".{name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return name, f"downloaded {len(data) / 1e6:.1f} MB" + ("" if want else " (no CHECKSUM file)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default=",".join(COINS))
    ap.add_argument("--start", default="2026-08-20")
    ap.add_argument("--end", default="2026-08-24")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    days = [str(d.date()) for d in pd.date_range(a.start, a.end, freq="D")]
    jobs = [(c, d) for c in a.coins.split(",") for d in days]
    with ThreadPoolExecutor(a.workers) as ex:
        for (c, d), (name, status) in zip(jobs, ex.map(lambda j: get(*j), jobs)):
            print(f"{name}: {status}", flush=True)


if __name__ == "__main__":
    main()
