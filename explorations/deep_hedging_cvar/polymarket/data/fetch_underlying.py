#!/usr/bin/env python3
"""Fetch 1-second underlying prices for every coin in the order-book files.

Source: https://data.binance.vision (public daily dumps; api.binance.com is
geo-blocked from here).
  * btc eth sol xrp bnb doge zec: spot <COIN>USDT 1s klines
        data/spot/daily/klines/<SYM>/1s/<SYM>-1s-<day>.zip
  * hype: NOT listed on Binance spot (404 for HYPEUSDT/HYPEUSDC/HYPEFDUSD) and
    Binance USD-M futures has no 1s klines; Hyperliquid's candleSnapshot only
    serves the most recent 5000 candles (empty for Aug 2026). We therefore
    build 1s closes from Binance USD-M perpetual aggTrades
        data/futures/um/daily/aggTrades/HYPEUSDT/HYPEUSDT-aggTrades-<day>.zip
    close(open_time t) = price of the last aggTrade with transact_time in
    [t, t+1s), forward-filled over seconds without trades (same convention as
    Binance 1s klines, whose empty seconds repeat the previous close).
    This is a perp price, not spot; flagged via the `source` column.

Note the markets resolve on Chainlink TWAP data streams (see markets.parquet
resolution_source / price_to_beat), so Binance is a proxy for the underlying.

Output: data/raw/underlying.parquet with columns
  coin (dictionary str), open_time (int64, unix SECONDS, kline open), close (float64),
  source (dictionary str: 'spot_kline_1s' | 'um_perp_aggtrades_1s').
A kline with open_time t closes at t+1s: its close is first KNOWABLE at t+1.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import random
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
CACHE = RAW / "cache" / "binance"
BASE = "https://data.binance.vision/data"
COINS = ["btc", "eth", "sol", "xrp", "bnb", "doge", "hype", "zec"]
PERP_AGG = {"hype"}


def url_for(coin: str, day: str) -> str:
    sym = coin.upper() + "USDT"
    if coin in PERP_AGG:
        return f"{BASE}/futures/um/daily/aggTrades/{sym}/{sym}-aggTrades-{day}.zip"
    return f"{BASE}/spot/daily/klines/{sym}/1s/{sym}-1s-{day}.zip"


def download(url: str, tries: int = 6) -> bytes | None:
    delay = 1.0
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
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


def get_zip(coin: str, day: str) -> Path | None:
    url = url_for(coin, day)
    path = CACHE / url.rsplit("/", 1)[1]
    if path.exists():
        return path
    data = download(url)
    if data is None:
        print("missing", url)
        return None
    chk = download(url + ".CHECKSUM")
    if chk is not None:
        want = chk.split()[0].decode()
        got = hashlib.sha256(data).hexdigest()
        if want != got:
            raise RuntimeError(f"checksum mismatch {url}")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


def to_seconds(ts: np.ndarray) -> np.ndarray:
    """Binance switched spot dumps to microseconds in 2025; handle ms/us/s."""
    ts = ts.astype(np.int64)
    if ts.max() > 10**15:
        return ts // 1_000_000
    if ts.max() > 10**12:
        return ts // 1000
    return ts


def parse(coin: str, day: str, path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp())
    grid = np.arange(day0, day0 + 86400, dtype=np.int64)
    if coin in PERP_AGG:
        first = raw[:200].split(b"\n", 1)[0]
        header = 0 if first.startswith(b"agg_trade_id") else None
        df = pd.read_csv(io.BytesIO(raw), header=header, usecols=[1, 5],
                         names=None if header == 0 else ["agg_trade_id", "price", "quantity", "first_trade_id",
                                                         "last_trade_id", "transact_time", "is_buyer_maker"])
        df.columns = ["price", "transact_time"]
        sec = to_seconds(df.transact_time.values)
        last = pd.Series(df.price.values.astype(np.float64)).groupby(sec).last()
        close = last.reindex(grid).ffill()
        src = "um_perp_aggtrades_1s"
    else:
        df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 4])
        df.columns = ["open_time", "close"]
        if isinstance(df.open_time.iloc[0], str):  # header row present
            df = df.iloc[1:].astype({"open_time": np.int64, "close": np.float64})
        sec = to_seconds(df.open_time.values)
        close = pd.Series(df.close.values.astype(np.float64), index=sec)
        close = close[~close.index.duplicated()].reindex(grid).ffill()
        src = "spot_kline_1s"
    out = pd.DataFrame({"coin": coin, "open_time": grid, "close": close.values, "source": src})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="first UTC day (default: derived from order books)")
    ap.add_argument("--end", default=None, help="last UTC day inclusive")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.start is None or args.end is None:
        ends = pd.concat(pd.read_parquet(RAW / f"updown_{cl}.parquet", columns=["market_end_at"]) for cl in ("5m", "15m", "4h"))
        lo = (ends.market_end_at.min() - pd.Timedelta(hours=5)).floor("D")   # 4h contracts + margin
        hi = ends.market_end_at.max().floor("D")
        start, end = args.start or str(lo.date()), args.end or str(hi.date())
    else:
        start, end = args.start, args.end
    days = [str(d.date()) for d in pd.date_range(start, end, freq="D")]
    print("days:", days)
    jobs = [(c, d) for c in COINS for d in days]
    with ThreadPoolExecutor(args.workers) as ex:
        paths = list(ex.map(lambda cd: get_zip(*cd), jobs))
    frames = []
    for (c, d), p in zip(jobs, paths):
        if p is None:
            continue
        frames.append(parse(c, d, p))
    df = pd.concat(frames, ignore_index=True).sort_values(["coin", "open_time"])
    # carry the previous day's last close over a leading gap at a day boundary
    df["close"] = df.groupby("coin").close.ffill()
    n_nan = int(df.close.isna().sum())
    df = df.dropna(subset=["close"])
    tbl = pa.table({
        "coin": pa.array(df.coin.values).dictionary_encode(),
        "open_time": pa.array(df.open_time.values.astype(np.int64)),
        "close": pa.array(df.close.values.astype(np.float64)),
        "source": pa.array(df.source.values).dictionary_encode(),
    })
    out = RAW / "underlying.parquet"
    pq.write_table(tbl, out, compression="zstd", compression_level=19,
                   use_byte_stream_split=["close"], row_group_size=1 << 20)
    print(df.groupby("coin").agg(n=("close", "size"), t0=("open_time", "min"), t1=("open_time", "max"),
                                 src=("source", "first")))
    print(f"dropped leading NaN seconds: {n_nan}; {out.name}: {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
