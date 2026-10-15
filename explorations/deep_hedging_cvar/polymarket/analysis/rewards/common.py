"""Shared helpers for the rewards/rebates study.

LEAKAGE RULE: only TRAIN markets are ever loaded: the trading window
[event_start, end] must lie inside 2026-08-21 00:00 .. 2026-08-23 00:00 UTC.
Every loader below filters on that *before* returning anything; per-market
cache files of other markets are never opened (the slug epoch = event start).
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "data" / "raw"          # read-only
CACHE = HERE / "cache"
OUT = HERE / "out"
TRAIN_LO = pd.Timestamp("2026-08-21", tz="UTC")
TRAIN_HI = pd.Timestamp("2026-08-23", tz="UTC")
LEN = {"5m": pd.Timedelta("5min"), "15m": pd.Timedelta("15min"), "4h": pd.Timedelta("4h")}
LEN_S = {k: v.total_seconds() for k, v in LEN.items()}
UA = {"User-Agent": "Mozilla/5.0 (research; polite)", "Accept": "application/json"}


def slug_window(slug: str):
    """(contract_length, start, end) parsed from the slug alone."""
    parts = slug.split("-")
    cl, ep = parts[-2], int(parts[-1])
    start = pd.Timestamp(ep, unit="s", tz="UTC")
    return cl, start, start + LEN[cl]


def is_train(slug: str) -> bool:
    try:
        cl, s, e = slug_window(slug)
    except Exception:  # noqa: BLE001
        return False
    return s >= TRAIN_LO and e <= TRAIN_HI


def load_l2(cl: str, columns=None) -> pd.DataFrame:
    """L2 snapshots of TRAIN markets only (filtered at scan time)."""
    d = ds.dataset(RAW / f"updown_{cl}.parquet")
    f = (ds.field("market_end_at") <= TRAIN_HI) & (ds.field("market_end_at") >= TRAIN_LO + LEN[cl])
    df = d.to_table(filter=f, columns=columns).to_pandas()
    df = df[df.market_slug.map(is_train)]
    return df.reset_index(drop=True)


def train_slugs() -> dict[str, list[str]]:
    out = {}
    for cl in LEN:
        d = ds.dataset(RAW / f"updown_{cl}.parquet")
        s = d.to_table(columns=["market_slug"]).column(0).unique().to_pylist()
        out[cl] = sorted(x for x in s if is_train(x))
    return out


def load_market_cache(slug: str) -> dict | None:
    """Per-market gamma + fills from the data agent's cache (train only)."""
    assert is_train(slug), f"refusing non-train market {slug}"
    p = RAW / "cache" / f"{slug}.json.gz"
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001  (file may be mid-write)
        return None


def http_json(url: str, tries: int = 6):
    delay = 1.0
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise RuntimeError(f"{url}: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 30)
