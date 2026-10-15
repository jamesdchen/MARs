"""Shared constants, on-disk format and train-only data access for the synthetic-market generator.

The on-disk format (ticks.bin columns, markets.bin record, flags) is imported from ../data/build.py so the
generator writes byte-for-byte the layout the C replay env reads. The fair-value pricer is imported from
../analysis/fair_value/fair_value.py (Student-t nu=4, EWMA half-life 300 s on overlapping 30 s returns).

LEAKAGE: `train_*` loaders only ever read split-0 markets (open 2026-08-21..22 UTC) and Binance data
< 2026-08-23 00:00 UTC (through fair_value.load_underlying, which asserts it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
POLY = HERE.parent
DATA = POLY / "data"
FVDIR = POLY / "analysis" / "fair_value"
BUILD = DATA / "build"
RAW = DATA / "raw"
OUT_DEFAULT = DATA / "build_sim"
PARAMS = HERE / "params.json"

for p in (str(DATA), str(FVDIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build as _build  # noqa: E402  (constants only; main() is guarded)
import fair_value as fv  # noqa: E402

COLS, C, F, NLEV = _build.COLS, _build.C, _build.F, _build.NLEV
MARKET_DTYPE = _build.MARKET_DTYPE
C_TYPES = _build.C_TYPES
COINS = _build.COINS
CLEN = _build.CLEN                      # {"5m": 300, "15m": 900, "4h": 14400}
LEN_NAME = {v: k for k, v in CLEN.items()}
LENS = [300, 900, 14400]
LEN_IDX = {L: i for i, L in enumerate(LENS)}
FLAG_HAS_TRADES, FLAG_HAS_UNDERLYING = _build.FLAG_HAS_TRADES, _build.FLAG_HAS_UNDERLYING
FLAG_UL_PERP_PROXY, FLAG_HAS_STRIKE = _build.FLAG_UL_PERP_PROXY, _build.FLAG_HAS_STRIKE
FLAG_SYNTHETIC = 16                     # new bit: row data produced by sim/generate.py

PRICER = dict(halflife_s=300, sample_s=30, nu=4)
TWAP_S = 60

# conditioning buckets (documented in README / params.json)
TAU_EDGES = np.array([0, 10, 30, 60, 120, 300, 900, 3600, 14401])          # seconds to close
EXT_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5001])         # |p_model - 0.5|
GAP_EDGES = np.array([-1.0, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 1.0])  # p_model - mid
NTAU, NEXT, NGAP = len(TAU_EDGES) - 1, len(EXT_EDGES) - 1, len(GAP_EDGES) - 1
# coin groups for book tables: BTC/ETH books are much tighter and deeper; ZEC books are mostly hollow
COIN_GROUP = {"btc": 0, "eth": 1, "sol": 2, "xrp": 2, "bnb": 2, "doge": 2, "hype": 2, "zec": 3}
CG_NAMES = ["btc", "eth", "alt", "zec"]
NCG = 4
HOLLOW_SPREAD = 0.3                     # two-sided book with spread > 0.3 = "hollow" (stub quotes only)
NORMAL, HOLLOW, ONESIDED = 0, 1, 2


def tau_bucket(tau):
    return np.clip(np.searchsorted(TAU_EDGES, np.asarray(tau), side="right") - 1, 0, NTAU - 1)


def ext_bucket(p):
    return np.clip(np.searchsorted(EXT_EDGES, np.abs(np.asarray(p) - 0.5), side="right") - 1, 0, NEXT - 1)


def gap_bucket(g):
    return np.clip(np.searchsorted(GAP_EDGES, np.asarray(g), side="right") - 1, 0, NGAP - 1)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def load_params(path=PARAMS) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------------------------- pricing
def coin_sigma(close: np.ndarray) -> np.ndarray:
    """EWMA per-sqrt-second vol, same settings as the fair-value study (causal; index = kline index)."""
    return fv.ewma_sigma(close, PRICER["halflife_s"], PRICER["sample_s"])


def market_pricing(close: np.ndarray, sigma: np.ndarray, t0: int, o: int, e: int) -> dict:
    """Everything the generator needs from the underlying for one market [o, e].

    close[i] is the close of the kline with open_time t0+i (first knowable at t0+i+1).
    Row t (t = o..e) sees S_t = close of kline t-1 (build.py convention).
    K = Binance TWAP60 at open (mean of closes of klines o-60..o-1); outcome on Binance TWAP60(end).
    p uses fair_value.model_at (time-shifted so the module's T0 maps onto t0).
    """
    lo = o - TWAP_S - 2 - t0
    assert lo >= 0 and e - t0 <= len(close), "underlying does not cover the market"
    close, sigma, t0 = close[lo: e - t0 + 1], sigma[lo: e - t0 + 1], t0 + lo   # local slice (model_at cumsums)
    t = np.arange(o, e + 1, dtype=np.int64)
    i = t - 1 - t0
    K = float(close[o - TWAP_S - t0: o - t0].mean())
    twap_end = float(close[e - TWAP_S - t0: e - t0].mean())
    shift = fv.T0 - t0
    p = fv.model_at(close, sigma, t + shift, np.full(len(t), e + shift), K, nu=PRICER["nu"])
    return dict(t=t, S=close[i], S_prev=close[i - 1], p=p, K=K, twap_end=twap_end, sigma=sigma[i])


# --------------------------------------------------------------------------------------------- train data
def train_build():
    """(ticks memmap, train market records, meta) from data/build, split 0 only."""
    from load import load
    ticks, mk, meta = load(BUILD)
    tr = mk[mk["split"] == 0]
    assert len(tr) and (tr["split"] == 0).all()
    # leakage check: every train market lies inside 2026-08-20..22 UTC
    assert tr["end_ts"].max() <= fv.T_TRAIN_END, "LEAKAGE: train market ends after 2026-08-23"
    return ticks, tr, meta


def train_underlying():
    """{coin: (t0, close, sigma)} for 2026-08-20 .. 08-22 (fair_value.load_underlying asserts < Aug 23)."""
    u = fv.load_underlying()
    return {c: (fv.T0, a, coin_sigma(a)) for c, a in u.items()}


def train_panel():
    """Per-second panel of all train markets: book/trade columns of ticks.bin + model p, sigma, S.

    Returns (P: dict of column arrays, mk: DataFrame of market records with extra fields).
    """
    ticks, tr, meta = train_build()
    ul = train_underlying()
    coins = {v: k for k, v in meta["coin_ids"].items()}
    mrows, parts = [], []
    for r in tr:
        c = coins[int(r["coin_id"])]
        o, e = int(r["open_ts"]), int(r["end_ts"])
        t0, a, sg = ul[c]
        pr = market_pricing(a, sg, t0, o, e)
        X = np.asarray(ticks[int(r["row_offset"]): int(r["row_offset"]) + int(r["n_rows"])], dtype=np.float64)
        parts.append((X, pr))
        mrows.append(dict(mi=int(r["market_idx"]), slug=meta["slugs"][int(r["market_idx"])], coin=c,
                          coin_id=int(r["coin_id"]), L=int(r["contract_len_s"]), open_ts=o, end_ts=e,
                          winner_up=int(r["winner_up"]), K=pr["K"], twap_end=pr["twap_end"],
                          log_strike_vs_ul_open=float(r["log_strike_vs_ul_open"]), flags=int(r["flags"]),
                          rewards_min_size=float(r["rewards_min_size"]),
                          rewards_max_spread=float(r["rewards_max_spread"]),
                          taker_fee_rate=float(r["taker_fee_rate"]), maker_rebate_rate=float(r["maker_rebate_rate"]),
                          tick_size_final=float(r["tick_size_final"]), n_rows=int(r["n_rows"])))
    mk = pd.DataFrame(mrows)
    X = np.concatenate([x for x, _ in parts])
    P = {name: X[:, j] for name, j in C.items()}
    P["p"] = np.concatenate([pr["p"] for _, pr in parts])
    P["sigma"] = np.concatenate([pr["sigma"] for _, pr in parts])
    P["S"] = np.concatenate([pr["S"] for _, pr in parts])
    P["t"] = np.concatenate([pr["t"] for _, pr in parts])
    P["m"] = np.concatenate([np.full(len(x), k) for k, (x, _) in enumerate(parts)])   # row -> position in mk
    P["L"] = mk.L.values[P["m"]]
    return P, mk


def book_state(P: dict):
    """Regime per row (NORMAL / HOLLOW / ONESIDED), displayed mid and spread (book_valid rows)."""
    bp0, ap0 = P["bid_px_0"], P["ask_px_0"]
    hb, ha = bp0 > 0, ap0 < 1
    two = hb & ha
    spread = np.where(two, ap0 - bp0, np.nan)
    reg = np.where(~two, ONESIDED, np.where(spread > HOLLOW_SPREAD, HOLLOW, NORMAL))
    mid = np.where(two, 0.5 * (bp0 + ap0), np.where(hb, 0.5 * (bp0 + 1), np.where(ha, 0.5 * ap0, 0.5)))
    return reg, mid, spread


ONCHAIN = POLY / "onchain" / "data" / "crypto-updown"
PRE_PERIOD_LAST_DAY = "2026-08-20"          # pre-period trades end before the train days (08-21..22)


def pre_period_trades(start: str = "2026-07-21", end: str = PRE_PERIOD_LAST_DAY) -> pd.DataFrame:
    """On-chain fills converted by onchain/to_trades.py (data/raw/trades.parquet schema) for markets that open
    AND end inside [start, end] (UTC days), i.e. strictly before the train days. Flow-only data (no L2 book):
    use it only for parameters that need no book or Binance state (taker sizes, tokens, legs, sweep depth,
    sign persistence, trades per market). Never reads anything dated 2026-08-21 or later for this purpose,
    and asserts nothing from 2026-08-22 on is returned."""
    assert end <= PRE_PERIOD_LAST_DAY, "pre-period trades must end by 2026-08-20 (train starts 08-21)"
    t_lo = int(pd.Timestamp(start, tz="UTC").timestamp())
    t_hi = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())      # exclusive
    parts = []
    for f in sorted(ONCHAIN.glob("trades_*_*.parquet")):
        s, e = f.stem.split("_")[1:3]
        if e < start or s > end:
            continue
        parts.append(pd.read_parquet(f))
    if not parts:
        return pd.DataFrame(columns=["slug", "timestamp", "outcome_idx", "side", "price", "size", "is_taker",
                                     "aggressor_up", "tx"])
    t = pd.concat(parts, ignore_index=True)
    t["slug"] = t.slug.astype(str)
    sl = t.slug.str.extract(r"^(\w+)-updown-(5m|15m|4h)-(\d+)$")
    t = t[sl[0].notna()].copy()
    sl = sl[sl[0].notna()]
    t["coin"] = sl[0].values
    t["L"] = sl[1].map(CLEN).values.astype(np.int64)
    t["open_ts"] = sl[2].astype(np.int64).values
    t["end_ts"] = t.open_ts + t.L
    t = t[(t.open_ts >= t_lo) & (t.end_ts <= t_hi) & t.coin.isin(COINS)
          & (t.timestamp >= t.open_ts) & (t.timestamp <= t.end_ts)]
    assert (t.timestamp < int(pd.Timestamp("2026-08-21", tz="UTC").timestamp())).all() or end < "2026-08-21"
    assert not (t.timestamp >= int(pd.Timestamp("2026-08-22", tz="UTC").timestamp())).any(), "LEAKAGE"
    t["px_up"] = np.where(t.outcome_idx == 0, t.price.astype(np.float64), 1.0 - t.price.astype(np.float64))
    return t.reset_index(drop=True)


def train_trades(slugs) -> pd.DataFrame:
    """Train-split fill legs (data/raw/trades.parquet) for the given slugs, with Up-frame price."""
    sl = set(slugs)
    splits = json.loads((BUILD / "splits.json").read_text())
    assert all(splits[s] == 0 for s in sl), "LEAKAGE: non-train slug requested"
    t = pd.read_parquet(RAW / "trades.parquet")
    t["slug"] = t.slug.astype(str)
    t = t[t.slug.isin(sl)].copy()
    t["px_up"] = np.where(t.outcome_idx == 0, t.price.astype(np.float64), 1.0 - t.price.astype(np.float64))
    return t
