#!/usr/bin/env python3
"""Precompute every market-only quantity the C market-making env (../mm.h) needs.

Inputs : build/ticks.bin, build/markets.bin, build/meta.json (from build.py),
         raw/trades.parquet (maker legs), raw/underlying.parquet (Binance 1 s closes).
Outputs (build/, git-ignored; build.py's outputs are not touched):
  feat.bin        float32 [N, G] market-only features, causal (row t uses data <= t only)
  book.bin        64-byte record per row: top-5 ladder with prices in integer mils, flags, tick
  fills.bin       20-byte record per maker fill leg (Up frame), sorted by row
  fill_index.bin  int32 [N + 1]: legs of row r are fills[fill_index[r] : fill_index[r + 1]]
  feat_meta.json  column docs, record layouts, pricer source, sanity-check numbers

Fair value: the pricer of ../analysis/fair_value/fair_value.py (digital_up_prob,
digital_up_delta, ewma_sigma, model_at), with the settings that study fixed on train
(REPORT.md section 2): Student-t nu = 4, EWMA half-life 300 s on overlapping 30 s Binance
log returns, vol multiplier 1, K = Binance TWAP60 at market open, closing-TWAP adjustment
for tau < 60 s. Nothing here is fitted; nothing reads val/test outcomes.

That module imports scipy, so run this with a Python that has scipy, pandas and pyarrow
(e.g. the system python3):   python3 data/features.py      (~1-2 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
RAW = HERE / "raw"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "analysis" / "fair_value"))
from load import load  # noqa: E402
import fair_value as fv  # noqa: E402  (pricer of the fair-value study)

FEATURES_VERSION = 1
NU = 4.0                 # Student-t dof (REPORT.md section 2)
HALFLIFE_S = 300.0       # EWMA half-life of the vol
SAMPLE_S = 30            # overlapping return sampling for the vol
TWAP_S = 60
STALE_S = 10.0           # book_age above this sets the stale flag
CLIP = 5.0
CLEN_ALL = (300, 900, 14400)

# Market-only observation block (first N_OBS columns), then auxiliary raw columns that the
# C step reads (never copied into the observation).
OBS_COLS = [
    ("p_logit", "logit(p_model) / 3, p clipped to [1e-4, 1 - 1e-4]"),
    ("mark_logit", "logit(mark) / 3; mark = book mid if two-sided and valid, else p_model"),
    ("gap_c", "(p_model - mark) in cents / 5"),
    ("dgap5_c", "change of (p_model - mark) over the last 5 s, cents / 2"),
    ("z", "ln(F/K) / sqrt(V) / 2 (+-5 at tau = 0)"),
    ("sdv_bp", "log1p(sqrt(V) in bp) / 3: sd of log price to the closing TWAP"),
    ("sigma_1m_bp", "log1p(sigma * sqrt(60) in bp) / 3: EWMA vol, 1-minute scale"),
    ("tau_frac", "seconds_to_close / contract_len_s"),
    ("log_tau", "log1p(seconds_to_close) / log1p(14400)"),
    ("in_twap", "1 if seconds_to_close < 60 (closing TWAP window)"),
    ("delta_usd", "log1p(dp/dlnS in USD of underlying per Up share) / 5"),
    ("spread_ticks", "ln(spread / tick) / 2 if two-sided, else 0"),
    ("bid_present", "best bid exists (book valid and bid_px_0 > 0)"),
    ("ask_present", "best ask exists (book valid and ask_px_0 < 1)"),
    ("imb_l1", "(bid_sz_0 - ask_sz_0) / (bid_sz_0 + ask_sz_0), 0 if both empty"),
    ("imb_l5", "same over the top-5 sizes"),
    ("bid_sz0", "log1p(bid_sz_0) / 5"),
    ("ask_sz0", "log1p(ask_sz_0) / 5"),
    ("flow5", "signed taker flow over (t-5, t]: (buy - sell) / (buy + sell + 5)"),
    ("flow30", "same over 30 s"),
    ("flow120", "same over 120 s"),
    ("mret5_c", "mark change over 5 s in cents / 2"),
    ("mret30_c", "mark change over 30 s in cents / 5"),
    ("mret120_c", "mark change over 120 s in cents / 5"),
    ("ulz5", "Binance log return over 5 s / (sigma sqrt 5)"),
    ("ulz30", "Binance log return over 30 s / (sigma sqrt 30)"),
    ("ulz120", "Binance log return over 120 s / (sigma sqrt 120)"),
    ("len_5m", "contract length one-hot"),
    ("len_15m", "contract length one-hot"),
    ("len_4h", "contract length one-hot"),
    ("tick_fine", "1 once the data shows sub-cent prices (tick 0.001), sticky"),
    ("book_age", "log1p(book_age s) / 4"),
    ("crossed_raw", "latest raw snapshot crossed (last uncrossed book forward-filled)"),
    ("mark_is_fv", "1 if the mark fell back to p_model (one-sided or no valid book)"),
    ("gross120", "log1p(taker volume over 120 s, shares) / 6"),
]
AUX_COLS = [
    ("p_model", "fair value of Up (probability), raw"),
    ("mark", "mark price (mid, or p_model fallback), raw"),
    ("delta_usd_raw", "dp/dlnS: USD notional of underlying per Up share"),
    ("sigma", "EWMA vol per sqrt(second)"),
    ("sqrt_v", "sqrt(V): sd of ln(TWAP_end / F)"),
]
N_OBS = len(OBS_COLS)
COLS = [c for c, _ in OBS_COLS] + [c for c, _ in AUX_COLS]
COLS += [f"pad_{i}" for i in range((-len(COLS)) % 4)]
G = len(COLS)
CI = {c: i for i, c in enumerate(COLS)}

BOOK_DTYPE = np.dtype([("bid_px", "<i2", 5), ("ask_px", "<i2", 5), ("bid_sz", "<f4", 5),
                       ("ask_sz", "<f4", 5), ("flags", "u1"), ("tick", "u1"), ("pad", "u1", 2)])
assert BOOK_DTYPE.itemsize == 64
BOOK_VALID, BOOK_CROSSED, BOOK_STALE, BOOK_FROM_DOWN = 1, 2, 4, 8
FILL_DTYPE = np.dtype([("row", "<i4"), ("side", "i1"), ("first", "u1"), ("px", "<i2"),
                       ("price", "<f4"), ("size", "<f4"), ("tx_size", "<f4")])
assert FILL_DTYPE.itemsize == 20


def clip(x, lo=-CLIP, hi=CLIP):
    return np.clip(x, lo, hi)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def lag(x, k):
    """x[t - k] within one market, clamped at the first row."""
    i = np.maximum(np.arange(len(x)) - k, 0)
    return x[i]


def window_sum(x, k):
    """sum of x over rows (t - k, t] within one market."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(1, len(x) + 1)
    return c[i] - c[np.maximum(i - k, 0)]


def underlying():
    """{coin: close array on the 1 s grid}, index i <-> kline open_time fv.T0 + i."""
    t = pd.read_parquet(RAW / "underlying.parquet", columns=["coin", "open_time", "close"])
    out = {}
    for c, g in t.groupby("coin", observed=True):
        g = g.sort_values("open_time")
        n = int(g.open_time.max()) - fv.T0 + 1
        arr = np.full(n, np.nan)
        arr[g.open_time.values - fv.T0] = g.close.values
        missing = int(np.isnan(arr).sum())
        arr = pd.Series(arr).ffill().bfill().values
        out[str(c)] = (arr, missing)
    return out


def fair_value_rows(close, sigma, o, e):
    """Model p, dp/dlnS, sigma, sqrt(V), z at rows t = o..e (causal: kline closes <= t-1)."""
    t = np.arange(o, e + 1, dtype=np.int64)
    K = fv.twap_binance(close, o, TWAP_S)                       # Binance TWAP60 at open
    p, dlt, S = fv.model_at(close, sigma, t, e, K, shift=0, nu=NU, twap=TWAP_S, with_delta=True)
    # replicate model_at's inputs to expose F, V (and z) for the observation
    i = t - 1 - fv.T0
    sg = sigma[i]
    tau = (e - t).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(close)])
    lo, hi = e - TWAP_S - fv.T0, i + 1
    n = hi - lo
    known = np.where(n > 0, (cs[np.clip(hi, 0, len(close))] - cs[np.clip(lo, 0, len(close))])
                     / np.maximum(n, 1), np.nan)
    F, V, _ = fv._forward_and_var(S, sg, np.maximum(tau, 0.5), known, TWAP_S)
    sd = np.sqrt(V)
    z = np.log(F / K) / sd
    done = tau <= 0
    z = np.where(done, np.where(p >= 0.5, CLIP * 2, -CLIP * 2), z)
    sd = np.where(done, 0.0, sd)
    return p, dlt * S, sg, sd, z, S, K


def main():
    t0 = time.time()
    ticks, markets, meta = load()
    N, F = ticks.shape
    c = meta["column_index"]
    slugs = meta["slugs"]
    coins = {v: k for k, v in meta["coin_ids"].items()}
    ul = underlying()
    sig = {k: fv.ewma_sigma(a, HALFLIFE_S, sample_s=SAMPLE_S) for k, (a, _) in ul.items()}
    print(f"underlying + EWMA vol {time.time() - t0:.1f}s")

    # ---------------- maker fill legs (Up frame), per row
    tr = pd.read_parquet(RAW / "trades.parquet")
    tr["slug"] = tr.slug.astype(str)
    tr["px_up"] = np.where(tr.outcome_idx == 0, tr.price.astype(np.float64), 1.0 - tr.price.astype(np.float64))
    taker = tr[tr.is_taker]
    maker = tr[~tr.is_taker].copy()
    assert taker.tx.is_unique, "more than one taker leg in a tx"
    tx_size = taker.set_index("tx")["size"].astype(np.float64)
    tx_side = taker.set_index("tx")["aggressor_up"]
    msum = maker.groupby("tx")["size"].sum()
    dev = (msum - tx_size.reindex(msum.index)).abs()
    rel = dev / tx_size.reindex(msum.index)
    # aggressor sign of every maker leg must match its taker leg
    assert (maker.aggressor_up.values == tx_side.reindex(maker.tx).values).all()
    sanity = {"taker_txs": int(len(taker)), "maker_legs": int(len(maker)),
              "txs_without_maker_legs": int(len(set(taker.tx) - set(maker.tx))),
              "max_abs_maker_sum_minus_taker_size": float(dev.max()),
              "txs_abs_dev_gt_1e-3": int((dev > 1e-3).sum()),
              "max_rel_dev": float(rel.max()),
              "maker_px_off_0.001_grid": int((np.abs(maker.px_up * 1000 - np.rint(maker.px_up * 1000)) > 1e-3).sum()),
              "max_maker_px_dist_from_0.001_grid": float(np.abs(maker.px_up * 1000 - np.rint(maker.px_up * 1000)).max() / 1000)}
    print("maker-leg sanity:", sanity)
    # sizes are float32 with ~6 digits; tiny txs (0.02 shares) carry relative rounding up to ~1%
    assert sanity["txs_without_maker_legs"] == 0 and sanity["max_abs_maker_sum_minus_taker_size"] < 0.01
    maker["tx_size"] = maker.tx.map(tx_size).astype(np.float64)
    maker["px_mil"] = np.rint(maker.px_up * 1000).astype(np.int64)
    maker["ord"] = np.arange(len(maker))
    mk_by = {s: g for s, g in maker.groupby("slug", sort=False)}

    feat = np.zeros((N, G), np.float32)
    book = np.zeros(N, BOOK_DTYPE)
    fill_parts = []
    stats = {"rows": N, "rows_tick_fine": 0, "legs_in_window": 0, "legs_after_end": 0, "legs_before_open": 0,
             "markets_tick_fine": 0, "rows_mark_is_fv": 0, "underlying_missing_seconds": {k: m for k, (_, m) in ul.items()}}
    for m in markets:
        o, e, L = int(m["open_ts"]), int(m["end_ts"]), int(m["contract_len_s"])
        r0, n = int(m["row_offset"]), int(m["n_rows"])
        slug = slugs[m["market_idx"]]
        X = np.asarray(ticks[r0:r0 + n], dtype=np.float64)
        bp, bs = X[:, c["bid_px_0"]:c["bid_px_0"] + 5], X[:, c["bid_sz_0"]:c["bid_sz_0"] + 5]
        ap, asz = X[:, c["ask_px_0"]:c["ask_px_0"] + 5], X[:, c["ask_sz_0"]:c["ask_sz_0"] + 5]
        valid = X[:, c["book_valid"]] > 0
        crossed = X[:, c["crossed_raw"]] > 0
        age = X[:, c["book_age"]]
        tau = X[:, c["seconds_to_close"]]
        coin = coins[int(m["coin_id"])]
        close, _ = ul[coin]
        p, dusd, sg, sdv, z, S, K = fair_value_rows(close, sig[coin], o, e)

        # ---- maker legs of this market
        g = mk_by.get(slug)
        if g is not None:
            stats["legs_after_end"] += int((g.timestamp > e).sum())
            stats["legs_before_open"] += int((g.timestamp < o).sum())
            g = g[(g.timestamp >= o) & (g.timestamp <= e)]
        legs = None
        sub_trade_row = np.zeros(n, bool)
        if g is not None and len(g):
            g = g.assign(row=(g.timestamp.values - o).astype(np.int64))
            # first appearance of each tx orders the txs within a row; within a tx the legs go
            # from the best price for the taker to the worst (sell: high to low; buy: low to high)
            first_ord = g.groupby("tx")["ord"].transform("min")
            g = g.assign(first_ord=first_ord, key=np.where(g.aggressor_up > 0, g.px_mil, -g.px_mil))
            g = g.sort_values(["row", "first_ord", "key", "ord"], kind="stable")
            first = (g.tx.values != np.roll(g.tx.values, 1))
            first[0] = True
            legs = np.zeros(len(g), FILL_DTYPE)
            legs["row"] = r0 + g.row.values
            legs["side"] = g.aggressor_up.values
            legs["first"] = first
            legs["px"] = g.px_mil.values
            legs["price"] = g.px_up.values
            legs["size"] = g["size"].values
            legs["tx_size"] = g.tx_size.values
            fill_parts.append(legs)
            stats["legs_in_window"] += len(legs)
            sub = (g.px_mil.values % 10) != 0
            sub_trade_row[g.row.values[sub]] = True

        # ---- tick: 0.001 from the first row where the book (top 5) or a maker fill shows a
        # sub-cent price, sticky (the exchange switches near 0.04/0.96 and never switches back)
        bm, am = np.rint(bp * 1000).astype(np.int64), np.rint(ap * 1000).astype(np.int64)
        sub_book = (((bm % 10) != 0) & (bm > 0)).any(1) | (((am % 10) != 0) & (am < 1000)).any(1)
        fine = np.maximum.accumulate(sub_book | sub_trade_row)
        tick = np.where(fine, 0.001, 0.01)
        stats["rows_tick_fine"] += int(fine.sum())
        stats["markets_tick_fine"] += int(fine.any())

        # ---- book record
        rec = book[r0:r0 + n]
        rec["bid_px"] = bm
        rec["ask_px"] = am
        rec["bid_sz"] = bs
        rec["ask_sz"] = asz
        rec["flags"] = (valid * BOOK_VALID | crossed * BOOK_CROSSED | (age > STALE_S) * BOOK_STALE
                        | (X[:, c["book_from_down"]] > 0) * BOOK_FROM_DOWN)
        rec["tick"] = np.where(fine, 1, 10)

        # ---- mark, gap
        has_b = valid & (bp[:, 0] > 0)
        has_a = valid & (ap[:, 0] < 1)
        two = has_b & has_a
        mid = 0.5 * (bp[:, 0] + ap[:, 0])
        mark = np.where(two, mid, p).astype(np.float32).astype(np.float64)   # stored as float32
        stats["rows_mark_is_fv"] += int((~two).sum())
        gap = p - mark
        bvol, svol = X[:, c["buy_vol"]], X[:, c["sell_vol"]]
        out = np.zeros((n, G))
        out[:, CI["p_logit"]] = logit(p) / 3
        out[:, CI["mark_logit"]] = logit(mark) / 3
        out[:, CI["gap_c"]] = clip(gap * 100 / 5)
        out[:, CI["dgap5_c"]] = clip((gap - lag(gap, 5)) * 100 / 2)
        out[:, CI["z"]] = clip(z / 2)
        out[:, CI["sdv_bp"]] = np.log1p(sdv * 1e4) / 3
        out[:, CI["sigma_1m_bp"]] = np.log1p(sg * np.sqrt(60) * 1e4) / 3
        out[:, CI["tau_frac"]] = tau / L
        out[:, CI["log_tau"]] = np.log1p(tau) / np.log1p(14400)
        out[:, CI["in_twap"]] = tau < TWAP_S
        out[:, CI["delta_usd"]] = np.log1p(np.maximum(dusd, 0)) / 5
        spread_t = np.where(two, (ap[:, 0] - bp[:, 0]) / tick, 1.0)
        out[:, CI["spread_ticks"]] = np.where(two, clip(np.log(np.maximum(spread_t, 1.0)) / 2), 0.0)
        out[:, CI["bid_present"]] = has_b
        out[:, CI["ask_present"]] = has_a
        s1 = bs[:, 0] + asz[:, 0]
        out[:, CI["imb_l1"]] = np.where(s1 > 0, (bs[:, 0] - asz[:, 0]) / np.where(s1 > 0, s1, 1), 0)
        s5b, s5a = bs.sum(1), asz.sum(1)
        out[:, CI["imb_l5"]] = np.where(s5b + s5a > 0, (s5b - s5a) / np.maximum(s5b + s5a, 1e-12), 0)
        out[:, CI["bid_sz0"]] = np.log1p(bs[:, 0]) / 5
        out[:, CI["ask_sz0"]] = np.log1p(asz[:, 0]) / 5
        for k in (5, 30, 120):
            b, s_ = window_sum(bvol, k), window_sum(svol, k)
            out[:, CI[f"flow{k}"]] = (b - s_) / (b + s_ + 5)
        out[:, CI["gross120"]] = np.log1p(window_sum(bvol + svol, 120)) / 6
        out[:, CI["mret5_c"]] = clip((mark - lag(mark, 5)) * 100 / 2)
        out[:, CI["mret30_c"]] = clip((mark - lag(mark, 30)) * 100 / 5)
        out[:, CI["mret120_c"]] = clip((mark - lag(mark, 120)) * 100 / 5)
        lc = np.log(close)
        i = np.arange(o, e + 1) - 1 - fv.T0
        for k in (5, 30, 120):
            out[:, CI[f"ulz{k}"]] = clip((lc[i] - lc[i - k]) / (sg * np.sqrt(k)))
        for L_, name in zip(CLEN_ALL, ("len_5m", "len_15m", "len_4h")):
            out[:, CI[name]] = float(L == L_)
        out[:, CI["tick_fine"]] = fine
        out[:, CI["book_age"]] = np.log1p(age) / 4
        out[:, CI["crossed_raw"]] = crossed
        out[:, CI["mark_is_fv"]] = ~two
        out[:, CI["p_model"]] = p
        out[:, CI["mark"]] = mark
        out[:, CI["delta_usd_raw"]] = dusd
        out[:, CI["sigma"]] = sg
        out[:, CI["sqrt_v"]] = sdv
        assert np.isfinite(out).all(), slug
        assert np.abs(out[:, :N_OBS]).max() <= CLIP + 1e-9, slug
        feat[r0:r0 + n] = out.astype(np.float32)
    fills = np.concatenate(fill_parts) if fill_parts else np.zeros(0, FILL_DTYPE)
    assert (np.diff(fills["row"]) >= 0).all()
    index = np.searchsorted(fills["row"], np.arange(N + 1), side="left").astype(np.int32)
    assert index[-1] == len(fills)
    # per-row taker volume from the legs must equal ticks buy_vol + sell_vol
    vol = np.bincount(fills["row"], weights=fills["size"].astype(np.float64), minlength=N)
    tv = np.asarray(ticks[:, c["buy_vol"]], np.float64) + np.asarray(ticks[:, c["sell_vol"]], np.float64)
    sanity["max_abs_row_volume_vs_ticks"] = float(np.abs(vol - tv).max())
    assert sanity["max_abs_row_volume_vs_ticks"] < 1e-2

    feat.tofile(BUILD / "feat.bin")
    book.tofile(BUILD / "book.bin")
    fills.tofile(BUILD / "fills.bin")
    index.tofile(BUILD / "fill_index.bin")
    stats["n_fills"] = int(len(fills))
    fmeta = {
        "features_version": FEATURES_VERSION,
        "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "N": int(N), "G": G, "n_obs_features": N_OBS, "dtype": "float32", "order": "row-major [N, G]",
        "columns": COLS, "column_index": CI,
        "column_docs": dict(OBS_COLS + AUX_COLS),
        "obs_block": f"columns 0..{N_OBS - 1} are copied into the observation; the rest are raw "
                     "auxiliaries read by the C step (never observed)",
        "causality": "row t uses ticks rows <= t of the same market, Binance klines with open_time <= t-1 "
                     "(also from before the market open: vol warm-up, K, lagged returns) and maker fills with "
                     "block time <= t (tick flag only). No outcome information.",
        "pricer": {"source": "analysis/fair_value/fair_value.py: model_at, digital_up_prob, digital_up_delta, "
                             "ewma_sigma, twap_binance, _forward_and_var (imported, not copied)",
                   "nu": NU, "ewma_halflife_s": HALFLIFE_S, "return_sampling_s": SAMPLE_S, "vol_multiplier": 1.0,
                   "K": "Binance TWAP60 at open (mean of closes of klines open_time open-60..open-1)",
                   "twap_window_s": TWAP_S, "shift_s": 0,
                   "note": "settings fixed by the fair-value study on train only (REPORT.md section 2)"},
        "tick_rule": "0.01 until the first row whose top-5 book or a maker fill (block time <= t) shows a sub-cent "
                     "price, then 0.001 for the rest of the market (sticky)",
        "book_record": {"itemsize": 64, "fields": [
            {"name": "bid_px", "type": "int16_t[5]", "offset": 0, "doc": "mils (price * 1000), 0 = empty"},
            {"name": "ask_px", "type": "int16_t[5]", "offset": 10, "doc": "mils, 1000 = empty"},
            {"name": "bid_sz", "type": "float[5]", "offset": 20, "doc": "shares, 0 = empty"},
            {"name": "ask_sz", "type": "float[5]", "offset": 40, "doc": "shares"},
            {"name": "flags", "type": "uint8_t", "offset": 60,
             "doc": f"bit0 book_valid, bit1 crossed_raw, bit2 book_age > {STALE_S:g} s, bit3 book_from_down"},
            {"name": "tick", "type": "uint8_t", "offset": 61, "doc": "tick in mils: 10 or 1"},
            {"name": "pad", "type": "uint8_t[2]", "offset": 62}]},
        "fill_record": {"itemsize": 20, "fields": [
            {"name": "row", "type": "int32_t", "offset": 0, "doc": "global row in ticks.bin = block timestamp"},
            {"name": "side", "type": "int8_t", "offset": 4,
             "doc": "+1 taker bought Up (hits asks), -1 taker sold Up (hits bids)"},
            {"name": "first", "type": "uint8_t", "offset": 5, "doc": "1 on the first leg of a taker tx"},
            {"name": "px", "type": "int16_t", "offset": 6, "doc": "maker resting price in mils, Up frame"},
            {"name": "price", "type": "float", "offset": 8, "doc": "maker resting price, Up frame"},
            {"name": "size", "type": "float", "offset": 12, "doc": "shares of this maker leg"},
            {"name": "tx_size", "type": "float", "offset": 16, "doc": "taker size of the whole tx"}],
            "order": "by row; within a row by tx (first appearance in trades.parquet); within a tx from the "
                     "best price for the taker to the worst (sweep order)",
            "window": "block timestamp in [open, end]; legs after end never fill"},
        "fill_index": "int32 [N + 1]: legs of row r are fills[fill_index[r]:fill_index[r+1]]",
        "sanity": sanity, "stats": stats,
    }
    (BUILD / "feat_meta.json").write_text(json.dumps(fmeta, indent=1))
    print(f"feat.bin {N}x{G}, book.bin, fills.bin {len(fills)} legs; {time.time() - t0:.1f}s")
    print("stats:", stats)


if __name__ == "__main__":
    main()
