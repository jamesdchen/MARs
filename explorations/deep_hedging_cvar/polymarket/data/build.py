#!/usr/bin/env python3
"""Build flat, memory-mappable arrays for the C market-making env.

Inputs  (data/raw/):  updown_{5m,15m,4h}.parquet, markets.parquet, trades.parquet,
                      underlying.parquet
Outputs (data/build/): ticks.bin, markets.bin, meta.json, splits.json

Causality conventions (row t = unix second t, t in [open, end]):
  * book      : latest snapshot with captured_at <= t (Up frame; crossed
                snapshots skipped, i.e. the last UNCROSSED book is forward-filled;
                `crossed_raw` says whether the most recent raw snapshot was crossed)
  * trades    : fills with timestamp in (t-1, t]  (timestamps are integer seconds)
  * underlying: close of the last 1s kline that has CLOSED by t, i.e. the kline
                with open_time t-1 (it covers [t-1, t))
  * price_to_beat (strike) is fixed at market open, so it is known for all rows.
  * winner / final price are NEVER in ticks.bin; winner_up lives in markets.bin.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "build"
BUILD_VERSION = 1
NLEV = 5
BASE_TICK = 0.01   # Polymarket base tick; shrinks to 0.001 once price goes extreme (see README)
COINS = ["btc", "eth", "sol", "xrp", "bnb", "doge", "hype", "zec"]
CLEN = {"5m": 300, "15m": 900, "4h": 14400}
SPLIT_DAYS = {"2026-08-20": 0, "2026-08-21": 0, "2026-08-22": 0, "2026-08-23": 1, "2026-08-24": 2}
SPLIT_NAMES = {0: "train", 1: "val", 2: "test", 3: "purged"}

COLS = (["seconds_to_close"]
        + [f"bid_px_{i}" for i in range(NLEV)] + [f"bid_sz_{i}" for i in range(NLEV)]
        + [f"ask_px_{i}" for i in range(NLEV)] + [f"ask_sz_{i}" for i in range(NLEV)]
        + ["bid_depth_total", "ask_depth_total", "book_age", "book_valid", "crossed_raw", "book_from_down",
           "buy_vol", "buy_vwap", "buy_max_px", "sell_vol", "sell_vwap", "sell_min_px", "trade_count",
           "ul_logp_rel_open", "ul_logret_1s", "ul_logp_vs_strike", "const_one"])
COLS += [f"pad_{i}" for i in range((-len(COLS)) % 4)]
F = len(COLS)
C = {c: i for i, c in enumerate(COLS)}

MARKET_DTYPE = np.dtype([
    ("row_offset", "<i8"), ("open_ts", "<i8"), ("end_ts", "<i8"),
    ("n_rows", "<i4"), ("contract_len_s", "<i4"), ("coin_id", "<i4"), ("winner_up", "<i4"),
    ("split", "<i4"), ("flags", "<i4"),
    ("tick_size", "<f4"), ("taker_fee_rate", "<f4"), ("fee_exponent", "<f4"), ("min_order_size", "<f4"),
    ("rewards_min_size", "<f4"), ("rewards_max_spread", "<f4"), ("log_strike_vs_ul_open", "<f4"),
    ("maker_rebate_rate", "<f4"), ("market_idx", "<i4"), ("tick_size_final", "<f4"),
])
assert MARKET_DTYPE.itemsize == 88
FLAG_HAS_TRADES, FLAG_HAS_UNDERLYING, FLAG_UL_PERP_PROXY, FLAG_HAS_STRIKE = 1, 2, 4, 8
C_TYPES = {"<i8": "int64_t", "<i4": "int32_t", "<f4": "float"}


# ----------------------------------------------------------------------------- books
def ladder_topk(tbl, price_col, size_col):
    """Return top-NLEV prices/sizes (NaN where missing), total depth and level count."""
    pc, sc = tbl.column(price_col).combine_chunks(), tbl.column(size_col).combine_chunks()
    off = pc.offsets.to_numpy()
    pv = pc.values.to_numpy(zero_copy_only=False)
    sv = sc.values.to_numpy(zero_copy_only=False)
    n = len(off) - 1
    ln = np.diff(off)
    px = np.full((n, NLEV), np.nan)
    sz = np.zeros((n, NLEV))
    for k in range(NLEV):
        m = ln > k
        px[m, k] = pv[off[:-1][m] + k]
        sz[m, k] = sv[off[:-1][m] + k]
    cs = np.concatenate([[0.0], np.cumsum(sv)])
    depth = cs[off[1:]] - cs[off[:-1]]
    return px, sz, depth, ln


def load_books():
    parts = []
    for cl in ("5m", "15m", "4h"):
        t = pq.read_table(RAW / f"updown_{cl}.parquet",
                          columns=["market_slug", "outcome", "captured_at", "bid_prices", "bid_sizes",
                                   "ask_prices", "ask_sizes", "crossed"])
        bpx, bsz, bdep, bn = ladder_topk(t, "bid_prices", "bid_sizes")
        apx, asz, adep, an = ladder_topk(t, "ask_prices", "ask_sizes")
        df = pd.DataFrame({
            "slug": t.column("market_slug").to_numpy(zero_copy_only=False),
            "is_up": t.column("outcome").to_numpy(zero_copy_only=False) == "Up",
            "ts": t.column("captured_at").cast("int64").to_numpy() // 1_000_000,
            "crossed": t.column("crossed").to_numpy(zero_copy_only=False),
            "bdep": bdep, "adep": adep, "bn": bn, "an": an,
        })
        for k in range(NLEV):
            df[f"bp{k}"], df[f"bs{k}"], df[f"ap{k}"], df[f"as{k}"] = bpx[:, k], bsz[:, k], apx[:, k], asz[:, k]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def to_up_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Map Down-token snapshots into the Up frame: Up bid = 1 - Down ask, Up ask = 1 - Down bid."""
    o = d.copy()
    for k in range(NLEV):
        o[f"bp{k}"], o[f"bs{k}"] = 1 - d[f"ap{k}"], d[f"as{k}"]
        o[f"ap{k}"], o[f"as{k}"] = 1 - d[f"bp{k}"], d[f"bs{k}"]
    o["bdep"], o["adep"], o["bn"], o["an"] = d.adep, d.bdep, d.an, d.bn
    return o


def mirror_stats(books: pd.DataFrame) -> dict:
    up = books[books.is_up].set_index(["slug", "ts"])
    dn = to_up_frame(books[~books.is_up]).set_index(["slug", "ts"])
    common = up.index.intersection(dn.index)
    u, d = up.loc[common], dn.loc[common]

    def eq(a, b):
        return (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) < 1e-9)
    top = eq(u.bp0.values, d.bp0.values) & eq(u.ap0.values, d.ap0.values)
    full5 = top.copy()
    for k in range(NLEV):
        full5 &= eq(u[f"bp{k}"].values, d[f"bp{k}"].values) & eq(u[f"ap{k}"].values, d[f"ap{k}"].values)
        full5 &= (np.abs(u[f"bs{k}"].values - d[f"bs{k}"].values) < 1e-6) & (np.abs(u[f"as{k}"].values - d[f"as{k}"].values) < 1e-6)
    both_ok = ~u.crossed.values & ~d.crossed.values
    return {
        "up_snapshots": int(len(up)), "down_snapshots": int(len(dn)), "common_timestamps": int(len(common)),
        "up_only_timestamps": int(len(up.index.difference(dn.index))),
        "down_only_timestamps": int(len(dn.index.difference(up.index))),
        "top_of_book_mirror_frac": float(top.mean()),
        "top5_px_and_size_mirror_frac": float(full5.mean()),
        "top_of_book_mirror_frac_both_uncrossed": float(top[both_ok].mean()),
        "up_crossed_frac": float(up.crossed.mean()), "down_crossed_frac": float(dn.crossed.mean()),
        "both_crossed_frac_common": float((u.crossed.values & d.crossed.values).mean()),
    }


def merged_snapshots(mb: pd.DataFrame):
    """Per captured timestamp pick one Up-frame book: Up if uncrossed, else mapped
    Down if uncrossed, else the timestamp is 'crossed' (no valid book)."""
    num = [f"{a}{k}" for k in range(NLEV) for a in ("bp", "bs", "ap", "as")] + ["bdep", "adep"]
    up = mb[mb.is_up].set_index("ts")
    dn = to_up_frame(mb[~mb.is_up]).set_index("ts")
    ts = np.union1d(up.index.values, dn.index.values)
    up_ok = up.crossed.reindex(ts).eq(False).values
    dn_ok = dn.crossed.reindex(ts).eq(False).values
    U = up[num].astype(np.float64).reindex(ts)
    D = dn[num].astype(np.float64).reindex(ts)
    use_dn = ~up_ok & dn_ok
    valid = up_ok | dn_ok
    book = pd.DataFrame(np.where(use_dn[:, None], D.values, U.values), index=ts, columns=num)
    book["from_down"] = use_dn
    book["valid"] = valid
    book["raw_crossed"] = ~valid
    return book


# ----------------------------------------------------------------------------- trades
def load_trades():
    t = pd.read_parquet(RAW / "trades.parquet", columns=["slug", "timestamp", "outcome_idx", "side", "price",
                                                        "size", "is_taker", "aggressor_up", "tx"])
    t["slug"] = t.slug.astype(str)
    t["px_up"] = np.where(t.outcome_idx == 0, t.price.astype(np.float64), 1.0 - t.price.astype(np.float64))
    # use maker legs (resting limit prices); fall back to the taker leg if a tx has no maker legs
    has_maker = t[~t.is_taker].groupby("tx").size()
    t["use"] = ~t.is_taker | ~t.tx.isin(has_maker.index)
    return t


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    mk = pd.read_parquet(RAW / "markets.parquet")
    books = load_books()
    print(f"books loaded {len(books)} rows  {time.time()-t0:.1f}s")
    mstats = mirror_stats(books)
    print("mirror:", mstats)
    trades = load_trades()
    tr_by = {s: g for s, g in trades[trades.use].groupby("slug", sort=False)}
    ul = pd.read_parquet(RAW / "underlying.parquet")
    ul_by = {c: g for c, g in ul.groupby("coin", observed=True)}
    print(f"trades {len(trades)}  underlying {len(ul)}  {time.time()-t0:.1f}s")

    raw_slugs = books.slug.unique()
    mk = mk[mk.slug.isin(raw_slugs)].copy()
    missing_meta = sorted(set(raw_slugs) - set(mk.slug))
    if missing_meta:
        print("WARNING: markets without gamma metadata (skipped):", len(missing_meta))
    mk["contract_len_s"] = mk.contract_length.map(CLEN).astype(int)
    mk["end_ts"] = mk.end.map(lambda x: int(pd.Timestamp(x).timestamp()))
    mk["open_ts"] = mk.end_ts - mk.contract_len_s
    ev_open = mk.event_start.map(lambda x: int(pd.Timestamp(x).timestamp()))
    assert (ev_open == mk.open_ts).all(), "eventStartTime != end - contract length"
    # ---- splits: by open date UTC; purge if [open, end) touches another split's day
    open_day = pd.to_datetime(mk.open_ts, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    last_day = pd.to_datetime(mk.end_ts - 1, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    s_open = open_day.map(SPLIT_DAYS)
    s_last = last_day.map(SPLIT_DAYS)
    assert s_open.notna().all() and s_last.notna().all(), "market outside split calendar"
    mk["split"] = np.where(s_open == s_last, s_open, 3).astype(int)
    mk = mk.sort_values(["split", "open_ts", "slug"]).reset_index(drop=True)
    mk["coin_id"] = mk.coin.map({c: i for i, c in enumerate(COINS)}).astype(int)

    n_rows = (mk.contract_len_s + 1).values
    offsets = np.concatenate([[0], np.cumsum(n_rows)[:-1]]).astype(np.int64)
    N = int(n_rows.sum())
    ticks = np.zeros((N, F), dtype=np.float32)
    recs = np.zeros(len(mk), dtype=MARKET_DTYPE)
    book_groups = {s: g for s, g in books.groupby("slug", sort=False)}
    per_market = []
    for i, m in mk.iterrows():
        L, o, e = int(m.contract_len_s), int(m.open_ts), int(m.end_ts)
        n = L + 1
        tgrid = np.arange(o, e + 1, dtype=np.int64)
        X = np.zeros((n, F), dtype=np.float64)
        X[:, C["seconds_to_close"]] = e - tgrid
        X[:, C["const_one"]] = 1.0
        # ---- book
        snap = merged_snapshots(book_groups[m.slug])
        sts = snap.index.values.astype(np.int64)
        j_raw = np.searchsorted(sts, tgrid, side="right") - 1
        X[:, C["crossed_raw"]] = np.where(j_raw >= 0, snap.raw_crossed.values[np.maximum(j_raw, 0)], 0)
        vsnap = snap[snap.valid.values]
        vts = vsnap.index.values.astype(np.int64)
        j = np.searchsorted(vts, tgrid, side="right") - 1
        ok = j >= 0
        jj = np.maximum(j, 0)
        X[:, C["book_valid"]] = ok
        X[:, C["book_age"]] = np.where(ok, tgrid - vts[jj], 0)
        X[:, C["book_from_down"]] = np.where(ok, vsnap.from_down.values[jj], 0)
        for k in range(NLEV):
            bp = np.nan_to_num(vsnap[f"bp{k}"].values[jj], nan=0.0)
            ap = np.nan_to_num(vsnap[f"ap{k}"].values[jj], nan=1.0)
            bsz = vsnap[f"bs{k}"].values[jj]
            asz = vsnap[f"as{k}"].values[jj]
            bsz = np.where(bp > 0, bsz, 0.0)
            asz = np.where(ap < 1, asz, 0.0)
            X[:, C[f"bid_px_{k}"]] = np.where(ok, bp, 0.0)
            X[:, C[f"ask_px_{k}"]] = np.where(ok, ap, 1.0)
            X[:, C[f"bid_sz_{k}"]] = np.where(ok, bsz, 0.0)
            X[:, C[f"ask_sz_{k}"]] = np.where(ok, asz, 0.0)
        X[:, C["bid_depth_total"]] = np.where(ok, vsnap.bdep.values[jj], 0)
        X[:, C["ask_depth_total"]] = np.where(ok, vsnap.adep.values[jj], 0)
        # ---- trades: row = timestamp - open (timestamp in (t-1, t] <=> == t for integer seconds)
        tg = tr_by.get(m.slug)
        n_tx_window = 0
        if tg is not None:
            tg = tg[(tg.timestamp >= o) & (tg.timestamp <= e)]
            if len(tg):
                r = (tg.timestamp.values - o).astype(np.int64)
                sz = tg["size"].values.astype(np.float64)
                px = tg.px_up.values
                ag = tg.aggressor_up.values
                for sgn, vcol, wcol, xcol, fn in ((1, "buy_vol", "buy_vwap", "buy_max_px", np.maximum),
                                                   (-1, "sell_vol", "sell_vwap", "sell_min_px", np.minimum)):
                    msk = ag == sgn
                    vol = np.bincount(r[msk], weights=sz[msk], minlength=n)
                    pv = np.bincount(r[msk], weights=(px * sz)[msk], minlength=n)
                    ext = np.full(n, -np.inf if sgn == 1 else np.inf)
                    fn.at(ext, r[msk], px[msk])
                    has = vol > 0
                    X[:, C[vcol]] = vol
                    X[:, C[wcol]] = np.where(has, pv / np.where(has, vol, 1), 0)
                    X[:, C[xcol]] = np.where(np.isfinite(ext) & has, ext, 0)
                txr = pd.DataFrame({"r": r, "tx": tg.tx.values}).drop_duplicates("tx")
                X[:, C["trade_count"]] = np.bincount(txr.r.values, minlength=n)
                n_tx_window = len(txr)
        # ---- underlying: close known at t = close of kline with open_time t-1
        has_ul, strike_rel = False, 0.0
        ug = ul_by.get(m.coin)
        if ug is not None:
            ot = ug.open_time.values
            cl = ug.close.values
            need = np.arange(o - 2, e, dtype=np.int64)      # open_time t-1 for t in [o-1, e]
            idx = np.searchsorted(ot, need)
            if (idx < len(ot)).all() and (ot[np.minimum(idx, len(ot) - 1)] == need).all():
                has_ul = True
                c = cl[idx]                  # c[0] -> known at o-1, c[1] -> known at o, ...
                lp = np.log(c)
                X[:, C["ul_logp_rel_open"]] = lp[1:] - lp[1]
                X[:, C["ul_logret_1s"]] = lp[1:] - lp[:-1]
                if np.isfinite(m.price_to_beat) and m.price_to_beat > 0:
                    strike_rel = float(lp[1] - np.log(m.price_to_beat))
                    X[:, C["ul_logp_vs_strike"]] = lp[1:] - np.log(m.price_to_beat)
        assert np.isfinite(X).all(), m.slug
        ticks[offsets[i]: offsets[i] + n] = X.astype(np.float32)
        flags = ((FLAG_HAS_TRADES if n_tx_window > 0 else 0) | (FLAG_HAS_UNDERLYING if has_ul else 0)
                 | (FLAG_UL_PERP_PROXY if m.coin == "hype" else 0)
                 | (FLAG_HAS_STRIKE if np.isfinite(m.price_to_beat) and m.price_to_beat > 0 else 0))
        recs[i] = (offsets[i], o, e, n, L, m.coin_id, int(m.winner_up), int(m.split), flags,
                   BASE_TICK, m.fee_rate if m.fees_enabled else 0.0, m.fee_exponent, m.min_size,
                   m.rewards_min_size, m.rewards_max_spread, strike_rel, m.fee_rebate_rate, i, m.tick_size)
        per_market.append({"slug": m.slug, "n_tx_window": n_tx_window})
    assert np.isfinite(ticks).all()
    ticks.tofile(OUT / "ticks.bin")
    recs.tofile(OUT / "markets.bin")

    # ---- train-only normalisation statistics (recorded, NOT applied)
    tr = recs[recs["split"] == 0]
    norm = {}
    for cid, coin in enumerate(COINS):
        rr = tr[(tr["coin_id"] == cid) & ((tr["flags"] & FLAG_HAS_UNDERLYING) > 0)]
        if len(rr) == 0:
            continue
        rows = np.concatenate([np.arange(r["row_offset"] + 1, r["row_offset"] + r["n_rows"]) for r in rr])
        x = ticks[rows, C["ul_logret_1s"]].astype(np.float64)
        norm[coin] = {"ul_logret_1s_mean": float(x.mean()), "ul_logret_1s_std": float(x.std()), "n": int(len(x))}
    # ---- test-split hash
    te = recs[recs["split"] == 2]
    if len(te):
        lo, hi = int(te["row_offset"].min()), int((te["row_offset"] + te["n_rows"]).max())
        assert hi - lo == int(te["n_rows"].sum()), "test rows not contiguous"
        test_sha = hashlib.sha256(ticks[lo:hi].tobytes()).hexdigest()
    else:
        lo = hi = 0
        test_sha = None

    splits = {s: int(v) for s, v in zip(mk.slug, mk.split)}
    (OUT / "splits.json").write_text(json.dumps(splits, indent=0, sort_keys=True))
    off = 0
    layout = []
    for name in MARKET_DTYPE.names:
        dt, o_ = MARKET_DTYPE.fields[name]
        layout.append({"name": name, "offset": o_, "type": C_TYPES[dt.str], "size": dt.itemsize})
    split_counts = {SPLIT_NAMES[k]: {"markets": int((recs["split"] == k).sum()),
                                     "rows": int(recs["n_rows"][recs["split"] == k].sum()),
                                     **{cl: int(((recs["split"] == k) & (recs["contract_len_s"] == L)).sum())
                                        for cl, L in CLEN.items()}}
                    for k in SPLIT_NAMES}
    meta = {
        "build_version": BUILD_VERSION,
        "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "F": F, "N": N, "n_markets": int(len(recs)), "dtype": "float32", "order": "row-major [N, F]",
        "columns": COLS, "column_index": C,
        "market_record": {"itemsize": MARKET_DTYPE.itemsize, "fields": layout,
                          "flags": {"HAS_TRADES": FLAG_HAS_TRADES, "HAS_UNDERLYING": FLAG_HAS_UNDERLYING,
                                    "UL_PERP_PROXY": FLAG_UL_PERP_PROXY, "HAS_STRIKE": FLAG_HAS_STRIKE}},
        "coin_ids": {c: i for i, c in enumerate(COINS)},
        "contract_len_s": CLEN,
        "splits": {"ids": SPLIT_NAMES, "rule": "by market open date UTC: train 2026-08-21..22 (also 08-20 opens), "
                   "val 08-23, test 08-24; purged (3) if [open, end) touches a day of another split",
                   "counts": split_counts},
        "market_order": "sorted by (split, open_ts, slug); each split's rows are contiguous",
        "slugs": list(mk.slug),
        "test_rows": {"row_lo": lo, "row_hi": hi, "sha256": test_sha},
        "normalisation_train_only": norm,
        "sentinels": {"empty_bid_px": 0.0, "empty_ask_px": 1.0, "empty_size": 0.0,
                      "no_trade_vwap_maxmin": 0.0, "no_book": "book_valid=0, book_age=0, empty ladders"},
        "conventions": {
            "grid": "row t for every unix second t in [open, end] inclusive (contract_len_s + 1 rows); "
                    "open = end - contract_len_s; seconds_to_close = end - t",
            "frame": "Up token. Down snapshots mapped as Up bid = 1 - Down ask, Up ask = 1 - Down bid (sizes carried).",
            "book": "latest snapshot with captured_at <= t. Per timestamp the Up snapshot is used if uncrossed, "
                    "else the mapped Down snapshot if uncrossed (book_from_down=1), else the timestamp is crossed "
                    "and the last uncrossed book is forward-filled. book_age = t - captured_at of the book used. "
                    "crossed_raw=1 if the most recent raw timestamp <= t had no uncrossed book.",
            "trades": "fills with integer timestamp == t, i.e. in (t-1, t]. Aggressor side in the Up frame from the "
                      "taker leg; prices are the MAKER legs' resting prices mapped to the Up frame (Down p -> 1-p); "
                      "volumes in shares; trade_count = distinct taker transactions.",
            "underlying": "close of the 1s Binance kline with open_time t-1 (the last kline closed by t). "
                          "ul_logp_rel_open = log(c_t / c_open); ul_logret_1s = log(c_t / c_{t-1}); "
                          "ul_logp_vs_strike = log(c_t / price_to_beat) (Binance vs Chainlink strike; strike is "
                          "fixed at open). HYPE uses Binance USD-M perp aggTrades (flag UL_PERP_PROXY).",
            "leakage": "winner/is_winning_outcome/final_price never appear in ticks.bin; winner_up is only in "
                       "markets.bin for terminal settlement.",
        },
        "mirror_stats": mstats,
        "row_counts": {"ticks": N, "markets": int(len(recs))},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"ticks.bin {N}x{F} ({(OUT/'ticks.bin').stat().st_size/1e6:.1f} MB), markets {len(recs)}, "
          f"splits {json.dumps(split_counts)}  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
