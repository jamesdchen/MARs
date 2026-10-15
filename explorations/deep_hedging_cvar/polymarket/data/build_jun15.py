#!/usr/bin/env python3
"""Protocol v2 TEST build (DECISIONS D7): every BTC 5-minute up/down market of 2026-06-15 (288),
from the Marketlens `polymarket-btc-5m-l2-depth` dataset, converted to the data/build format.

THIS IS THE UNTOUCHED v2 TEST SET. Only integrity checks may run on it (see TEST_LOCK.md written
next to the data): no policy, baseline, tuning or strategy statistic before the v2 plan is frozen.

Inputs
  data/raw/marketlens/{markets,snapshots,deltas,trades}.parquet  HF revision REV below (sha256 checked)
  data/build_jun15/raw/{markets,trades,underlying}.parquet        data/fetch_jun15.py (gamma, data-api,
                                                                  Binance BTCUSDT 1 s), same schemas as
                                                                  data/raw/{markets,trades,underlying}
Outputs (data/build_jun15/, git-ignored; every file written atomically)
  ticks.bin markets.bin meta.json splits.json   EXACTLY the data/build.py schema (40 float32 columns,
                                                88-byte market records), split = 2 for every market
  feat.bin book.bin fills.bin fill_index.bin feat_meta.json
                                                data/features.py, UNMODIFIED, run with its module paths
                                                redirected here (pattern of sim/features_sim.py)
  book_recon.json                               book-reconstruction diagnostics (counts only)
  env_check.json                                C env stepped with a constant no-quote action (load check)
  TEST_LOCK.md                                  lock note with the sha256 of every file

Conventions (row t = unix second t, t in [open, end], as data/build.py):
  * book: the Marketlens book (one Up-frame book per market; Down-token events are folded in by the
    dataset) at exchange time t*1000 ms: latest snapshot with collected_at_ms <= t*1000, plus every
    delta of its chain with platform_timestamp_ms <= t*1000 (data/marketlens_book.py). Levels that a
    Marketlens taker trade at or before t proves empty, and that were last updated more than 50 ms
    before that trade, are hidden (removals that never arrived as deltas, see marketlens_book.py;
    Marketlens trades.parquet is used for nothing else). If the state at t is crossed, the last
    uncrossed row's book is forward-filled and crossed_raw = 1 (data/build.py rule). book_valid = 0 before the first snapshot.
    book_age = whole seconds since the latest feed message (snapshot or delta) behind the book used.
    book_from_down = 0 (the dataset has no per-token books).
  * trades: data-api fill legs with block timestamp == t, i.e. in (t-1, t] (data/build.py code).
  * underlying: close of the Binance BTCUSDT 1 s kline with open_time t-1 (data/build.py code).

    python data/fetch_jun15.py                          # inputs (gamma, data-api, Binance), ~5 min
    python data/build_jun15.py                          # ticks, features, trade agreement, verify, lock (~5 min)
    MM_FINAL=1 python data/build_jun15.py --env-check   # C env, constant no-quote action only
    python data/build_jun15.py --stage lock             # re-hash with env_check.json, rewrite TEST_LOCK.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402  (data/build.py: schema, record layout, trade loader; main() not run)
import marketlens_book as mlb  # noqa: E402

ML = HERE / "raw" / "marketlens"
OUT = HERE / "build_jun15"
RAW = OUT / "raw"
REV = "e4856cdda097e0897d66f0d383e1940f91309d97"       # HF dataset revision downloaded
ML_SHA256 = {                                          # = the Git LFS oids of that revision
    "markets.parquet": "5bc119e26a3184968a518b624fb639e4e15d0c0d5f0a9b9ea714c6a4cf76dfd1",
    "snapshots.parquet": "df6614d7b958a2f0a7980c2547359c8ca5b5aa1b2986ba60ce9db46c5ac1a55d",
    "deltas.parquet": "5796c8ceed13bc3021e141ae75e93be51f1fc385ef4ad0a05abedae9620ce6cf",
    "trades.parquet": "33f677d62398c688f8594e25c63efc352d44c36571bc8629d054c3bffa4edff1",
}
PRE_S = 600                 # replay from 10 min before open (snapshots start <= 171 s before open)
STALE_S = 10
T0_UNDERLYING = int(pd.Timestamp("2026-06-14", tz="UTC").timestamp())   # fv.T0 for the features
SPLIT = 2
NOTE = ("v2 TEST SET (DECISIONS D7): Marketlens polymarket-btc-5m-l2-depth, every BTC 5m up/down market "
        "of 2026-06-15; untouched until the v2 plan is frozen; evaluated once")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_atomic(path: Path, data):
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    if isinstance(data, np.ndarray):
        data.tofile(tmp)
    elif isinstance(data, bytes):
        tmp.write_bytes(data)
    else:
        tmp.write_text(data)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- inputs
def check_inputs():
    for f, want in ML_SHA256.items():
        got = sha256_file(ML / f)
        assert got == want, f"{ML / f}: sha256 {got} != HF LFS oid {want} (LFS pointer file?)"
    for f in ("markets.parquet", "trades.parquet", "underlying.parquet"):
        assert (RAW / f).exists(), f"missing {RAW / f}: run data/fetch_jun15.py first"


def delta_row_groups(pf: pq.ParquetFile) -> dict:
    """market_id -> row groups of deltas.parquet (the file is written one market at a time)."""
    out = {}
    for i in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(i).column(0).statistics
        assert st.min == st.max, "row group spans markets"
        out.setdefault(st.min, []).append(i)
    return out


# ----------------------------------------------------------------------------- book
def market_book(snaps, deltas, ml_trades, o, e, clean=True):
    """Book columns of ticks.bin for rows t = o..e (+ diagnostics)."""
    tgrid = np.arange(o - PRE_S, e + 1, dtype=np.int64)
    Bms = tgrid * 1000
    r = mlb.replay(snaps, deltas, Bms, ml_trades if clean else None)
    have = r["k"] >= 0
    bq0, aq0 = r["bq"][:, 0], r["aq"][:, 0]
    crossed = have & (bq0 >= 0) & (aq0 >= 0) & (bq0 >= aq0)
    good = have & ~crossed
    j = np.where(good, np.arange(len(tgrid)), -1)
    np.maximum.accumulate(j, out=j)
    sl = slice(PRE_S, PRE_S + (e - o) + 1)
    j, crossed, have = j[sl], crossed[sl], have[sl]
    ok = j >= 0
    jj = np.maximum(j, 0)
    n = e - o + 1
    X = {}
    X["book_valid"] = ok.astype(np.float64)
    age = (Bms[sl] - r["last_msg"][jj]) // 1000
    assert (age[ok] >= 0).all()
    X["book_age"] = np.where(ok, age, 0).astype(np.float64)
    X["crossed_raw"] = (have & crossed).astype(np.float64)
    X["book_from_down"] = np.zeros(n)
    for k in range(B.NLEV):
        bq, bs = r["bq"][jj, k], r["bs"][jj, k]
        aq, as_ = r["aq"][jj, k], r["as"][jj, k]
        X[f"bid_px_{k}"] = np.where(ok & (bq >= 0), bq / mlb.PX, 0.0)
        X[f"bid_sz_{k}"] = np.where(ok & (bq >= 0), bs, 0.0)
        X[f"ask_px_{k}"] = np.where(ok & (aq >= 0), aq / mlb.PX, 1.0)
        X[f"ask_sz_{k}"] = np.where(ok & (aq >= 0), as_, 0.0)
    X["bid_depth_total"] = np.where(ok, r["bdep"][jj], 0.0)
    X["ask_depth_total"] = np.where(ok, r["adep"][jj], 0.0)
    diag = {"rows_no_book": int((~ok).sum()), "first_book_s": int(np.argmax(ok)) if ok.any() else n,
            "rows_crossed_raw": int((have & crossed).sum()), "rows_stale": int((X["book_age"] > STALE_S).sum()),
            "rows_with_hidden_levels": int((r["n_hidden"][sl] > 0).sum()), "orphan_deltas": int(r["n_orphan"])}
    return X, diag


def anchor_agreement(snaps, deltas, ml_trades, o, e, clean=True):
    """Replay of chain k with all its deltas vs snapshot k+1, for snapshots inside (open, end]:
    counts of exact best-bid/ask agreement and of exact top-5 price agreement."""
    snaps = snaps.sort_values(["collected_at_ms", "snapshot_id"], kind="stable").reset_index(drop=True)
    res = {"cont": [0, 0, 0], "reseed": [0, 0, 0]}
    for k1 in range(1, len(snaps)):
        sk = snaps.iloc[k1]
        if not (o * 1000 < sk.collected_at_ms <= e * 1000):
            continue
        dk = deltas[deltas.snapshot_id == snaps.snapshot_id.iloc[k1 - 1]]
        t_end = max(int(dk.platform_timestamp_ms.max()) if len(dk) else 0, int(sk.collected_at_ms) - 1)
        r = mlb.replay(snaps.iloc[:k1], dk, np.array([t_end], np.int64), ml_trades if clean else None)
        bq, _ = mlb.snapshot_levels(sk.bids)
        aq, _ = mlb.snapshot_levels(sk.asks)
        tb = np.full(B.NLEV, -1, np.int64)
        ta = np.full(B.NLEV, -1, np.int64)
        tb[:min(B.NLEV, len(bq))] = bq[::-1][:B.NLEV]
        ta[:min(B.NLEV, len(aq))] = aq[:B.NLEV]
        c = res["reseed" if sk.is_reseed else "cont"]
        c[0] += 1
        c[1] += int(r["bq"][0, 0] == tb[0] and r["aq"][0, 0] == ta[0])
        c[2] += int((r["bq"][0] == tb).all() and (r["aq"][0] == ta).all())
    return res


# ----------------------------------------------------------------------------- ticks
def build_ticks():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    check_inputs()
    C, F = B.C, B.F
    mk = pd.read_parquet(RAW / "markets.parquet")
    mk["contract_len_s"] = mk.contract_length.map(B.CLEN).astype(int)
    mk["end_ts"] = mk.end.map(lambda x: int(pd.Timestamp(x).timestamp()))
    mk["open_ts"] = mk.end_ts - mk.contract_len_s
    ev_open = mk.event_start.map(lambda x: int(pd.Timestamp(x).timestamp()))
    assert (ev_open == mk.open_ts).all(), "eventStartTime != end - contract length"
    assert (mk.coin == "btc").all() and (mk.contract_len_s == 300).all()
    open_day = pd.to_datetime(mk.open_ts, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    assert (open_day == "2026-06-15").all()
    mk["split"] = SPLIT
    mk = mk.sort_values(["split", "open_ts", "slug"]).reset_index(drop=True)
    mk["coin_id"] = mk.coin.map({c: i for i, c in enumerate(B.COINS)}).astype(int)

    mlm = pd.read_parquet(ML / "markets.parquet").set_index("market_id")
    assert set(mlm.index) == set(mk.condition_id) and len(mlm) == len(mk) == 288
    snaps = pd.read_parquet(ML / "snapshots.parquet")
    assert snaps.snapshot_id.is_unique
    snap_by = {m: g for m, g in snaps.groupby("market_id", sort=False)}
    mlt = pd.read_parquet(ML / "trades.parquet", columns=["market_id", "platform_timestamp_ms", "price", "size", "side"])
    assert np.isin(mlt.side.values, ["BUY", "SELL"]).all()
    mlt_by = {m: g for m, g in mlt.groupby("market_id", sort=False)}
    pf = pq.ParquetFile(ML / "deltas.parquet")
    rg = delta_row_groups(pf)

    B.RAW = RAW                                  # build.load_trades() reads RAW / "trades.parquet"
    trades = B.load_trades()
    tr_by = {s: g for s, g in trades[trades.use].groupby("slug", sort=False)}
    ul = pd.read_parquet(RAW / "underlying.parquet")
    ul_by = {c: g for c, g in ul.groupby("coin", observed=True)}
    print(f"inputs loaded {time.time() - t0:.1f}s", flush=True)

    n_rows = (mk.contract_len_s + 1).values
    offsets = np.concatenate([[0], np.cumsum(n_rows)[:-1]]).astype(np.int64)
    N = int(n_rows.sum())
    ticks = np.zeros((N, F), dtype=np.float32)
    recs = np.zeros(len(mk), dtype=B.MARKET_DTYPE)
    per_market, tot = [], {}
    anchors = {"clean": {"cont": [0, 0, 0], "reseed": [0, 0, 0]}, "plain": {"cont": [0, 0, 0], "reseed": [0, 0, 0]}}
    n_deltas = n_sub_cent = 0
    for i, m in mk.iterrows():
        L, o, e = int(m.contract_len_s), int(m.open_ts), int(m.end_ts)
        n = L + 1
        tgrid = np.arange(o, e + 1, dtype=np.int64)
        X = np.zeros((n, F), dtype=np.float64)
        X[:, C["seconds_to_close"]] = e - tgrid
        X[:, C["const_one"]] = 1.0
        # ---- book (Marketlens replay)
        mid = m.condition_id
        sn = snap_by[mid]
        d = pf.read_row_groups(rg[mid], columns=["snapshot_id", "platform_timestamp_ms", "price", "size", "side"]).to_pandas()
        n_deltas += len(d)
        n_sub_cent += int((np.rint(d.price.values * mlb.PX) % 100 != 0).sum())
        mt = mlt_by.get(mid, mlt.iloc[:0])
        cols, diag = market_book(sn, d, mt, o, e, clean=True)
        for k, v in cols.items():
            X[:, C[k]] = v
        _, diag_plain = market_book(sn, d, None, o, e, clean=False)
        diag["rows_crossed_raw_without_trade_cleaning"] = diag_plain["rows_crossed_raw"]
        for var, cl in (("clean", True), ("plain", False)):
            a = anchor_agreement(sn, d, mt, o, e, clean=cl)
            for kk in ("cont", "reseed"):
                anchors[var][kk] = [x + y for x, y in zip(anchors[var][kk], a[kk])]
        # ---- trades: row = timestamp - open (timestamp in (t-1, t] <=> == t for integer seconds)
        #      (verbatim from data/build.py main())
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
        # ---- underlying: close known at t = close of kline with open_time t-1 (verbatim from build.py)
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
        flags = ((B.FLAG_HAS_TRADES if n_tx_window > 0 else 0) | (B.FLAG_HAS_UNDERLYING if has_ul else 0)
                 | (B.FLAG_UL_PERP_PROXY if m.coin == "hype" else 0)
                 | (B.FLAG_HAS_STRIKE if np.isfinite(m.price_to_beat) and m.price_to_beat > 0 else 0))
        recs[i] = (offsets[i], o, e, n, L, m.coin_id, int(m.winner_up), int(m.split), flags,
                   B.BASE_TICK, m.fee_rate if m.fees_enabled else 0.0, m.fee_exponent, m.min_size,
                   m.rewards_min_size, m.rewards_max_spread, strike_rel, m.fee_rebate_rate, i, m.tick_size)
        per_market.append({"slug": m.slug, "condition_id": mid, "n_snapshots": int(len(sn)),
                           "n_reseed": int(sn.is_reseed.sum()), "n_deltas": int(len(d)),
                           "n_tx_window": n_tx_window, "has_underlying": has_ul, **diag})
        for k, v in diag.items():
            if k != "first_book_s":
                tot[k] = tot.get(k, 0) + v
        if (i + 1) % 48 == 0:
            print(f"  {i + 1}/{len(mk)} markets  {time.time() - t0:.0f}s", flush=True)
    assert np.isfinite(ticks).all()
    assert (recs["winner_up"] == (mlm.loc[mk.condition_id, "winning_outcome"] == "Up").astype(int).values).all()

    # ---- test-split hash (every row is test)
    te = recs[recs["split"] == SPLIT]
    lo, hi = int(te["row_offset"].min()), int((te["row_offset"] + te["n_rows"]).max())
    assert lo == 0 and hi == N and hi - lo == int(te["n_rows"].sum())
    test_sha = hashlib.sha256(ticks[lo:hi].tobytes()).hexdigest()
    write_atomic(OUT / "ticks.bin", ticks)
    write_atomic(OUT / "markets.bin", recs)
    splits = {s: int(v) for s, v in zip(mk.slug, mk.split)}
    write_atomic(OUT / "splits.json", json.dumps(splits, indent=0, sort_keys=True))

    v1 = json.loads((HERE / "build" / "meta.json").read_text())
    layout = []
    for name in B.MARKET_DTYPE.names:
        dt, o_ = B.MARKET_DTYPE.fields[name]
        layout.append({"name": name, "offset": o_, "type": B.C_TYPES[dt.str], "size": dt.itemsize})
    split_counts = {B.SPLIT_NAMES[k]: {"markets": int((recs["split"] == k).sum()),
                                       "rows": int(recs["n_rows"][recs["split"] == k].sum()),
                                       **{cl: int(((recs["split"] == k) & (recs["contract_len_s"] == L)).sum())
                                          for cl, L in B.CLEN.items()}}
                    for k in B.SPLIT_NAMES}
    pm = pd.DataFrame(per_market)
    recon = {
        "dataset": {"hf": "marketlens/polymarket-btc-5m-l2-depth", "revision": REV, "sha256": ML_SHA256},
        "markets": int(len(mk)), "rows": N, "snapshots": int(len(snaps)), "snapshots_reseed": int(snaps.is_reseed.sum()),
        "deltas": int(n_deltas), "deltas_sub_cent_price": int(n_sub_cent),
        "marketlens_trades": int(len(mlt)),
        "totals": tot,
        "markets_without_book_at_open": int((pm.first_book_s > 0).sum()),
        "first_book_s_of_those": sorted(pm.first_book_s[pm.first_book_s > 0].tolist()),
        "anchor_agreement": {
            "what": "snapshot k + all deltas of its chain (replay) vs snapshot k+1, for snapshots inside "
                    "(open, end]; [anchors, best bid and best ask equal, top-5 prices equal]",
            "with_trade_cleaning": anchors["clean"], "without_trade_cleaning": anchors["plain"]},
        "per_market": per_market,
    }
    write_atomic(OUT / "book_recon.json", json.dumps(recon, indent=1))
    conv = dict(v1["conventions"])
    conv["frame"] = ("Up token. The Marketlens book is YES (= Up) centric: Down-token events are folded in by "
                     "the dataset as 1 - price with the side flipped; book_from_down is always 0.")
    conv["book"] = ("Marketlens replay at exchange time t*1000 ms: latest snapshot with collected_at_ms <= t*1000 "
                    "(re-anchored on every snapshot) plus every delta of its chain with platform_timestamp_ms <= "
                    "t*1000 (time order, ties in file order). Levels proven empty by a Marketlens taker trade at "
                    "or before t (BUY at p: no ask below p; SELL at p: no bid above p) and last updated more than "
                    f"{mlb.GUARD_MS} ms before that trade are hidden. If the state at t is crossed, the last "
                    "uncrossed row's book is forward-filled and "
                    "crossed_raw=1. book_valid=0 before the first snapshot. book_age = whole seconds since the "
                    "latest snapshot or delta behind the book used (data/marketlens_book.py, data/build_jun15.py).")
    conv["underlying"] = ("close of the 1s Binance BTCUSDT spot kline with open_time t-1 (the last kline closed "
                          "by t). ul_logp_rel_open = log(c_t / c_open); ul_logret_1s = log(c_t / c_{t-1}); "
                          "ul_logp_vs_strike = log(c_t / price_to_beat) (Binance vs Chainlink strike).")
    meta = {
        "build_version": B.BUILD_VERSION,
        "built_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "F": F, "N": N, "n_markets": int(len(recs)), "dtype": "float32", "order": "row-major [N, F]",
        "columns": B.COLS, "column_index": C,
        "market_record": {"itemsize": B.MARKET_DTYPE.itemsize, "fields": layout,
                          "flags": {"HAS_TRADES": B.FLAG_HAS_TRADES, "HAS_UNDERLYING": B.FLAG_HAS_UNDERLYING,
                                    "UL_PERP_PROXY": B.FLAG_UL_PERP_PROXY, "HAS_STRIKE": B.FLAG_HAS_STRIKE}},
        "coin_ids": {c: i for i, c in enumerate(B.COINS)},
        "contract_len_s": B.CLEN,
        "splits": {"ids": B.SPLIT_NAMES, "rule": "v2 test (DECISIONS D7): every market is split 2 (test)",
                   "counts": split_counts},
        "market_order": "sorted by (split, open_ts, slug); each split's rows are contiguous",
        "slugs": list(mk.slug),
        "test_rows": {"row_lo": lo, "row_hi": hi, "sha256": test_sha},
        "normalisation_train_only": v1["normalisation_train_only"],
        "normalisation_note": "copied from data/build/meta.json (train split of v1, Aug 21-22); not applied",
        "sentinels": v1["sentinels"],
        "conventions": conv,
        "protocol_version": 2,
        "v2_note": NOTE,
        "source": {
            "book": f"Hugging Face marketlens/polymarket-btc-5m-l2-depth @ {REV} (CC BY 4.0, Marketlens; "
                    "doi 10.5281/zenodo.21933084): markets, snapshots, deltas, trades (trades only for ghost-level "
                    "removal)",
            "metadata": "gamma-api /markets?slug=<slug>&closed=true (data/fetch_jun15.py -> raw/markets.parquet)",
            "fills": "data-api /trades?market=<conditionId>&takerOnly=true|false (raw/trades.parquet)",
            "underlying": "data.binance.vision BTCUSDT 1s klines 2026-06-14..15 (raw/underlying.parquet)"},
        "book_reconstruction": {k: v for k, v in recon.items() if k != "per_market"},
        "first_book_s": {s: int(x) for s, x in zip(pm.slug, pm.first_book_s)},
        "row_counts": {"ticks": N, "markets": int(len(recs))},
    }
    write_atomic(OUT / "meta.json", json.dumps(meta, indent=1))
    print(f"ticks.bin {N}x{F}, markets {len(recs)}  {time.time() - t0:.0f}s")
    print("totals:", tot)
    print("anchors:", anchors)


# ----------------------------------------------------------------------------- features
def run_features():
    """data/features.py, unmodified, with its paths redirected to this build."""
    import features as Fe               # data/features.py
    from load import load
    Fe.BUILD = OUT
    Fe.RAW = RAW                        # markets.parquet (created_start, end), trades.parquet, underlying.parquet
    Fe.load = lambda: load(OUT)
    Fe.fv.T0 = T0_UNDERLYING            # pricer indexing: close[i] <-> kline open_time T0 + i (2026-06-14 00:00)
    Fe.main()


# ----------------------------------------------------------------------------- integrity
def integrity():
    """Integrity checks only (no strategy statistics)."""
    from load import load, verify
    verify(OUT)
    ticks, markets, meta = load(OUT)
    fm = json.loads((OUT / "feat_meta.json").read_text())
    v1f = json.loads((HERE / "build" / "feat_meta.json").read_text())
    N = meta["N"]
    assert fm["N"] == N and fm["G"] == v1f["G"] and fm["column_index"] == v1f["column_index"]
    assert fm["book_record"] == v1f["book_record"] and fm["fill_record"] == v1f["fill_record"]
    assert meta["columns"] == json.loads((HERE / "build" / "meta.json").read_text())["columns"]
    feat = np.memmap(OUT / "feat.bin", np.float32, "r", shape=(N, fm["G"]))
    assert np.isfinite(feat).all(), "NaN/inf in feat.bin"
    assert np.abs(feat[:, :fm["n_obs_features"]]).max() <= 5.0 + 1e-6
    bd = np.dtype([("bid_px", "<i2", 5), ("ask_px", "<i2", 5), ("bid_sz", "<f4", 5), ("ask_sz", "<f4", 5),
                   ("flags", "u1"), ("tick", "u1"), ("tick_alt", "u1"), ("same_run", "u1")])
    book = np.fromfile(OUT / "book.bin", bd)
    assert len(book) == N
    for f in ("bid_sz", "ask_sz"):
        assert np.isfinite(book[f]).all() and (book[f] >= 0).all()
    c = meta["column_index"]
    assert ((book["flags"] & 1) == (np.asarray(ticks[:, c["book_valid"]]) > 0)).all()
    assert (((book["flags"] & 2) > 0) == (np.asarray(ticks[:, c["crossed_raw"]]) > 0)).all()
    assert np.isin(book["tick"], [1, 10]).all()
    fd = np.dtype([("row", "<i4"), ("side", "i1"), ("first", "u1"), ("px", "<i2"), ("price", "<f4"),
                   ("size", "<f4"), ("tx_size", "<f4")])
    fills = np.fromfile(OUT / "fills.bin", fd)
    idx = np.fromfile(OUT / "fill_index.bin", "<i4")
    assert len(idx) == N + 1 and idx[0] == 0 and idx[-1] == len(fills) and (np.diff(idx) >= 0).all()
    assert (np.diff(fills["row"]) >= 0).all() and (fills["row"] >= 0).all() and (fills["row"] < N).all()
    assert (np.searchsorted(fills["row"], np.arange(N + 1), side="left") == idx).all()
    assert np.isin(fills["side"], [-1, 1]).all() and (fills["size"] > 0).all()
    assert ((fills["px"] > 0) & (fills["px"] < 1000)).all() and np.isfinite(fills["price"]).all()
    # every fill row lies inside its market and inside [open, end]
    mo = np.searchsorted(markets["row_offset"], fills["row"], side="right") - 1
    assert (fills["row"] < markets["row_offset"][mo] + markets["n_rows"][mo]).all()
    stc = np.asarray(ticks[:, c["seconds_to_close"]])
    assert (stc[fills["row"]] >= 0).all()
    vol = np.bincount(fills["row"], weights=fills["size"].astype(np.float64), minlength=N)
    tv = np.asarray(ticks[:, c["buy_vol"]], np.float64) + np.asarray(ticks[:, c["sell_vol"]], np.float64)
    assert np.abs(vol - tv).max() < 1e-2
    assert (markets["split"] == SPLIT).all()
    return {"verify": "ok", "N": int(N), "G": int(fm["G"]), "n_markets": int(len(markets)),
            "n_fills": int(len(fills)), "markets_with_fills": int((np.bincount(mo, minlength=len(markets)) > 0).sum()),
            "checks": ["load.verify (finite, offsets, seconds_to_close monotone, ladders, test sha256)",
                       "feat.bin finite, |obs| <= 5, same column_index as data/build",
                       "book.bin flags == ticks book_valid / crossed_raw; tick in {1, 10} mils",
                       "fills sorted by row, inside their market window; fill_index consistent",
                       "per-row fill volume == ticks buy_vol + sell_vol"]}


def env_check():
    """Load check of the C env on this build: allow_test=1 + MM_FINAL=1, CONSTANT NO-QUOTE action only.
    Records counts only (steps, finished episodes, finite observations)."""
    assert os.environ.get("MM_FINAL") == "1", "run with MM_FINAL=1"
    sys.path.insert(0, str(HERE.parent))
    import env as E
    n = 288
    envs = E.MmEnv(num_envs=n, data_dir=str(OUT), seed=0, log_interval=10 ** 9, split=SPLIT, allow_test=1,
                   cycle_stride=n, fixed_start=0)
    obs, _ = envs.reset(seed=0)
    act = np.zeros((n, 3), np.int32)           # bid off, ask off, size 0: no quote, ever
    done = np.zeros(n, np.int64)
    finite = bool(np.isfinite(obs).all())
    steps = 0
    while steps < 2000 and done.min() < 1:
        obs, rew, term, trunc, _ = envs.step(act)
        assert (act == 0).all()
        finite &= bool(np.isfinite(obs).all() and np.isfinite(rew).all())
        done += (term | trunc).astype(np.int64)
        steps += 1
    envs.close()
    out = {"action": "constant [0, 0, 0] (no bid, no ask)", "num_envs": n, "cycle_stride": n, "fixed_start": 0,
           "steps": steps, "envs_with_finished_episode": int((done > 0).sum()), "obs_and_reward_finite": finite,
           "note": "load/step check only; no policy, baseline or learned model was run; no P&L recorded"}
    write_atomic(OUT / "env_check.json", json.dumps(out, indent=1))
    print(out)
    assert out["envs_with_finished_episode"] == n and finite


def trade_agreement():
    """data-api taker fills vs Marketlens trades.parquet: counts, volumes, one-to-one matching.
    A Marketlens trade is one taker order: exchange time (ms), taker side in the Up frame, size, and the
    taker's average price rounded to the cent. A data-api taker leg is one tx at block time (s)."""
    mk = pd.read_parquet(RAW / "markets.parquet")
    mk["end_ts"] = mk.end.map(lambda x: int(pd.Timestamp(x).timestamp()))
    mk["open_ts"] = mk.end_ts - 300
    da = pd.read_parquet(RAW / "trades.parquet")
    da = da[da.is_taker].copy()
    da["slug"] = da.slug.astype(str)
    da["px_up"] = np.where(da.outcome_idx == 0, da.price.astype(np.float64), 1.0 - da.price.astype(np.float64))
    da_by = {s: g for s, g in da.groupby("slug", sort=False)}
    ml = pd.read_parquet(ML / "trades.parquet", columns=["market_id", "platform_timestamp_ms", "price", "size", "side"])
    ml_by = {m: g for m, g in ml.groupby("market_id", sort=False)}
    first_snap = pd.read_parquet(ML / "snapshots.parquet", columns=["market_id", "collected_at_ms"]) \
        .groupby("market_id").collected_at_ms.min()
    tot = {"ml_all": [0, 0.0], "da_all": [0, 0.0], "ml_win": [0, 0.0], "da_win": [0, 0.0], "ml_matched": [0, 0.0],
           "da_win_matched": [0, 0.0]}
    lags, ratio_n, ratio_v, unmatched = [], [], [], []
    for m in mk.itertuples():
        o, e = int(m.open_ts), int(m.end_ts)
        a = ml_by.get(m.condition_id, ml.iloc[:0])
        b = da_by.get(m.slug, da.iloc[:0])
        at_all = np.sort(a.platform_timestamp_ms.values) / 1000.0
        tot["ml_all"][0] += len(a); tot["ml_all"][1] += float(a["size"].sum())
        tot["da_all"][0] += len(b); tot["da_all"][1] += float(b["size"].sum())
        a = a[(a.platform_timestamp_ms >= o * 1000) & (a.platform_timestamp_ms < (e + 1) * 1000)]
        bw = b[(b.timestamp >= o) & (b.timestamp <= e)]
        tot["ml_win"][0] += len(a); tot["ml_win"][1] += float(a["size"].sum())
        tot["da_win"][0] += len(bw); tot["da_win"][1] += float(bw["size"].sum())
        if len(bw):
            ratio_n.append(len(a) / len(bw))
            ratio_v.append(float(a["size"].sum()) / float(bw["size"].sum()))
        # one-to-one: same Up-frame side, size within 0.006, price within half a cent, block time in
        # [match time - 1 s, match time + 30 s]; earliest unmatched candidate
        used = np.zeros(len(b), bool)
        bt = b.timestamp.values.astype(np.int64)
        order = np.argsort(bt, kind="stable")
        bt, bsz, bpx, bsd = bt[order], b["size"].values[order].astype(np.float64), b.px_up.values[order], b.aggressor_up.values[order]
        inwin = (bt >= o) & (bt <= e)
        for ts, px, sz, sd in zip(a.platform_timestamp_ms.values, a.price.values, a["size"].values, a.side.values):
            s = 1 if sd == "BUY" else -1
            lo = np.searchsorted(bt, ts // 1000 - 1, side="left")
            hi = np.searchsorted(bt, ts // 1000 + 30, side="right")
            for j in range(lo, hi):
                if not used[j] and bsd[j] == s and abs(bsz[j] - sz) <= 0.006 and abs(bpx[j] - px) <= 0.0051:
                    used[j] = True
                    tot["ml_matched"][0] += 1; tot["ml_matched"][1] += float(sz)
                    lags.append(int(bt[j]) - ts / 1000.0)
                    break
        tot["da_win_matched"][0] += int((used & inwin).sum()); tot["da_win_matched"][1] += float(bsz[used & inwin].sum())
        unmatched.append((bt[~used & inwin], first_snap.get(m.condition_id, np.inf) / 1000.0, at_all))
    lags = np.array(lags)
    # where are the data-api fills without a Marketlens trade? exchange time ~ block time - median lag
    med = float(np.median(lags)) if len(lags) else 0.0
    why = {"before_first_marketlens_snapshot": 0, "in_marketlens_trade_gap_gt_5s": 0, "inside_marketlens_coverage": 0}
    for bts, fs, at_all in unmatched:
        ex = bts - med
        before = ex < fs
        if len(at_all):
            k = np.searchsorted(at_all, ex)
            prev_t = np.where(k > 0, at_all[np.maximum(k - 1, 0)], -np.inf)
            next_t = np.where(k < len(at_all), at_all[np.minimum(k, len(at_all) - 1)], np.inf)
            gap = (next_t - prev_t) > 5
        else:
            gap = np.ones(len(ex), bool)
        why["before_first_marketlens_snapshot"] += int(before.sum())
        why["in_marketlens_trade_gap_gt_5s"] += int((~before & gap).sum())
        why["inside_marketlens_coverage"] += int((~before & ~gap).sum())
    out = {
        "definitions": {
            "ml": "Marketlens trades.parquet rows (one per taker order, exchange time in ms)",
            "da": "data-api taker legs (takerOnly=true rows = one per tx, block time in s)",
            "win": "inside the market window: Marketlens exchange time in [open, end+1) s; data-api block time in [open, end]",
            "match": "same Up-frame taker side, |size diff| <= 0.006, |Marketlens price - data-api avg price (Up frame)| "
                     "<= 0.0051, block time in [exchange time - 1 s, + 30 s], earliest unmatched data-api leg"},
        "counts_volumes": {k: {"n": int(v[0]), "shares": round(v[1], 2)} for k, v in tot.items()},
        "matched_frac_of_ml_win": round(tot["ml_matched"][0] / max(tot["ml_win"][0], 1), 4),
        "matched_frac_of_da_win": round(tot["da_win_matched"][0] / max(tot["da_win"][0], 1), 4),
        "matched_volume_frac_of_ml_win": round(tot["ml_matched"][1] / max(tot["ml_win"][1], 1e-9), 4),
        "unmatched_da_win_by_where": why,
        "unmatched_da_win_note": "exchange time of a data-api fill estimated as block time - median lag; a gap is "
                                 "> 5 s between consecutive Marketlens trades of the market",
        "block_minus_exchange_time_s_quantiles": {str(q): round(float(np.quantile(lags, q)), 3)
                                                  for q in (0.01, 0.1, 0.5, 0.9, 0.99)} if len(lags) else None,
        "per_market_count_ratio_ml_over_da_win_quantiles": {str(q): round(float(np.quantile(ratio_n, q)), 3)
                                                            for q in (0.05, 0.5, 0.95)},
        "per_market_volume_ratio_ml_over_da_win_quantiles": {str(q): round(float(np.quantile(ratio_v, q)), 3)
                                                             for q in (0.05, 0.5, 0.95)},
    }
    write_atomic(OUT / "trade_agreement.json", json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def lock(integ: dict):
    files = sorted(p.name for p in OUT.iterdir() if p.is_file() and p.name not in ("meta.json", "TEST_LOCK.md")
                   and not p.name.startswith("."))
    files += sorted(f"raw/{p.name}" for p in RAW.iterdir() if p.is_file() and not p.name.startswith("."))
    meta = json.loads((OUT / "meta.json").read_text())
    meta["integrity"] = integ
    if (OUT / "trade_agreement.json").exists():
        meta["trade_agreement"] = json.loads((OUT / "trade_agreement.json").read_text())
    meta["file_sha256"] = {f: sha256_file(OUT / f) for f in files}
    meta["file_sha256_note"] = "every file of data/build_jun15 except meta.json and TEST_LOCK.md (see TEST_LOCK.md)"
    meta["inputs_sha256"] = {f"data/raw/marketlens/{k}": v for k, v in ML_SHA256.items()}
    write_atomic(OUT / "meta.json", json.dumps(meta, indent=1))
    write_atomic(OUT / "TEST_LOCK.md", lock_text(meta, sha256_file(OUT / "meta.json")))
    return meta


def trade_line(meta: dict) -> str:
    if "trade_agreement" not in meta:
        return ""
    ta = meta["trade_agreement"]
    cv = ta["counts_volumes"]
    lag = ta["block_minus_exchange_time_s_quantiles"]
    w = ta["unmatched_da_win_by_where"]
    return (f": inside the market windows, {100 * ta['matched_frac_of_ml_win']:.2f}% of the "
            f"{cv['ml_win']['n']:,} Marketlens taker trades ({100 * ta['matched_volume_frac_of_ml_win']:.2f}% of their "
            f"shares) match a data-api taker fill one-to-one (block time - exchange time median {lag['0.5']} s, "
            f"1-99% {lag['0.01']}-{lag['0.99']} s); the other way, {100 * ta['matched_frac_of_da_win']:.1f}% of the "
            f"{cv['da_win']['n']:,} data-api taker fills have a Marketlens trade. Of the rest, "
            f"{w['before_first_marketlens_snapshot']:,} fall before Marketlens' first snapshot of the market, "
            f"{w['in_marketlens_trade_gap_gt_5s']:,} in gaps of its trade feed and {w['inside_marketlens_coverage']:,} "
            "inside its covered time, so the trade-implied cleaning above misses some removals")


def lock_text(meta: dict, meta_sha: str) -> str:
    br = meta["book_reconstruction"]
    tot = br["totals"]
    an = br["anchor_agreement"]
    env = json.loads((OUT / "env_check.json").read_text()) if (OUT / "env_check.json").exists() else None
    rows = "\n".join(f"| `{f}` | {(OUT / f).stat().st_size:,} | `{h}` |" for f, h in meta["file_sha256"].items())
    ins = "\n".join(f"| `{f}` | `{h}` |" for f, h in meta["inputs_sha256"].items())
    late = br["first_book_s_of_those"]
    return f"""# TEST LOCK: `data/build_jun15/` is the untouched v2 test set (DECISIONS D7)

**What.** Every BTC 5-minute up/down market of 2026-06-15 UTC: {meta['n_markets']} markets,
{meta['N']:,} one-second rows, all split 2 (test), in exactly the `data/build` format (ticks.bin,
markets.bin, meta.json, splits.json from `data/build_jun15.py`; feat.bin, book.bin, fills.bin,
fill_index.bin, feat_meta.json from the unmodified `data/features.py`).

**Rule.** Nothing may be run on these files before the v2 plan is frozen and hashed: no policy,
baseline, learned model or tuning run, and no strategy statistic (spreads, fill rates, P&L, markouts,
distribution comparisons). The test runs once and every result is reported (D7). The C env refuses
split 2 unless `allow_test=1` and `MM_FINAL=1` are both set.

**What has touched it so far** (built {meta['built_at_utc']}):
1. The conversion itself (`data/fetch_jun15.py`, `data/build_jun15.py`, `data/marketlens_book.py`).
2. Integrity checks only: {'; '.join(meta['integrity']['checks'])}.
3. {'The C env stepped with a constant no-quote action [0, 0, 0] (allow_test=1, MM_FINAL=1), ' + str(env['num_envs']) + ' envs x ' + str(env['steps']) + ' steps, every market played once, observations finite (env_check.json). No policy, no P&L.' if env else 'C env load check: not run yet.'}

Counts recorded during the conversion (data integrity, not strategy statistics) are in
`book_recon.json` and `meta.json` (`book_reconstruction`).

## Sources and license

* Order book: Hugging Face `marketlens/polymarket-btc-5m-l2-depth`, revision `{REV}`,
  **CC BY 4.0** (attribution required): Marketlens, "Polymarket Limit Order Book: 5-Minute Bitcoin
  Markets, One Complete Day (2026-06-15)", Zenodo, doi:10.5281/zenodo.21933084. Derived from publicly
  available Polymarket data. Raw files in `data/raw/marketlens/` (git-ignored); the dataset README is
  kept there.
* Market metadata: Polymarket gamma API by slug (`closed=true`). Fills: Polymarket data-api
  `/trades` (`takerOnly=true` and `false`), the same source and code as the training data
  (`data/fetch_trades.py`). Underlying: Binance BTCUSDT 1 s klines from data.binance.vision.

## Conversion caveats (stated before any result)

* Different capture: Marketlens WebSocket snapshots (about every 60 s) plus millisecond deltas,
  replayed to 1 s rows, instead of PolyOrderBooks 1 s snapshots. `book_age` is the time since the
  last feed message, so it is 0 while the feed is live; `book_from_down` is always 0.
* Not every level removal arrives as a delta. Levels proven empty by a Marketlens taker trade
  (and last updated more than {mlb.GUARD_MS} ms before it) are hidden (rows with a hidden level:
  {tot['rows_with_hidden_levels']:,}); Marketlens trades are used for nothing else. Rows crossed and
  forward-filled (`crossed_raw`): {tot['rows_crossed_raw']:,} with this cleaning,
  {tot['rows_crossed_raw_without_trade_cleaning']:,} without it.
* Replay vs the next snapshot (anchors inside the market window; [anchors, best bid and ask equal,
  top-5 prices equal]): continuous chains {an['with_trade_cleaning']['cont']}, chains ending in a
  reseed {an['with_trade_cleaning']['reseed']} (without cleaning {an['without_trade_cleaning']['cont']},
  {an['without_trade_cleaning']['reseed']}). Some snapshots disagree with the deltas and trades around
  them; snapshots are still treated as authoritative (dataset README), so errors last at most one chain.
* {len(late)} markets have no Marketlens book at open (first snapshot {min(late) if late else 0}-{max(late) if late else 0} s
  after open; {tot['rows_no_book']:,} rows with `book_valid = 0`). They are kept, with the same
  convention as data/build (v1 5m markets: 57 of 471 started without a book). `meta.json`
  `first_book_s` lists the offset per market.
* Fills come from data-api maker legs at block time, as in training. Agreement with Marketlens
  trades.parquet is in `trade_agreement.json`{trade_line(meta)}.
* No market was dropped.

## sha256

`meta.json`: `{meta_sha}`

| file | bytes | sha256 |
|---|---:|---|
{rows}

Inputs:

| file | sha256 |
|---|---|
{ins}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-check", action="store_true", help="only the constant no-quote env load check")
    ap.add_argument("--stage", choices=["all", "ticks", "features", "trades", "lock"], default="all")
    args = ap.parse_args()
    if args.env_check:
        env_check()
        return
    if args.stage in ("all", "ticks"):
        build_ticks()
    if args.stage in ("all", "features"):
        run_features()
    if args.stage in ("all", "trades"):
        trade_agreement()
    if args.stage in ("all", "lock"):
        integ = integrity()
        print("integrity:", integ)
        lock(integ)


if __name__ == "__main__":
    main()
