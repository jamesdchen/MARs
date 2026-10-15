#!/usr/bin/env python3
"""Panels for the order-flow study: decision rows and touch fills, with every candidate
sub-second Binance feature (data/features_v2.py) and the targets that make fills toxic.

LEAKAGE RULES (DECISIONS.md D7, task brief):
  * `train` panels read only train markets (split 0, Aug 21-22) and Binance trades before
    2026-08-23 00:00 UTC (asserted).
  * `val` panels (Aug 23, plus Aug 24 when data/build_v2 relabels it val) are built only for the
    final confirmation, after every choice is frozen on train.
  * Jun 15 (data/build_jun15) is never read here.

Rows (decision grid of the env): every row of 5m/15m markets, every 5th row of 4h (dt_4h = 5),
minus the last row. Fills: taker txs whose sweep reached the touch of the match-row book
(central model: match row tau = print row - 4), with features at the decision row tau - 1
(latency 1), i.e. what the agent knew when it chose to leave that quote in place.

    /usr/bin/python3 -I analysis/orderflow/panel.py --split train
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
POLY = HERE.parents[1]
sys.path.insert(0, str(POLY / "data"))
sys.path.insert(0, str(POLY))
import features_v2 as f2  # noqa: E402
from reference import Data  # noqa: E402

CACHE = HERE / "cache"
D_MATCH = 4          # central fill model: print row u matched at tau = u - 4
LAT = 1              # latency: the quote live at match row tau was decided at tau - 1
T_TRAIN_END = int(pd.Timestamp("2026-08-23", tz="UTC").timestamp())
BASE_COLS = list(range(33)) + [41, 42]          # every observed feat.bin column
COINS = {0: "btc", 1: "eth", 2: "sol", 3: "xrp", 4: "bnb", 5: "doge", 6: "hype", 7: "zec"}


def candidates(g, gb, sg, dusd, lnp_snap, lnp_chg, since_chg, gi, gbi):
    """Every candidate column at the given grid indices (row t <-> grid index t - g.t0)."""
    s = np.maximum(sg, 1e-7)
    vol = g.vol_ewma[gi]
    C = {}
    C["r250_z"] = g.ret[1][gi] / (s * np.sqrt(0.25))
    C["r500_z"] = g.ret[2][gi] / (s * np.sqrt(0.5))
    C["r1_z"] = g.ret[4][gi] / s
    C["r10_z"] = g.ret[40][gi] / (s * np.sqrt(10))
    for k, w in ((1, 0.25), (4, 1), (20, 5), (40, 10)):
        C[f"sv{w:g}"] = np.arcsinh(g.net[k][gi] / (vol * w + 1.0))
    for k, w in ((1, 0.25), (4, 1), (20, 5)):
        C[f"ofi{w:g}"] = g.net[k][gi] / (g.tot[k][gi] + 0.25 * vol * w + 1.0)
    C["act1"] = np.log((g.tot[4][gi] + 1.0) / (vol + 1.0))
    C["cnt1"] = np.log((g.cnt[4][gi] + 1.0) / (g.cnt_ewma[gi] + 1.0))
    C["hl1_z"] = g.hl1[gi] / s
    C["rv10_z"] = g.rv[40][gi] / (s * np.sqrt(10))
    sb = np.maximum(gb.sigma[gbi], 1e-7)
    C["btc_r250_z"] = gb.ret[1][gbi] / (sb * 0.5)
    C["btc_r1_z"] = gb.ret[4][gbi] / sb
    C["btc_sv1"] = np.arcsinh(gb.net[4][gbi] / (gb.vol_ewma[gbi] + 1.0))
    C["btc_ofi1"] = gb.net[4][gbi] / (gb.tot[4][gbi] + 0.25 * gb.vol_ewma[gbi] + 1.0)
    lnp = g.lnp[gi]
    C["since_touch"] = np.log1p(since_chg) / 4
    C["ul_since_touch_z"] = (lnp - lnp_chg) / (s * np.sqrt(np.maximum(since_chg, 1)))
    C["dp_since_touch_c"] = dusd * (lnp - lnp_chg) * 100 / 5
    C["dp_since_snap_c"] = dusd * (lnp - lnp_snap) * 100 / 5
    C["dp_1s_c"] = dusd * g.ret[4][gi] * 100 / 2         # fair-value move of the last second
    C["dp_250_c"] = dusd * g.ret[1][gi] * 100 / 2
    return C


GROUPS = {
    "ret_sub": ["r250_z", "r500_z", "r1_z", "r10_z"],
    "signed_vol": ["sv0.25", "sv1", "sv5", "sv10"],
    "ofi": ["ofi0.25", "ofi1", "ofi5"],
    "activity": ["act1", "cnt1", "hl1_z", "rv10_z"],
    "btc_lead": ["btc_r250_z", "btc_r1_z", "btc_sv1", "btc_ofi1"],
    "staleness": ["since_touch", "ul_since_touch_z", "dp_since_touch_c", "dp_since_snap_c"],
    "dp_recent": ["dp_1s_c", "dp_250_c"],
}
CAND = [c for g in GROUPS.values() for c in g]


def build(split: str, build_dir: Path, dmatch: int = D_MATCH):
    t_start = time.time()
    d = Data(build_dir)
    ticks_meta = json.loads((build_dir / "meta.json").read_text())
    tcol = ticks_meta["column_index"]
    F = ticks_meta["F"]
    ticks = np.memmap(build_dir / "ticks.bin", np.float32, "r", shape=(d.N, F))
    mk = d.markets
    sid = {"train": [0], "val": [1]}[split]
    sel = np.flatnonzero(np.isin(mk["split"], sid))
    if split == "train":
        assert (mk["end_ts"][sel] <= T_TRAIN_END).all(), "LEAKAGE: a non-train market in split 0"
    t0, t1 = int(mk["open_ts"][sel].min()), int(mk["end_ts"][sel].max())
    print(f"{split}: {len(sel)} markets, {pd.Timestamp(t0, unit='s')} .. {pd.Timestamp(t1, unit='s')}", flush=True)
    grids = {}
    for c in sorted({COINS[int(x)] for x in mk["coin_id"][sel]} | {"btc"}):
        grids[c] = f2.CoinGrid(c, t0, t1)
        if split == "train":
            assert grids[c].end_unix <= T_TRAIN_END, f"LEAKAGE: {c} grid loaded data after Aug 22"
    fc = d.col
    rows, fills = [], []
    for i in sel:
        m = mk[i]
        r0, n, o, e = int(m["row_offset"]), int(m["n_rows"]), int(m["open_ts"]), int(m["end_ts"])
        L = int(m["contract_len_s"])
        coin = COINS[int(m["coin_id"])]
        g, gb = grids[coin], grids["btc"]
        Fm = np.asarray(d.feat[r0:r0 + n], np.float64)
        Bk = d.book[r0:r0 + n]
        T = np.asarray(ticks[r0:r0 + n], np.float64)
        t = o + np.arange(n)
        gi, gbi = t - g.t0, t - gb.t0
        sg, dusd = Fm[:, fc["sigma"]], Fm[:, fc["delta_usd_raw"]]
        bb, ba = Bk["bid_px"][:, 0].astype(np.int64), Bk["ask_px"][:, 0].astype(np.int64)
        loc = np.arange(n)
        chg = np.r_[True, (bb[1:] != bb[:-1]) | (ba[1:] != ba[:-1])]
        last_chg = np.maximum.accumulate(np.where(chg, loc, 0))
        since_chg = loc - last_chg
        snap = np.maximum(loc - T[:, tcol["book_age"]].astype(np.int64), -10**6)
        lnp_all = g.lnp[gi]
        # the snapshot may predate the open: read the grid directly at its second
        lnp_snap = g.lnp[np.clip(t[0] + snap - g.t0, 0, len(g.lnp) - 1)]
        lnp_chg = lnp_all[last_chg]
        C = candidates(g, gb, sg, dusd, lnp_snap, lnp_chg, since_chg, gi, gbi)
        base = Fm[:, BASE_COLS]
        mark = Fm[:, fc["mark"]]
        two = Fm[:, fc["mark_is_fv"]] == 0
        winner = float(m["winner_up"])
        # ---------------- decision rows
        dt = 5 if L == 14400 else 1
        rr = loc[(loc % dt == 0) & (loc < n - 1)]
        Y = {}
        for h in (1, 5, 30):
            j = rr + h
            ok = (j <= n - 1)
            jj = np.minimum(j, n - 1)
            ok &= two[rr] & two[jj]
            Y[f"dmid{h}"] = np.where(ok, (mark[jj] - mark[rr]) * 100, np.nan)
        bv, sv = T[:, tcol["buy_vol"]], T[:, tcol["sell_vol"]]
        cnet = np.r_[0.0, np.cumsum(bv - sv)]
        ctot = np.r_[0.0, np.cumsum(bv + sv)]
        for h in (1, 5):
            a, b = rr + D_MATCH, rr + D_MATCH + h           # print rows (a, b] = match rows (t, t+h]
            ok = b <= n - 1
            bb_, aa = np.minimum(b, n - 1), np.minimum(a, n - 1)
            net = cnet[bb_ + 1] - cnet[aa + 1]                # rows a+1..b
            tot = ctot[bb_ + 1] - ctot[aa + 1]
            Y[f"fnet{h}"] = np.where(ok, net, np.nan)
            Y[f"ftot{h}"] = np.where(ok, tot, np.nan)
        # Binance forward returns (in sigma units), for the fair-value-change predictor
        for h in (1, 2, 5):
            Y[f"ulfwd{h}"] = g.fwd_ret(h)[gi[rr]] / (np.maximum(sg[rr], 1e-7) * np.sqrt(h))
        rows.append(dict(market=np.full(len(rr), i), t=t[rr], clen=np.full(len(rr), L),
                         coin=np.full(len(rr), int(m["coin_id"])), base=base[rr],
                         cand=np.column_stack([C[c][rr] for c in CAND]),
                         dusd=dusd[rr], sigma=sg[rr], **{k: v for k, v in Y.items()}))
        # ---------------- touch fills (one per taker tx that reached the touch)
        lo, hi = int(d.fill_index[r0]), int(d.fill_index[r0 + n])
        if hi > lo:
            fl = d.fills[lo:hi]
            txid = np.cumsum(fl["first"].astype(np.int64)) - 1
            u = fl["row"].astype(np.int64) - r0
            side = fl["side"].astype(np.int64)
            px = fl["px"].astype(np.int64)
            ntx = txid[-1] + 1
            u_tx = np.zeros(ntx, np.int64)
            u_tx[txid] = u
            s_tx = np.zeros(ntx, np.int64)
            s_tx[txid] = side
            ext = np.where(s_tx < 0, 10**6, -10**6)
            np.minimum.at(ext, txid[side < 0], px[side < 0])
            np.maximum.at(ext, txid[side > 0], px[side > 0])
            size_tx = np.bincount(txid, weights=fl["size"].astype(np.float64), minlength=ntx)
            tau = u_tx - dmatch
            tdec = tau - LAT
            ok = tdec >= 0
            tau_c = np.clip(tau, 0, n - 1)
            valid = (Bk["flags"][tau_c] & 1) > 0
            crossed = (Bk["flags"][tau_c] & 2) > 0
            two_t = (bb[tau_c] > 0) & (ba[tau_c] < 1000)
            reach = np.where(s_tx < 0, ext <= bb[tau_c], ext >= ba[tau_c])
            ok &= valid & ~crossed & two_t & reach
            k = np.flatnonzero(ok)
            if len(k):
                tau_k, td = tau[k], tdec[k]
                mm = np.where(s_tx[k] < 0, 1.0, -1.0)          # maker sign: +1 our bid filled
                touch = np.where(s_tx[k] < 0, bb[tau_k], ba[tau_k]) / 1000.0
                mid0 = mark[tau_k]

                def mid_at(j):
                    return np.where(j <= n - 1, mark[np.minimum(j, n - 1)], winner)
                fills.append(dict(
                    market=np.full(len(k), i), t=t[tau_k], clen=np.full(len(k), L),
                    coin=np.full(len(k), int(m["coin_id"])), m=mm, size=size_tx[k],
                    base=base[td], cand=np.column_stack([C[c][td] for c in CAND]),
                    dusd=dusd[td], sigma=sg[td],
                    mo5=mm * (mid_at(tau_k + 5) - mid0) * 100,
                    mo30=mm * (mid_at(tau_k + 30) - mid0) * 100,
                    mo60=mm * (mid_at(tau_k + 60) - mid0) * 100,
                    spread=mm * (mid0 - touch) * 100,
                    edge60=mm * (mid_at(tau_k + 60) - touch) * 100))
    cat = lambda L_, k: np.concatenate([x[k] for x in L_])
    R = {k: cat(rows, k) for k in rows[0]}
    Fl = {k: cat(fills, k) for k in fills[0]}
    CACHE.mkdir(exist_ok=True)
    tag = "" if dmatch == D_MATCH else f"_D{dmatch}"
    np.savez(CACHE / f"panel_{split}{tag}.npz", **{f"r_{k}": v for k, v in R.items()},
             **{f"f_{k}": v for k, v in Fl.items()}, cand_names=np.array(CAND),
             base_names=np.array([{v: k for k, v in d.col.items()}[c] for c in BASE_COLS]))
    print(f"{split}: rows {len(R['t'])}, touch fills {len(Fl['t'])}; {time.time() - t_start:.0f}s", flush=True)
    for k in ("cand",):
        bad = ~np.isfinite(R[k])
        if bad.any():
            print("non-finite candidate values (rows):", dict(zip(CAND, bad.sum(0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--build", default=None, help="build dir (val: data/build_v2 when it exists)")
    ap.add_argument("--dmatch", type=int, default=D_MATCH,
                    help="what-if: match delay D (rows) of the touch-fill panel (env central = 4)")
    a = ap.parse_args()
    bd = Path(a.build) if a.build else POLY / "data" / "build"
    assert "jun15" not in str(bd), "the v2 test set is never analysed here"
    build(a.split, bd, a.dmatch)


if __name__ == "__main__":
    main()
