#!/usr/bin/env python3
"""Sim-vs-replay consistency test on the TRAIN days (SIMULATOR.md §5), numpy version (no C env yet).

Real: the 431 train markets (data/build ticks + data/raw/trades.parquet legs).
Sim:  the same schedule on the same Binance paths, --reps replications (base generator, no randomisation).
Both go through the SAME statistics and the SAME numpy fill check (two fixed quoting rules), so any
difference is the generator's. Writes sim/validation.json and sim/VALIDATION.md.

Thresholds are fixed in GATE below (written before any comparison was run).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common as cm
import engine as en
import generate as gen
from fit import mid_moments, MID_H, P_LO, P_HI, EXCL_END

C = cm.C
LN = {300: "5m", 900: "15m", 14400: "4h"}
Q_SIZE = 10.0           # baseline quote size (shares)
MATCH_ROW_LAG = 4       # prints at block time T are matched against our quotes at row T - 4 (central d = 3.5 s)
MARKOUT_H = (5, 30, 60)

GATE = {
    "G1_pnl_interval": "real total and median per-market P&L inside the sim 5-95% interval, per baseline",
    "G1_ks_hourly": 0.15,          # KS D between real and pooled-sim hourly P&L series (markets by end hour)
    "G1_ks_market": 0.15,          # KS D per-market P&L (reported; iid-market version)
    "G2_rank": "rank order of {no-quote, touch, fv} by total P&L identical (real vs sim median)",
    "G3_band": "per-market means of fills, volume, spread capture, markout 5/30/60, resolution inside sim 5-95%",
    "G4_spread_median_ticks": 1.0,   # |median spread sim - real| <= 1 tick (0.01) per (len, tau bucket)
    "G4_ks_dist": 0.15,              # KS D for pooled spread / log top size / log top-5 depth / trades per market
    "G4_onesided_abs": 0.10,         # |one-sided fraction sim - real| per (len, tau bucket)
    "G4_rate_ratio": (0.67, 1.5),    # trade rate per second sim/real per (len, tau bucket) with >= 200 real trades
    "G4_pbuy_abs": 0.06,             # |P(buy|gap) sim - real| per (len, gap bucket) with >= 300 real trades
    "G4_acf_abs": 0.06,              # |lag-1 autocorrelation of trade signs sim - real| per len
}


# --------------------------------------------------------------------------------------------- data sets
class DS:
    """One data set (real or one sim replication) in the train markets' order."""

    def __init__(self, X, mk, legs, p):
        self.X, self.mk, self.legs, self.p = X, mk, legs, p
        self.off = np.concatenate([[0], np.cumsum(mk.n_rows.values)])
        self.m = np.repeat(np.arange(len(mk)), mk.n_rows.values)
        self.L = mk.L.values[self.m]
        self.tau = X[:, C["seconds_to_close"]].astype(np.int64)
        bp0, ap0 = X[:, C["bid_px_0"]].astype(float), X[:, C["ask_px_0"]].astype(float)
        self.valid = X[:, C["book_valid"]] == 1
        hb, ha = (bp0 > 0) & self.valid, (ap0 < 1) & self.valid
        self.two = hb & ha
        self.spread = np.where(self.two, ap0 - bp0, np.nan)
        self.reg = np.where(~self.two, cm.ONESIDED, np.where(self.spread > cm.HOLLOW_SPREAD, cm.HOLLOW, cm.NORMAL))
        self.mid = np.where(self.two, 0.5 * (bp0 + ap0), np.where(hb, 0.5 * (bp0 + 1), np.where(ha, 0.5 * ap0, 0.5)))
        self.bp0, self.ap0 = bp0, ap0
        self.tb = cm.tau_bucket(np.minimum(self.tau, self.L - 1))
        self.li = np.searchsorted(cm.LENS, self.L)


def real_dataset(P, mk):
    ticks, tr, meta = cm.train_build()
    X = np.concatenate([np.asarray(ticks[int(r["row_offset"]): int(r["row_offset"]) + int(r["n_rows"])])
                        for r in tr])
    t = cm.train_trades(mk.slug)
    pos = pd.Series(np.arange(len(mk)), index=mk.slug)
    t["m"] = pos.loc[t.slug].values
    t["tx_id"] = pd.factorize(t.tx)[0]
    t = t[(t.timestamp >= mk.open_ts.values[t.m]) & (t.timestamp <= mk.end_ts.values[t.m])]
    legs = pd.DataFrame(dict(m=t.m.values, stamp=t.timestamp.values, is_taker=t.is_taker.values,
                             sign=t.aggressor_up.values.astype(int), px=t.px_up.values, size=t["size"].values,
                             tx=t.tx_id.values))
    return DS(X, mk, legs, P["p"])


def sim_datasets(model, mk, prices, reps, seed, chunk):
    """Yield DS per replication (base generator, no randomisation)."""
    rng = np.random.default_rng(seed)
    base = pd.DataFrame(dict(coin=mk.coin.values, L=mk.L.values, open_ts=mk.open_ts.values))
    off = np.concatenate([[0], np.cumsum(mk.n_rows.values)])
    N = off[-1]
    p_all = np.concatenate([pr["p"] for pr in prices])
    for r0 in range(0, reps, chunk):
        k = min(chunk, reps - r0)
        sch = pd.concat([base.assign(rep=j) for j in range(k)], ignore_index=True)
        pr = prices * k
        X = np.zeros((k, N, cm.F), np.float32)
        legs_parts = []
        winners = np.zeros((k, len(mk)), np.int32)

        def on_batch(idx, res, knobs):
            rep, mpos = idx // len(mk), idx % len(mk)
            for j in range(len(idx)):
                X[rep[j], off[mpos[j]]: off[mpos[j] + 1]] = res["ticks"][j]
            winners[rep, mpos] = res["winner_up"]
            lg = en.trade_legs(model, res["trades"], rng)
            if len(lg["b"]):
                bb = idx[lg["b"]]
                legs_parts.append(pd.DataFrame(dict(
                    rep=bb // len(mk), m=bb % len(mk), stamp=mk.open_ts.values[bb % len(mk)] + lg["stamp"],
                    is_taker=lg["is_taker"], sign=lg["aggressor_up"].astype(int),
                    px=np.where(lg["outcome_idx"] == 0, lg["price"], 1 - lg["price"].astype(float)),
                    size=lg["size"], tx=(bb.astype(np.int64) << 24) + lg["trade"])))

        gen.run_batches(model, sch, pr, rng, False, on_batch)
        legs = pd.concat(legs_parts, ignore_index=True)
        for j in range(k):
            lj = legs[legs.rep == j]
            lj = lj[lj.stamp <= mk.end_ts.values[lj.m]]
            mkj = mk.copy()
            mkj["winner_up"] = winners[j]
            yield DS(X[j], mkj, lj.drop(columns="rep"), p_all)


# --------------------------------------------------------------------------------------------- statistics
def taker_table(d: DS):
    """One row per taker tx: market, stamp, sign, vwap, worst maker price, size."""
    mkr = d.legs[~d.legs.is_taker]
    g = mkr.groupby("tx")
    t = pd.DataFrame(dict(m=g.m.first(), stamp=g.stamp.first(), sign=g.sign.first(), size=g["size"].sum(),
                          vwap=(mkr.px * mkr["size"]).groupby(mkr.tx).sum() / g["size"].sum(),
                          worst_hi=g.px.max(), worst_lo=g.px.min(), nlev=g.px.nunique()))
    t["row"] = d.off[t.m.values] + (t.stamp.values - d.mk.open_ts.values[t.m.values])
    return t.reset_index(drop=True)


def book_stats(d: DS) -> dict:
    out = {}
    nr = d.valid & (d.reg == cm.NORMAL)
    for li, L in enumerate(cm.LENS):
        for k in range(cm.NTAU):
            sel = d.valid & (d.li == li) & (d.tb == k)
            if sel.sum() < 50:
                continue
            key = f"{LN[L]}|{k}"
            s = sel & nr
            out[key] = dict(
                onesided=float((d.reg[sel] == cm.ONESIDED).mean()), hollow=float((d.reg[sel] == cm.HOLLOW).mean()),
                spread_med=float(np.median(d.spread[s])) if s.any() else np.nan,
                spread_mean=float(np.mean(d.spread[s])) if s.any() else np.nan,
                p_1tick=float((d.spread[s] <= 0.0101).mean()) if s.any() else np.nan,
                top_sz_med=float(np.median(d.X[s, C["bid_sz_0"]])) if s.any() else np.nan,
                top5_med=float(np.median(d.X[s, C["bid_sz_0"]:C["bid_sz_0"] + 5].sum(1))) if s.any() else np.nan,
                age0=float((d.X[sel, C["book_age"]] == 0).mean()),
                trade_rate=float(d.X[sel, C["trade_count"]].mean()))
    return out


def dist_samples(d: DS, rng, n=20000) -> dict:
    """Subsamples for pooled KS tests."""
    out = {}
    nr = d.valid & (d.reg == cm.NORMAL)
    for li, L in enumerate(cm.LENS):
        s = np.flatnonzero(nr & (d.li == li))
        s = rng.choice(s, min(n, len(s)), replace=False)
        out[f"spread|{LN[L]}"] = d.spread[s]
        out[f"log_top_sz|{LN[L]}"] = np.log(np.maximum(d.X[s, C["bid_sz_0"]], 1e-3))
        out[f"log_top5|{LN[L]}"] = np.log(np.maximum(d.X[s, C["bid_sz_0"]:C["bid_sz_0"] + 5].sum(1), 1e-3))
        mm = np.flatnonzero(d.mk.L.values == L)
        tc = np.add.reduceat(d.X[:, C["trade_count"]].astype(float), d.off[:-1])[mm]
        out[f"trades_per_market|{LN[L]}"] = tc
    return out


def trade_stats(d: DS, tk: pd.DataFrame) -> dict:
    out = {}
    first = d.off[tk.m.values]
    Lm = d.mk.L.values[tk.m.values]
    # alignment: worst price vs best opposite quote at row - k
    for k in (0, 1, 2, 3, 4, 6):
        r = np.maximum(tk.row.values - k, first)
        buy = tk.sign.values == 1
        ask, bid = d.ap0[r], d.bp0[r]
        has = np.where(buy, ask < 1, bid > 0)
        at_thr = np.where(buy, tk.worst_hi.values >= ask - 1e-9, tk.worst_lo.values <= bid + 1e-9) & has
        thr = np.where(buy, tk.worst_hi.values > ask + 1e-9, tk.worst_lo.values < bid - 1e-9) & has
        for L in cm.LENS:
            s = (Lm == L) & has
            out[f"align_at_or_through|{LN[L]}|k{k}"] = float(at_thr[s].mean())
            out[f"align_through|{LN[L]}|k{k}"] = float(thr[s].mean())
    # P(buy | gap at row - 3)
    r3 = np.maximum(tk.row.values - 3, first)
    gap = d.p[r3] - d.mid[r3]
    gb = cm.gap_bucket(gap)
    for L in cm.LENS:
        for g in range(cm.NGAP):
            s = (Lm == L) & (gb == g)
            out[f"pbuy|{LN[L]}|{g}"] = (float((tk.sign.values[s] == 1).mean()) if s.sum() else np.nan, int(s.sum()))
    # sign autocorrelation (sequence order by stamp within market)
    o = np.lexsort((tk.stamp.values, tk.m.values))
    sg, mm, LL = tk.sign.values[o].astype(float), tk.m.values[o], Lm[o]
    for L in cm.LENS:
        for lag in (1, 2, 5):
            same = (mm[lag:] == mm[:-lag]) & (LL[lag:] == L)
            out[f"sign_acf|{LN[L]}|{lag}"] = float(np.corrcoef(sg[lag:][same], sg[:-lag][same])[0, 1])
    # maker markouts (cents/share, count weighted), from match row ~ stamp - 3
    last = d.off[tk.m.values + 1] - 1
    for h in (0,) + MARKOUT_H:
        r = np.clip(r3 + h, first, last)
        mko = -tk.sign.values * (d.mid[r] - tk.vwap.values) * 100
        for L in cm.LENS:
            out[f"markout|{LN[L]}|{h}"] = float(np.mean(mko[Lm == L]))
    for L in cm.LENS:
        s = Lm == L
        out[f"taker_size_med|{LN[L]}"] = float(np.median(tk["size"].values[s]))
        out[f"taker_size_p90|{LN[L]}"] = float(np.quantile(tk["size"].values[s], 0.9))
        out[f"levels_per_tx|{LN[L]}"] = float(tk.nlev.values[s].mean())
        out[f"buy_share|{LN[L]}"] = float((tk.sign.values[s] == 1).mean())
    return out


def mid_stats(d: DS) -> dict:
    x = cm.logit(d.p, 1e-4)
    y = cm.logit(np.clip(d.mid, 1e-4, 1 - 1e-4), 1e-4)
    good = (d.reg == cm.NORMAL) & d.valid & (d.p > P_LO) & (d.p < P_HI) & (d.mid > P_LO) & (d.mid < P_HI) & (d.tau > EXCL_END)
    out = {}
    for L in cm.LENS:
        mom = mid_moments(y, x, good & (d.L == L), d.m, H=MID_H[L])
        out.update({f"mid|{LN[L]}|{k}": v for k, v in mom.items()})
    return out


# --------------------------------------------------------------------------------------------- fill check
def quotes(d: DS, rule: str):
    """Our bid/ask per row r (decided on the book at row r-1: 1 s latency). NaN = not quoting."""
    r1 = np.r_[0, np.arange(len(d.p) - 1)]
    r1 = np.maximum(r1, d.off[d.m])                 # first row of a market uses its own row
    ok = d.two[r1] & (d.spread[r1] <= 0.1)
    bid, ask = d.bp0[r1].copy(), d.ap0[r1].copy()
    if rule == "fv":
        p, mid, spr = d.p[r1], d.mid[r1], d.spread[r1]
        gap = p - mid
        c = np.clip(0.5 * (p + mid), 0.01, 0.99)
        h = np.maximum(0.01, 0.5 * spr) + 0.5 * np.abs(gap)
        bid = np.floor((c - h) * 100 + 1e-9) / 100
        ask = np.ceil((c + h) * 100 - 1e-9) / 100
        bid = np.minimum(bid, d.ap0[r1] - 0.01)
        ask = np.maximum(ask, d.bp0[r1] + 0.01)
        ok &= (np.abs(gap) <= 0.1) & ~((d.tau < 30) & (d.L <= 900)) & (bid > 0) & (ask < 1)
    return np.where(ok, bid, np.nan), np.where(ok, ask, np.nan), r1


def displayed_at(d: DS, r1, px, side):
    """Displayed size at price px on our side in book row r1 (0 if none)."""
    pc = "bid_px_" if side == 1 else "ask_px_"
    sc = "bid_sz_" if side == 1 else "ask_sz_"
    D = np.zeros(len(px))
    for k in range(5):
        D += np.where(np.abs(d.X[r1, C[pc + str(k)]] - px) < 1e-6, d.X[r1, C[sc + str(k)]], 0.0)
    return D


def fill_check(d: DS, rule: str):
    """Per-market P&L of a fixed quoting rule; central-like rule shared by real and sim."""
    qb, qa, r1 = quotes(d, rule)
    mkr = d.legs[~d.legs.is_taker]
    row = d.off[mkr.m.values] + (mkr.stamp.values - d.mk.open_ts.values[mkr.m.values]) - MATCH_ROW_LAG
    first = d.off[mkr.m.values]
    valid = row >= first
    row = np.maximum(row, first)
    buy = mkr.sign.values == 1          # taker buys Up -> can hit our ask (we sell)
    our = np.where(buy, qa[row], qb[row])
    px, sz = mkr.px.values, mkr["size"].values
    hit = valid & np.isfinite(our)
    thr = hit & np.where(buy, px > our + 1e-9, px < our - 1e-9)
    at = hit & (np.abs(px - our) < 1e-9)
    D = np.zeros(len(px))
    if at.any():
        ia = np.flatnonzero(at)
        Da = displayed_at(d, r1[row[ia]], our[ia], -1)     # taker buy -> our ask, queue = displayed ask size
        Db = displayed_at(d, r1[row[ia]], our[ia], 1)
        D[ia] = np.where(buy[ia], Da, Db)
    pot = np.where(thr, sz, 0.0) + np.where(at, sz * Q_SIZE / (Q_SIZE + D), 0.0)
    f = pd.DataFrame(dict(m=mkr.m.values, row=row, buy=buy, pot=pot, px=our))
    f = f[f.pot > 0]
    # capacity Q per (row, side)
    f["cum"] = f.groupby(["row", "buy"]).pot.cumsum()
    f["qty"] = np.clip(Q_SIZE - (f.cum - f.pot), 0.0, f.pot)
    f = f[f.qty > 0]
    side = np.where(f.buy.values, -1.0, 1.0)            # we sell when the taker buys
    m = f.m.values
    win = d.mk.winner_up.values[m].astype(float)
    mid0 = d.mid[r1[f.row.values]]
    last = d.off[m + 1] - 1
    res = dict(pnl=side * f.qty.values * (win - f.px.values), capture=side * f.qty.values * (mid0 - f.px.values))
    for h in MARKOUT_H:
        res[f"mk{h}"] = side * f.qty.values * (d.mid[np.minimum(f.row.values + h, last)] - mid0)
    res["resolution"] = res["pnl"] - res["capture"] - res["mk60"]
    nm = len(d.mk)
    out = {k: np.bincount(m, weights=v, minlength=nm) for k, v in res.items()}
    out["volume"] = np.bincount(m, weights=f.qty.values, minlength=nm)
    out["fills"] = np.bincount(m, minlength=nm).astype(float)
    quoting = np.isfinite(qb) | np.isfinite(qa)
    out["quoted_s"] = np.bincount(d.m, weights=quoting.astype(float), minlength=nm)
    return out


def hourly(d: DS, pnl):
    h = (d.mk.end_ts.values - 1) // 3600
    s = pd.Series(pnl).groupby(h).sum()
    return s


def ks(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    v = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b))))


def all_stats(d: DS, rng):
    tk = taker_table(d)
    st = dict(book=book_stats(d), trade=trade_stats(d, tk), mid=mid_stats(d), dist=dist_samples(d, rng))
    st["fills"] = {rule: fill_check(d, rule) for rule in ("touch", "fv")}
    st["hourly"] = {rule: hourly(d, st["fills"][rule]["pnl"]) for rule in ("touch", "fv")}
    return st


# --------------------------------------------------------------------------------------------- gate
def band(vals, q=(5, 95)):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))) if len(v) else (np.nan, np.nan)


def evaluate(real, sims):
    res = {"G1": {}, "G2": {}, "G3": {}, "G4": {}}
    # ---- G1 / G2 / G3: baselines
    totals_real = {"no-quote": 0.0}
    totals_sim = {"no-quote": [0.0] * len(sims)}
    for rule in ("touch", "fv"):
        fr = real["fills"][rule]
        tot_s = [s["fills"][rule]["pnl"].sum() for s in sims]
        med_s = [np.median(s["fills"][rule]["pnl"]) for s in sims]
        b_tot, b_med = band(tot_s), band(med_s)
        tr, mr = fr["pnl"].sum(), np.median(fr["pnl"])
        pooled = np.concatenate([s["fills"][rule]["pnl"] for s in sims])
        hourly_pooled = np.concatenate([s["hourly"][rule].values for s in sims])
        d_mk = ks(fr["pnl"], pooled)
        d_hr = ks(real["hourly"][rule].values, hourly_pooled)
        res["G1"][rule] = dict(
            real_total=tr, sim_total_band=b_tot, sim_total_median=float(np.median(tot_s)),
            total_in=bool(b_tot[0] <= tr <= b_tot[1]),
            real_median=mr, sim_median_band=b_med, median_in=bool(b_med[0] <= mr <= b_med[1]),
            ks_market=d_mk, ks_hourly=d_hr,
            passed=bool(b_tot[0] <= tr <= b_tot[1] and b_med[0] <= mr <= b_med[1] and d_hr < GATE["G1_ks_hourly"]))
        totals_real[rule] = tr
        totals_sim[rule] = tot_s
        comp = {}
        for k in ("fills", "volume", "capture", "mk5", "mk30", "mk60", "resolution"):
            rv = float(fr[k].mean())
            b = band([s["fills"][rule][k].mean() for s in sims])
            comp[k] = dict(real=rv, band=b, inside=bool(b[0] <= rv <= b[1]))
        # markout curve per share
        vol_r = fr["volume"].sum()
        curve = {}
        for h in MARKOUT_H:
            rv = float(fr[f"mk{h}"].sum() / max(vol_r, 1e-9) * 100)
            b = band([s["fills"][rule][f"mk{h}"].sum() / max(s["fills"][rule]["volume"].sum(), 1e-9) * 100
                      for s in sims])
            curve[h] = dict(real_c_per_share=rv, band=b, inside=bool(b[0] <= rv <= b[1]))
        comp["markout_curve"] = curve
        res["G3"][rule] = comp
    order_r = sorted(totals_real, key=lambda k: -totals_real[k])
    med = {k: float(np.median(v)) for k, v in totals_sim.items()}
    order_s = sorted(med, key=lambda k: -med[k])
    res["G2"] = dict(real_order=order_r, sim_order=order_s, passed=order_r == order_s)
    g3_pass = all(v["inside"] for r in res["G3"].values() for k, v in r.items() if k != "markout_curve")
    g3_pass &= all(c["inside"] for c in res["G3"]["touch"]["markout_curve"].values())
    res["G3"]["passed"] = bool(g3_pass)

    # ---- G4: stylised facts
    g4 = {}
    rows = []
    for key, rv in real["book"].items():
        L, k = key.split("|")
        sv = [s["book"].get(key) for s in sims]
        sv = [x for x in sv if x is not None]
        if not sv:
            continue
        row = dict(len=L, tau_b=int(k))
        for f in ("onesided", "hollow", "spread_med", "spread_mean", "p_1tick", "top_sz_med", "top5_med", "age0",
                  "trade_rate"):
            row[f"{f}_real"] = rv[f]
            row[f"{f}_sim"] = float(np.nanmedian([x[f] for x in sv]))
        rows.append(row)
    bt = pd.DataFrame(rows)
    ok_sp = (np.abs(bt.spread_med_sim - bt.spread_med_real) <= GATE["G4_spread_median_ticks"] * 0.01 + 1e-9) | bt.spread_med_real.isna()
    ok_os = np.abs(bt.onesided_sim - bt.onesided_real) <= GATE["G4_onesided_abs"]
    rr = bt.trade_rate_sim / bt.trade_rate_real.replace(0, np.nan)
    lo, hi = GATE["G4_rate_ratio"]
    ok_rate = ((rr >= lo) & (rr <= hi)) | (bt.trade_rate_real < 0.005)
    g4["spread_median_by_tau"] = dict(passed=bool(ok_sp.all()), n_fail=int((~ok_sp).sum()), n=int(len(bt)))
    g4["onesided_by_tau"] = dict(passed=bool(ok_os.all()), n_fail=int((~ok_os).sum()), n=int(len(bt)))
    g4["trade_rate_by_tau"] = dict(passed=bool(ok_rate.all()), n_fail=int((~ok_rate).sum()), n=int(len(bt)))
    for name in ("spread", "log_top_sz", "log_top5", "trades_per_market"):
        for L in ("5m", "15m", "4h"):
            key = f"{name}|{L}"
            D = ks(real["dist"][key], np.concatenate([s["dist"][key] for s in sims]))
            g4[f"ks_{key}"] = dict(D=D, passed=bool(D < GATE["G4_ks_dist"]))
    pb = []
    for key, (rv, n) in ((k, v) for k, v in real["trade"].items() if k.startswith("pbuy|")):
        if n < 300:
            continue
        sv = float(np.nanmedian([s["trade"][key][0] for s in sims]))
        pb.append(dict(key=key, real=rv, sim=sv, n=n, ok=abs(sv - rv) <= GATE["G4_pbuy_abs"]))
    pb = pd.DataFrame(pb)
    g4["pbuy_gap_curve"] = dict(passed=bool(pb.ok.all()), n_fail=int((~pb.ok).sum()), n=int(len(pb)))
    for L in ("5m", "15m", "4h"):
        key = f"sign_acf|{L}|1"
        rv = real["trade"][key]
        sv = float(np.nanmedian([s["trade"][key] for s in sims]))
        g4[f"sign_acf1|{L}"] = dict(real=rv, sim=sv, passed=bool(abs(sv - rv) <= GATE["G4_acf_abs"]))
    res["G4"] = g4
    res["G4"]["passed"] = bool(all(v["passed"] for v in g4.values() if isinstance(v, dict)))
    res["tables"] = dict(book=bt, pbuy=pb)
    res["passed"] = bool(all(res["G1"][r]["passed"] for r in ("touch", "fv")) and res["G2"]["passed"]
                         and res["G3"]["passed"] and res["G4"]["passed"])
    return res


def compare_table(real, sims, section, keys=None):
    rows = []
    for k, rv in real[section].items():
        if keys and not any(k.startswith(x) for x in keys):
            continue
        if isinstance(rv, tuple):
            rv = rv[0]
        sv = [s[section][k][0] if isinstance(s[section][k], tuple) else s[section][k] for s in sims]
        b = band(sv)
        rows.append(dict(stat=k, real=rv, sim_median=float(np.nanmedian(sv)), sim_lo=b[0], sim_hi=b[1],
                         inside=bool(b[0] <= rv <= b[1]) if np.isfinite(rv) else None))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- report
def fmt(x, nd=3):
    if isinstance(x, (tuple, list)):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(res, real, sims, args, elapsed, path):
    pf = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    L = []
    L.append("# Generator validation: sim vs replay on the train days\n")
    L.append(f"Generated by `sim/validate.py --reps {args.reps} --seed {args.seed}` in {elapsed/60:.1f} min, with "
             f"params.json sha256 `{args.params_sha[:16]}`. Real data: the 431 train markets. Sim: the same schedule "
             f"on the same Binance paths, {args.reps} replications of the base generator (no randomisation).\n")
    L.append("The C env does not exist yet, so the two fixed quoting rules run through a numpy fill check. "
             "Real and sim use the same code for this check and for every statistic. The fill rule: a print with "
             f"block time T can fill our quote from row T-{MATCH_ROW_LAG} (quotes are set on row r-1, so 1 s "
             f"latency). A print through our price fills us in full; a print at our price fills pro-rata "
             f"Q/(Q+displayed). Q = {Q_SIZE:.0f} shares per side, and at most Q fill per side per second. "
             "There is no inventory control, fees or rebates. Settlement is at winner_up.\n")
    L.append("- **touch** joins the best bid and best ask when the book is two-sided with spread <= 0.10.")
    L.append("- **fv** quotes around c = (p_model + mid) / 2 with half-spread h = max(1c, spread/2) + |gap|/2, on the "
             "0.01 grid and never crossing. It pulls quotes when |gap| > 0.10, and in the last 30 s of 5m and 15m "
             "markets.\n")
    L.append(f"## Overall: **{pf(res['passed'])}**\n")
    L.append("| gate | result |\n|---|---|")
    L.append(f"| G1 P&L inside sim 90% interval + hourly KS < {GATE['G1_ks_hourly']} | touch {pf(res['G1']['touch']['passed'])}, fv {pf(res['G1']['fv']['passed'])} |")
    L.append(f"| G2 rank order of baselines | {pf(res['G2']['passed'])} (real {res['G2']['real_order']}, sim {res['G2']['sim_order']}) |")
    L.append(f"| G3 attribution, fill rates and markout curve inside the sim 90% band | {pf(res['G3']['passed'])} |")
    L.append(f"| G4 stylised facts | {pf(res['G4']['passed'])} |")
    L.append("| G5 out-of-sample val day | not run. It needs the C env and a ledger entry (EVALUATION.md T4) |\n")
    L.append("Thresholds were fixed before the first comparison (`GATE` in validate.py):\n")
    for k, v in GATE.items():
        L.append(f"- `{k}`: {v}")
    L.append("\n## G1/G2: baseline P&L ($, per market or total over 431 markets)\n")
    L.append("| rule | real total | sim total median [5%, 95%] | real median/mkt | sim median/mkt [5%, 95%] | KS per-market | KS hourly | pass |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in ("touch", "fv"):
        g = res["G1"][r]
        L.append(f"| {r} | {g['real_total']:.1f} | {g['sim_total_median']:.1f} {fmt(g['sim_total_band'], 1)} | "
                 f"{g['real_median']:.3f} | {fmt(g['sim_median_band'])} | {g['ks_market']:.3f} | {g['ks_hourly']:.3f} | "
                 f"{pf(g['passed'])} |")
    L.append("\n## G3: attribution per market ($), fills, markout curve (c/share)\n")
    L.append("| rule | component | real | sim [5%, 95%] | inside |\n|---|---|---|---|---|")
    for r in ("touch", "fv"):
        for k, v in res["G3"][r].items():
            if k == "markout_curve":
                for h, c in v.items():
                    L.append(f"| {r} | markout {h}s (c/share) | {c['real_c_per_share']:.3f} | {fmt(c['band'])} | {c['inside']} |")
            else:
                L.append(f"| {r} | {k} | {v['real']:.3f} | {fmt(v['band'])} | {v['inside']} |")
    L.append("\n## G4: stylised facts\n")
    L.append("| check | result |\n|---|---|")
    for k, v in res["G4"].items():
        if k == "passed":
            continue
        det = ", ".join(f"{a}={fmt(b)}" for a, b in v.items() if a != "passed")
        L.append(f"| {k} | {pf(v['passed'])} ({det}) |")
    bt = res["tables"]["book"]
    L.append("\n### Book and trade rate by contract length and time-to-close bucket (sim = median over replications)\n")
    L.append(f"tau buckets (s): {list(cm.TAU_EDGES)}\n")
    L.append("| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | "
             "top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in bt.itertuples():
        L.append(f"| {r.len} | {r.tau_b} | {r.onesided_real:.3f}/{r.onesided_sim:.3f} | {r.hollow_real:.3f}/{r.hollow_sim:.3f} | "
                 f"{r.spread_med_real:.3f}/{r.spread_med_sim:.3f} | {r.p_1tick_real:.2f}/{r.p_1tick_sim:.2f} | "
                 f"{r.top_sz_med_real:.0f}/{r.top_sz_med_sim:.0f} | {r.top5_med_real:.0f}/{r.top5_med_sim:.0f} | "
                 f"{r.age0_real:.2f}/{r.age0_sim:.2f} | {r.trade_rate_real:.3f}/{r.trade_rate_sim:.3f} |")
    pb = res["tables"]["pbuy"]
    L.append("\n### P(taker buys Up | gap = p_model - mid at block time - 3 s)\n")
    L.append(f"gap buckets: {list(cm.GAP_EDGES)}\n")
    L.append("| len|bucket | n real | real | sim |\n|---|---|---|---|")
    for r in pb.itertuples():
        L.append(f"| {r.key[5:]} | {r.n} | {r.real:.3f} | {r.sim:.3f} |")
    L.append("\n### Other trade and mid statistics (reported, not gated)\n")
    t = compare_table(real, sims, "trade", keys=["align_", "markout|", "sign_acf", "taker_size", "levels_per", "buy_share"])
    m = compare_table(real, sims, "mid")
    L.append("| stat | real | sim median | sim [5%, 95%] |\n|---|---|---|---|")
    for df in (t, m):
        for r in df.itertuples():
            L.append(f"| {r.stat} | {r.real:.3f} | {r.sim_median:.3f} | [{r.sim_lo:.3f}, {r.sim_hi:.3f}] |")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=5, help="replications simulated per engine call")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(cm.HERE / "VALIDATION.md"))
    a = ap.parse_args()
    t0 = time.time()
    import hashlib
    a.params_sha = hashlib.sha256(cm.PARAMS.read_bytes()).hexdigest()
    model = en.Model(cm.load_params())
    P, mk = cm.train_panel()
    rng = np.random.default_rng(a.seed + 1)
    real = all_stats(real_dataset(P, mk), rng)
    print(f"real stats {time.time()-t0:.1f}s")
    ul = cm.train_underlying()
    prices = [cm.market_pricing(ul[r.coin][1], ul[r.coin][2], ul[r.coin][0], r.open_ts, r.end_ts) for r in mk.itertuples()]
    sims = []
    for i, d in enumerate(sim_datasets(model, mk, prices, a.reps, a.seed, a.chunk)):
        sims.append(all_stats(d, rng))
        if (i + 1) % a.chunk == 0:
            print(f"  sim rep {i+1}/{a.reps}  {time.time()-t0:.0f}s", flush=True)
    res = evaluate(real, sims)
    write_report(res, real, sims, a, time.time() - t0, cm.Path(a.out))
    js = {k: v for k, v in res.items() if k != "tables"}
    js["tables"] = {k: v.to_dict(orient="records") for k, v in res["tables"].items()}
    (cm.HERE / "validation.json").write_text(json.dumps(js, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"overall {'PASS' if res['passed'] else 'FAIL'}  ({time.time()-t0:.0f}s) -> {a.out}")


if __name__ == "__main__":
    main()
