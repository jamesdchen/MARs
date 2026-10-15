"""Statistics shared by validate.py (sim-vs-replay gate) and fit.py (informed-taker calibration).

A `DS` is one data set in the train markets' order: ticks rows, market table, fill legs, model p per row.
Real and simulated data go through exactly the same functions, including the numpy fill check of two
fixed quoting rules.
"""
from __future__ import annotations

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


def alignment_curve(d: DS, tk: pd.DataFrame, ks=range(-10, 11)):
    """Pooled fractions of taker prints at-or-through / through the best opposite quote of the book at row - k."""
    first, last = d.off[tk.m.values], d.off[tk.m.values + 1] - 1
    buy = tk.sign.values == 1
    at_thr, thr = [], []
    for k in ks:
        r = np.clip(tk.row.values - k, first, last)
        ask, bid = d.ap0[r], d.bp0[r]
        has = np.where(buy, ask < 1, bid > 0)
        a = np.where(buy, tk.worst_hi.values >= ask - 1e-9, tk.worst_lo.values <= bid + 1e-9) & has
        t = np.where(buy, tk.worst_hi.values > ask + 1e-9, tk.worst_lo.values < bid - 1e-9) & has
        at_thr.append(a[has].mean())
        thr.append(t[has].mean())
    return np.array(list(ks)), np.array(at_thr), np.array(thr)


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
            out[f"align_at_or_through|{LN[L]}|k{k}"] = float(at_thr[s].mean()) if s.any() else np.nan
            out[f"align_through|{LN[L]}|k{k}"] = float(thr[s].mean()) if s.any() else np.nan
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
            out[f"sign_acf|{LN[L]}|{lag}"] = (float(np.corrcoef(sg[lag:][same], sg[:-lag][same])[0, 1])
                                              if same.sum() > 10 else np.nan)
    # maker markouts (cents/share, count weighted), from match row ~ stamp - 3
    last = d.off[tk.m.values + 1] - 1
    for h in (0,) + MARKOUT_H:
        r = np.clip(r3 + h, first, last)
        mko = -tk.sign.values * (d.mid[r] - tk.vwap.values) * 100
        for L in cm.LENS:
            out[f"markout|{LN[L]}|{h}"] = float(np.mean(mko[Lm == L])) if (Lm == L).any() else np.nan
    for L in cm.LENS:
        s = Lm == L
        if not s.any():
            continue
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


