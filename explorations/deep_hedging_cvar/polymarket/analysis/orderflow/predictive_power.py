#!/usr/bin/env python3
"""Incremental predictive power of sub-second Binance order flow over the observed feat.bin
columns, for the targets that make our fills toxic. TRAIN ONLY unless --confirm.

Targets (panel.py):
  rows : dmid{1,5,30}  book-mid change over the next h s (cents; two-sided at both ends)
         fnet{1,5}     signed taker flow matched in (t, t+h] (print rows t+5..t+4+h, shares)
         ftot{1,5}     taker volume in the same window
         ulfwd{1,2,5}  Binance log return over (t, t+h] / (sigma sqrt h)
  fills: mo60 / mo30 / mo5  maker markout of a touch fill vs the mid at the match row (cents)
         edge60             spread capture + 60 s markout (cents)
         quote_edge         per 5m/15m decision row and side: edge60 summed over the touch fills a
                            touch quote decided at t would face (match row t + 1)
Models: ridge / logistic on standardised, winsorised inputs (fills: x and m*x, m = maker sign),
and a small LightGBM (fixed settings, not tuned). CV: 6 contiguous 8-hour blocks of Aug 21-22,
training rows within 300 s of a test block purged. --confirm: fit on all train, score val
(panel_val.npz = data/build_v2 split 1, Aug 23 + 24).

Outcome (2026-10-15): no consistent gain on the fill-toxicity targets (markouts, toxic-fill AUC,
quote value) or on the taker-flow sign, on train or val, linear or GBM; so no v2 feature columns
or env observations were added. timing.py shows why: takers react to Binance 2-3.5 s before the
print's block time, after the env's last decision for that print (u - 5 at L = 1, D = 4).

Run order: panel.py --split train, pred_dp.py, predictive_power.py [--models lin,gbm];
then panel.py --split val --build data/build_v2, pred_dp.py --confirm, predictive_power.py --confirm.
LightGBM is not in this repo's Pythons: pip install --target DIR lightgbm narwhals, then run
without -I and with DIR on sys.path (e.g. PYTHONPATH=DIR).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
T0 = 1787270400                 # 2026-08-21 00:00 UTC
PURGE_S = 300
WINS = (0.001, 0.999)
sys.path.insert(0, str(HERE))
from panel import GROUPS, CAND  # noqa: E402

PRED = ["pred_dp1", "pred_dp2", "pred_dp5"]   # analysis/orderflow/pred_dp.py

GBM = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=2, verbose=-1, seed=0)
ROUNDS = 300


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def auc(y, s):
    y = y.astype(bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class Prep:
    """Winsorise at train quantiles, standardise with train moments."""

    def __init__(self, X):
        self.lo, self.hi = np.quantile(X, WINS[0], 0), np.quantile(X, WINS[1], 0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu, self.sd = Xc.mean(0), Xc.std(0) + 1e-9

    def __call__(self, X):
        return (np.clip(X, self.lo, self.hi) - self.mu) / self.sd


def ridge_fit(X, y, lam=1.0):
    n, k = X.shape
    A = X.T @ X + lam * n * 1e-4 * np.eye(k)
    b = np.linalg.solve(A, X.T @ (y - y.mean()))
    return lambda Z: y.mean() + Z @ b


def logit_fit(X, y, lam=1.0):
    n, k = X.shape
    Xa = np.column_stack([np.ones(n), X])
    reg = lam * 1e-4 * n

    def f(w):
        z = Xa @ w
        l = np.logaddexp(0, z) - y * z
        g = Xa.T @ (1 / (1 + np.exp(-z)) - y)
        return l.sum() + 0.5 * reg * w[1:] @ w[1:], g + reg * np.r_[0, w[1:]]
    w = minimize(f, np.zeros(k + 1), jac=True, method="L-BFGS-B", options=dict(maxiter=500)).x
    return lambda Z: np.column_stack([np.ones(len(Z)), Z]) @ w


def gbm_fit(X, y, binary):
    import lightgbm as lgb
    p = dict(GBM, objective="binary" if binary else "regression")
    b = lgb.train(p, lgb.Dataset(X, y), num_boost_round=ROUNDS)
    return b.predict


def add_quotes(P):
    """Quote-level target on 5m/15m decision rows (dt = 1): for row t and side s (+1 bid, -1 ask),
    q_y = sum of edge60 (cents) of the touch fills on that side whose match row is t + 1, i.e. what
    a touch quote decided at t (latency 1) would have faced; 0 if none. Two entries per row."""
    keep = P["r_clen"] < 14400
    rid = np.flatnonzero(keep)
    key_r = P["r_market"][rid].astype(np.int64) * 10**7 + (P["r_t"][rid] - T0 + 10**5)
    o = np.argsort(key_r)
    key_f = P["f_market"].astype(np.int64) * 10**7 + (P["f_t"] - 1 - T0 + 10**5)   # decision row tau - 1
    pos = np.searchsorted(key_r[o], key_f)
    pos = np.minimum(pos, len(o) - 1)
    hit = key_r[o][pos] == key_f
    yb = np.zeros(len(rid))
    ya = np.zeros(len(rid))
    j = o[pos[hit]]
    e, mm = P["f_edge60"][hit], P["f_m"][hit]
    np.add.at(yb, j[mm > 0], e[mm > 0])
    np.add.at(ya, j[mm < 0], e[mm < 0])
    P["q_row"] = np.r_[rid, rid]
    P["q_m"] = np.r_[np.ones(len(rid)), -np.ones(len(rid))]
    P["q_y"] = np.r_[yb, ya]
    P["q_t"] = P["r_t"][P["q_row"]]
    P["q_mapped_fills"] = int(hit.sum())


def folds(t):
    blk = ((t - T0) // 3600) // 8
    return blk


def design(P, kind, idx, cand_idx, linear):
    """Feature matrix: base columns + selected candidate columns; fills add m (and m*x if linear)."""
    if kind == "quotes":
        rows = P["q_row"][idx]
        X = [P["r_base"][rows]]
        if len(cand_idx):
            X.append(P["r_cand"][rows][:, cand_idx])
        X = np.column_stack(X)
        m = P["q_m"][idx][:, None]
        return np.column_stack([m, X, m * X]) if linear else np.column_stack([m, X])
    pre = "r_" if kind == "rows" else "f_"
    X = [P[pre + "base"][idx]]
    if len(cand_idx):
        X.append(P[pre + "cand"][idx][:, cand_idx])
    X = np.column_stack(X)
    if kind == "fills":
        m = P["f_m"][idx][:, None]
        X = np.column_stack([m, X, m * X]) if linear else np.column_stack([m, X])
    return X


TARGETS = {
    # name: (kind, panel key, transform, type)
    "dmid1": ("rows", "dmid1", None, "reg"),
    "dmid5": ("rows", "dmid5", None, "reg"),
    "dmid30": ("rows", "dmid30", None, "reg"),
    "dmid5_sign": ("rows", "dmid5", "sign", "cls"),
    "fnet1_sign": ("rows", "fnet1", "sign", "cls"),
    "fnet5_sign": ("rows", "fnet5", "sign", "cls"),
    "fnet5": ("rows", "fnet5", "asinh10", "reg"),
    "ftot5": ("rows", "ftot5", "log1p", "reg"),
    "ulfwd1": ("rows", "ulfwd1", None, "reg"),
    "ulfwd5": ("rows", "ulfwd5", None, "reg"),
    "mo60": ("fills", "mo60", None, "reg"),
    "mo5": ("fills", "mo5", None, "reg"),
    "mo60_toxic": ("fills", "mo60", "neg", "cls"),
    "edge60": ("fills", "edge60", None, "reg"),
    "quote_edge": ("quotes", "y", None, "reg"),
}
PREFIX = {"rows": "r_", "fills": "f_", "quotes": "q_"}


def target(P, name):
    kind, key, tr, typ = TARGETS[name]
    pre = PREFIX[kind]
    y = P[pre + key].astype(np.float64)
    ok = np.isfinite(y)
    if tr == "sign":
        ok &= y != 0
        y = (y > 0).astype(float)
    elif tr == "neg":
        ok &= y != 0
        y = (y < 0).astype(float)
    elif tr == "asinh10":
        y = np.arcsinh(y / 10)
    elif tr == "log1p":
        y = np.log1p(np.maximum(y, 0))
    return kind, np.flatnonzero(ok), y, typ


def evaluate(P, Pv, name, cand_idx, model):
    """OOF metric on train (blocked CV) or, with Pv, the train-fit metric on val. Returns
    (metric, oof predictions or val predictions, index)."""
    kind, idx, y, typ = target(P, name)
    pre = PREFIX[kind]
    linear = model == "lin"
    fit = (lambda X, yy: gbm_fit(X, yy, typ == "cls")) if not linear else \
          (logit_fit if typ == "cls" else ridge_fit)
    if Pv is None:
        t = P[pre + "t"][idx]
        fb = folds(t)
        X = design(P, kind, idx, cand_idx, linear)
        pred = np.zeros(len(idx))
        for f in np.unique(fb):
            te = fb == f
            lo_t, hi_t = t[te].min(), t[te].max()
            tr = ~te & ((t < lo_t - PURGE_S) | (t > hi_t + PURGE_S))
            ytr = y[idx][tr]
            if typ == "reg":
                ql, qh = np.quantile(ytr, WINS)
                ytr = np.clip(ytr, ql, qh)
            if linear:
                pp = Prep(X[tr])
                pred[te] = fit(pp(X[tr]), ytr)(pp(X[te]))
            else:
                pred[te] = fit(X[tr], ytr)(X[te])
        yy = y[idx]
        if typ == "reg":                       # score against y winsorised at the pooled train quantiles
            ql, qh = np.quantile(yy, WINS)
            return r2(np.clip(yy, ql, qh), pred), pred, idx
        return auc(yy, pred), pred, idx
    kv, idv, yv, _ = target(Pv, name)
    X = design(P, kind, idx, cand_idx, linear)
    Xv = design(Pv, kind, idv, cand_idx, linear)
    ytr = y[idx]
    if typ == "reg":
        ql, qh = np.quantile(ytr, WINS)
        ytr = np.clip(ytr, ql, qh)
    if linear:
        pp = Prep(X)
        pred = fit(pp(X), ytr)(pp(Xv))
    else:
        pred = fit(X, ytr)(Xv)
    yy = yv[idv]
    if typ == "reg":
        return r2(np.clip(yy, ql, qh), pred), pred, idv
    return auc(yy, pred), pred, idv


def quote_value(P, pred, idx, frac):
    """Total quote edge (cents, summed over kept (row, side) pairs, per 1000 pairs) when only the
    top (1 - frac) of pairs by predicted value are quoted, and the share of all touch-fill loss
    avoided."""
    y = P["q_y"][idx]
    keep = pred > np.quantile(pred, frac)
    return y[keep].sum() / len(y) * 1000, 1 - y[keep].sum() / y.sum()


def selection_value(P, pred, idx, frac):
    """Edge (cents/share, per touch fill) kept when the fills with the lowest predicted edge
    (fraction frac) are skipped, vs keeping all."""
    e = P["f_edge60"][idx]
    cut = np.quantile(pred, frac)
    keep = pred > cut
    return e[keep].sum() / len(e), e.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbm", action="store_true")
    ap.add_argument("--models", default=None, help="comma list of lin,gbm (default: lin, plus gbm with --gbm)")
    ap.add_argument("--confirm", action="store_true", help="fit on train, score val (frozen choices)")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--groups", default="all," + ",".join(GROUPS))
    ap.add_argument("--select", default=None, help="comma list of candidate columns (the chosen set)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="panel variant, e.g. _D1 (panel.py --dmatch 1 what-if)")
    a = ap.parse_args()
    P = dict(np.load(CACHE / f"panel_train{a.tag}.npz"))
    Pv = dict(np.load(CACHE / f"panel_val{a.tag}.npz")) if a.confirm else None
    add_quotes(P)
    if Pv is not None:
        add_quotes(Pv)
    if "r_pred" in P:                      # pred_dp.py columns (out-of-fold on train)
        for Q in (P, Pv) if Pv is not None else (P,):
            for pre in ("r_", "f_"):
                Q[pre + "cand"] = np.column_stack([Q[pre + "cand"], Q[pre + "pred"]])
        if "pred_dp1" not in CAND:
            CAND.extend(PRED)
            GROUPS["pred_dp"] = list(PRED)
    sets = {"base": []}
    for g in a.groups.split(","):
        if g == "pred_dp" and "pred_dp" not in GROUPS:
            continue
        if g == "all":
            sets["+all"] = list(range(len(CAND)))
        elif g:
            sets["+" + g] = [CAND.index(c) for c in GROUPS[g]]
    if a.select:
        sets["+chosen"] = [CAND.index(c) for c in a.select.split(",")]
    models = a.models.split(",") if a.models else ["lin"] + (["gbm"] if a.gbm else [])
    res = {}
    for name in a.targets.split(","):
        for model in models:
            row = {}
            for sname, ci in sets.items():
                t1 = time.time()
                met, pred, idx = evaluate(P, Pv, name, ci, model)
                row[sname] = met
                if name == "quote_edge" and sname in ("base", "+all", "+chosen", "+staleness", "+pred_dp"):
                    src = Pv if a.confirm else P
                    for frac in (0.5, 0.9, 0.99):
                        v, avoided = quote_value(src, pred, idx, frac)
                        row[f"{sname}:top{round((1 - frac) * 100)}%"] = v
                    row["keep_all"] = src["q_y"][idx].sum() / len(idx) * 1000
                if name == "edge60" and sname in ("base", "+all", "+chosen"):
                    src = Pv if a.confirm else P
                    for frac in (0.1, 0.25, 0.5):
                        kept, allv = selection_value(src, pred, idx, frac)
                        row[f"{sname}:skip{int(frac * 100)}"] = kept
                    row["keep_all"] = allv
            res[f"{name}/{model}"] = row
            b = row["base"]
            line = "  ".join(f"{k} {v:+.4f}" if k == "base" else f"{k} {v - b:+.4f}"
                             for k, v in row.items() if ":" not in k and k != "keep_all")
            extra = "  ".join(f"{k} {v:+.3f}" for k, v in row.items() if ":" in k or k == "keep_all")
            print(f"{name:12s} {model}: {line}" + (f"\n    selection (edge60: cents/fill; quote_edge: cents per 1000 row-sides): {extra}" if extra else ""),
                  flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
