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
Models: ridge / logistic on standardised, winsorised inputs (fills: x and m*x, m = maker sign),
and a small LightGBM (fixed settings, not tuned). CV: 6 contiguous 8-hour blocks of Aug 21-22,
training rows within 300 s of a test block purged. --confirm: fit on all train, score val.

    PYTHONPATH=<lightgbm> /usr/bin/python3 analysis/orderflow/predictive_power.py [--gbm] [--confirm]
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


def folds(t):
    blk = ((t - T0) // 3600) // 8
    return blk


def design(P, kind, idx, cand_idx, linear):
    """Feature matrix: base columns + selected candidate columns; fills add m (and m*x if linear)."""
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
}


def target(P, name):
    kind, key, tr, typ = TARGETS[name]
    pre = "r_" if kind == "rows" else "f_"
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
    pre = "r_" if kind == "rows" else "f_"
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
    ap.add_argument("--confirm", action="store_true", help="fit on train, score val (frozen choices)")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--groups", default="all," + ",".join(GROUPS))
    ap.add_argument("--select", default=None, help="comma list of candidate columns (the chosen set)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = dict(np.load(CACHE / "panel_train.npz"))
    Pv = dict(np.load(CACHE / "panel_val.npz")) if a.confirm else None
    sets = {"base": []}
    for g in a.groups.split(","):
        if g == "all":
            sets["+all"] = list(range(len(CAND)))
        elif g:
            sets["+" + g] = [CAND.index(c) for c in GROUPS[g]]
    if a.select:
        sets["+chosen"] = [CAND.index(c) for c in a.select.split(",")]
    models = ["lin"] + (["gbm"] if a.gbm else [])
    res = {}
    for name in a.targets.split(","):
        for model in models:
            row = {}
            for sname, ci in sets.items():
                t1 = time.time()
                met, pred, idx = evaluate(P, Pv, name, ci, model)
                row[sname] = met
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
            print(f"{name:12s} {model}: {line}" + (f"\n    selection (cents/fill): {extra}" if extra else ""),
                  flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
