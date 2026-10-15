#!/usr/bin/env python3
"""Candidate 'predicted fair-value change over the next h s' (h = 1, 2, 5): a ridge model of
the Binance log return over (t, t+h] (in sigma sqrt h units) on the order-flow candidates,
mapped to cents of fair value through dp/dlnS:  pred_dp_h_c = dusd * sigma * sqrt(h) * zhat * 100 / 5.

Fitted out-of-fold on train (the same 6 contiguous 8-hour blocks and 300 s purge as
predictive_power.py), so the train CV of the Polymarket targets stays clean; for --confirm the
model is fitted on all train rows and applied to val. Writes the columns into the panels as
r_pred / f_pred (and prints the Binance-return R2).

    /usr/bin/python3 analysis/orderflow/pred_dp.py [--confirm]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from panel import CAND, GROUPS  # noqa: E402
from predictive_power import CACHE, PURGE_S, Prep, folds, r2, ridge_fit  # noqa: E402

FLOW = GROUPS["ret_sub"] + GROUPS["signed_vol"] + GROUPS["ofi"] + GROUPS["btc_lead"] + GROUPS["activity"]
HS = (1, 2, 5)


def xmat(P, pre):
    ci = [CAND.index(c) for c in FLOW]
    base = P[pre + "base"]
    names = list(P["base_names"])
    ulz = base[:, [names.index("ulz5"), names.index("ulz30")]]      # 5 s / 30 s Binance z-returns
    return np.column_stack([P[pre + "cand"][:, ci], ulz])


def to_cents(P, pre, z, h):
    return np.clip(P[pre + "dusd"] * P[pre + "sigma"] * np.sqrt(h) * z * 100 / 5, -5, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    P = dict(np.load(CACHE / "panel_train.npz"))
    X = xmat(P, "r_")
    t = P["r_t"]
    fb = folds(t)
    Xf, tf = xmat(P, "f_"), P["f_t"]
    ff = folds(tf)
    out_r = np.zeros((len(t), len(HS)))
    out_f = np.zeros((len(tf), len(HS)))
    for j, h in enumerate(HS):
        y = P[f"r_ulfwd{h}"]
        ok = np.isfinite(y)
        ql, qh = np.quantile(y[ok], (0.001, 0.999))
        if not a.confirm:
            oof = np.zeros(len(t))
            for f in np.unique(fb):
                te = fb == f
                lo_t, hi_t = t[te].min(), t[te].max()
                tr = ~te & ok & ((t < lo_t - PURGE_S) | (t > hi_t + PURGE_S))
                pp = Prep(X[tr])
                model = ridge_fit(pp(X[tr]), np.clip(y[tr], ql, qh))
                oof[te] = model(pp(X[te]))
                out_f[ff == f, j] = model(pp(Xf[ff == f]))
            out_r[:, j] = oof
            print(f"ulfwd{h}: OOF R2 of the order-flow model {r2(np.clip(y[ok], ql, qh), oof[ok]):.4f}")
        else:
            pp = Prep(X[ok])
            model = ridge_fit(pp(X[ok]), np.clip(y[ok], ql, qh))
            V = dict(np.load(CACHE / "panel_val.npz"))
            Xv, Xvf = xmat(V, "r_"), xmat(V, "f_")
            zv = model(pp(Xv))
            yv = V[f"r_ulfwd{h}"]
            okv = np.isfinite(yv)
            print(f"ulfwd{h}: val R2 of the train-fitted order-flow model {r2(np.clip(yv[okv], ql, qh), zv[okv]):.4f}")
            V.setdefault("r_pred", np.zeros((len(Xv), len(HS))))
            V.setdefault("f_pred", np.zeros((len(Xvf), len(HS))))
            V["r_pred"][:, j] = to_cents(V, "r_", zv, h)
            V["f_pred"][:, j] = to_cents(V, "f_", model(pp(Xvf)), h)
            if j == len(HS) - 1:
                np.savez(CACHE / "panel_val.npz", **V)
    if not a.confirm:
        P["r_pred"] = np.column_stack([to_cents(P, "r_", out_r[:, j], h) for j, h in enumerate(HS)])
        P["f_pred"] = np.column_stack([to_cents(P, "f_", out_f[:, j], h) for j, h in enumerate(HS)])
        np.savez(CACHE / "panel_train.npz", **P)


if __name__ == "__main__":
    main()
