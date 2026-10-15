#!/usr/bin/env python3
"""When do Polymarket takers react to Binance, relative to what the env lets us see? TRAIN ONLY
(split 0, Aug 21-22; Binance trades before 2026-08-23 are the only ones loaded, asserted).

1. Lead-lag: corr(taker side, Binance 250 ms return in [u + k/4, u + (k+1)/4)) per taker tx,
   u = block second of the print (fills.bin row).
2. AUC of the taker side against the Binance return over the 1 s before u - k. In the env's
   central model with latency 1, the quote facing a print at row u was decided at u - 5
   (L + D = 1 + 4), so Binance is visible only up to u - 5.
3. Book timestamps: corr(book mid change over (t-1, t], delta-scaled Binance 250 ms return in
   [t + k/4, t + (k+1)/4)) on two-sided 5m/15m rows. A peak after t means the book row t holds
   information from after t on Binance's clock (whole-second captured_at stamps).

    /usr/bin/python3 analysis/orderflow/timing.py        (~2 min)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
POLY = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(POLY))
import binance_flow as bf  # noqa: E402
from reference import Data  # noqa: E402

COINS = {0: "btc", 1: "eth", 2: "sol", 3: "xrp", 4: "bnb", 5: "doge", 6: "hype", 7: "zec"}
T_TRAIN_END = 1787443200        # 2026-08-23 00:00 UTC


def auc(y, s):
    y = y.astype(bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def corr_cols(a, M):
    a = a - a.mean()
    M = M - M.mean(0)
    return (a[:, None] * M).mean(0) / (a.std() * M.std(0) + 1e-18)


def main():
    d = Data()
    mk = d.markets
    sel = np.flatnonzero(mk["split"] == 0)
    t0, t1 = int(mk["open_ts"][sel].min()), int(mk["end_ts"][sel].max())
    assert t1 <= T_TRAIN_END, "LEAKAGE: non-train market"
    grids = {}
    offs = np.arange(-64, 9)                       # 250 ms buckets from u - 16 s to u + 2 s
    lags = [6, 5, 4, 3, 2, 1, 0]
    tx = {}                                        # contract length -> (side, returns by offset)
    side_all, last1 = [], []
    bk_dm, bk_r = [], []
    boffs = np.arange(-16, 9)
    for i in sel:
        m = mk[i]
        r0, n, o, L = int(m["row_offset"]), int(m["n_rows"]), int(m["open_ts"]), int(m["contract_len_s"])
        c = COINS[int(m["coin_id"])]
        if c not in grids:
            grids[c] = bf.CoinGrid(c, t0, t1)
            assert grids[c].end_unix <= T_TRAIN_END, "LEAKAGE: Binance data after Aug 22 loaded"
        g = grids[c]
        lnp = g._lnp_full
        lo, hi = int(d.fill_index[r0]), int(d.fill_index[r0 + n])
        fl = d.fills[lo:hi]
        fl = fl[fl["first"] == 1]                  # one row per taker tx
        u = o + (fl["row"].astype(np.int64) - r0)
        bu = 4 * (u - g.base)
        ok = (bu + offs.max() + 1 < len(lnp)) & (bu + offs.min() - 1 >= 0)
        bu, u, s = bu[ok], u[ok], fl["side"][ok].astype(float)
        R = np.stack([lnp[bu + k] - lnp[bu + k - 1] for k in offs], 1)
        tx.setdefault(L, []).append((s, R))
        sg = np.maximum(g.sigma[np.clip(u[:, None] - np.array(lags)[None, :] - g.t0, 0, len(g.sigma) - 1)], 1e-7)
        B = np.stack([lnp[bu - 4 * k - 1] - lnp[bu - 4 * k - 5] for k in lags], 1) / sg
        side_all.append(s)
        last1.append(B)
        if L < 14400:                              # book timing on 5m / 15m rows
            F = np.asarray(d.feat[r0:r0 + n], np.float64)
            mark, fv, dl = F[:, d.col["mark"]], F[:, d.col["mark_is_fv"]], F[:, d.col["delta_usd_raw"]]
            k = np.arange(1, n - 3)
            k = k[(fv[k] == 0) & (fv[k - 1] == 0)]
            bt = 4 * (o + k - g.base)
            okb = (bt + boffs.max() + 1 < len(lnp)) & (bt + boffs.min() - 1 >= 0)
            k, bt = k[okb], bt[okb]
            bk_dm.append(mark[k] - mark[k - 1])
            bk_r.append(np.stack([(lnp[bt + j] - lnp[bt + j - 1]) * dl[k] for j in boffs], 1))
    print("1. corr(taker side, Binance 250 ms return in [u + k/4, u + (k+1)/4)), u = block second")
    for L, parts in sorted(tx.items()):
        s = np.concatenate([p[0] for p in parts])
        R = np.concatenate([p[1] for p in parts])
        cr = corr_cols(s, R)
        top = np.argsort(-cr)[:4]
        print(f"   {L // 60}m: {len(s)} txs; peak " + ", ".join(f"k/4={offs[j] / 4:+.2f}s {cr[j]:.3f}" for j in top))
        print("      " + " ".join(f"{offs[j] / 4:+.1f}:{cr[j]:.3f}" for j in range(32, len(offs), 2)))
    s = np.concatenate(side_all)
    B = np.concatenate(last1)
    print("2. AUC(taker buys Up | Binance return over [u-k-1, u-k)), all txs:")
    print("   " + "  ".join(f"k={k}: {auc(s > 0, B[:, j]):.3f}" for j, k in enumerate(lags)))
    print("   env central, latency 1: the quote facing print u was decided at u-5 -> column k=5")
    dm = np.concatenate(bk_dm)
    R = np.concatenate(bk_r)
    cr = corr_cols(dm, R)
    print(f"3. corr(book mid change over (t-1, t], Binance 250 ms return in [t + k/4, t + (k+1)/4)), {len(dm)} rows:")
    print("   " + " ".join(f"{o_ / 4:+.2f}:{c_:.3f}" for o_, c_ in zip(boffs, cr)))


if __name__ == "__main__":
    main()
