"""Q3: lead-lag between Binance-implied model p and the Polymarket Up mid (TRAIN split only).

* cross-correlation corr(dmid over [t_i, t_{i+1}], dp over [t_i - L, t_{i+1} - L]) for L in -60..60 s
  (L > 0: Binance move happens L s before the Polymarket move -> Polymarket lags)
* distributed-lag regression (joint OLS over L=-10..60) -> cumulative impulse response
* predictive power of gap = p - mid for mid changes over the next 5 / 30 s
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fair_value as fv

CFG = json.load(open(fv.CACHE / "model_cfg.json"))
P = pd.read_parquet(fv.CACHE / "panel_scored.parquet")
u = fv.load_underlying()
CLS = ["5m", "15m", "4h"]
LAGS = np.arange(-60, 61)
EXCL_END = 30  # drop the last 30 s (p saturates, books one-sided)


def p_grid(slug_rows):
    """model p at every second of [open-60, end] for each market -> dict slug -> (t0, array)."""
    out = {}
    for c, g in slug_rows.groupby("coin"):
        sig = np.load(fv.CACHE / f"sigma_{c}.npz")[CFG["sigma"]].astype(float) * CFG["mult"]
        for r in g.itertuples():
            ts = np.arange(r.t_open - 60, r.t_end + 1)
            ts = ts[ts < fv.T_TRAIN_END]
            out[r.slug] = (ts[0], fv.model_at(u[c], sig, ts, np.full(len(ts), r.t_end), np.full(len(ts), r.K),
                                              shift=CFG["shift"], nu=CFG["nu"]))
    return out


mk = P.groupby("slug").agg(coin=("coin", "first"), cl=("cl", "first"), t_end=("t_end", "first"),
                           K=("k_bin0", "first")).reset_index()
mk["t_open"] = mk.t_end - mk.cl.map(fv.LEN_S)
PG = p_grid(mk)
json.dump({k: [int(v[0]), v[1].round(6).tolist()] for k, v in PG.items()}, open(fv.CACHE / "p_grid.json", "w"))


def p_at(slug, t):
    t0, a = PG[slug]
    i = np.asarray(t) - t0
    ok = (i >= 0) & (i < len(a))
    return np.where(ok, a[np.clip(i, 0, len(a) - 1)], np.nan)


# ---------- consecutive snapshot pairs ----------
Q = P[P.two_sided & (P.tau > EXCL_END)].sort_values(["slug", "t"]).copy()
Q["t_next"] = Q.groupby("slug").t.shift(-1)
Q["mid_next"] = Q.groupby("slug").mid.shift(-1)
Q = Q[Q.t_next.notna() & (Q.t_next - Q.t <= 20)].copy()
Q["t_next"] = Q.t_next.astype(np.int64)
Q["dmid"] = Q.mid_next - Q.mid
Q["t_open"] = Q.t_end - Q.cl.map(fv.LEN_S)
print("TRAIN pairs:", Q.groupby("cl").size().to_dict(), "median gap s:", Q.groupby("cl").apply(lambda x: (x.t_next - x.t).median()).to_dict())

XL = {}
for L in LAGS:
    a = np.concatenate([p_at(s, g.t_next.values - L) - p_at(s, g.t.values - L) for s, g in Q.groupby("slug", sort=True)])
    XL[L] = a
Qs = Q.sort_values(["slug", "t"])  # same order as groupby(sort=True) concat
dm = Qs.dmid.values
cl_arr = Qs.cl.values
ok_all = np.all(np.isfinite(np.column_stack([XL[L] for L in LAGS])), axis=1)

res = {}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
for cl, col in zip(CLS, ("C0", "C1", "C2")):
    sel = (cl_arr == cl) & ok_all
    cc = np.array([np.corrcoef(XL[L][sel], dm[sel])[0, 1] for L in LAGS])
    Lmax = int(LAGS[np.argmax(cc)])
    # distributed lag regression L=-10..60
    Ls = np.arange(-10, 61)
    X = np.column_stack([XL[L][sel] for L in Ls])
    beta, *_ = np.linalg.lstsq(X, dm[sel], rcond=None)
    cum = np.cumsum(beta)
    tot = cum[-1]
    half = int(Ls[np.argmax(cum >= 0.5 * tot)])
    p90 = int(Ls[np.argmax(cum >= 0.9 * tot)])
    resid = dm[sel] - X @ beta
    r2 = 1 - resid.var() / dm[sel].var()
    res[cl] = dict(n_pairs=int(sel.sum()), peak_lag_s=Lmax, peak_corr=float(cc.max()), corr_lag0=float(cc[LAGS == 0][0]),
                   corr_lag_m5=float(cc[LAGS == -5][0]), dl_total_passthrough=float(tot), dl_lag_50pct=half,
                   dl_lag_90pct=p90, dl_r2=float(r2),
                   cum_at={int(L): float(cum[Ls == L][0] / tot) for L in (0, 2, 5, 10, 20, 30, 60)})
    axs[0].plot(LAGS, cc, color=col, label=f"{cl} (peak {Lmax:+d}s)")
    axs[1].plot(Ls, cum, color=col, label=f"{cl}: 50% by {half}s, 90% by {p90}s")
axs[0].axvline(0, color="k", lw=0.6)
axs[0].set_xlabel("lag L (s); L>0 = Binance leads")
axs[0].set_ylabel("corr(d mid, d model p shifted by L)")
axs[0].set_title("Cross-correlation of changes (train)")
axs[0].legend()
axs[1].set_xlabel("lag L (s)")
axs[1].set_ylabel("cumulative pass-through of model-p move into mid")
axs[1].set_title("Distributed-lag impulse response (train)")
axs[1].axhline(0, color="k", lw=0.5)
axs[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(fv.FIG / "q3_leadlag.png", dpi=120)

# ---------- gap predictability ----------
G = P[P.two_sided & (P.tau > EXCL_END)].sort_values(["slug", "t"]).copy()
G["gap"] = G.p_model - G.mid
rows = []
for h in (5, 30):
    fut = []
    for s, g in G.groupby("slug", sort=True):
        t = g.t.values
        j = np.searchsorted(t, t + h, side="left")
        jj = np.clip(j, 0, len(t) - 1)
        okj = (j < len(t)) & (t[jj] <= t + h + 5)
        fut.append(np.where(okj, g.mid.values[jj], np.nan))
    G[f"mid_f{h}"] = np.concatenate(fut)
    G[f"dmid{h}"] = G[f"mid_f{h}"] - G.mid
    G[f"dp{h}"] = np.concatenate([p_at(s, g.t.values + h) for s, g in G.groupby("slug", sort=True)]) - G.p_model
    for cl in CLS + ["all"]:
        d = G if cl == "all" else G[G.cl == cl]
        d = d[d[f"dmid{h}"].notna()]
        x, yv = d.gap.values, d[f"dmid{h}"].values
        b = np.sum(x * yv) / np.sum(x * x)
        r2 = 1 - np.var(yv - b * x) / np.var(yv)
        # add contemporaneous future Binance move (not tradable, upper-bound reference)
        X2 = np.column_stack([x, d[f"dp{h}"].values, np.ones(len(x))])
        okk = np.isfinite(X2).all(1)
        bb, *_ = np.linalg.lstsq(X2[okk], yv[okk], rcond=None)
        r2b = 1 - np.var(yv[okk] - X2[okk] @ bb) / np.var(yv[okk])
        dpv = d[f"dp{h}"].values
        okp = np.isfinite(dpv)
        b_p = np.sum(x[okp] * dpv[okp]) / np.sum(x[okp] ** 2)
        rows.append(dict(h=h, cl=cl, n=len(d), slope_on_gap=b, slope_dp_on_gap=b_p, R2_gap=r2, corr=np.corrcoef(x, yv)[0, 1],
                         R2_gap_plus_future_dp=r2b, sd_dmid=yv.std(), mean_abs_gap=np.abs(x).mean()))
gp = pd.DataFrame(rows)
print("\nTRAIN: lead-lag summary")
print(pd.DataFrame(res).T.drop(columns=["cum_at"]).to_string())
print({k: v["cum_at"] for k, v in res.items()})
print("\nTRAIN: predictive regressions dmid(t->t+h) = b*gap(t)  (no intercept)")
print(gp.round(4).to_string(index=False))
gp.to_csv(fv.CACHE / "q3_gap_pred.csv", index=False)
json.dump(res, open(fv.CACHE / "q3_leadlag.json", "w"), indent=1)

# gap half-life: AR(1) of gap at snapshots
G["gap_next"] = G.groupby("slug").gap.shift(-1)
G["dt"] = G.groupby("slug").t.shift(-1) - G.t
for cl in CLS:
    d = G[(G.cl == cl) & G.gap_next.notna() & (G.dt <= 20)]
    rho = np.sum(d.gap * d.gap_next) / np.sum(d.gap ** 2)
    print(f"{cl}: gap AR coef per snapshot {rho:.3f}, mean dt {d.dt.mean():.1f}s -> half-life {np.log(0.5) / np.log(rho) * d.dt.mean():.1f}s;"
          f" mean gap {d.gap.mean():+.4f}, sd gap {d.gap.std():.4f}")
