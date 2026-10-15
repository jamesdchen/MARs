"""Q2: fair-value model vs Polymarket mid vs blend, TRAIN split only (markets of 2026-08-21/22).

Weighting: each snapshot weighted 1/(#snapshots of its market in its time-to-close bucket), so
every market counts equally within a bucket; pooled numbers average buckets equally.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import fair_value as fv

EPS = 1e-3          # clip probabilities to [tick, 1-tick] before log loss
EDGES = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0001]
LABELS = ["0-5%", "5-10%", "10-25%", "25-50%", "50-75%", "75-100%"]
CLS = ["5m", "15m", "4h"]

P = pd.read_parquet(fv.CACHE / "panel.parquet")
P = P[P.mid.notna()].reset_index(drop=True)
u = fv.load_underlying()
P["bucket"] = pd.cut(P.frac, EDGES, labels=LABELS, right=True)
P["w"] = 1.0 / P.groupby(["slug", "bucket"], observed=True).t.transform("size")
P["day"] = (P.t_open // 86400).astype(int)
y = P.winner_up.values.astype(float)


def ll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p, y):
    return (p - y) ** 2


def wmean(x, w):
    return np.sum(x * w) / np.sum(w)


def pooled(loss, df):
    """mean over cl x bucket cells of weighted mean (equal weight per cell)."""
    d = df.assign(L=loss)
    cells = d.groupby(["cl", "bucket"], observed=True).apply(lambda g: wmean(g.L.values, g.w.values))
    return cells.mean()


def model_p(df, sigkey, shift=0, nu=None, mult=1.0):
    out = np.empty(len(df))
    for c, idx in df.groupby("coin").indices.items():
        g = df.iloc[idx]
        sig = np.load(fv.CACHE / f"sigma_{c}.npz")[sigkey].astype(float) * mult
        out[idx] = fv.model_at(u[c], sig, g.t.values, g.t_end.values, g[f"k_bin{shift}"].values,
                               shift=shift, nu=nu)
    return out


res = {}
# ---------- 1. choose sigma estimator (half-life x sampling) and window alignment ----------
rows = []
keys = [k for k in P.columns if k.startswith("sig_")]
for key in keys:
    k = key[4:]
    for sh in (0, -4):
        p = model_p(P, k, sh)
        rows.append(dict(sigma=k, shift=sh, logloss=pooled(ll(p, y), P), brier=pooled(brier(p, y), P),
                         **{f"ll_{cl}": pooled(ll(p, y)[P.cl == cl], P[P.cl == cl]) for cl in CLS}))
grid = pd.DataFrame(rows).sort_values("logloss")
print("TRAIN: sigma/alignment grid (pooled, equal weight per cl x bucket cell)")
print(grid.round(4).to_string(index=False))
best = grid.iloc[0]
# prefer the longer (less noisy) half-life when within 0.001 nats of the best
ok = grid[grid.logloss <= best.logloss + 0.001].copy()
ok["hl"] = ok.sigma.str.extract(r"h(\d+)_")[0].astype(int)
ch = ok.sort_values(["hl", "logloss"], ascending=[False, True]).iloc[0]
SIG, SH = ch["sigma"], int(ch["shift"])
print("chosen:", SIG, SH, "LL", round(ch["logloss"], 4), "(best", best["sigma"], int(best["shift"]), round(best["logloss"], 4), ")")
grid.to_csv(fv.CACHE / "q2_sigma_grid.csv", index=False)

# vol multiplier and Student-t on the chosen sigma
rows = []
for mult in (0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
    for nu in (None, 3, 4, 6, 10):
        p = model_p(P, SIG, SH, nu=nu, mult=mult)
        rows.append(dict(mult=mult, nu=nu or 0, logloss=pooled(ll(p, y), P), brier=pooled(brier(p, y), P)))
tg = pd.DataFrame(rows).sort_values("logloss")
print("\nTRAIN: vol multiplier x Student-t dof (0 = Gaussian), sigma=", SIG, "shift=", SH)
print(tg.round(4).to_string(index=False))
tg.to_csv(fv.CACHE / "q2_mult_nu_grid.csv", index=False)
bt = tg.iloc[0]
MULT, NU = float(bt.mult), (None if bt.nu == 0 else int(bt.nu))
json.dump(dict(sigma=SIG, shift=SH, mult=MULT, nu=NU), open(fv.CACHE / "model_cfg.json", "w"))

# ---------- 2. model vs mid vs blend ----------
P["p_gauss"] = model_p(P, SIG, SH, None, 1.0)
P["p_model"] = model_p(P, SIG, SH, NU, MULT)
lg = lambda p: np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))
X = np.column_stack([lg(P.p_model.values), lg(P.mid.values), np.ones(len(P))])


def fit_blend(idx):
    w = P.w.values[idx]

    def f(b):
        z = X[idx] @ b
        q = 1 / (1 + np.exp(-z))
        return wmean(ll(q, y[idx]), w)
    return minimize(f, np.array([0.5, 0.5, 0.0]), method="Nelder-Mead").x


P["p_blend"] = np.nan
coefs = {}
for cl in CLS:
    for d in sorted(P.day.unique()):
        tr = np.where((P.cl == cl) & (P.day != d))[0]
        te = np.where((P.cl == cl) & (P.day == d))[0]
        b_ = fit_blend(tr)
        coefs[f"{cl}_fit_excl_day{d}"] = b_.round(3).tolist()
        P.loc[te, "p_blend"] = 1 / (1 + np.exp(-(X[te] @ b_)))
print("\nBlend coefs [a_model, b_mid, c] (logit space, 2-fold by day):", coefs)
res["blend_coefs"] = coefs

tab = []
for cl in CLS + ["all"]:
    for bk in LABELS + ["pooled"]:
        d = P if cl == "all" else P[P.cl == cl]
        if bk != "pooled":
            d = d[d.bucket == bk]
        if not len(d):
            continue
        r = dict(cl=cl, bucket=bk, n_mkt=d.slug.nunique(), n_snap=len(d))
        yy = d.winner_up.values
        for name in ("p_model", "mid", "p_blend", "p_gauss"):
            if bk == "pooled":
                r[f"LL_{name}"] = pooled(ll(d[name].values, yy), d)
                r[f"BS_{name}"] = pooled(brier(d[name].values, yy), d)
            else:
                r[f"LL_{name}"] = wmean(ll(d[name].values, yy), d.w.values)
                r[f"BS_{name}"] = wmean(brier(d[name].values, yy), d.w.values)
        # market-clustered bootstrap of LL(model)-LL(mid)
        dd = d.assign(diff=(ll(d.p_model.values, yy) - ll(d.mid.values, yy)) * d.w.values)
        per = dd.groupby("slug").agg(s=("diff", "sum"), w=("w", "sum"))
        rng = np.random.default_rng(0)
        bs = [per.s.values[i].sum() / per.w.values[i].sum()
              for i in (rng.integers(0, len(per), len(per)) for _ in range(500))]
        r["dLL_model_minus_mid_ci90"] = f"[{np.percentile(bs, 5):+.3f},{np.percentile(bs, 95):+.3f}]"
        r["mean_abs_gap"] = wmean(np.abs(d.p_model - d.mid).values, d.w.values)
        tab.append(r)
tab = pd.DataFrame(tab)
pd.set_option("display.width", 250)
print("\nTRAIN: log loss (LL) and Brier (BS) by contract length x time-to-close bucket (frac of life left)")
print(tab.round(4).to_string(index=False))
tab.to_csv(fv.CACHE / "q2_scores.csv", index=False)
P[["slug", "coin", "cl", "t", "t_end", "tau", "frac", "bucket", "mid", "best_bid", "best_ask", "two_sided",
   "spread", "p_model", "p_gauss", "p_blend", "winner_up", "w", "S", "k_bin0", "k_bin-4"]].to_parquet(
    fv.CACHE / "panel_scored.parquet", index=False)

# ---------- 3. calibration plots ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
bins = np.linspace(0, 1, 11)
for ax, cl in zip(axs, CLS):
    d = P[P.cl == cl]
    for name, col, lab in (("p_model", "#1f77b4", "model p"), ("mid", "#d62728", "Polymarket mid"),
                           ("p_blend", "#2ca02c", "blend (CV)")):
        k = np.clip(np.digitize(d[name], bins) - 1, 0, 9)
        g = pd.DataFrame({"k": k, "p": d[name], "y": d.winner_up, "w": d.w}).groupby("k").apply(
            lambda x: pd.Series({"p": wmean(x.p, x.w), "y": wmean(x.y, x.w), "n": x.slug.nunique() if "slug" in x else len(x)}))
        ax.plot(g.p, g.y, "o-", color=col, label=lab, ms=4)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title(f"{cl}: calibration (train, {d.slug.nunique()} mkts)")
    ax.set_xlabel("predicted P(Up)")
axs[0].set_ylabel("realised Up frequency (market-weighted)")
axs[0].legend()
plt.tight_layout()
plt.savefig(fv.FIG / "q2_calibration.png", dpi=120)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for name, ls in (("p_model", "-"), ("mid", "--"), ("p_blend", ":")):
    for cl, col in zip(CLS, ("#1f77b4", "#ff7f0e", "#2ca02c")):
        d = tab[(tab.cl == cl) & (tab.bucket != "pooled")]
        axs[0].plot(d.bucket, d[f"LL_{name}"], ls, color=col, marker="o", ms=3, label=f"{cl} {name}")
        axs[1].plot(d.bucket, d[f"BS_{name}"], ls, color=col, marker="o", ms=3)
axs[0].set_title("log loss by fraction of life remaining (train)")
axs[1].set_title("Brier by fraction of life remaining (train)")
axs[0].legend(fontsize=7, ncol=3)
for a in axs:
    a.set_xlabel("time-to-close bucket (fraction of contract life left)")
plt.tight_layout()
plt.savefig(fv.FIG / "q2_scores_by_bucket.png", dpi=120)

# example paths
fig, axs = plt.subplots(3, 1, figsize=(11, 9))
for ax, cl in zip(axs, CLS):
    s = P[P.cl == cl].slug.unique()[3]
    d = P[P.slug == s]
    ax.plot(d.tau, d.p_model, label="model p (Binance)")
    ax.plot(d.tau, d.mid, label="Polymarket Up mid", alpha=0.8)
    ax.fill_between(d.tau, d.best_bid, d.best_ask, alpha=0.15, color="C1", step="post")
    ax.invert_xaxis()
    ax.set_title(f"{s} (train) winner_up={int(d.winner_up.iloc[0])}")
    ax.set_xlabel("seconds to close")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(fv.FIG / "q2_example_paths.png", dpi=110)
print("cfg:", dict(sigma=SIG, shift=SH, mult=MULT, nu=NU))
