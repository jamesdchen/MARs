"""Q4: adverse selection of taker flow vs the model-p / mid gap (TRAIN split only).

Taker fills (data-api takerOnly=true; one row per matched tx, aggressor side). Timestamps are Polygon
block times, so "pre-trade" state is taken at ts - PRE seconds.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fair_value as fv

PRE = 2
CLS = ["5m", "15m", "4h"]
P = pd.read_parquet(fv.CACHE / "panel_scored.parquet")
PG = {k: (v[0], np.asarray(v[1])) for k, v in json.load(open(fv.CACHE / "p_grid.json")).items()}
mk = P.groupby("slug").agg(cl=("cl", "first"), coin=("coin", "first"), t_end=("t_end", "first"),
                           winner_up=("winner_up", "first")).reset_index()

rows = []
for r in mk.itertuples():
    for tx, ts, wallet, side, oi, price, size, asset in fv.load_market_meta(r.slug)["taker"]:
        s = 1 if side == "BUY" else -1
        rows.append((r.slug, ts, wallet, s if oi == 0 else -s, price if oi == 0 else 1 - price, size, oi))
T = pd.DataFrame(rows, columns=["slug", "ts", "wallet", "agg", "px", "size", "oi"]).merge(mk, on="slug")
T["t_open"] = T.t_end - T.cl.map(fv.LEN_S)
T = T[(T.ts >= T.t_open) & (T.ts < T.t_end)].sort_values(["slug", "ts"]).reset_index(drop=True)
assert T.ts.max() < fv.T_TRAIN_END
T["tau"] = T.t_end - T.ts
print("TRAIN taker fills in window:", T.groupby("cl").size().to_dict(),
      "usd notional:", T.assign(n=T.px * T["size"]).groupby("cl").n.sum().round(0).to_dict())


def p_at(slug, t):
    t0, a = PG[slug]
    i = np.asarray(t) - t0
    return np.where((i >= 0) & (i < len(a)), a[np.clip(i, 0, len(a) - 1)], np.nan)


Pg = {s: g.sort_values("t") for s, g in P.groupby("slug")}


def mid_asof(slug, t, max_age=30):
    g = Pg[slug]
    tt = g.t.values
    j = np.searchsorted(tt, t, side="right") - 1
    ok = (j >= 0) & (t - tt[np.clip(j, 0, None)] <= max_age)
    return np.where(ok, g.mid.values[np.clip(j, 0, None)], np.nan)


def mid_after(slug, t, slack=10):
    g = Pg[slug]
    tt = g.t.values
    j = np.searchsorted(tt, t, side="left")
    jj = np.clip(j, 0, len(tt) - 1)
    ok = (j < len(tt)) & (tt[jj] <= t + slack)
    return np.where(ok, g.mid.values[jj], np.nan)


out = []
for s, g in T.groupby("slug", sort=True):
    g = g.copy()
    g["p_pre"] = p_at(s, g.ts.values - PRE)
    g["mid_pre"] = mid_asof(s, g.ts.values - PRE)
    for h in (5, 30, 60):
        g[f"mid_{h}"] = mid_after(s, g.ts.values + h)
        g[f"p_{h}"] = p_at(s, g.ts.values + h)
    out.append(g)
T = pd.concat(out, ignore_index=True)
T["gap"] = T.p_pre - T.mid_pre
T = T[T.gap.notna() & (T.tau > 5)].copy()
T["agree"] = np.sign(T.gap) == T["agg"]
T["agree_"] = T["agree"]
T["absgap_b"] = pd.cut(T.gap.abs(), [0, 0.01, 0.03, 0.05, 0.1, 1], labels=["<1c", "1-3c", "3-5c", "5-10c", ">10c"])
for h in (5, 30, 60):
    T[f"mk_{h}"] = -T["agg"] * (T[f"mid_{h}"] - T.px)       # maker markout, $/share
T["mk_settle"] = -T["agg"] * (T.winner_up - T.px)
T["fee"] = 0.07 * T.px * (1 - T.px)  # ASSUMED taker fee/share from feeSchedule rate=0.07, exponent=1

pd.set_option("display.width", 250)


def summ(d):
    w = d["size"].values
    r = dict(n=len(d), share_agree_cnt=d.agree_.mean(), share_agree_vol=np.average(d.agree_, weights=w))
    for k in ("mk_5", "mk_30", "mk_60", "mk_settle"):
        x = d[k].values
        ok = np.isfinite(x)
        r[f"{k}_c"] = 100 * x[ok].mean()
        r[f"{k}_vw_c"] = 100 * np.average(x[ok], weights=w[ok])
    return pd.Series(r)


print("\nTRAIN: maker markout (cents/share, + = maker gains) of taker fills, by agreement with gap sign")
tab = T.groupby(["cl", "agree"]).apply(summ)
print(tab.round(3).to_string())
print("\nTRAIN: by |gap| bucket (all lengths): share of fills whose direction agrees with gap; maker markouts")
tab2 = T.groupby(["cl", "absgap_b"], observed=True).apply(summ)
print(tab2.round(3).to_string())
# logistic-ish: P(taker buys Up) vs gap
T["gb"] = pd.cut(T.gap, [-1, -0.1, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.1, 1])
buy = T.groupby(["cl", "gb"], observed=True).apply(lambda d: pd.Series({"n": len(d), "P_buyUp": (d["agg"] > 0).mean(),
                                                                          "vw_P_buyUp": np.average(d["agg"] > 0, weights=d["size"])}))
print("\nTRAIN: P(taker buys Up | gap bucket)")
print(buy.round(3).to_string())

# ---------- simulated maker: quote centre = mid vs model p (vs blend) ----------
sim = []
for cl in CLS + ["all"]:
    d = T if cl == "all" else T[T.cl == cl]
    for hs in (0.005, 0.01, 0.02, 0.03):
        for cname, c in (("mid", d.mid_pre.values), ("model_p", d.p_pre.values), ("avg(mid,p)", 0.5 * (d.mid_pre.values + d.p_pre.values))):
            ask, bid = c + hs, c - hs
            buyf = (d["agg"].values > 0) & (ask <= d.px.values)       # taker buy would have lifted our ask
            sellf = (d["agg"].values < 0) & (bid >= d.px.values)      # taker sell would have hit our bid
            for V, vn in ((d.winner_up.values, "settle"), (d.mid_60.values, "mid60")):
                pnl = np.where(buyf, ask - V, 0.0) + np.where(sellf, V - bid, 0.0)
                fill = buyf | sellf
                ok = np.isfinite(pnl) & fill
                sim.append(dict(cl=cl, half_spread=hs, centre=cname, value=vn, fills=int(ok.sum()),
                                fill_rate=fill.mean(), pnl_per_fill_c=100 * pnl[ok].mean(),
                                total_pnl_per_mkt=pnl[ok].sum() / d.slug.nunique(),
                                se_c=100 * pnl[ok].std() / np.sqrt(max(ok.sum(), 1))))
sim = pd.DataFrame(sim)
print("\nTRAIN: unit-size maker sim (fill if taker price crosses our quote). pnl per fill in cents; value = settlement or mid+60s")
print(sim[sim.value == "settle"].round(3).to_string(index=False))
print(sim[(sim.value == "mid60") & (sim.cl == "all")].round(3).to_string(index=False))
sim.to_csv(fv.CACHE / "q4_maker_sim.csv", index=False)
tab.to_csv(fv.CACHE / "q4_markout_agree.csv")
tab2.to_csv(fv.CACHE / "q4_markout_absgap.csv")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for cl, col in zip(CLS, ("C0", "C1", "C2")):
    b = buy.loc[cl]
    axs[0].plot([iv.mid for iv in b.index], b.P_buyUp, "o-", color=col, label=cl)
    for ag, ls in ((True, "-"), (False, "--")):
        r = tab.loc[(cl, ag)]
        axs[1].plot([5, 30, 60], [r.mk_5_c, r.mk_30_c, r.mk_60_c], ls, marker="o", color=col,
                    label=f"{cl} {'agrees' if ag else 'disagrees'} w/ gap")
axs[0].axhline(0.5, color="k", lw=0.5)
axs[0].axvline(0, color="k", lw=0.5)
axs[0].set_xlabel("gap = model p - mid just before fill")
axs[0].set_ylabel("P(taker buys Up)")
axs[0].set_title("Taker direction vs gap (train)")
axs[0].legend()
axs[1].axhline(0, color="k", lw=0.5)
axs[1].set_xlabel("markout horizon (s)")
axs[1].set_ylabel("maker markout vs mid (cents/share)")
axs[1].set_title("Maker markout, fills split by gap agreement (train)")
axs[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(fv.FIG / "q4_adverse_selection.png", dpi=120)

# market-clustered bootstrap of (model_p-centred minus mid-centred) maker P&L per market
print("\nTRAIN: per-market P&L difference (centre=model_p minus centre=mid), $ per market at unit size, 90% market-bootstrap CI")
rng = np.random.default_rng(1)
for cl in CLS + ["all"]:
    d = T if cl == "all" else T[T.cl == cl]
    for hs in (0.01, 0.02):
        for vn in ("winner_up", "mid_60"):
            per = {}
            for cname, c in (("mid", d.mid_pre.values), ("model_p", d.p_pre.values)):
                ask, bid = c + hs, c - hs
                V = d[vn].values
                pnl = np.where((d["agg"].values > 0) & (ask <= d.px.values), ask - V, 0.0) + \
                    np.where((d["agg"].values < 0) & (bid >= d.px.values), V - bid, 0.0)
                per[cname] = pd.Series(np.nan_to_num(pnl)).groupby(d.slug.values).sum()
            diff = (per["model_p"] - per["mid"]).values
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            print(f"  {cl:4s} h={hs:.2f} value={vn:9s}: mean diff {diff.mean():+.3f} $/mkt  CI90 [{np.percentile(bs,5):+.3f},{np.percentile(bs,95):+.3f}]  n_mkt={len(diff)}")
