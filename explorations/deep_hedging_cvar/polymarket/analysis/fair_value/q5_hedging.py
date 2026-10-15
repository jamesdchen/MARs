"""Q5: hedging economics for +1 Up share (TRAIN split only, Binance 1 s prices of 2026-08-21/22).

(a) delta surface; (b) discrete delta-hedge backtest on (i) the actual train markets (Chainlink
outcome) and (ii) synthetic windows on every aligned 5m/15m/4h slot of Aug 21-22 (Binance-TWAP
outcome, same pricer); (c) cross-market offsets (5m vs 15m vs 4h of the same coin).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fair_value as fv

CFG = json.load(open(fv.CACHE / "model_cfg.json"))
u = fv.load_underlying()
SIG = {c: np.load(fv.CACHE / f"sigma_{c}.npz")[CFG["sigma"]].astype(float) * CFG["mult"] for c in u}
CLS = ["5m", "15m", "4h"]
NU = CFG["nu"]
rng = np.random.default_rng(0)


def grids(coins, t_open, t_end_scalar_len, K):
    """p, delta(coins/share), S matrices [n, T+1] on the 1 s grid t_open..t_end."""
    T = t_end_scalar_len
    n = len(coins)
    p = np.empty((n, T + 1))
    D = np.empty((n, T + 1))
    S = np.empty((n, T + 1))
    sg = np.empty(n)
    off = np.arange(T + 1)
    for c in np.unique(coins):
        idx = np.where(coins == c)[0]
        ts = (t_open[idx, None] + off[None, :]).ravel()
        te = np.repeat(t_open[idx] + T, T + 1)
        kk = np.repeat(K[idx], T + 1)
        pp, dd, ss = fv.model_at(u[c], SIG[c], ts, te, kk, shift=CFG["shift"], nu=NU, with_delta=True)
        p[idx] = pp.reshape(len(idx), T + 1)
        D[idx] = dd.reshape(len(idx), T + 1)
        S[idx] = ss.reshape(len(idx), T + 1)
        sg[idx] = SIG[c][t_open[idx] - 1 - fv.T0]
    return p, D, S, sg


def hedge_sim(p, D, S, Y, e, rule, fee_bps, sig, T):
    """Hold +1 Up from step e to settlement. rule: ('every', k) | ('band', b) | ('none',).
    Returns dict of per-market arrays: pnl (net), gross hedge pnl, fees, n_rebal, max |delta usd|."""
    n, m = p.shape
    hold = np.zeros(n)
    hpnl = np.zeros(n)
    fees = np.zeros(n)
    nreb = np.zeros(n)
    # reference delta for the band: ATM delta at entry
    Dref = fv.digital_up_delta(S[:, e], S[:, e], sig, max(T - e, 1), nu=NU)
    for j in range(e, m):
        tgt = D[:, j] if j < m - 1 else np.zeros(n)
        if rule[0] == "capped":   # ('capped', k, cap_mult, stop_tau): |target| <= cap_mult*Dref, flat when tau<stop_tau
            tgt = np.clip(tgt, -rule[2] * Dref, rule[2] * Dref)
            if T - j < rule[3]:
                tgt = np.zeros(n)
        if rule[0] == "none":
            new = hold
        elif rule[0] in ("every", "capped"):
            new = tgt if ((j - e) % rule[1] == 0 or j == m - 1) else hold
        else:
            trig = np.abs(tgt - hold) > rule[1] * Dref
            new = np.where(trig | (j == m - 1), tgt, hold)
        tr = np.abs(new - hold)
        fees += fee_bps * 1e-4 * tr * S[:, j]
        nreb += tr > 0
        hold = new
        if j < m - 1:
            hpnl += -hold * (S[:, j + 1] - S[:, j])
    unh = Y - p[:, e]
    return dict(unhedged=unh, hedged_gross=unh + hpnl, hedged_net=unh + hpnl - fees, fees=fees, nreb=nreb)


def cvar(x, a=0.05):
    q = np.quantile(x, a)
    return -x[x <= q].mean()


RULES = [("none",), ("every", 1), ("every", 5), ("every", 30), ("band", 0.05), ("band", 0.1), ("band", 0.25),
         ("every", 120), ("capped", 30, 1.0, 60), ("capped", 30, 1.5, 120)]
FEES = {"maker2bp": 2.0, "taker5bp": 5.0}
ENTRY_FRAC = [1.0, 0.5, 0.25, 0.1]   # fraction of life remaining at entry


def run_book(name, coins, t_open, K, Y, T, extra=None):
    p, D, S, sg = grids(coins, t_open, T, K)
    rows = []
    for f in ENTRY_FRAC:
        e = int(round(T * (1 - f)))
        for rule in RULES:
            for fn, fb in FEES.items():
                if rule[0] == "none" and fn != "maker2bp":
                    continue
                r = hedge_sim(p, D, S, Y, e, rule, fb, sg, T)
                x = r["hedged_net"] if rule[0] != "none" else r["unhedged"]
                rows.append(dict(set=name, entry_frac_left=f, rule="/".join(map(str, rule)),
                                 fee="-" if rule[0] == "none" else fn, n=len(x), mean=x.mean(), sd=x.std(),
                                 var=x.var(), cvar95=cvar(x), mean_fee=r["fees"].mean(),
                                 sd_gross=r["hedged_gross"].std(), rebal=r["nreb"].mean()))
    return pd.DataFrame(rows), (p, D, S, sg)


pd.set_option("display.width", 250)
allres = []
# ---------- (i) actual train markets, Chainlink outcome ----------
mk = pd.read_parquet(fv.CACHE / "markets_train.parquet")
for cl in CLS:
    d = mk[mk.cl == cl]
    T = fv.LEN_S[cl]
    t_open = d.t_open.values.astype(np.int64)
    res, _ = run_book(f"actual_{cl}", d.coin.values, t_open, d.k_bin.values, d.winner_up.values.astype(float), T)
    res_b, _ = run_book(f"actual_{cl}_binY", d.coin.values, t_open, d.k_bin.values, d.win_bin.values.astype(float), T)
    allres += [res, res_b]
# ---------- (ii) synthetic aligned windows on Binance ----------
coins_all = np.array(sorted(u))
syn_cache = {}
for cl in CLS:
    T = fv.LEN_S[cl]
    step = T if cl != "4h" else 3600       # 4h: overlapping hourly starts for sample size
    starts = np.arange(fv.T_TRAIN_START, fv.T_TRAIN_END - T + 1, step)
    co = np.repeat(coins_all, len(starts))
    to = np.tile(starts, len(coins_all)).astype(np.int64)
    K = np.array([fv.twap_binance(u[c], t) for c, t in zip(co, to)])
    Y = np.array([float(fv.twap_binance(u[c], t + T) >= k) for c, t, k in zip(co, to, K)])
    res, mats = run_book(f"synthetic_{cl}", co, to, K, Y, T)
    syn_cache[cl] = (co, to, K, Y, mats)
    allres.append(res)
R = pd.concat(allres, ignore_index=True)
R["var_reduction"] = np.nan
for keys, g in R.groupby(["set", "entry_frac_left"]):
    base = g[g.rule == "none"]["var"].values[0]
    R.loc[g.index, "var_reduction"] = 1 - g["var"] / base
R.to_csv(fv.CACHE / "q5_hedge_backtest.csv", index=False)
show = R[R.set.str.startswith("synthetic") | (R.set.str.startswith("actual") & ~R.set.str.endswith("binY"))]
print("TRAIN: hedge backtest, +1 Up share bought at model p, P&L in $/share (net of perp fees)")
print(show[show.entry_frac_left.isin([1.0, 0.25])].round(4).to_string(index=False))
print("\nTRAIN: best net CVaR95 per set/entry (and the unhedged CVaR95)")
for (st_, ef), g in R.groupby(["set", "entry_frac_left"]):
    if st_.endswith("binY"):
        continue
    base = g[g.rule == "none"].iloc[0]
    for fn in FEES:
        h = g[g.fee == fn].sort_values("cvar95").iloc[0]
        print(f"  {st_:14s} entry@{ef:.2f}  {fn}: unhedged sd {base.sd:.3f} CVaR95 {base.cvar95:.3f} | best {h.rule:16s}"
              f" sd {h.sd:.3f} CVaR95 {h.cvar95:.3f} fee {100*h.mean_fee:.1f}c gross-sd {h.sd_gross:.3f}")
print("\nbasis effect (actual markets): sd of net P&L, Chainlink outcome vs Binance-TWAP outcome, rule every/5 maker")
b = R[(R.rule == "every/5") & (R.fee == "maker2bp") & R.set.str.startswith("actual")]
print(b[["set", "entry_frac_left", "sd", "cvar95"]].round(4).to_string(index=False))

# ---------- (a) delta surface ----------
sig_med = np.median(np.concatenate([SIG[c][fv.T_TRAIN_START - fv.T0:] for c in u if c != "hype"]))
print(f"\nmedian train sigma (excl hype) per sqrt(s) {sig_med:.3e} = {sig_med * np.sqrt(365 * 86400):.0%} annualised")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
mny = np.linspace(-60, 60, 241)  # bp
tab = []
for ax, cl in zip(axs, CLS):
    T = fv.LEN_S[cl]
    for fl in (1.0, 0.5, 0.25, 0.1, 0.05, 30 / T, 10 / T):
        tau = fl * T
        S = 1.0
        K = np.exp(-mny / 1e4)
        known = np.where(tau < 60, 1.0, np.nan)
        dusd = fv.digital_up_delta(S, K, sig_med, tau, known_avg=known, nu=NU) * S
        ax.plot(mny, dusd, label=f"tau={tau:.0f}s")
        tab.append(dict(cl=cl, tau_s=tau, atm_delta_usd=float(dusd[120]),
                        delta_usd_at_10bp=float(dusd[mny == 10][0]), delta_usd_at_30bp=float(dusd[mny == 30][0])))
    ax.set_title(f"{cl}: $ delta per Up share (sigma={sig_med * np.sqrt(365 * 86400):.0%}/yr)")
    ax.set_xlabel("moneyness ln(S/K) (bp)")
    ax.legend(fontsize=7)
axs[0].set_ylabel("USD notional of coin per 1 Up share")
plt.tight_layout()
plt.savefig(fv.FIG / "q5_delta_surface.png", dpi=120)
dt = pd.DataFrame(tab)
print("\nDelta in USD notional per Up share (median train sigma):")
print(dt.round(1).to_string(index=False))
dt.to_csv(fv.CACHE / "q5_delta_table.csv", index=False)

# realised delta distribution from synthetic grids
print("\nRealised |delta| USD per share on synthetic train windows: median / p90 / p99 over all seconds, by time left")
for cl in CLS:
    co, to, K, Y, (p, D, S, sg) = syn_cache[cl]
    T = fv.LEN_S[cl]
    du = np.abs(D * S)
    for fl in (1.0, 0.5, 0.1, 30 / T):
        j = int(round(T * (1 - fl)))
        x = du[:, j]
        print(f"  {cl:4s} tau={T - j:6d}s  median ${np.median(x):7.1f}  p90 ${np.quantile(x, .9):7.1f}  p99 ${np.quantile(x, .99):8.1f}")

# variance/CVaR vs fee figure
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, cl in zip(axs, CLS):
    g = R[(R.set == f"synthetic_{cl}") & (R.entry_frac_left == 1.0)]
    for fn, mk_ in (("maker2bp", "o"), ("taker5bp", "s")):
        h = g[g.fee == fn]
        ax.scatter(h.mean_fee * 100, h.cvar95, marker=mk_, label=fn)
        for r in h.itertuples():
            ax.annotate(r.rule, (r.mean_fee * 100, r.cvar95), fontsize=7)
    base = g[g.rule == "none"].iloc[0]
    ax.axhline(base.cvar95, color="k", ls="--", lw=0.8, label="unhedged CVaR95")
    ax.set_xlabel("mean perp fees paid (cents per Up share)")
    ax.set_ylabel("CVaR95 of P&L ($/share)")
    ax.set_title(f"{cl}: hold from open, synthetic train windows (n={len(syn_cache[cl][0])})")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(fv.FIG / "q5_hedge_frontier.png", dpi=120)

# ---------- (c) cross-market offsets ----------
print("\nCross-market (same coin, simultaneous windows; synthetic, model p): corr of 5 s changes in p, and static cross-hedge")
co5, to5, K5, Y5, (p5, D5, S5, _) = syn_cache["5m"]
co15, to15, K15, Y15, (p15, D15, S15, _) = syn_cache["15m"]
co4, to4, K4, Y4, (p4, D4, S4, _) = syn_cache["4h"]
idx15 = {(c, t): i for i, (c, t) in enumerate(zip(co15, to15))}
idx4 = {(c, t): i for i, (c, t) in enumerate(zip(co4, to4))}
rows = []
cc = {"5m~15m": [], "5m~4h": [], "15m~4h": []}
static = []
for i, (c, t) in enumerate(zip(co5, to5)):
    t15 = t - (t - fv.T_TRAIN_START) % 900
    t4 = t - (t - fv.T_TRAIN_START) % 3600  # an hourly-start 4h window containing t (the one started at t4)
    if t4 + 14400 > fv.T_TRAIN_END or (c, t4) not in idx4 or (c, t15) not in idx15:
        continue
    j15, j4 = idx15[(c, t15)], idx4[(c, t4)]
    a5 = p5[i, :300:5]
    a15 = p15[j15, t - t15: t - t15 + 300: 5]
    a4 = p4[j4, t - t4: t - t4 + 300: 5]
    d5, d15, d4 = np.diff(a5), np.diff(a15), np.diff(a4)
    cc["5m~15m"].append((d5, d15))
    cc["5m~4h"].append((d5, d4))
    cc["15m~4h"].append((d15, d4))
    # static cross hedge over the 5m life: +1 5m Up, short h shares of the 15m / 4h Up, h = delta ratio at 5m open
    for other, jj, off, pp, DD in (("15m", j15, t - t15, p15, D15), ("4h", j4, t - t4, p4, D4)):
        h = D5[i, 0] / max(DD[jj, off], 1e-12)
        pnl5 = Y5[i] - p5[i, 0]
        pnl_o = pp[jj, off + 300] - pp[jj, off]   # mark the other leg at model p at 5m expiry
        static.append(dict(other=other, h=h, unhedged=pnl5, hedged=pnl5 - h * pnl_o, coin=c))
for k, v in cc.items():
    x = np.concatenate([a for a, b in v])
    y = np.concatenate([b for a, b in v])
    ok = np.isfinite(x) & np.isfinite(y)
    print(f"  corr 5s dp {k}: {np.corrcoef(x[ok], y[ok])[0, 1]:.3f}  (n={ok.sum()})")
st = pd.DataFrame(static)
for o, g in st.groupby("other"):
    print(f"  static hedge of 5m Up with {o} Up: median ratio h={g.h.median():.2f} shares (p90 {g.h.quantile(.9):.1f});"
          f" sd unhedged {g.unhedged.std():.3f} -> hedged {g.hedged.std():.3f};"
          f" CVaR95 {cvar(g.unhedged.values):.3f} -> {cvar(g.hedged.values):.3f}")
st.to_csv(fv.CACHE / "q5_cross_static.csv", index=False)

# empirical: actual overlapping mids in the sample (same coin, simultaneous snapshots)
Pn = pd.read_parquet(fv.CACHE / "panel_scored.parquet")
Pn = Pn[Pn.two_sided & (Pn.tau > 30)]
piv = {}
for cl in CLS:
    d = Pn[Pn.cl == cl]
    piv[cl] = d.groupby(["coin", "t"]).mid.mean()
for a, b_ in (("5m", "15m"), ("5m", "4h"), ("15m", "4h")):
    j = pd.concat([piv[a].rename("a"), piv[b_].rename("b")], axis=1).dropna().reset_index().sort_values(["coin", "t"])
    j["da"] = j.groupby("coin").a.diff()
    j["db"] = j.groupby("coin").b.diff()
    j["dt"] = j.groupby("coin").t.diff()
    jj = j[(j.dt > 0) & (j.dt <= 10)]
    print(f"  empirical Polymarket mid: corr(d mid {a}, d mid {b_}) = {jj[['da', 'db']].corr().iloc[0, 1]:.3f}"
          f" over n={len(jj)} common snapshot intervals (coins {jj.coin.nunique()})")
