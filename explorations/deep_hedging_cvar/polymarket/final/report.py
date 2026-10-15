#!/usr/bin/env python3
"""results/<split>/ -> RESULTS.md + figures (EVALUATION.md sections 3, 4 and 5.5).

    python final/report.py                    # val dry run (split 1, the default)
    python final/report.py --split 2          # the test report -> final/RESULTS.md

Default outputs: split 2 -> final/RESULTS.md and final/figures/; split 1 ->
final/RESULTS_dryrun_val.md and final/figures/dryrun_val/ (a dry run can never overwrite the
test report). Everything in the plan is reported, including failed endpoints and every variant:
headline table (central fills, 1 s), primary endpoint and decision rule, paired RL-minus-baseline
CIs (with and without rewards+rebates), attribution and markouts, exploit detectors, directional
verdict and mirror test, concentration, fill/latency grid and size sweep, contract-length and coin
breakdowns, seed spread, trial count and deflated Sharpe, regime comparison (T9), run integrity.

All CIs: stats.paired_difference_ci / stats.block_bootstrap on the zero-filled hourly MTM grid
summed over every market, stationary bootstrap, Politis-White block length floored at 2 h,
plan["bootstrap"]["n_boot"] replicates (10,000) with the plan's seed. RL = the selected config's
seed mean (seeds are never selected individually).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CODE_ROOT, FINAL, LENGTH_NAMES, MARKOUT_HORIZONS, SPLIT_NAMES, S, Dataset,  # noqa: E402
                    read_jsonl, resolve, utcnow)
from reduce import (ALL_FLAGS, FLAG_DOCS, RULE_FLAGS, directional_verdict, exploit_tables,  # noqa: E402
                    market_making)

COMPONENTS = ["spread_capture", "adverse_selection", "inventory_carry", "resolution", "fees", "rebates",
              "rewards", "total"]
GROUP_LABEL = {"rl": "RL (seed mean)", "rl5": "RL 5.0 (seed mean)"}
# reference palette (dataviz skill): categorical slots 1-3, ink, surface, diverging blue<->red
C = dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", ink="#0b0b0b", ink2="#52514e", muted="#898781",
         grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb", other="#b8b6ae", pos="#2a78d6", neg="#e34948",
         mid="#f0efec")


# ============================================================================ data
class Results:
    def __init__(self, results_dir: Path, split: int, plan_path: Path | None):
        self.split = split
        self.dir = results_dir / SPLIT_NAMES[split]
        if not self.dir.exists():
            raise SystemExit(f"no results for split {split} in {self.dir}: run run_test.py --split {split}")
        rep = "test_report.json" if split == 2 else "dryrun_report.json"
        self.report = json.loads((self.dir / rep).read_text())
        self.mk = pd.read_parquet(self.dir / "markets.parquet")
        self.mins = pd.read_parquet(self.dir / "minutes.parquet")
        self.fills = pd.read_parquet(self.dir / "fills.parquet")
        self.port = pd.read_parquet(self.dir / "portfolio.parquet")
        self.aggs = {(a["method"], a["variant"]): a for a in json.loads((self.dir / "runs.json").read_text())["runs"]}
        self.plan_path = Path(plan_path or self.report["plan_path"])
        self.plan = json.loads(self.plan_path.read_text())
        self.plan_hash_now = S.sha256_file(self.plan_path)
        self.t0, self.t1 = self.report["t_range"]
        self.methods = [m for m in self.plan["methods"] if m["id"] in self.report["methods"]]
        self.variants = {v["id"]: v for v in self.plan["variants"] if v["id"] in self.report["variants"]}
        self.head = self.plan["headline_variant"]
        self.best = self.plan["best_baseline"]["id"]
        self.groups = {}
        for m in self.methods:
            self.groups.setdefault(m["group"], []).append(m["id"])
        self.boot = dict(self.plan["bootstrap"])
        self._cache = {}

    # --- method keys: a real method id, or "rl/mean" / "rl5/mean"
    def members(self, key):
        if key.endswith("/mean"):
            return self.groups.get(key.split("/")[0], [])
        return [key]

    def has(self, key, vid):
        mem = self.members(key)
        return bool(mem) and vid in self.variants and all((m, vid) in self.aggs for m in mem)

    def label(self, key):
        if key.endswith("/mean"):
            return GROUP_LABEL.get(key.split("/")[0], key)
        if key.startswith("baseline_c/"):
            return key[len("baseline_c/"):] + " (active)"
        return key.replace("baseline/", "")

    def hourly(self, key, vid, col="dpnl", bin_s=3600):
        ck = ("h", key, vid, col, bin_s)
        if ck in self._cache:
            return self._cache[ck]
        nb = int(math.ceil((self.t1 - self.t0) / bin_s))
        arrs = []
        for m in self.members(key):
            d = self.mins[(self.mins.method == m) & (self.mins.variant == vid)]
            idx = ((d.minute_ts.to_numpy() - self.t0) // bin_s).astype(np.int64)
            arrs.append(np.bincount(idx, weights=d[col].to_numpy(float), minlength=nb)[:nb])
        out = np.mean(arrs, axis=0) if arrs else np.zeros(nb)
        self._cache[ck] = out
        return out

    def per_market(self, key, vid, col="pnl"):
        d = self.mk[(self.mk.method.isin(self.members(key))) & (self.mk.variant == vid)]
        return d.groupby("market")[col].mean()

    def mk_rows(self, key, vid):
        return self.mk[(self.mk.method.isin(self.members(key))) & (self.mk.variant == vid)]

    def total(self, key, vid, col="pnl"):
        return float(self.per_market(key, vid, col).sum())

    def agg_mean(self, key, vid, path):
        vals = []
        for m in self.members(key):
            a = self.aggs[(m, vid)]
            for p in path:
                a = a[p]
            vals.append(a)
        return vals

    def ci(self, a, b=None, statistic="total"):
        kw = dict(n_boot=int(self.boot["n_boot"]), seed=int(self.boot["seed"]),
                  alpha=float(self.boot["alpha"]), kind=self.boot["kind"], min_block_len=2.0)
        if b is None:
            return S.block_bootstrap(a, "sum" if statistic == "total" else "mean", **kw)
        return S.paired_difference_ci(a, b, statistic=statistic, **kw)

    def sharpe_ci(self, x):
        if np.std(x) == 0:
            return dict(estimate=float("nan"), lo=float("nan"), hi=float("nan"))
        return S.block_bootstrap(x, lambda z: S.sharpe(z) if np.std(z) > 0 else 0.0,
                                 n_boot=int(self.boot["n_boot"]), seed=int(self.boot["seed"]),
                                 alpha=float(self.boot["alpha"]), kind=self.boot["kind"], min_block_len=2.0)


# ============================================================================ formatting
def f2(x, signed=True, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+,.{nd}f}" if signed else f"{x:,.{nd}f}"


def pct(x, nd=0):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


def cis(r, nd=2):
    return f"[{f2(r['lo'], nd=nd)}, {f2(r['hi'], nd=nd)}]"


def table(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def verdict(ok):
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


# ============================================================================ sections
def method_exploit(R, key, vid):
    """T5 tables (reduce.exploit_tables) of a method, or of a group's seeds pooled (= seed mean)."""
    if not key.endswith("/mean"):
        return R.aggs[(key, vid)]["exploit"]
    f = R.fills[(R.fills.method.isin(R.members(key))) & (R.fills.variant == vid)]
    return exploit_tables(f)


def method_mm(R, key, vid):
    """DECISIONS.md D1.3 classification; for a group, on the seed-mean quantities."""
    if not key.endswith("/mean"):
        return R.aggs[(key, vid)]["market_making"]
    mk = R.mk_rows(key, vid)
    g = mk.groupby("market").agg(n_fills=("n_fills", "mean"), uptime_two=("uptime_two", "mean"),
                                 uptime_any=("uptime_any", "mean"), contract_len_s=("contract_len_s", "first"))
    return market_making(g.reset_index(), group_attr(R, key, vid), method_exploit(R, key, vid)["verdict"])


def group_attr(R, key, vid):
    tots = R.agg_mean(key, vid, ["attribution", "totals"])
    keys = set().union(*[t.keys() for t in tots])
    return {k: float(np.mean([t.get(k, 0.0) for t in tots])) for k in keys}


def headline_rows(R, keys, vid):
    rows = []
    n_days = max(1, round((R.t1 - R.t0) / 86400))
    for key in keys:
        if not R.has(key, vid):
            continue
        pm = R.per_market(key, vid)
        h = R.hourly(key, vid)
        sh = R.sharpe_ci(h)
        mk = R.mk_rows(key, vid)
        nmem = len(R.members(key))
        port = R.port[(R.port.method.isin(R.members(key))) & (R.port.variant == vid)]
        maxdelta = float(port.groupby("method").max_abs_delta_usd.max().mean()) if len(port) else float("nan")
        qm = mk.quoted_minutes.sum()
        w = mk.contract_len_s.to_numpy(float)                 # live seconds per market
        up2 = float((mk.uptime_two * w).sum() / w.sum()) if w.sum() else float("nan")
        up1 = float((mk.uptime_one * w).sum() / w.sum()) if w.sum() else float("nan")
        rows.append([
            f"**{R.label(key)}**" if key.endswith("/mean") else R.label(key),
            f2(pm.sum()), f2(pm.sum() / n_days), f2(pm.mean(), nd=3), f2(pm.median(), nd=3),
            f2(R.total(key, vid, "pnl_ex_rr")),
            f"{f2(sh['estimate'], nd=2)} {cis(sh)}" if np.isfinite(sh["estimate"]) else "n/a",
            f2(S.cvar(pm.to_numpy(), 0.95), nd=3),
            f"{mk.n_fills.sum() / nmem:,.0f}", f2(mk.n_fills.sum() / qm if qm else float('nan'), False),
            pct(mk.volume.sum() / mk.taker_volume.sum() if mk.taker_volume.sum() > 0 else float("nan"), 2),
            pct(up2, 1), pct(up1, 1),
            f2(float(mk.groupby("method").max_abs_inv.max().mean()), False, 1), f2(maxdelta, False, 0)])
    return rows


def decision(R):
    """Primary endpoint and every success criterion of the plan."""
    rl, best, head = "rl/mean", R.best, R.head
    out = dict(criteria={}, primary=None)
    if not (R.has(rl, head) and R.has(best, head)):
        out["missing"] = True
        return out
    a, b = R.hourly(rl, head), R.hourly(best, head)
    prim = R.ci(a, b, "total")
    out["primary"] = prim
    crit = out["criteria"]
    crit["ci_lower_gt_0"] = (prim["lo"] > 0, f"CI lower bound {f2(prim['lo'])}")
    for cid, vid in (("pessimistic_1s_gt_0", "pessimistic_lat1_x1"), ("central_2s_gt_0", "central_lat2_x1")):
        if R.has(rl, vid) and R.has(best, vid):
            d = float(R.hourly(rl, vid).sum() - R.hourly(best, vid).sum())
            crit[cid] = (d > 0, f"point estimate {f2(d)}")
        else:
            crit[cid] = (None, f"variant {vid} not run")
    dv = directional_verdict(group_attr(R, rl, head))
    crit["not_directional"] = (not dv["directional"], f"inventory share of total {pct(dv['share_of_total'])}, "
                                                      f"of positive components {pct(dv['share_of_positive'])}")
    ev = method_exploit(R, rl, head)["verdict"]
    crit["no_exploit_flag"] = (not ev["invalid"], f"worst artifact flag {ev['worst_single_flag']} "
                                                  f"{pct(ev['worst_single_share'])} of the fill edge; edge excl. "
                                                  f"flagged {f2(ev['edge_excluding_flagged'])} (D2 flags)")
    mm = method_mm(R, rl, head)
    crit["counts_as_market_making"] = (mm["counts_as_market_making"],
                                       f"{mm['description']}: {mm['fills_per_market']:.1f} fills/market, "
                                       f"two-sided uptime {pct(mm['uptime_two_sided'])} (D1.3)")
    cm = S.concentration(R.per_market(rl, head).to_numpy(), ks=(1, 3, 5, 10))
    chh = S.concentration(a, ks=(1, 3, 5, 10))
    crit["ex_top_positive"] = (cm["ex_top5"] > 0 and chh["ex_top3"] > 0,
                               f"ex-top-5 markets {f2(cm['ex_top5'])}, ex-top-3 hours {f2(chh['ex_top3'])}")
    vals = [v[0] for v in crit.values()]
    out["success"] = None if any(v is None for v in vals) else all(vals)
    return out


def regime_stats(ds: Dataset, splits):
    """T9: per split realised vol, trades per market-hour, two-sided spread, one-sided share."""
    c = ds.ci
    rows = []
    for s in splits:
        idx = ds.split_markets(s)
        if not idx.size:
            continue
        vols, trades, hours, spr, one, n = [], 0.0, 0.0, [], 0.0, 0
        for m in idx:
            t = ds.ticks[ds.market_rows(int(m))]
            r = t[:, c["ul_logret_1s"]].astype(np.float64)
            vols.append(np.sum(r ** 2))
            n += r.size
            trades += float(t[:, c["trade_count"]].sum())
            hours += t.shape[0] / 3600
            bid, ask = t[:, c["bid_px_0"]], t[:, c["ask_px_0"]]
            two = (t[:, c["book_valid"]] > 0.5) & (bid > 0) & (ask < 1)
            spr.append((ask - bid)[two].astype(np.float64))
            one += float((~two).sum())
        sp = np.concatenate(spr) if spr else np.zeros(0)
        rows.append([SPLIT_NAMES[s], int(idx.size), f"{1e4 * math.sqrt(np.sum(vols) / n * 3600):.1f}",
                     f"{trades / hours:,.0f}", f"{100 * sp.mean():.2f}" if sp.size else "n/a", pct(one / n, 1)])
    return rows


# ============================================================================ figures
def _style(ax):
    ax.set_facecolor(C["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["axis"])
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=C["muted"], labelsize=8, length=0)
    ax.grid(True, color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def figures(R, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": C["ink"],
                         "axes.labelcolor": C["ink2"], "figure.facecolor": C["surface"]})
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    head = R.head
    hours = (np.arange(int(math.ceil((R.t1 - R.t0) / 3600))) + 1)
    # 1 cumulative P&L by method (> 4 series: legend; comparator and RL emphasised, others grey)
    nb = int(math.ceil((R.t1 - R.t0) / 3600))
    hours = np.arange(nb + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    _style(ax)
    cum = lambda key: np.r_[0.0, np.cumsum(R.hourly(key, head))]  # noqa: E731
    others = [m["id"] for m in R.methods if m["group"] == "baseline" and m["id"] != R.best]
    greys = ["#b8b6ae", "#898781", "#52514e", "#d4d2ca", "#6f6d68"]
    for i, key in enumerate(k for k in others if R.has(k, head)):
        ax.plot(hours, cum(key), color=greys[i % len(greys)], lw=1.2, label=R.label(key), zorder=2)
    if R.has(R.best, head):
        y = cum(R.best)
        ax.plot(hours, y, color=C["s2"], lw=2, label=f"{R.label(R.best)} (comparator)", zorder=3)
    for g, col in (("rl", C["s1"]), ("rl5", C["s3"])):
        key = f"{g}/mean"
        if not R.has(key, head):
            continue
        seeds = np.array([np.r_[0.0, np.cumsum(R.hourly(m, head))] for m in R.members(key)])
        ax.fill_between(hours, seeds.min(0), seeds.max(0), color=col, alpha=0.13, lw=0,
                        label=f"{R.label(key)}: seed min-max")
        y = cum(key)
        ax.plot(hours, y, color=col, lw=2, label=R.label(key), zorder=4)
    ax.axhline(0, color=C["axis"], lw=0.8)
    ax.set_xlim(0, hours[-1])
    ax.set_xticks(hours[:: max(1, len(hours) // 12)])
    ax.set_xlabel("hours since 00:00 UTC of the evaluation day")
    ax.set_ylabel("cumulative MTM P&L (USDC)")
    ax.set_title(f"Cumulative P&L, central fills, 1 s latency ({SPLIT_NAMES[R.split]})", loc="left",
                 fontsize=10, color=C["ink"])
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7.5)
    for t in leg.get_texts():
        t.set_color(C["ink2"])
    fig.subplots_adjust(left=0.09, right=0.68, bottom=0.12)
    p = fig_dir / "cumulative_pnl.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out["cumulative_pnl"] = p

    # 2 attribution bars: small multiples, one panel per method, blue gain / red loss
    keys = [k for k in ["rl/mean", "rl5/mean"] + [m["id"] for m in R.methods if m["group"] == "baseline"
                                                  and m["id"] != "baseline/no_quote"] if R.has(k, head)]
    comps = COMPONENTS[:-1]
    if keys:
        fig, axes = plt.subplots(1, len(keys), figsize=(2.3 * len(keys) + 1.4, 3.4), sharey=True, sharex=True)
        axes = np.atleast_1d(axes)
        vals = {k: (group_attr(R, k, head) if k.endswith("/mean") else R.aggs[(k, head)]["attribution"]["totals"])
                for k in keys}
        lim = max(1e-9, max(abs(vals[k].get(c, 0.0)) for k in keys for c in comps + ["total"]))
        for ax, k in zip(axes, keys):
            _style(ax)
            v = [vals[k].get(c, 0.0) for c in comps]
            yy = np.arange(len(comps))[::-1]
            ax.barh(yy, v, height=0.62, color=[C["pos"] if x >= 0 else C["neg"] for x in v],
                    edgecolor=C["surface"], linewidth=2)
            ax.axvline(0, color=C["axis"], lw=0.8)
            ax.axvline(vals[k]["total"], color=C["ink"], lw=1.2)
            ax.set_title(f"{R.label(k)}\ntotal {f2(vals[k]['total'])}", fontsize=8.5, loc="left", color=C["ink"])
            ax.set_xlim(-1.1 * lim, 1.1 * lim)
        axes[0].set_yticks(np.arange(len(comps))[::-1])
        axes[0].set_yticklabels([c.replace("_", " ") for c in comps], color=C["ink2"])
        fig.supxlabel("USDC (blue gain, red loss; black line = total)", fontsize=8, color=C["ink2"])
        fig.tight_layout()
        p = fig_dir / "attribution.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out["attribution"] = p

    # 3 sensitivity heatmaps: RL - best baseline, and RL alone, over fill model x latency
    fms, lats = ["optimistic", "central", "pessimistic"], [0, 1, 2]
    grids = {}
    for name, fn in (("RL (seed mean) - best baseline", lambda vid: R.total("rl/mean", vid) - R.total(R.best, vid)),
                     ("RL (seed mean)", lambda vid: R.total("rl/mean", vid))):
        g = np.full((3, 3), np.nan)
        for i, fm in enumerate(fms):
            for j, lat in enumerate(lats):
                vid = f"{fm}_lat{lat}_x1"
                if R.has("rl/mean", vid) and R.has(R.best, vid):
                    g[i, j] = fn(vid)
        grids[name] = g
    if any(np.isfinite(g).any() for g in grids.values()):
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("div", [C["neg"], C["mid"], C["pos"]])
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
        for ax, (name, g) in zip(axes, grids.items()):
            lim = np.nanmax(np.abs(g)) if np.isfinite(g).any() else 1.0
            lim = lim if lim > 0 else 1.0
            im = ax.imshow(g, cmap=cmap, vmin=-lim, vmax=lim)
            for i in range(3):
                for j in range(3):
                    tag = "\nheadline" if (i, j) == (1, 1) else ("\n(never headline)" if j == 0 else "")
                    if np.isfinite(g[i, j]):
                        ink = C["surface"] if abs(g[i, j]) > 0.6 * lim else C["ink"]
                        ax.text(j, i, f2(g[i, j], nd=1) + tag, ha="center", va="center", fontsize=8, color=ink)
                    else:
                        ax.text(j, i, "not run", ha="center", va="center", fontsize=7.5, color=C["muted"])
            ax.set_xticks(range(3), [f"{x} s" for x in lats])
            ax.set_yticks(range(3), fms)
            ax.tick_params(length=0, colors=C["ink2"], labelsize=8)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel("quote/cancel latency")
            ax.set_title(f"{name}, total P&L (USDC)", fontsize=9, loc="left", color=C["ink"])
            cb = fig.colorbar(im, ax=ax, shrink=0.8)
            cb.ax.tick_params(labelsize=7, colors=C["muted"], length=0)
            cb.outline.set_visible(False)
        fig.tight_layout()
        p = fig_dir / "sensitivity_heatmap.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out["sensitivity_heatmap"] = p
    return out


# ============================================================================ document
def build(R: Results, fig_dir: Path, out_path: Path, ds: Dataset | None) -> str:
    P = R.plan
    head = R.head
    figs = getattr(R, "_figs", {})
    L = []
    dry = R.split != 2
    rep = R.report
    title = "Final test results" if not dry else "DRY RUN on the validation split (not a test result)"
    L.append(f"# Polymarket market making: {title}\n")
    banner = []
    if dry:
        banner.append("**DRY RUN** of the final pipeline on split 1 (val). These numbers are NOT test results "
                      "and were produced after the plan was frozen; they are not used for selection.")
    if not rep["backend"]["of_record"]:
        banner.append(f"**Stand-in env** (`{rep['backend']['name']}`): not the C env of record; no number here "
                      "is a number of record.")
    if rep["dataset"]["synthetic"]:
        banner.append("**Synthetic dataset** (unit-test data).")
    if P["groups"]["rl"].get("standin"):
        banner.append("**Stand-in RL methods** (callable stand-ins, not trained checkpoints).")
    if not rep.get("complete_plan", True):
        banner.append("**Partial run**: a subset of methods/variants/markets (dry-run options).")
    rp = rep.get("replay") or {}
    if rp.get("failures"):
        banner.append(f"**Replay mismatch** in {len(rp['failures'])} runs: per-fill detail (attribution "
                      "split, exploit shares) of those markets is not verified against the C core; see section 13.")
    for b in banner:
        L.append(f"> {b}\n")
    L.append(f"Generated {utcnow()} by `final/report.py` from `{R.dir}`; plan sha256 `{rep['plan_sha256']}`"
             f"{'' if R.plan_hash_now == rep['plan_sha256'] else ' (**the plan file on disk now hashes to ' + R.plan_hash_now + '**)'}"
             f"; run `{rep['run_id']}`; {rep['n_markets']} markets; bootstrap {P['bootstrap']['kind']}, "
             f"{R.boot['n_boot']} replicates, block length Politis-White floored at 2 h.\n")

    # ---- primary endpoint
    D = decision(R)
    L.append("## 1. Primary endpoint and decision rule\n")
    pe = P["primary_endpoint"]
    L.append(f"{pe['description']}. Comparator (fixed in the plan): **{R.label(R.best)}** "
             f"({P['best_baseline']['rule']}).\n")
    am = (P.get("protocol") or {}).get("amendments") or {}
    if am.get("applied"):
        L.append(f"Protocol amendments applied ({am['path']}, sha256 `{(am.get('sha256') or '')[:16]}`): "
                 + "; ".join(am["applied"]) + ".\n")
    if D.get("missing"):
        L.append("_Not evaluable: the RL group or the comparator was not run at the headline variant._\n")
    else:
        p = D["primary"]
        L.append(table(["statistic", "estimate", "95% CI", "block length (h)", "hours", "corr(RL, baseline)"],
                       [["RL seed mean - best baseline, total P&L", f2(p["estimate"]), cis(p),
                         f"{p['block_len']:.1f}", p["n"], f2(p["corr_ab"], False)]]) + "\n")
        rows = []
        texts = {c["id"]: c["text"] for c in P["decision_rule"]["success_requires_all"]}
        for cid, (ok, detail) in D["criteria"].items():
            rows.append([texts.get(cid, cid), verdict(ok), detail])
        L.append(table(["criterion (all required)", "verdict", "detail"], rows, ["l", "l", "l"]) + "\n")
        s = D["success"]
        L.append(f"**Overall: {'SUCCESS' if s else ('NOT EVALUABLE' if s is None else 'FAIL')}** "
                 "(the decision rule of EVALUATION.md section 4).\n")

    # ---- headline table
    L.append("## 2. Headline: central fills, 1 s latency, size x1\n")
    keys = [m["id"] for m in R.methods if m["group"] in ("baseline", "baseline_c")] + \
        [f"{g}/mean" for g in ("rl", "rl5") if g in R.groups]
    L.append(table(["method", "P&L total", "P&L/day", "mean/market", "median/market", "P&L ex rewards+rebates",
                    "hourly Sharpe [95% CI]", "CVaR95/market", "fills", "fills per quoted min",
                    "share of taker vol", "uptime 2-sided", "uptime 1-sided", "max |inv|", "max portfolio delta $"],
                   headline_rows(R, keys, head)) + "\n")
    L.append("Sharpe is per hour (not annualised) on the zero-filled hourly MTM grid summed over all markets. "
             "CVaR95 is the mean of the worst 5% of per-market P&L. Uptime is over live seconds. For the RL "
             "rows every number is the seed mean (max |inv| and portfolio delta: mean over seeds of the max).\n")

    # ---- paired CIs
    L.append("## 3. Paired differences vs every baseline (hourly blocks, identical markets)\n")
    rows = []
    for g in ("rl", "rl5"):
        key = f"{g}/mean"
        for b in [m["id"] for m in R.methods if m["group"] in ("baseline", "baseline_c")]:
            if not (R.has(key, head) and R.has(b, head)):
                continue
            r1 = R.ci(R.hourly(key, head), R.hourly(b, head))
            r2 = R.ci(R.hourly(key, head, "dpnl_ex_rr"), R.hourly(b, head, "dpnl_ex_rr"))
            rows.append([R.label(key), R.label(b) + (" (comparator)" if b == R.best else
                                                     (" (secondary, D1.2)" if b.startswith("baseline_c/") else "")),
                         f2(r1["estimate"]), cis(r1), f2(r2["estimate"]), cis(r2), f2(r1["corr_ab"], False)])
    best_test = None
    tuned = [m["id"] for m in R.methods if m["group"] == "baseline" and m.get("tuned")]
    if tuned and all(R.has(b, head) for b in tuned):
        best_test = max(tuned, key=lambda b: R.total(b, head))
    L.append(table(["RL", "baseline", "diff (with rewards+rebates)", "95% CI", "diff (floor: without)",
                    "95% CI", "corr"], rows) + "\n")
    if best_test:
        L.append(f"Post hoc (secondary, conservative): the best tuned baseline on this split is "
                 f"**{R.label(best_test)}** ({f2(R.total(best_test, head))}); the pre-registered comparator is "
                 f"{R.label(R.best)}.\n")

    # ---- attribution
    L.append("## 4. Attribution (T6) and markouts\n")
    rows, mrows = [], []
    akeys = [k for k in ["rl/mean", "rl5/mean"] + [m["id"] for m in R.methods if m["group"] in ("baseline", "baseline_c")]
             if R.has(k, head)]
    for k in akeys:
        if k.endswith("/mean"):
            t = group_attr(R, k, head)
            idr = max(R.agg_mean(k, head, ["attribution", "identity_error"]))
            cc = max(R.agg_mean(k, head, ["attribution", "cash_check_max_abs"]))
            mo = R.agg_mean(k, head, ["attribution", "markouts"])
            mo = {h: float(np.mean([x[str(h)] for x in mo])) for h in MARKOUT_HORIZONS}
        else:
            a = R.aggs[(k, head)]["attribution"]
            t, idr, cc = a["totals"], a["identity_error"], a["cash_check_max_abs"]
            mo = {h: a["markouts"][str(h)] for h in MARKOUT_HORIZONS}
        rows.append([R.label(k)] + [f2(t.get(c, 0.0)) for c in COMPONENTS] + [f"{idr:.1e}", f"{cc:.1e}"])
        mrows.append([R.label(k)] + [f2(mo[h]) for h in MARKOUT_HORIZONS])
    L.append(table(["method"] + [c.replace("_", " ") for c in COMPONENTS] + ["identity err", "vs cash"], rows) + "\n")
    L.append("spread capture = sq (mid at fill - price); adverse selection = sq (mid +60 s - mid at fill); "
             f"inventory carry = sq (mid at {P['attribution']['mid_close_seconds_to_close']} s before close - "
             "mid +60 s); resolution = sq (settlement - that mid); fees < 0; rebates and liquidity rewards. "
             "Components sum to the total; 'identity err' is stats.attribute_pnl's check and 'vs cash' the "
             "max per-market difference from the env's own cash accounting.\n")
    L.append("Markouts (sum of sq x (mid at +h - mid at fill), USDC):\n")
    L.append(table(["method"] + [f"+{h} s" for h in MARKOUT_HORIZONS], mrows) + "\n")
    if "attribution" in figs:
        L.append(f"![attribution]({_relfig(figs['attribution'], out_path)})\n")

    # ---- exploit detectors
    L.append("## 5. Exploit detectors (T5, DECISIONS.md D2) and market-making status (D1.3)\n")
    rows, rows2, rows3 = [], [], []
    for k in akeys:
        ex = method_exploit(R, k, head)
        rf, ev, af, vs = ex["rule_flags"], ex["verdict"], ex["flags"], ex["volume_share"]
        rows.append([R.label(k)] + [f"{pct(rf[f]['share_of_total'])} ({rf[f]['n']})" for f in RULE_FLAGS]
                    + [f2(ev["edge_total"]), f2(ev["edge_excluding_flagged"]),
                       "**INVALID**" if ev["invalid"] else "valid"])
        rows2.append([R.label(k)] + [f"{pct(af[f]['share_of_total'])} / {pct(vs[f])}" for f in ALL_FLAGS])
        mm = method_mm(R, k, head)
        rows3.append([R.label(k), "yes" if mm["counts_as_market_making"] else "**no**", mm["description"],
                      f"{mm['fills_per_market']:.1f}", pct(mm["uptime_two_sided"]), pct(mm["uptime_any"]),
                      pct(mm["frac_markets_without_fills"]),
                      ", ".join(c for c, ok in mm["criteria"].items() if not ok) or "-"])
    L.append(table(["method"] + RULE_FLAGS + ["fill edge", "excl. flagged", "verdict"], rows) + "\n")
    L.append("Artifact flags of D2 (the T5 rule): share of the total fill edge (per fill: spread capture + "
             "60 s markout, the env's pnl_X) from fills under the flag (number of fills); n/a when the total "
             "edge is <= 0. Invalid if one of them carries > 25%, or the edge excluding all of them is <= 0 "
             "while the total is > 0. RL rows pool the seeds' fills.\n")
    L.append("Every flag, edge share / volume share (the through-the-touch `outside` flag is reported, not a "
             "detector, D2):\n")
    L.append(table(["method"] + ALL_FLAGS, rows2) + "\n")
    L.append("Flags (judged on the book at the match second, as the env does): "
             + "; ".join(f"`{k}` {v}" for k, v in FLAG_DOCS.items()) + ".\n")
    L.append("D1.3: a policy counts as making markets only with >= 5 fills per market, T6 directional share "
             "<= 50% and T5 artifact share <= 25%; otherwise this table says what it does instead.\n")
    L.append(table(["method", "makes markets", "what it does", "fills/market", "uptime 2-sided", "uptime any",
                    "markets without fills", "failed"], rows3) + "\n")
    tt = R.fills[R.fills.variant == head]
    if len(tt):
        L.append("Fills by time to close (all methods, headline; a policy that only trades in the last seconds "
                 "is suspect):\n")
        bins = [-1, 10, 30, 60, 300, 900, 1e9]
        labs = ["<=10 s", "10-30 s", "30-60 s", "1-5 min", "5-15 min", ">15 min"]
        rows = []
        for k in akeys:
            f = tt[tt.method.isin(R.members(k))]
            c = pd.cut(f.seconds_to_close, bins, labels=labs).value_counts(normalize=True).reindex(labs).fillna(0)
            rows.append([R.label(k)] + [pct(x, 1) for x in c])
        L.append(table(["method"] + labs, rows) + "\n")

    # ---- directional
    L.append("## 6. Directional gambling (T6) and the mirror test\n")
    rows = []
    mir = [v for v in R.variants if R.variants[v].get("mirror")]
    for k in akeys:
        if k.endswith("/mean"):
            dv = directional_verdict(group_attr(R, k, head))
            up = float(np.mean(R.agg_mean(k, head, ["directional", "pnl_up_won"])))
            dn = float(np.mean(R.agg_mean(k, head, ["directional", "pnl_down_won"])))
            corr = float(np.nanmean([x if x is not None else np.nan for x in
                                     R.agg_mean(k, head, ["directional", "corr_inventory_next60s_return"])]))
        else:
            d = R.aggs[(k, head)]["directional"]
            dv, up, dn = d, d["pnl_up_won"], d["pnl_down_won"]
            corr = d["corr_inventory_next60s_return"]
            corr = float("nan") if corr is None else corr
        mtxt = "not run"
        if mir and R.has(k, mir[0]):
            r = R.ci(R.hourly(k, mir[0]), R.hourly(k, head))
            tol = 1e-9 * (1.0 + float(np.abs(R.hourly(k, head)).sum()))
            same = r["lo"] <= tol and r["hi"] >= -tol
            mtxt = f"{f2(r['estimate'])} {cis(r)} {'unchanged' if same else '**CHANGED**'}"
        rows.append([R.label(k), f2(dv["inventory_pnl"]), pct(dv["share_of_total"]), pct(dv["share_of_positive"]),
                     "**DIRECTIONAL**" if dv["directional"] else "market making", f2(up), f2(dn),
                     f2(corr, nd=3), mtxt])
    L.append(table(["method", "inventory P&L", "share of total", "share of positive", "class",
                    "P&L Up-won mkts", "P&L Down-won mkts", "corr(inv, next 60 s return)",
                    "mirror - original [95% CI]"], rows) + "\n")

    # ---- concentration
    L.append("## 7. Concentration (T7)\n")
    rows = []
    for k in akeys:
        cm = S.concentration(R.per_market(k, head).to_numpy(), ks=(1, 3, 5, 10))
        ch = S.concentration(R.hourly(k, head), ks=(1, 3, 5, 10))
        rows.append([R.label(k), f2(cm["total"]), pct(cm["top1_share_of_total"]), pct(cm["top5_share_of_total"]),
                     pct(cm["top10_share_of_total"]), f2(cm["ex_top5"]), f2(cm.get("n_effective"), False, 1),
                     cm["n_for_half_profit"], pct(cm["frac_positive"]), pct(ch["top1_share_of_total"]),
                     f2(ch["ex_top3"]), pct(ch["frac_positive"])])
    L.append(table(["method", "total", "top-1 mkt", "top-5 mkts", "top-10 mkts", "ex top-5 mkts",
                    "eff. # mkts", "# mkts for half profit", "mkts > 0", "top-1 hour", "ex top-3 hours",
                    "hours > 0"], rows) + "\n")

    # ---- sensitivity
    L.append("## 8. Fill model x latency grid, size sweep\n")
    grid = [f"{fm}_lat{lat}_x1" for fm in ("optimistic", "central", "pessimistic") for lat in (0, 1, 2)]
    gk = keys
    rows = []
    for k in gk:
        rows.append([R.label(k)] + [f2(R.total(k, v)) if R.has(k, v) else "-" for v in grid])
    L.append(table(["method"] + [v.replace("_x1", "").replace("_lat", " ") + " s" for v in grid], rows) + "\n")
    rows = []
    for v in grid:
        if R.has("rl/mean", v) and R.has(R.best, v):
            r = R.ci(R.hourly("rl/mean", v), R.hourly(R.best, v))
            rows.append([v.replace("_x1", ""), f2(r["estimate"]), cis(r),
                         "headline" if v == head else ("never headline" if "lat0" in v else "")])
    if rows:
        L.append(f"RL (seed mean) minus {R.label(R.best)} per cell:\n")
        L.append(table(["variant", "diff", "95% CI", ""], rows) + "\n")
    if "sensitivity_heatmap" in figs:
        L.append(f"![sensitivity]({_relfig(figs['sensitivity_heatmap'], out_path)})\n")
    sz = ["central_lat1_x0.5", "central_lat1_x1", "central_lat1_x2"]
    rows = []
    for k in gk:
        t = [R.total(k, v) if R.has(k, v) else float("nan") for v in sz]
        ratio = t[2] / t[1] if np.isfinite(t[1]) and abs(t[1]) > 1e-12 else float("nan")
        rows.append([R.label(k)] + [f2(x) for x in t] + [f2(ratio, False)])
    L.append("Size sweep (central, 1 s). P&L that grows linearly in size well beyond displayed depth is "
             "replay optimism, not edge:\n")
    L.append(table(["method", "x0.5", "x1", "x2", "x2 / x1"], rows) + "\n")

    # ---- breakdowns
    L.append("## 9. By contract length and by coin (headline)\n")
    bk = [k for k in keys if R.has(k, head)]
    for col, name in (("length", "contract length"), ("coin", "coin")):
        rows = []
        base = R.mk_rows(bk[0], head) if bk else pd.DataFrame()
        cats = list(base[col].drop_duplicates()) if len(base) else []
        if col == "length":
            order = list(LENGTH_NAMES.values())
            cats = sorted(cats, key=lambda x: order.index(x) if x in order else 99)
        else:
            cats = sorted(cats)
        for cval in cats:
            r = [cval, int((base[col] == cval).sum() / max(1, len(R.members(bk[0]))))]
            for k in bk:
                d = R.mk_rows(k, head)
                r.append(f2(d[d[col] == cval].groupby("market").pnl.mean().sum()))
            rows.append(r)
        L.append(f"P&L by {name}:\n")
        L.append(table([name, "markets"] + [R.label(k) for k in bk], rows) + "\n")

    # ---- seeds
    L.append("## 10. Seed spread\n")
    for g in ("rl", "rl5"):
        if g not in R.groups:
            continue
        rows, tots = [], []
        for m in R.groups[g]:
            t = [R.total(m, v) if R.has(m, v) else float("nan") for v in (head, "pessimistic_lat1_x1", "central_lat2_x1")]
            tots.append(t)
            ci_txt = "n/a"
            if R.has(m, head) and R.has(R.best, head):
                r = R.ci(R.hourly(m, head), R.hourly(R.best, head))
                ci_txt = f"{f2(r['estimate'])} {cis(r)}"
            rows.append([m] + [f2(x) for x in t] + [ci_txt])
        a = np.array(tots, float)
        for nm, fn in (("mean", np.nanmean), ("std", lambda x, axis: np.nanstd(x, axis=axis, ddof=1)),
                       ("min", np.nanmin), ("max", np.nanmax)):
            rows.append([f"**{nm}**"] + [f2(x) for x in fn(a, axis=0)] + [""])
        L.append(f"{GROUP_LABEL[g]}: {len(R.groups[g])} seeds of the selected config.\n")
        L.append(table(["seed", "central 1 s", "pessimistic 1 s", "central 2 s", f"minus {R.label(R.best)} [95% CI]"],
                       rows) + "\n")

    # ---- multiple testing
    L.append("## 11. Multiple testing (T4)\n")
    led = P["ledger"]
    L.append(f"Configurations evaluated on val before the freeze (all `trials.jsonl` ledgers): **N = "
             f"{led['n_trials']}** evaluations, {led['n_distinct_configs']} distinct configs "
             f"({led['n_assumed_val']} records without a split field counted as val; "
             f"{led.get('n_annotation_rows', 0)} annotation rows not counted), by kind "
             f"{json.dumps(led['by_kind'])}; files: " + ", ".join(f"`{f['path']}` ({f['n_val_records']})"
                                                                for f in led["files"]) + ".\n")
    if R.has("rl/mean", head):
        h = R.hourly("rl/mean", head)
        var = led.get("var_trial_sharpes")
        if var is not None and np.std(h) > 0:
            d = S.deflated_sharpe(h, max(int(led["n_trials"]), 1), float(var))
            L.append(f"Deflated Sharpe of RL (seed mean), hourly, on this split: SR {f2(d['sharpe'], nd=3)}, "
                     f"SR0 {f2(d['sr0'], nd=3)} (N = {d['n_trials']}, var of trial Sharpes {var:.4g}), "
                     f"P(SR > 0 | selection) = {d['dsr']:.3f}. A sanity check, not a test of record.\n")
        else:
            L.append("Deflated Sharpe: not available (no hourly Sharpe values in the ledgers, or a flat series).\n")
    runs = [r for r in read_jsonl(R.dir.parent / "runs.jsonl")[0] if r.get("split") == R.split]
    started = [r for r in runs if r.get("status") == "started"]
    L.append(f"Run records of this split: {len(started)} started, "
             f"{sum(r.get('status') == 'completed' for r in runs)} completed, "
             f"{sum(r.get('status') == 'crashed' for r in runs)} crashed. Forced reruns: "
             + ("; ".join(f"`{r['run_id']}`: {r['force_rerun_reason']}" for r in started if r.get("force_rerun_reason"))
                or "none") + ".\n")
    if dry:
        nd = len(read_jsonl(R.dir / "dryrun_trials.jsonl")[0])
        L.append(f"Post-freeze dry-run evaluations on val logged in `dryrun_trials.jsonl`: {nd} (not part of N; "
                 "they cannot influence a frozen plan).\n")

    # ---- regime
    L.append("## 12. Regime of the evaluation day (T9)\n")
    if ds is not None:
        splits = [0, 1] + ([2] if R.split == 2 else [])
        L.append(table(["split", "markets", "Binance 1 s vol (bp per sqrt hour)", "taker trades per market-hour",
                        "two-sided spread (c)", "rows without a two-sided book"], regime_stats(ds, splits)) + "\n")
        if R.split != 2:
            L.append("(The dry run never reads the test split, so the test day is not in this table.)\n")

    # ---- integrity
    L.append("## 13. Run integrity\n")
    ck = rep["checks"]
    rows = [["plan sha256 (results / recorded)", rep["plan_sha256"]],
            ["backend", f"{rep['backend']['name']} (of record: {rep['backend']['of_record']})"],
            ["data files vs plan", f"{ck['data_files']['n']} files match"],
            ["test rows sha256", ck["test_rows"].get("sha256") or ck["test_rows"].get("note")],
            ["code vs plan", "unchanged" if ck["code"]["ok"] else
             f"CHANGED {ck['code']['changed']} (accepted: {ck['code']['accepted_reason']})"],
            ["registration", json.dumps(ck.get("registration")) if "registration" in ck else "n/a (dry run)"],
            ["reruns", json.dumps(rep["reruns"]) if rep["reruns"] else "none"],
            ["data snapshot", json.dumps(ck.get("snapshot")) if ck.get("snapshot") else "n/a"],
            ["mirrored data (sha256)", ", ".join(f"{k} {v[:12]}" for k, v in (ck.get("mirror_data") or {}).items())
             or "not built"],
            ["fill flags vs the C Log (vol_X, pnl_X)", flag_check_text(R)],
            ["C replay (harness + reference)",
             "n/a (stand-in env)" if not rp.get("checked") else
             f"max |state diff| {rp['state_max_abs_diff']:.1e}, max |C Log P&L - trace| "
             f"{rp['log_pnl_max_abs_diff']:.1e}, runs with mismatches: {len(rp['failures'])}"],
            ["git commit at plan / now", f"{P['git']['commit']} / "
                                          f"{(runs[-1]['git']['commit'] if runs else 'n/a')}"],
            ["runs", f"{len(rep['methods'])} methods x {len(rep['variants'])} variants in {rep['seconds']} s"]]
    L.append(table(["check", "value"], rows, ["l", "l"]) + "\n")
    if "cumulative_pnl" in figs:
        L.insert(L.index("## 3. Paired differences vs every baseline (hourly blocks, identical markets)\n"),
                 f"![cumulative P&L]({_relfig(figs['cumulative_pnl'], out_path)})\n")
    return "\n".join(L)


def flag_check_text(R) -> str:
    fc = [a["flag_check_vs_c_log"] for a in R.aggs.values() if a.get("flag_check_vs_c_log")]
    if not fc:
        return "n/a (stand-in env)"
    return (f"max |volume diff| {max(x['vol_max_abs_diff'] for x in fc):.1e} shares, max |edge diff| "
            f"{max(x['pnl_max_abs_diff'] for x in fc):.1e} USD over {len(fc)} runs (float32 Log)")


def _relfig(p: Path, out_path: Path) -> str:
    try:
        return Path(p).resolve().relative_to(out_path.parent.resolve()).as_posix()
    except ValueError:
        return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=int, default=1, choices=[1, 2])
    ap.add_argument("--results-dir", default=str(FINAL / "results"))
    ap.add_argument("--plan", default=None, help="default: the plan path recorded in the results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--n-boot", type=int, default=None, help="override (recorded in the report)")
    ap.add_argument("--no-regime", action="store_true", help="skip the T9 data scan")
    a = ap.parse_args(argv)
    R = Results(Path(a.results_dir), a.split, a.plan)
    if a.n_boot:
        R.boot["n_boot"] = a.n_boot
    out = Path(a.out) if a.out else FINAL / ("RESULTS.md" if a.split == 2 else "RESULTS_dryrun_val.md")
    figd = Path(a.figures) if a.figures else FINAL / ("figures" if a.split == 2 else "figures/dryrun_val")
    if a.split == 1 and out.resolve() == (FINAL / "RESULTS.md").resolve():
        print("report: refusing to write a dry-run report to final/RESULTS.md", file=sys.stderr)
        return 2
    R._figs = figures(R, figd)
    ds = None if a.no_regime else Dataset(resolve(R.plan["data"]["build_dir"], CODE_ROOT))
    text = build(R, figd, out, ds)
    if a.n_boot and a.n_boot != R.plan["bootstrap"]["n_boot"]:
        text = text.replace("\n", f"\n> **Bootstrap replicates overridden: {a.n_boot}** (plan: "
                                  f"{R.plan['bootstrap']['n_boot']}).\n", 1)
    out.write_text(text)
    print(f"wrote {out} and {len(R._figs)} figures in {figd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
