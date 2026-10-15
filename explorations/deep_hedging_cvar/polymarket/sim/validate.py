#!/usr/bin/env python3
"""Sim-vs-replay consistency test on the TRAIN days, numpy-only version (fast; secondary to validate_env.py).

The gate of record is validate_env.py (C env). This script keeps the original numpy fill check and the
G4 stylised-fact evaluation (`evaluate`, `compare_table`), which validate_env.py reuses.

Real: the 431 train markets (data/build ticks + data/raw/trades.parquet legs).
Sim:  the same schedule on the same Binance paths, --reps replications (base generator, no randomisation).
Both go through the SAME statistics and the SAME numpy fill check (two fixed quoting rules), so any
difference is the generator's. Writes sim/cache/validation_numpy.json and sim/cache/VALIDATION_numpy.md.

Thresholds are fixed in GATE below (written before any comparison was run).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common as cm
import engine as en
from metrics import (DS, real_dataset, sim_datasets, all_stats, ks, LN, Q_SIZE, MATCH_ROW_LAG,  # noqa: F401
                     MARKOUT_H)

C = cm.C

GATE = {
    "G1_pnl_interval": "real total and median per-market P&L inside the sim 5-95% interval, per baseline",
    "G1_ks_hourly": 0.15,          # KS D between real and pooled-sim hourly P&L series (markets by end hour)
    "G1_ks_market": 0.15,          # KS D per-market P&L (reported; iid-market version)
    "G2_rank": "rank order of {no-quote, touch, fv} by total P&L identical (real vs sim median)",
    "G3_band": "per-market means of fills, volume, spread capture, markout 5/30/60, resolution inside sim 5-95%",
    "G4_spread_median_ticks": 1.0,   # |median spread sim - real| <= 1 tick (0.01) per (len, tau bucket)
    "G4_ks_dist": 0.15,              # KS D for pooled spread / log top size / log top-5 depth / trades per market
    "G4_onesided_abs": 0.10,         # |one-sided fraction sim - real| per (len, tau bucket)
    "G4_rate_ratio": (0.67, 1.5),    # trade rate per second sim/real per (len, tau bucket) with >= 200 real trades
    "G4_pbuy_abs": 0.06,             # |P(buy|gap) sim - real| per (len, gap bucket) with >= 300 real trades
    "G4_acf_abs": 0.06,              # |lag-1 autocorrelation of trade signs sim - real| per len
}


# --------------------------------------------------------------------------------------------- gate
def band(vals, q=(5, 95)):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))) if len(v) else (np.nan, np.nan)


def evaluate(real, sims):
    res = {"G1": {}, "G2": {}, "G3": {}, "G4": {}}
    # ---- G1 / G2 / G3: baselines
    totals_real = {"no-quote": 0.0}
    totals_sim = {"no-quote": [0.0] * len(sims)}
    for rule in ("touch", "fv"):
        fr = real["fills"][rule]
        tot_s = [s["fills"][rule]["pnl"].sum() for s in sims]
        med_s = [np.median(s["fills"][rule]["pnl"]) for s in sims]
        b_tot, b_med = band(tot_s), band(med_s)
        tr, mr = fr["pnl"].sum(), np.median(fr["pnl"])
        pooled = np.concatenate([s["fills"][rule]["pnl"] for s in sims])
        hourly_pooled = np.concatenate([s["hourly"][rule].values for s in sims])
        d_mk = ks(fr["pnl"], pooled)
        d_hr = ks(real["hourly"][rule].values, hourly_pooled)
        res["G1"][rule] = dict(
            real_total=tr, sim_total_band=b_tot, sim_total_median=float(np.median(tot_s)),
            total_in=bool(b_tot[0] <= tr <= b_tot[1]),
            real_median=mr, sim_median_band=b_med, median_in=bool(b_med[0] <= mr <= b_med[1]),
            ks_market=d_mk, ks_hourly=d_hr,
            passed=bool(b_tot[0] <= tr <= b_tot[1] and b_med[0] <= mr <= b_med[1] and d_hr < GATE["G1_ks_hourly"]))
        totals_real[rule] = tr
        totals_sim[rule] = tot_s
        comp = {}
        for k in ("fills", "volume", "capture", "mk5", "mk30", "mk60", "resolution"):
            rv = float(fr[k].mean())
            b = band([s["fills"][rule][k].mean() for s in sims])
            comp[k] = dict(real=rv, band=b, inside=bool(b[0] <= rv <= b[1]))
        # markout curve per share
        vol_r = fr["volume"].sum()
        curve = {}
        for h in MARKOUT_H:
            rv = float(fr[f"mk{h}"].sum() / max(vol_r, 1e-9) * 100)
            b = band([s["fills"][rule][f"mk{h}"].sum() / max(s["fills"][rule]["volume"].sum(), 1e-9) * 100
                      for s in sims])
            curve[h] = dict(real_c_per_share=rv, band=b, inside=bool(b[0] <= rv <= b[1]))
        comp["markout_curve"] = curve
        res["G3"][rule] = comp
    order_r = sorted(totals_real, key=lambda k: -totals_real[k])
    med = {k: float(np.median(v)) for k, v in totals_sim.items()}
    order_s = sorted(med, key=lambda k: -med[k])
    res["G2"] = dict(real_order=order_r, sim_order=order_s, passed=order_r == order_s)
    g3_pass = all(v["inside"] for r in res["G3"].values() for k, v in r.items() if k != "markout_curve")
    g3_pass &= all(c["inside"] for c in res["G3"]["touch"]["markout_curve"].values())
    res["G3"]["passed"] = bool(g3_pass)

    # ---- G4: stylised facts
    g4 = {}
    rows = []
    for key, rv in real["book"].items():
        L, k = key.split("|")
        sv = [s["book"].get(key) for s in sims]
        sv = [x for x in sv if x is not None]
        if not sv:
            continue
        row = dict(len=L, tau_b=int(k))
        for f in ("onesided", "hollow", "spread_med", "spread_mean", "p_1tick", "top_sz_med", "top5_med", "age0",
                  "trade_rate"):
            row[f"{f}_real"] = rv[f]
            row[f"{f}_sim"] = float(np.nanmedian([x[f] for x in sv]))
        rows.append(row)
    bt = pd.DataFrame(rows)
    ok_sp = (np.abs(bt.spread_med_sim - bt.spread_med_real) <= GATE["G4_spread_median_ticks"] * 0.01 + 1e-9) | bt.spread_med_real.isna()
    ok_os = np.abs(bt.onesided_sim - bt.onesided_real) <= GATE["G4_onesided_abs"]
    rr = bt.trade_rate_sim / bt.trade_rate_real.replace(0, np.nan)
    lo, hi = GATE["G4_rate_ratio"]
    ok_rate = ((rr >= lo) & (rr <= hi)) | (bt.trade_rate_real < 0.005)
    g4["spread_median_by_tau"] = dict(passed=bool(ok_sp.all()), n_fail=int((~ok_sp).sum()), n=int(len(bt)))
    g4["onesided_by_tau"] = dict(passed=bool(ok_os.all()), n_fail=int((~ok_os).sum()), n=int(len(bt)))
    g4["trade_rate_by_tau"] = dict(passed=bool(ok_rate.all()), n_fail=int((~ok_rate).sum()), n=int(len(bt)))
    for name in ("spread", "log_top_sz", "log_top5", "trades_per_market"):
        for L in ("5m", "15m", "4h"):
            key = f"{name}|{L}"
            D = ks(real["dist"][key], np.concatenate([s["dist"][key] for s in sims]))
            g4[f"ks_{key}"] = dict(D=D, passed=bool(D < GATE["G4_ks_dist"]))
    pb = []
    for key, (rv, n) in ((k, v) for k, v in real["trade"].items() if k.startswith("pbuy|")):
        if n < 300:
            continue
        sv = float(np.nanmedian([s["trade"][key][0] for s in sims]))
        pb.append(dict(key=key, real=rv, sim=sv, n=n, ok=abs(sv - rv) <= GATE["G4_pbuy_abs"]))
    pb = pd.DataFrame(pb)
    g4["pbuy_gap_curve"] = dict(passed=bool(pb.ok.all()), n_fail=int((~pb.ok).sum()), n=int(len(pb)))
    for L in ("5m", "15m", "4h"):
        key = f"sign_acf|{L}|1"
        rv = real["trade"][key]
        sv = float(np.nanmedian([s["trade"][key] for s in sims]))
        g4[f"sign_acf1|{L}"] = dict(real=rv, sim=sv, passed=bool(abs(sv - rv) <= GATE["G4_acf_abs"]))
    res["G4"] = g4
    res["G4"]["passed"] = bool(all(v["passed"] for v in g4.values() if isinstance(v, dict)))
    res["tables"] = dict(book=bt, pbuy=pb)
    res["passed"] = bool(all(res["G1"][r]["passed"] for r in ("touch", "fv")) and res["G2"]["passed"]
                         and res["G3"]["passed"] and res["G4"]["passed"])
    return res


def compare_table(real, sims, section, keys=None):
    rows = []
    for k, rv in real[section].items():
        if keys and not any(k.startswith(x) for x in keys):
            continue
        if isinstance(rv, tuple):
            rv = rv[0]
        sv = [s[section][k][0] if isinstance(s[section][k], tuple) else s[section][k] for s in sims]
        b = band(sv)
        rows.append(dict(stat=k, real=rv, sim_median=float(np.nanmedian(sv)), sim_lo=b[0], sim_hi=b[1],
                         inside=bool(b[0] <= rv <= b[1]) if np.isfinite(rv) else None))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- report
def fmt(x, nd=3):
    if isinstance(x, (tuple, list)):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(res, real, sims, args, elapsed, path):
    pf = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    L = []
    L.append("# Generator validation: sim vs replay on the train days\n")
    L.append(f"Generated by `sim/validate.py --reps {args.reps} --seed {args.seed}` in {elapsed/60:.1f} min, with "
             f"params.json sha256 `{args.params_sha[:16]}`. Real data: the 431 train markets. Sim: the same schedule "
             f"on the same Binance paths, {args.reps} replications of the base generator (no randomisation).\n")
    L.append("The C env does not exist yet, so the two fixed quoting rules run through a numpy fill check. "
             "Real and sim use the same code for this check and for every statistic. The fill rule: a print with "
             f"block time T can fill our quote from row T-{MATCH_ROW_LAG} (quotes are set on row r-1, so 1 s "
             f"latency). A print through our price fills us in full; a print at our price fills pro-rata "
             f"Q/(Q+displayed). Q = {Q_SIZE:.0f} shares per side, and at most Q fill per side per second. "
             "There is no inventory control, fees or rebates. Settlement is at winner_up.\n")
    L.append("- **touch** joins the best bid and best ask when the book is two-sided with spread <= 0.10.")
    L.append("- **fv** quotes around c = (p_model + mid) / 2 with half-spread h = max(1c, spread/2) + |gap|/2, on the "
             "0.01 grid and never crossing. It pulls quotes when |gap| > 0.10, and in the last 30 s of 5m and 15m "
             "markets.\n")
    L.append(f"## Overall: **{pf(res['passed'])}**\n")
    L.append("| gate | result |\n|---|---|")
    L.append(f"| G1 P&L inside sim 90% interval + hourly KS < {GATE['G1_ks_hourly']} | touch {pf(res['G1']['touch']['passed'])}, fv {pf(res['G1']['fv']['passed'])} |")
    L.append(f"| G2 rank order of baselines | {pf(res['G2']['passed'])} (real {res['G2']['real_order']}, sim {res['G2']['sim_order']}) |")
    L.append(f"| G3 attribution, fill rates and markout curve inside the sim 90% band | {pf(res['G3']['passed'])} |")
    L.append(f"| G4 stylised facts | {pf(res['G4']['passed'])} |")
    L.append("| G5 out-of-sample val day | not run. It needs the C env and a ledger entry (EVALUATION.md T4) |\n")
    L.append("Thresholds were fixed before the first comparison (`GATE` in validate.py):\n")
    for k, v in GATE.items():
        L.append(f"- `{k}`: {v}")
    L.append("\n## G1/G2: baseline P&L ($, per market or total over 431 markets)\n")
    L.append("| rule | real total | sim total median [5%, 95%] | real median/mkt | sim median/mkt [5%, 95%] | KS per-market | KS hourly | pass |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in ("touch", "fv"):
        g = res["G1"][r]
        L.append(f"| {r} | {g['real_total']:.1f} | {g['sim_total_median']:.1f} {fmt(g['sim_total_band'], 1)} | "
                 f"{g['real_median']:.3f} | {fmt(g['sim_median_band'])} | {g['ks_market']:.3f} | {g['ks_hourly']:.3f} | "
                 f"{pf(g['passed'])} |")
    L.append("\n## G3: attribution per market ($), fills, markout curve (c/share)\n")
    L.append("| rule | component | real | sim [5%, 95%] | inside |\n|---|---|---|---|---|")
    for r in ("touch", "fv"):
        for k, v in res["G3"][r].items():
            if k == "markout_curve":
                for h, c in v.items():
                    L.append(f"| {r} | markout {h}s (c/share) | {c['real_c_per_share']:.3f} | {fmt(c['band'])} | {c['inside']} |")
            else:
                L.append(f"| {r} | {k} | {v['real']:.3f} | {fmt(v['band'])} | {v['inside']} |")
    L.append("\n## G4: stylised facts\n")
    L.append("| check | result |\n|---|---|")
    for k, v in res["G4"].items():
        if k == "passed":
            continue
        det = ", ".join(f"{a}={fmt(b)}" for a, b in v.items() if a != "passed")
        L.append(f"| {k} | {pf(v['passed'])} ({det}) |")
    bt = res["tables"]["book"]
    L.append("\n### Book and trade rate by contract length and time-to-close bucket (sim = median over replications)\n")
    L.append(f"tau buckets (s): {list(cm.TAU_EDGES)}\n")
    L.append("| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | "
             "top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in bt.itertuples():
        L.append(f"| {r.len} | {r.tau_b} | {r.onesided_real:.3f}/{r.onesided_sim:.3f} | {r.hollow_real:.3f}/{r.hollow_sim:.3f} | "
                 f"{r.spread_med_real:.3f}/{r.spread_med_sim:.3f} | {r.p_1tick_real:.2f}/{r.p_1tick_sim:.2f} | "
                 f"{r.top_sz_med_real:.0f}/{r.top_sz_med_sim:.0f} | {r.top5_med_real:.0f}/{r.top5_med_sim:.0f} | "
                 f"{r.age0_real:.2f}/{r.age0_sim:.2f} | {r.trade_rate_real:.3f}/{r.trade_rate_sim:.3f} |")
    pb = res["tables"]["pbuy"]
    L.append("\n### P(taker buys Up | gap = p_model - mid at block time - 3 s)\n")
    L.append(f"gap buckets: {list(cm.GAP_EDGES)}\n")
    L.append("| len|bucket | n real | real | sim |\n|---|---|---|---|")
    for r in pb.itertuples():
        L.append(f"| {r.key[5:]} | {r.n} | {r.real:.3f} | {r.sim:.3f} |")
    L.append("\n### Other trade and mid statistics (reported, not gated)\n")
    t = compare_table(real, sims, "trade", keys=["align_", "markout|", "sign_acf", "taker_size", "levels_per", "buy_share"])
    m = compare_table(real, sims, "mid")
    L.append("| stat | real | sim median | sim [5%, 95%] |\n|---|---|---|---|")
    for df in (t, m):
        for r in df.itertuples():
            L.append(f"| {r.stat} | {r.real:.3f} | {r.sim_median:.3f} | [{r.sim_lo:.3f}, {r.sim_hi:.3f}] |")
    path.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=5, help="replications simulated per engine call")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(cm.HERE / "cache" / "VALIDATION_numpy.md"))
    a = ap.parse_args()
    t0 = time.time()
    import hashlib
    a.params_sha = hashlib.sha256(cm.PARAMS.read_bytes()).hexdigest()
    model = en.Model(cm.load_params())
    P, mk = cm.train_panel()
    rng = np.random.default_rng(a.seed + 1)
    real = all_stats(real_dataset(P, mk), rng)
    print(f"real stats {time.time()-t0:.1f}s")
    ul = cm.train_underlying()
    prices = [cm.market_pricing(ul[r.coin][1], ul[r.coin][2], ul[r.coin][0], r.open_ts, r.end_ts) for r in mk.itertuples()]
    sims = []
    for i, d in enumerate(sim_datasets(model, mk, prices, a.reps, a.seed, a.chunk)):
        sims.append(all_stats(d, rng))
        if (i + 1) % a.chunk == 0:
            print(f"  sim rep {i+1}/{a.reps}  {time.time()-t0:.0f}s", flush=True)
    res = evaluate(real, sims)
    write_report(res, real, sims, a, time.time() - t0, cm.Path(a.out))
    js = {k: v for k, v in res.items() if k != "tables"}
    js["tables"] = {k: v.to_dict(orient="records") for k, v in res["tables"].items()}
    (cm.HERE / "cache" / "validation_numpy.json").write_text(json.dumps(js, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"overall {'PASS' if res['passed'] else 'FAIL'}  ({time.time()-t0:.0f}s) -> {a.out}")


if __name__ == "__main__":
    main()
