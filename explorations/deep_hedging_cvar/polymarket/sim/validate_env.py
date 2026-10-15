#!/usr/bin/env python3
"""Sim-vs-replay consistency gate (SIMULATOR.md section 5) through the C env, on the TRAIN markets.

Real:  the 431 train markets of data/build, played once by each policy through the C env
       (baselines/runner.py: one env per market, offset 0, central fills, latency 1 s by default).
Sim:   the same schedule on the same Binance paths (generate.py --schedule train, base generator, no
       randomisation), --reps replications built in chunks under data/build_sim/gate_c<k>/ (deleted
       after use), each passed through data/features.py (features_sim.py) and played identically.
Policies: baselines/baselines.py no_quote, sym_touch, fair_value at their untuned DEFAULTS.
G4 stylised facts come from the same chunks through metrics.py (validate.py's statistics).

The C core maps one data directory per process, so every (data dir) runs in a worker subprocess:
    python sim/validate_env.py --worker <data_dir> <out.json>
Orchestrator (writes sim/VALIDATION.md and sim/validation_env.json):
    python sim/validate_env.py --reps 20 --chunk 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

import common as cm

POLICIES = ["no_quote", "sym_touch", "fair_value"]
LOG_KEYS = ["pnl", "pnl_trading", "spread_capture", "markout_5", "markout_30", "markout_60", "inventory_pnl",
            "rebates", "fills", "fill_vol", "uptime_two_sided", "max_abs_inv"]
G3_KEYS = ["fills", "fill_vol", "spread_capture", "markout_5", "markout_30", "markout_60", "inventory_pnl",
           "rebates"]
GATE = {
    "G1": "per policy: real total and median per-market P&L inside the sim 5-95% band of replications, and "
          "KS D < 0.15 between the real hourly P&L series and the pooled sim hourly series (markets binned "
          "by end hour, so block structure is kept); per-market KS D is reported",
    "G1_ks": 0.15,
    "G2": "rank order of {no_quote, sym_touch, fair_value} by total P&L identical (real vs sim median)",
    "G3": "per-market means of fills, fill volume, spread capture, markout 5/30/60, inventory P&L and rebates, "
          "and sym_touch's markout curve (c/share), inside the sim 5-95% band",
    "G4": "stylised facts as in validate.py GATE (spread, depth, one-sided, trade rate, P(buy|gap), sign ACF)",
}


# --------------------------------------------------------------------------------------------- worker
def worker(data_dir: str, out: str, fill_model: int, latency: int):
    sys.path.insert(0, str(cm.POLY / "baselines"))
    sys.path.insert(0, str(cm.POLY))
    import runner
    import baselines as bl
    from obs_spec import load_spec
    runner.DATA_DIR = Path(data_dir)
    spec = load_spec(env_params=runner.core_params())
    r = runner.Runner(split=0, fill_model=fill_model, latency=latency)
    slugs = json.loads((Path(data_dir) / "meta.json").read_text())["slugs"]
    _, markets, _ = runner.load_data(Path(data_dir))
    res = {}
    for name in POLICIES:
        out_r = r.run(bl.make(name, spec, bl.DEFAULTS[name]))
        recs = []
        for rec in out_r["markets"]:
            gi = rec["market"]
            recs.append({"slug": slugs[int(markets[gi]["market_idx"])], "len": rec["len"], "coin": rec["coin"],
                         "end_ts": rec["end_ts"], **{k: rec.get(k, np.nan) for k in LOG_KEYS}})
        res[name] = recs
        print(f"  {Path(data_dir).name} {name}: total {sum(x['pnl'] for x in recs):.1f} ({out_r['wall_s']:.1f}s)",
              flush=True)
    Path(out).write_text(json.dumps(res))


def run_worker(data_dir: Path, out: Path, fill_model: int, latency: int):
    subprocess.run([sys.executable, str(cm.HERE / "validate_env.py"), "--worker", str(data_dir), str(out),
                    "--fill-model", str(fill_model), "--latency", str(latency)], check=True, cwd=str(cm.HERE))
    return json.loads(out.read_text())


# --------------------------------------------------------------------------------------------- numpy stats per rep
def chunk_datasets(build: Path, mk: pd.DataFrame, P: dict):
    """metrics.DS per replication of a train-schedule chunk, in the real train markets' order."""
    import metrics as mt
    from load import load
    ticks, markets, meta = load(build)
    slugs = np.array(meta["slugs"])
    fl = pd.read_parquet(build / "fills.parquet")
    fl["slug"] = fl.slug.astype(str)
    sm = pd.DataFrame({"slug": slugs[markets["market_idx"]], "off": markets["row_offset"],
                       "n": markets["n_rows"], "win": markets["winner_up"]})
    sm["rep"] = sm.slug.str.extract(r"-r(\d+)-")[0].astype(int)
    sm["real"] = sm.slug.str.replace(r"^sim[^-]*-r\d+-", "", regex=True)
    pos = pd.Series(np.arange(len(mk)), index=mk.slug)
    sm["m"] = pos.loc[sm.real].values
    fl = fl.merge(sm[["slug", "rep", "m"]], on="slug")
    fl["px"] = np.where(fl.outcome_idx == 0, fl.price.astype(float), 1 - fl.price.astype(float))
    fl["tx_id"] = pd.factorize(fl.tx)[0]
    for rep, g in sm.groupby("rep"):
        g = g.sort_values("m")
        assert (g.m.values == np.arange(len(mk))).all()
        X = np.concatenate([np.asarray(ticks[o:o + n]) for o, n in zip(g.off.values, g.n.values)])
        lg = fl[fl.rep == rep]
        lg = lg[(lg.timestamp >= mk.open_ts.values[lg.m]) & (lg.timestamp <= mk.end_ts.values[lg.m])]
        legs = pd.DataFrame(dict(m=lg.m.values, stamp=lg.timestamp.values, is_taker=lg.is_taker.values,
                                 sign=lg.aggressor_up.values.astype(int), px=lg.px.values, size=lg["size"].values,
                                 tx=lg.tx_id.values))
        mkj = mk.copy()
        mkj["winner_up"] = g.win.values
        yield mt.DS(X, mkj, legs, P["p"])


# --------------------------------------------------------------------------------------------- gate
def band(v, q=(5, 95)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, q[0])), float(np.percentile(v, q[1]))) if len(v) else (np.nan, np.nan)


def ks(a, b):
    from metrics import ks as _ks
    return _ks(a, b)


def evaluate_env(real: dict, sims: list[dict]) -> dict:
    """real: {policy: [per-market dicts]}; sims: list over replications of the same."""
    res = {"G1": {}, "G3": {}}
    tot_r, tot_s = {}, {}
    for pol in POLICIES:
        r = pd.DataFrame(real[pol])
        S = [pd.DataFrame(s[pol]) for s in sims]
        tr, mr = r.pnl.sum(), r.pnl.median()
        ts = [s.pnl.sum() for s in S]
        ms = [s.pnl.median() for s in S]
        hr = r.groupby((r.end_ts - 1) // 3600).pnl.sum().values
        hs = np.concatenate([s.groupby((s.end_ts - 1) // 3600).pnl.sum().values for s in S])
        bt, bm = band(ts), band(ms)
        d_h, d_m = ks(hr, hs), ks(r.pnl.values, np.concatenate([s.pnl.values for s in S]))
        tot_r[pol], tot_s[pol] = tr, float(np.median(ts))
        if pol == "no_quote":
            res["G1"][pol] = dict(real_total=tr, sim_total_median=tot_s[pol], passed=bool(abs(tr) < 1e-9 and
                                                                                       all(abs(x) < 1e-9 for x in ts)))
            continue
        res["G1"][pol] = dict(real_total=tr, sim_total_median=tot_s[pol], sim_total_band=bt,
                              total_in=bool(bt[0] <= tr <= bt[1]), real_median=mr, sim_median_band=bm,
                              median_in=bool(bm[0] <= mr <= bm[1]), ks_hourly=d_h, ks_market=d_m,
                              passed=bool(bt[0] <= tr <= bt[1] and bm[0] <= mr <= bm[1] and d_h < GATE["G1_ks"]))
        comp = {}
        for k in G3_KEYS:
            rv = float(r[k].mean())
            b = band([s[k].mean() for s in S])
            comp[k] = dict(real=rv, sim_median=float(np.median([s[k].mean() for s in S])), band=b,
                           inside=bool(b[0] <= rv <= b[1]))
        curve = {}
        for h in (5, 30, 60):
            rv = float(r[f"markout_{h}"].sum() / max(r.fill_vol.sum(), 1e-9) * 100)
            sv = [s[f"markout_{h}"].sum() / max(s.fill_vol.sum(), 1e-9) * 100 for s in S]
            b = band(sv)
            curve[h] = dict(real=rv, sim_median=float(np.median(sv)), band=b, inside=bool(b[0] <= rv <= b[1]))
        comp["markout_curve_c_per_share"] = curve
        by_len = {}
        for L in ("5m", "15m", "4h"):
            rv = float(r.pnl[r.len == L].sum())
            b = band([s.pnl[s.len == L].sum() for s in S])
            by_len[L] = dict(real=rv, band=b, sim_median=float(np.median([s.pnl[s.len == L].sum() for s in S])))
        comp["pnl_by_len"] = by_len
        res["G3"][pol] = comp
    order_r = sorted(tot_r, key=lambda k: -tot_r[k])
    order_s = sorted(tot_s, key=lambda k: -tot_s[k])
    res["G2"] = dict(real_order=order_r, sim_order=order_s, passed=order_r == order_s)
    res["G1"]["passed"] = bool(all(v["passed"] for v in res["G1"].values() if isinstance(v, dict)))
    g3 = all(c["inside"] for pol in res["G3"].values() for k, c in pol.items() if k in G3_KEYS)
    g3 &= all(c["inside"] for c in res["G3"]["sym_touch"]["markout_curve_c_per_share"].values())
    res["G3"]["passed"] = bool(g3)
    return res


# --------------------------------------------------------------------------------------------- report
def fmt(x, nd=2):
    if isinstance(x, (tuple, list)):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(env_res, np_res, args, elapsed, real_sum, sim_sum, path: Path):
    import validate as vd
    pf = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    g4 = np_res["G4"]
    overall = env_res["G1"]["passed"] and env_res["G2"]["passed"] and env_res["G3"]["passed"] and g4["passed"]
    L = ["# Generator validation: sim vs replay through the C env (train days)\n"]
    L.append(f"`sim/validate_env.py --reps {args.reps} --chunk {args.chunk} --fill-model {args.fill_model} "
             f"--latency {args.latency}` ran in {elapsed / 60:.0f} min. params.json sha256 "
             f"`{args.params_sha[:16]}`, mm.h sha256 `{args.mm_sha[:16]}`.\n")
    L.append("**Setup**\n")
    L.append("- Real: the 431 train markets of data/build (opens on 2026-08-21..22).")
    L.append(f"- Sim: the same schedule on the same Binance paths, {args.reps} replications of the base "
             "generator (no randomisation). Each was written in the replay format, passed through "
             "data/features.py unmodified (sim/features_sim.py) and played by the C env (mm.h through "
             "baselines/runner.py) with identical settings.")
    L.append(f"- Fill model {['optimistic', 'central', 'pessimistic'][args.fill_model]}, latency "
             f"{args.latency} s, offset 0, env defaults otherwise.")
    L.append("- Policies: baselines.py `no_quote`, `sym_touch` and `fair_value` at their untuned DEFAULTS.")
    L.append(f"- SIMULATOR.md asks for 200 replications. {args.reps} were run because CPU is shared with "
             f"PPO, so the 5-95% bands are rough.\n")
    L.append(f"## Overall: **{pf(overall)}**\n")
    L.append("| gate | result |\n|---|---|")
    L.append(f"| G1 P&L inside sim 90% band + hourly KS < {GATE['G1_ks']} | "
             + ", ".join(f"{p} {pf(env_res['G1'][p]['passed'])}" for p in POLICIES) + " |")
    L.append(f"| G2 rank order | {pf(env_res['G2']['passed'])}: real {env_res['G2']['real_order']}, sim "
             f"{env_res['G2']['sim_order']} |")
    L.append(f"| G3 attribution, fills, markout curve inside the 90% band | {pf(env_res['G3']['passed'])} |")
    L.append(f"| G4 stylised facts (numpy statistics on the same chunks) | {pf(g4['passed'])} |")
    L.append("| G5 out-of-sample on the val day | not run: it needs a ledger entry (EVALUATION.md T4) |\n")
    L.append("Gate definitions: " + "; ".join(f"**{k}** {v}" for k, v in GATE.items() if not k.endswith("_ks")) + "\n")
    L.append("## G1/G2: P&L through the C env ($)\n")
    L.append("| policy | real total | sim total median [5%, 95%] | real median/mkt | sim median/mkt [5%, 95%] | "
             "KS hourly | KS per-market | pass |\n|---|---|---|---|---|---|---|---|")
    for p in POLICIES[1:]:
        g = env_res["G1"][p]
        L.append(f"| {p} | {g['real_total']:.1f} | {g['sim_total_median']:.1f} {fmt(g['sim_total_band'], 1)} | "
                 f"{g['real_median']:.3f} | {fmt(g['sim_median_band'], 3)} | {g['ks_hourly']:.3f} | "
                 f"{g['ks_market']:.3f} | {pf(g['passed'])} |")
    L.append(f"| no_quote | {env_res['G1']['no_quote']['real_total']:.1f} | "
             f"{env_res['G1']['no_quote']['sim_total_median']:.1f} | 0 | 0 | | | "
             f"{pf(env_res['G1']['no_quote']['passed'])} |")
    L.append("\nP&L by contract length (total $):\n")
    L.append("| policy | len | real | sim median [5%, 95%] |\n|---|---|---|---|")
    for p in POLICIES[1:]:
        for Ln, v in env_res["G3"][p]["pnl_by_len"].items():
            L.append(f"| {p} | {Ln} | {v['real']:.1f} | {v['sim_median']:.1f} {fmt(v['band'], 1)} |")
    L.append("\n## G3: Log attribution per market ($, means over markets) and markout curve\n")
    L.append("| policy | component | real | sim median [5%, 95%] | inside |\n|---|---|---|---|---|")
    for p in POLICIES[1:]:
        for k, v in env_res["G3"][p].items():
            if k in G3_KEYS:
                L.append(f"| {p} | {k} | {v['real']:.3f} | {v['sim_median']:.3f} {fmt(v['band'], 3)} | {v['inside']} |")
        for h, v in env_res["G3"][p]["markout_curve_c_per_share"].items():
            L.append(f"| {p} | markout {h}s (c/share) | {v['real']:.3f} | {v['sim_median']:.3f} {fmt(v['band'], 3)} | "
                     f"{v['inside']} |")
    L.append("\n## G4: stylised facts\n")
    L.append("| check | result |\n|---|---|")
    for k, v in g4.items():
        if k == "passed":
            continue
        det = ", ".join(f"{a}={fmt(b, 3)}" for a, b in v.items() if a != "passed")
        L.append(f"| {k} | {pf(v['passed'])} ({det}) |")
    bt = np_res["tables"]["book"]
    L.append("\n### Book and trade rate by length and time-to-close bucket (sim = median over replications)\n")
    L.append(f"tau bucket edges (s): {[int(x) for x in cm.TAU_EDGES]}\n")
    L.append("| len | tau_b | one-sided real/sim | hollow real/sim | median spread real/sim | P(1 tick) real/sim | "
             "top size med real/sim | top-5 med real/sim | P(age=0) real/sim | trades/s real/sim |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in bt.itertuples():
        L.append(f"| {r.len} | {r.tau_b} | {r.onesided_real:.3f}/{r.onesided_sim:.3f} | {r.hollow_real:.3f}/{r.hollow_sim:.3f} | "
                 f"{r.spread_med_real:.3f}/{r.spread_med_sim:.3f} | {r.p_1tick_real:.2f}/{r.p_1tick_sim:.2f} | "
                 f"{r.top_sz_med_real:.0f}/{r.top_sz_med_sim:.0f} | {r.top5_med_real:.0f}/{r.top5_med_sim:.0f} | "
                 f"{r.age0_real:.2f}/{r.age0_sim:.2f} | {r.trade_rate_real:.3f}/{r.trade_rate_sim:.3f} |")
    pb = np_res["tables"]["pbuy"]
    L.append("\n### P(taker buys Up | gap = p_model - mid at block time - 3 s)\n")
    L.append(f"gap bucket edges: {[float(x) for x in cm.GAP_EDGES]}\n")
    L.append("| len / bucket | n real | real | sim |\n|---|---|---|---|")
    for r in pb.itertuples():
        L.append(f"| {r.key[5:]} | {r.n} | {r.real:.3f} | {r.sim:.3f} |")
    L.append("\n### Other statistics (reported, not gated; sim median and 5-95% band)\n")
    t = vd.compare_table(real_sum, sim_sum, "trade", keys=["align_", "markout|", "sign_acf", "taker_size",
                                                            "levels_per", "buy_share"])
    m = vd.compare_table(real_sum, sim_sum, "mid")
    L.append("| stat | real | sim median | sim [5%, 95%] |\n|---|---|---|---|")
    for df in (t, m):
        for r in df.itertuples():
            L.append(f"| {r.stat} | {r.real:.3f} | {r.sim_median:.3f} | [{r.sim_lo:.3f}, {r.sim_hi:.3f}] |")
    L.append("\n### Numpy fill check (secondary cross-check, validate.py rules)\n")
    L.append("| rule | real total | sim total median [5%, 95%] | KS hourly |\n|---|---|---|---|")
    for rule in ("touch", "fv"):
        g = np_res["G1"][rule]
        L.append(f"| {rule} | {g['real_total']:.1f} | {g['sim_total_median']:.1f} {fmt(g['sim_total_band'], 1)} | "
                 f"{g['ks_hourly']:.3f} |")
    path.write_text("\n".join(L) + "\n")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("DATA_DIR", "OUT_JSON"))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--fill-model", type=int, default=1)
    ap.add_argument("--latency", type=int, default=1)
    ap.add_argument("--keep", action="store_true", help="keep the chunk builds")
    ap.add_argument("--out", default=str(cm.HERE / "VALIDATION.md"))
    a = ap.parse_args()
    if a.worker:
        worker(a.worker[0], a.worker[1], a.fill_model, a.latency)
        return
    import metrics as mt
    import validate as vd
    t0 = time.time()
    a.params_sha = hashlib.sha256(cm.PARAMS.read_bytes()).hexdigest()
    a.mm_sha = hashlib.sha256((cm.POLY / "mm.h").read_bytes()).hexdigest()
    work = cm.HERE / "cache" / "gate"
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    P, mk = cm.train_panel()
    real_env = run_worker(cm.BUILD, work / "real_env.json", a.fill_model, a.latency)
    real_np = mt.all_stats(mt.real_dataset(P, mk), rng)
    print(f"real done {time.time() - t0:.0f}s", flush=True)
    sims_env, sims_np = [], []
    for c, r0 in enumerate(range(0, a.reps, a.chunk)):
        k = min(a.chunk, a.reps - r0)
        build = cm.OUT_DEFAULT / f"gate_c{c}"
        subprocess.run([sys.executable, str(cm.HERE / "generate.py"), "--schedule", "train", "--reps", str(k),
                        "--no-randomize", "--seed", str(a.seed + c), "--out", str(build)], check=True,
                       cwd=str(cm.HERE), stdout=subprocess.DEVNULL)
        env_c = run_worker(build, work / f"sim_env_c{c}.json", a.fill_model, a.latency)
        for j in range(k):
            pref = f"-r{j}-"
            sims_env.append({p: [x for x in v if pref in x["slug"]] for p, v in env_c.items()})
        for d in chunk_datasets(build, mk, P):
            sims_np.append(mt.all_stats(d, rng))
        if not a.keep:
            shutil.rmtree(build)
        print(f"chunk {c}: {len(sims_env)}/{a.reps} replications  {time.time() - t0:.0f}s", flush=True)
    env_res = evaluate_env(real_env, sims_env)
    np_res = vd.evaluate(real_np, sims_np)
    overall = write_report(env_res, np_res, a, time.time() - t0, real_np, sims_np, Path(a.out))
    js = dict(env=env_res, numpy={k: v for k, v in np_res.items() if k != "tables"},
              tables={k: v.to_dict(orient="records") for k, v in np_res["tables"].items()}, overall=overall,
              reps=a.reps, fill_model=a.fill_model, latency=a.latency, params_sha256=a.params_sha, mm_sha256=a.mm_sha)
    (cm.HERE / "validation_env.json").write_text(json.dumps(js, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"overall {'PASS' if overall else 'FAIL'}  ({time.time() - t0:.0f}s) -> {a.out}")


if __name__ == "__main__":
    main()
