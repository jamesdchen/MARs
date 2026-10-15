#!/usr/bin/env python3
"""Calibrate the flow's look-ahead informed takers (and impact) through the C env, on TRAIN only.

Each candidate parameter set runs the whole pipeline on one replication of the train schedule:
generate.py (candidate params) -> data/features.py (features_sim.py) -> C env (baselines sym_touch and
fair_value at their DEFAULTS, central fills, latency 1, validate_env.py worker) -> print-alignment statistics.
The loss compares with the real train markets:
  fair_value and sym_touch markout curves (c/share at 5/30/60 s), fill volume and spread capture per market,
  total P&L, the alignment of prints with the captured book (at-or-through / through at k = 0 and 4 s), and
  the taker-tx count per length (so arrival rates stay calibrated).
Results append to cache/calib_env.jsonl; --apply writes the best candidate into params.json.

    python sim/calibrate_env.py --grid lookahead      # then --apply
"""
from __future__ import annotations

import argparse
import copy
import itertools
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

WORK = cm.HERE / "cache" / "calib"
LOG = cm.HERE / "cache" / "calib_env.jsonl"
LENS = ("5m", "15m", "4h")


def env_metrics(res: dict) -> dict:
    out = {}
    for pol in ("sym_touch", "fair_value"):
        d = pd.DataFrame(res[pol])
        vol = d.fill_vol.sum()
        out[f"{pol}|total"] = float(d.pnl.sum())
        out[f"{pol}|fill_vol"] = float(d.fill_vol.mean())
        out[f"{pol}|capture"] = float(d.spread_capture.mean())
        for h in (5, 30, 60):
            out[f"{pol}|mk{h}"] = float(d[f"markout_{h}"].sum() / max(vol, 1e-9) * 100)
    return out


def align_metrics(d) -> dict:
    import metrics as mt
    tk = mt.taker_table(d)
    _, a, t = mt.alignment_curve(d, tk, ks=(0, 4))
    out = {"align|at0": float(a[0]), "align|thr0": float(t[0]), "align|at4": float(a[1]), "align|thr4": float(t[1])}
    for L, n in zip((300, 900, 14400), LENS):
        out[f"count|{n}"] = float(d.X[d.L == L, cm.C["trade_count"]].sum())
    return out


SCALE = {"total": 600.0, "fill_vol": None, "capture": 0.3, "mk5": 0.4, "mk30": 0.4, "mk60": 0.4,
         "at0": 0.03, "thr0": 0.03, "at4": 0.03, "thr4": 0.03, "count": None}


def loss(sim: dict, real: dict) -> tuple[float, dict]:
    parts = {}
    for k, rv in real.items():
        kind = k.split("|")[-1] if not k.startswith("count") else "count"
        sv = sim[k]
        if kind in ("fill_vol", "count"):
            parts[k] = (np.log(max(sv, 1e-9) / max(rv, 1e-9)) / (0.15 if kind == "fill_vol" else 0.1)) ** 2
        else:
            parts[k] = ((sv - rv) / SCALE[kind]) ** 2
    return float(sum(parts.values())), parts


def real_targets(mk, P) -> dict:
    import metrics as mt
    import validate_env as ve
    rj = WORK / "real_env.json"
    res = ve.run_worker(cm.BUILD, rj, 1, 1) if not rj.exists() else json.loads(rj.read_text())
    out = env_metrics(res)
    out.update(align_metrics(mt.real_dataset(P, mk)))
    return out


def evaluate(cand: dict, base: dict, mk, P, seed: int) -> dict:
    import validate_env as ve
    p = copy.deepcopy(base)
    for path, val in cand.items():
        node = p
        keys = path.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = val
    pf = WORK / "params_cand.json"
    pf.write_text(json.dumps(p))
    build = WORK / "build"
    if build.exists():
        shutil.rmtree(build)
    subprocess.run([sys.executable, str(cm.HERE / "generate.py"), "--schedule", "train", "--reps", "1",
                    "--no-randomize", "--seed", str(seed), "--params", str(pf), "--out", str(build)],
                   check=True, cwd=str(cm.HERE), stdout=subprocess.DEVNULL)
    res = ve.run_worker(build, WORK / "sim_env.json", 1, 1)
    out = env_metrics(res)
    d = next(ve.chunk_datasets(build, mk, P))
    out.update(align_metrics(d))
    shutil.rmtree(build)
    return out


def grid(name: str, base: dict) -> list[dict]:
    imp0 = {L: base["flow"]["impact"][L]["kappa"] for L in LENS}
    w0 = {L: base["flow"]["informed"][L]["w"] for L in LENS}
    cands = [{}]                                                      # current params
    if name == "lookahead":
        for h, k, im in itertools.product((2, 5), (0.1, 0.3, 1.0), (1.0, 0.5)):
            c = {f"flow.lookahead.{L}": {"h": h, "kappa": k} for L in LENS}
            c.update({f"flow.informed.{L}.w": 0.0 for L in LENS})
            c.update({f"flow.impact.{L}.kappa": imp0[L] * im for L in LENS})
            cands.append(c)
    elif name == "refine":
        for h, k, trk in itertools.product((5, 10, 20), (1.0, 3.0), (1.0, 0.5)):
            c = {f"flow.lookahead.{L}": {"h": h, "kappa": k} for L in LENS}
            c.update({f"flow.informed.{L}.w": 0.0 for L in LENS})
            c.update({f"flow.impact.{L}.kappa": imp0[L] * 0.5 for L in LENS})
            c["flow.actual_book_tracking"] = trk
            cands.append(c)
    elif name == "refill":
        for rf, k in itertools.product((0.5, 0.8, 0.95), (1.0, 3.0)):
            c = {f"flow.lookahead.{L}": {"h": 5, "kappa": k} for L in LENS}
            c.update({f"flow.informed.{L}.w": 0.0 for L in LENS})
            c.update({f"flow.impact.{L}.kappa": imp0[L] * 0.5 for L in LENS})
            c["flow.refill_prob"] = rf
            cands.append(c)
        cands = cands[1:]
    elif name == "refine2":
        for h, k, trk in itertools.product((10, 20, 40), (3.0, 10.0), (0.5, 0.25)):
            c = {f"flow.lookahead.{L}": {"h": h, "kappa": k} for L in LENS}
            c.update({f"flow.informed.{L}.w": 0.0 for L in LENS})
            c.update({f"flow.impact.{L}.kappa": imp0[L] * 0.5 for L in LENS})
            c["flow.actual_book_tracking"] = trk
            cands.append(c)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="lookahead")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--apply", action="store_true", help="write the best logged candidate into params.json")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.apply:
        rows = [json.loads(x) for x in LOG.read_text().splitlines()]
        rows = [r for r in rows if a.tag is None or r.get("tag") == a.tag]
        best = min(rows, key=lambda r: r["loss"])
        p = cm.load_params()
        for path, val in best["cand"].items():
            node = p
            keys = path.split(".")
            for kk in keys[:-1]:
                node = node.setdefault(kk, {})
            node[keys[-1]] = val
        p["flow"]["lookahead_calibration"] = dict(
            loss=best["loss"], sim=best["sim"], real=best["real"], cand=best["cand"], n_candidates=len(rows),
            note="calibrate_env.py: one train-schedule replication per candidate through features.py + C env; "
                 "loss on baseline markout curves, fill volume, capture, totals, print alignment, trade counts")
        cm.PARAMS.write_text(json.dumps(p, indent=1))
        print("applied", best["cand"], "loss", round(best["loss"], 1))
        return
    P, mk = cm.train_panel()
    base = cm.load_params()
    real = real_targets(mk, P)
    print("real", {k: round(v, 3) for k, v in real.items()}, flush=True)
    for i, cand in enumerate(grid(a.grid, base)):
        t0 = time.time()
        sim = evaluate(cand, base, mk, P, a.seed)
        L, parts = loss(sim, real)
        rec = dict(tag=a.tag or a.grid, i=i, cand=cand, loss=L, parts=parts, sim=sim, real=real)
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        short = {k.replace("flow.", ""): v for k, v in cand.items()
                 if k.endswith(".5m") or k.endswith("5m.kappa") or k.endswith("tracking") or k.endswith("refill_prob")}
        top = sorted(parts.items(), key=lambda kv: -kv[1])[:4]
        print(f"[{i}] loss {L:7.1f} {short} | fv mk {sim['fair_value|mk5']:.2f}/{sim['fair_value|mk30']:.2f}/"
              f"{sim['fair_value|mk60']:.2f} st vol {sim['sym_touch|fill_vol']:.0f} cap {sim['sym_touch|capture']:.2f} "
              f"tot {sim['sym_touch|total']:.0f}/{sim['fair_value|total']:.0f} at0 {sim['align|at0']:.3f} "
              f"| worst {[(k, round(v, 1)) for k, v in top]} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
