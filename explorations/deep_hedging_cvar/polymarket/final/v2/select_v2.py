#!/usr/bin/env python3
"""D8 selection across every v2 ledger (val v2 only), under both T6 definitions.

    python final/v2/select_v2.py            # -> final/v2/selection_v2.json (+ prints tables)

Reads only ledger rows (train/trials.jsonl, train/workers/w*/trials.jsonl). Kept rows: split 1,
val.n_markets == 374, data_dir ending in data/build_v2, central fills at 1 s (variant central_1s,
or a length-filter row derived from a central_1s evaluation). Rows that are exact duplicates
(same config hash, seed, filter and identical P&L) are counted once.

A method is a (source, config, length filter) group of single-seed evaluations, or one
ensemble evaluation (members listed). For each method, D1.3 (DECISIONS.md D1.3 + D2 + train/
evaluator.protocol_flags conventions: T6 and T5 count as 0 when the P&L / fill edge is <= 0)
is computed two ways:

  per-seed  ('old', train/v2search.py before commit 4f17e3a): fills = mean of per-seed
            fills/market; T6 = mean of per-seed t6_share (undefined -> 0); T5 = mean of per-seed
            t5_max_share (undefined -> 0; the old v2search check had no T5 term at all).
  pooled    ('new', as final/reduce.py judges the seed mean): fills = mean fills/market;
            T6 = mean inventory P&L / mean P&L when the mean P&L > 0, else 0; T5 = max over the D2
            flags of (summed flagged fill edge / summed fill edge) when the summed edge > 0, else 0.

  final     pooled as 'new', with the directional verdict of final/reduce.py directional_verdict
            (the rule frozen in the v1 TEST_PLAN.json and used by the test's D1.3 check):
            directional if inventory P&L > 0 and (> 50% of a positive total, or > 50% of the sum
            of the positive components spread capture, 60 s markout, inventory, rebates, rewards).
            t6_pos = that positive-component share.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[2]
LEDGERS = {"local": P / "train" / "trials.jsonl",
           **{f"w{i}": P / "train" / "workers" / f"w{i}" / "trials.jsonl" for i in range(1, 6)}}
T5_FLAGS = ("inspread", "crossed", "stale", "onesided", "last30")
ALL = "5m+15m+4h"


def filt_of(row):
    q = row.get("quote_lengths")
    if q:
        return "+".join(q)
    m = re.search(r"\|quote:([0-9mh+]+)", (row.get("name") or "") + "|" + (row.get("variant") or ""))
    return m.group(1) if m else ALL


def central_1s(row):
    v = row.get("variant") or ""
    return v == "central_1s" or v.startswith("central_1s|")


def method_key(src, r):
    """(method id, seed or None for an ensemble evaluation, members)."""
    tag, name, f = r.get("tag") or "", r.get("name") or "", filt_of(r)
    if src == "local" and tag in ("v2s1", "v2s1_filter"):
        return f"v2s1/{CFG_NAME.get(r['cfg_hash'], r['cfg_hash'][:8])}|{f}", r["seed"], None
    if src == "local" and tag in ("v2_ensemble", "v2s1_ens"):
        grp, sub = name.split("|")[0].split("/", 1)
        prefix = "v2s1_ens/" if tag == "v2s1_ens" else ""
        if sub.startswith("ensemble"):
            n = int(sub.rsplit("_", 1)[1])
            return f"{prefix}{grp}/{sub}|{f}", None, n
        seed = int(re.search(r"(\d+)$", sub).group(1))
        return f"{grp}/seeds|{f}", seed, None
    if src == "local" and tag == "v2_baselines":
        return None, None, None                      # comparators, not candidates (D1)
    if src == "w1":
        return f"w1/{W1_NAME.get(r['cfg_hash'], tag)}|{f}", r["seed"], None
    if src in ("w2", "w3", "w5") and r.get("cfg_hash"):
        return f"{src}/{tag[len(src) + 1:]}|{f}", r["seed"], None
    if src == "w4" and tag in ("w4_q1best", "w4_bc"):
        return f"w4/{tag[3:]}|{f}", r["seed"], None
    if src == "w4" and re.fullmatch(r"t\d+_m\d+", name):  # quoter TPE trials x 3 edge-model seeds (t<k>_m<seed>)
        trial, m = name.split("_m")
        return f"w4/{tag[3:]}:{trial}|{f}", int(m), None
    if src == "w4" and r.get("cfg_hash"):             # w4 RL runs (train_ppo.py rows)
        return f"w4/{tag[3:]}|{f}", r["seed"], None
    if src == "w5":                                   # teachers (baselines) / BC students, 1 eval each
        if tag == "w5_teachers":
            return None, None, None
        return f"w5/{tag[3:]}:{name}|{f}", 0, None
    return f"{src}/{tag}:{name}|{f}", r.get("seed"), None


def load_names():
    plan = json.load(open(P / "train" / "runs" / "v2" / "v2s1" / "plan.json"))
    import hashlib  # noqa: F401
    names = {}
    for d in (P / "train" / "runs" / "v2" / "v2s1").glob("c*_s0/result.json"):
        r = json.load(open(d))
        names[r["cfg_hash"]] = d.parent.name.split("_s")[0]
    w1 = {}
    for l in open(P / "train" / "workers" / "w1" / "trials.jsonl"):
        r = json.loads(l)
        w1.setdefault(r["cfg_hash"], r["tag"][3:])  # first tag that used the hash (t04 before t05)
    return names, w1, plan


CFG_NAME, W1_NAME, _ = load_names()


def flags(v):
    """Per-evaluation numbers used by both definitions."""
    pnl = v["pnl_total"]
    inv = v.get("sum/inventory_pnl") or 0.0
    comps = [v.get(k) or 0.0 for k in ("sum/spread_capture", "sum/markout_60", "sum/inventory_pnl",
                                         "sum/rebates", "sum/liq_rewards")]
    edge = (v.get("sum/spread_capture") or 0.0) + (v.get("sum/markout_60") or 0.0)
    fl = {f: v.get(f"sum/pnl_{f}") or 0.0 for f in T5_FLAGS}
    t6 = v.get("t6_share")
    t5 = v.get("t5_max_share")
    return dict(pnl=pnl, inv=inv, pos=sum(max(c, 0.0) for c in comps), edge=edge, fl=fl,
                fills=v.get("fills_per_market") or 0.0,
                t6=0.0 if (t6 is None or t6 != t6 or pnl <= 0) else t6,
                t5=0.0 if (t5 is None or t5 != t5) else t5,
                len={L: v.get(f"len/{L}/pnl_total") or 0.0 for L in ("5m", "15m", "4h")},
                n=v["n_markets"])


def summarise(mid, src, evals, members):
    pnl = np.array([e["pnl"] for e in evals])
    n = len(evals)
    mean = float(pnl.mean())
    fills = float(np.mean([e["fills"] for e in evals]))
    # per-seed ('old')
    t6_old = float(np.mean([e["t6"] for e in evals]))
    t5_old = float(np.mean([e["t5"] for e in evals]))
    # pooled ('new')
    inv = float(np.mean([e["inv"] for e in evals]))
    t6_new = inv / mean if mean > 0 else 0.0
    edge = float(np.sum([e["edge"] for e in evals]))
    t5_new = max((sum(e["fl"][f] for e in evals) / edge for f in T5_FLAGS), default=0.0) if edge > 0 else 0.0
    pos = float(np.mean([e["pos"] for e in evals]))
    t6_pos = max(inv, 0.0) / pos if pos > 0 else 0.0
    directional = bool(inv > 0 and ((mean > 0 and inv / mean > 0.5) or (pos > 0 and inv / pos > 0.5)))
    seeds_ok = (n >= 3) if members is None else (members >= 3)
    return dict(method=mid, source=src, n_evals=n, members=members, seeds_ok=bool(seeds_ok),
                ensemble=members is not None, mean=mean,
                se=float(pnl.std(ddof=1) / math.sqrt(n)) if n > 1 else None, per_seed=pnl.round(2).tolist(),
                fills=fills, t6_old=t6_old, t5_old=t5_old, t6_new=t6_new, t5_new=t5_new, t6_pos=t6_pos,
                d13_old=bool(fills >= 5 and t6_old <= 0.5 and t5_old <= 0.25),
                d13_new=bool(fills >= 5 and t6_new <= 0.5 and t5_new <= 0.25),
                directional_final=directional,
                d13_final=bool(fills >= 5 and not directional and t5_new <= 0.25),
                len={L: float(np.mean([e["len"][L] for e in evals])) for L in ("5m", "15m", "4h")})


def main():
    groups = defaultdict(dict)
    srcs = {}
    members = {}
    audit = defaultdict(lambda: defaultdict(int))
    for src, path in LEDGERS.items():
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("val") or {}
            if r.get("split") != 1 or "pnl_total" not in v:
                continue
            if not central_1s(r):
                audit[src]["other_variant"] += 1
                continue
            if v.get("n_markets") != 374 or not str(r.get("data_dir", "")).rstrip("/").endswith("data/build_v2"):
                audit[src]["not_v2_val"] += 1
                continue
            mid, seed, mem = method_key(src, r)
            if mid is None:
                audit[src]["comparator_rows"] += 1
                continue
            audit[src]["kept"] += 1
            key = ("ens",) if mem is not None else seed
            prev = groups[mid].get(key)
            if prev is not None and abs(prev["pnl"] - v["pnl_total"]) > 1e-9:
                raise SystemExit(f"{mid} seed {seed}: two different evaluations")
            if prev is not None:
                audit[src]["duplicate_rows"] += 1
            groups[mid][key] = flags(v)
            srcs[mid], members[mid] = src, mem
    rows = [summarise(m, srcs[m], list(g.values()), members[m]) for m, g in groups.items()]
    out = dict(rows=sorted(rows, key=lambda r: -r["mean"]), audit={k: dict(v) for k, v in audit.items()})
    for defn in ("old", "new", "final"):
        for ens in (False, True):
            ok = [r for r in rows if r["seeds_ok"] and r[f"d13_{defn}"] and (ens or not r["ensemble"])]
            best = max(ok, key=lambda r: r["mean"], default=None)
            out[f"winner_{defn}_{'with' if ens else 'no'}_ensembles"] = best and best["method"]
    (Path(__file__).parent / "selection_v2.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["audit"], indent=1))
    for k in [k for k in out if k.startswith("winner")]:
        print(k, out[k])
    fmt = lambda x: "-" if x is None else f"{x:+.1f}"  # noqa: E731
    print("\ntop eligible (>=3 seeds) by mean, either definition:")
    for r in out["rows"]:
        if r["seeds_ok"] and (r["d13_old"] or r["d13_new"] or r["d13_final"]):
            print(f"{r['method']:45s} n={r['n_evals']} mem={r['members']} mean={r['mean']:+8.1f} se={fmt(r['se'])}"
                  f" fills={r['fills']:5.1f} T6old={r['t6_old']:.2f} T6new={r['t6_new']:.2f} T5old={r['t5_old']:.2f}"
                  f" T5new={r['t5_new']:.2f} T6pos={r['t6_pos']:.2f} old={r['d13_old']} new={r['d13_new']} final={r['d13_final']}")


if __name__ == "__main__":
    main()
