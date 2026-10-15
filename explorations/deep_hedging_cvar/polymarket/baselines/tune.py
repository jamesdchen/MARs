"""Tune the baselines on the validation split and evaluate the tuned parameters under every fill
model and latency (EVALUATION.md sections 2-4).

Protocol:
  * Split 1 (val, Aug 23) only; the C env refuses test without the final-run switches and the
    Runner refuses split 2 outright.
  * Every val market played once from offset 0 (runner.Runner).
  * Objective: mean per-market P&L (USD, the env's `pnl`: trading + 20% maker rebates;
    liquidity rewards off, the headline preset) under CENTRAL fills at 1 s latency.
  * Budget: --trials (default 64, the budget of PPO's sweep) Optuna TPE trials per tunable
    baseline (Bergstra et al. 2011), sampler seed --seed. Trial 0 is the untuned point
    `baselines.DEFAULTS` (for fair_value, the REPORT section-6 spec), and it counts toward the
    budget. no_quote has no parameters and is evaluated once.
  * Every evaluation (each trial, and each variant afterwards) is appended to --ledger
    (default baselines/trials.jsonl) with its config, the config's sha256, the code's sha256,
    the sampler seed, the summary metrics and the per-market P&L. Nothing is evaluated on val
    outside the ledger.
  * After tuning, each baseline's best parameters (by the objective) are evaluated under
    {optimistic, central, pessimistic} fills x latency {0, 1, 2} s, plus a size sweep
    (size_mult 0.5 and 2 at central / 1 s), and written to --eval-out.

    python tune.py                                  # all baselines, 64 trials each
    python tune.py --baselines fair_value --trials 8
    python tune.py --eval-only                      # re-evaluate tuned.json's parameters
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baselines as bl  # noqa: E402
from obs_spec import load_spec  # noqa: E402
from runner import PM, Runner, config_hash, core_params  # noqa: E402

VAL = 1
OBJECTIVE = {"split": VAL, "fill_model": "central", "latency": 1, "metric": "pnl_mean"}
VARIANTS = [{"fill_model": f, "latency": L, "env": {}} for f in ("optimistic", "central", "pessimistic")
            for L in (0, 1, 2)]
VARIANTS += [{"fill_model": "central", "latency": 1, "env": {"size_mult": s}} for s in (0.5, 2.0)]


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("baselines.py", "obs_spec.py", "runner.py", "tune.py"):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()


def env_sha(env_dir: Path | None = None) -> dict:
    if env_dir is not None:
        return json.loads((env_dir / "manifest.json").read_text())["files"]
    out = {}
    for f in ["mm.h", "binding.c"] + sorted(x.name for x in PM.glob("binding*.so")):
        out[f] = hashlib.sha256((PM / f).read_bytes()).hexdigest()
    return out


def locked_append(path: Path, rec: dict):
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def locked_update(path: Path, key: str, value):
    """Read-modify-write one top-level key of a JSON file under an exclusive lock."""
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        txt = f.read()
        obj = json.loads(txt) if txt.strip() else {}
        obj[key] = value
        f.seek(0)
        f.truncate()
        f.write(json.dumps(obj, indent=1))
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def suggest(trial, space: dict) -> dict:
    p = {}
    for k, s in space.items():
        if s[0] == "cat":
            p[k] = trial.suggest_categorical(k, s[1])
        else:
            _, lo, hi, log = s
            p[k] = trial.suggest_float(k, lo, hi, log=log)
    return p


class Evaluator:
    """One Runner per (fill model, latency, env overrides); evaluations logged to the ledger."""

    def __init__(self, spec, ledger: Path, sampler_seed: int, split: int = VAL, env_dir=None):
        self.spec, self.ledger, self.seed, self.split = spec, ledger, sampler_seed, split
        self.env_dir = Path(env_dir) if env_dir else None
        self.runners = {}
        self.code = code_sha()
        self.env_sha = env_sha(self.env_dir)

    def runner(self, fill, lat, env):
        key = (fill, lat, json.dumps(env, sort_keys=True))
        if key not in self.runners:
            self.runners[key] = Runner(split=self.split, fill_model=fill, latency=lat, env=env,
                                       env_dir=self.env_dir)
        return self.runners[key]

    def __call__(self, name, params, kind, fill="central", lat=1, env=None, trial=None) -> dict:
        env = env or {}
        r = self.runner(fill, lat, env)
        res = r.run(bl.make(name, self.spec, params))
        cfg = {"baseline": name, "params": params, "env": r.params, "split": self.split,
               "obs_spec": self.spec.source}
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "baseline": name,
               "trial": trial, "params": params, "fill_model": fill, "latency": lat, "env_overrides": env,
               "split": self.split, "sampler_seed": self.seed, "config_sha256": config_hash(cfg),
               "code_sha256": self.code, "env_sha256": self.env_sha, "obs_spec": self.spec.source,
               "env_dir": str(self.env_dir) if self.env_dir else None, "steps": res["steps"], "wall_s": round(res["wall_s"], 3),
               "summary": res["summary"],
               "per_market": {"market": [m["market"] for m in res["markets"]],
                              "pnl": [round(m["pnl"], 6) for m in res["markets"]],
                              "pnl_trading": [round(m["pnl_trading"], 6) for m in res["markets"]],
                              "fills": [m["fills"] for m in res["markets"]]}}
        locked_append(self.ledger, rec)
        return res


def tune_one(ev: Evaluator, name: str, n_trials: int, seed: int, out: Path):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = bl.SEARCH[name]
    trials = []
    t0 = time.time()

    def objective(trial):
        params = {**bl.DEFAULTS[name], **suggest(trial, space)}
        res = ev(name, params, "tune", trial=trial.number)
        s = res["summary"]
        trials.append({"trial": trial.number, "params": params, "pnl_mean": s["pnl_mean"],
                       "cvar95": s["cvar95"], "fills_mean": s["fills_mean"]})
        print(json.dumps({"baseline": name, "trial": trial.number, "params": params,
                          "pnl_mean": round(s["pnl_mean"], 4), "pnl_trading_mean": round(s["pnl_trading_mean"], 4),
                          "fills": round(s["fills_mean"], 1), "cvar95": round(s["cvar95"], 3),
                          "wall_s": round(res["wall_s"], 2)}), flush=True)
        return s["pnl_mean"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial({k: bl.DEFAULTS[name][k] for k in space})
    study.optimize(objective, n_trials=n_trials)
    best = max(trials, key=lambda r: r["pnl_mean"])
    entry = {"best_params": best["params"], "best_trial": best["trial"], "val_pnl_mean": best["pnl_mean"],
             "untuned_params": bl.DEFAULTS[name], "untuned_pnl_mean": trials[0]["pnl_mean"],
             "n_trials": len(trials), "sampler": "TPE", "sampler_seed": seed, "search_space": space,
             "objective": OBJECTIVE, "tune_wall_s": round(time.time() - t0, 1)}
    locked_update(out, name, entry)
    return entry


def evaluate_variants(ev: Evaluator, name: str, params: dict) -> dict:
    out = {}
    for v in VARIANTS:
        res = ev(name, params, "variant", fill=v["fill_model"], lat=v["latency"], env=v["env"])
        tag = f'{v["fill_model"]}/L{v["latency"]}' + "".join(f"/{k}={x}" for k, x in v["env"].items())
        out[tag] = res["summary"]
        s = res["summary"]
        print(json.dumps({"baseline": name, "variant": tag, "pnl_mean": round(s["pnl_mean"], 4),
                          "fills": round(s["fills_mean"], 1)}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["sym_touch", "as_logit", "fair_value", "fv_smart"])
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="TPE sampler seed")
    ap.add_argument("--ledger", default=str(HERE / "trials.jsonl"))
    ap.add_argument("--out", default=str(HERE / "tuned.json"))
    ap.add_argument("--eval-out", default=str(HERE / "eval_val.json"))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-quote", action="store_true", help="also evaluate no_quote (once + variants)")
    ap.add_argument("--split", type=int, default=VAL, choices=[0, VAL],
                    help="1 = val (the protocol); 0 = train, only for pipeline dry runs with other outputs")
    ap.add_argument("--env-dir", default=None, help="env snapshot (runner.make_snapshot) to run")
    ap.add_argument("--make-snapshot", default=None, help="copy the current env to this dir and exit")
    args = ap.parse_args()
    if args.make_snapshot:
        from runner import make_snapshot
        print(json.dumps(make_snapshot(args.make_snapshot)["files"], indent=1))
        return
    if args.split != VAL and Path(args.ledger).resolve() == (HERE / "trials.jsonl").resolve():
        raise SystemExit("--split 0 is a dry run: pass a scratch --ledger/--out/--eval-out")

    ed = Path(args.env_dir) if args.env_dir else None
    if ed is not None:
        spec = load_spec(feat_meta=ed / "feat_meta.json", mm_h=ed / "mm.h", env_params=core_params(ed / "mm.h"))
    else:
        spec = load_spec(env_params=core_params())
    print(f"obs spec: {spec.source}", flush=True)
    ev = Evaluator(spec, Path(args.ledger), args.seed, args.split, ed)
    out = Path(args.out)
    names = list(args.baselines)
    if args.no_quote and "no_quote" not in names:
        names = ["no_quote"] + names
    for name in names:
        if name == "no_quote":
            if not args.eval_only:
                res = ev(name, {}, "tune", trial=0)
                locked_update(out, name, {"best_params": {}, "val_pnl_mean": res["summary"]["pnl_mean"],
                                          "n_trials": 1, "objective": OBJECTIVE})
            params = {}
        elif args.eval_only:
            params = json.loads(out.read_text())[name]["best_params"]
        else:
            params = tune_one(ev, name, args.trials, args.seed, out)["best_params"]
        if not args.no_eval:
            locked_update(Path(args.eval_out), name, {"params": params, "variants": evaluate_variants(ev, name, params)})


if __name__ == "__main__":
    main()
