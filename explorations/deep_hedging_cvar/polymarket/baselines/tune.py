"""Tune the baselines on the validation split and evaluate the tuned parameters under every fill
model and latency (EVALUATION.md sections 2-4, eval/DECISIONS.md D1-D2).

Protocol:
  * Split 1 (val, Aug 23) only. The C env refuses test without the final-run switches, and the
    Runner refuses split 2 outright.
  * The frozen env only: mm.h must hash to FROZEN_MM_H (D1.4), unless --allow-unfrozen is given.
    With --env-dir, a snapshot of the env (runner.make_snapshot) is run instead of the live
    folder; its manifest hashes go into every ledger row.
  * Every val market is played once from offset 0 (runner.Runner).
  * Objective: mean per-market P&L (USD, the env's `pnl`: trading + 20% maker rebates; liquidity
    rewards off, the headline preset) under CENTRAL fills at 1 s latency.
  * Phases (--phases):
      primary  unconstrained (D1.1). Writes tuned.json.
      active   activity-constrained (D1.2): mean fills per market >= 5 and mean two-sided uptime
               (per-market Log `uptime_two_sided`, averaged over markets) >= 0.25. Optuna's TPE
               gets the constraints (Trial.set_constraint; infeasible trials rank below feasible
               ones), and the best is the highest-P&L FEASIBLE trial. Writes tuned_active.json.
  * Budget: --trials (default 64, the budget of PPO's sweep) Optuna TPE trials per tunable
    baseline and phase (Bergstra et al. 2011). Trial 0 is the untuned point `baselines.DEFAULTS`
    (for fair_value, the REPORT section-6 spec), and it counts toward the budget. The sampler
    seed is --seed for primary and --seed + 1 for active, so the two phases do not repeat the same
    random start-up trials. no_quote has no parameters and is evaluated once (primary only).
  * Every evaluation (each trial, and each variant afterwards) is appended to --ledger
    (default baselines/trials.jsonl) with its phase, config, the config's sha256, the code's and
    the env's sha256, the sampler seed, the summary metrics (including the T5 P&L shares and the
    T6 directional share) and the per-market P&L. Nothing is evaluated on val outside the ledger.
    `--mark-pre-freeze` tags every earlier row whose env is not the frozen one with
    "pre_freeze": true (rows are kept; they count toward N).
  * After tuning, each baseline's best parameters are evaluated under {optimistic, central,
    pessimistic} fills x latency {0, 1, 2} s, plus a size sweep (size_mult 0.5 and 2 at central /
    1 s), and written to --eval-out under the phase's key.

    python tune.py --env-dir SNAP --phases primary active
    python tune.py --baselines fair_value --trials 8 --phases active
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 thread: PPO training needs the other cores

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baselines as bl  # noqa: E402
from obs_spec import load_spec  # noqa: E402
from runner import PM, Runner, config_hash, core_params  # noqa: E402

VAL = 1
FROZEN_MM_H = "9189f83adbb3f42ab803aca998ff97954c7cc092ffd65401d2b835abaa98a8e9"  # was 746dd7de... (D5: adds reward_mode 2; modes 0/1 bit-identical)
OBJECTIVE = {"split": VAL, "fill_model": "central", "latency": 1, "metric": "pnl_mean"}
PHASES = {
    "primary": {"out": "tuned.json", "constraints": {}},
    "active": {"out": "tuned_active.json", "constraints": {"fills_mean": 5.0, "uptime_two_sided_mean": 0.25}},
}
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


def locked_update(path: Path, keys: list, value):
    """Read-modify-write obj[keys[0]][keys[1]]... = value in a JSON file under an exclusive lock."""
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        txt = f.read()
        obj = json.loads(txt) if txt.strip() else {}
        d = obj
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        f.seek(0)
        f.truncate()
        f.write(json.dumps(obj, indent=1))
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def mark_pre_freeze(ledger: Path) -> int:
    """Add "pre_freeze": true to every row whose env mm.h is not the frozen one (and to notes
    written before the freeze). Rows are otherwise unchanged; the file is replaced atomically."""
    n = 0
    with open(ledger, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        rows = [json.loads(line) for line in f.read().splitlines() if line.strip()]
        for r in rows:
            if "pre_freeze" not in r:
                r["pre_freeze"] = r.get("env_sha256", {}).get("mm.h") != FROZEN_MM_H
                n += r["pre_freeze"]
        fd, tmp = tempfile.mkstemp(dir=ledger.parent, prefix=".trials.")
        with os.fdopen(fd, "w") as g:
            for r in rows:
                g.write(json.dumps(r, separators=(",", ":")) + "\n")
        os.replace(tmp, ledger)
        fcntl.flock(f, fcntl.LOCK_UN)
    return n


def suggest(trial, space: dict) -> dict:
    p = {}
    for k, s in space.items():
        if s[0] == "cat":
            p[k] = trial.suggest_categorical(k, s[1])
        else:
            _, lo, hi, log = s
            p[k] = trial.suggest_float(k, lo, hi, log=log)
    return p


def violations(summary: dict, constraints: dict) -> list:
    """Constraint values (<= 0 feasible): threshold - achieved, per constraint."""
    return [float(thr - summary[k]) for k, thr in constraints.items()]


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

    def __call__(self, name, params, kind, phase, fill="central", lat=1, env=None, trial=None,
                 seed=None, extra=None) -> dict:
        env = env or {}
        r = self.runner(fill, lat, env)
        res = r.run(bl.make(name, self.spec, params))
        cfg = {"baseline": name, "params": params, "env": r.params, "split": self.split,
               "obs_spec": self.spec.source}
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "phase": phase, "baseline": name,
               "trial": trial, "params": params, "fill_model": fill, "latency": lat, "env_overrides": env,
               "split": self.split, "sampler_seed": self.seed if seed is None else seed,
               "config_sha256": config_hash(cfg), "code_sha256": self.code, "env_sha256": self.env_sha,
               "pre_freeze": self.env_sha.get("mm.h") != FROZEN_MM_H, "obs_spec": self.spec.source,
               "env_dir": str(self.env_dir) if self.env_dir else None, **(extra or {}),
               "steps": res["steps"], "wall_s": round(res["wall_s"], 3), "summary": res["summary"],
               "per_market": {"market": [m["market"] for m in res["markets"]],
                              "pnl": [round(m["pnl"], 6) for m in res["markets"]],
                              "pnl_trading": [round(m["pnl_trading"], 6) for m in res["markets"]],
                              "fills": [m["fills"] for m in res["markets"]],
                              "uptime_two_sided": [round(m["uptime_two_sided"], 4) for m in res["markets"]]}}
        locked_append(self.ledger, rec)
        return res


def tune_one(ev: Evaluator, name: str, n_trials: int, seed: int, out: Path, phase: str):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = bl.SEARCH[name]
    cons = PHASES[phase]["constraints"]
    trials = []
    t0 = time.time()

    def objective(trial):
        params = {**bl.DEFAULTS[name], **suggest(trial, space)}
        res = ev(name, params, "tune", phase, trial=trial.number, seed=seed,
                 extra={"constraints": cons} if cons else None)
        s = res["summary"]
        v = violations(s, cons)
        for k, x in zip(cons, v):
            trial.set_constraint(k, x)                   # TPE ranks infeasible trials below feasible
        feasible = all(x <= 0 for x in v)
        trials.append({"trial": trial.number, "params": params, "pnl_mean": s["pnl_mean"], "cvar95": s["cvar95"],
                       "fills_mean": s["fills_mean"], "uptime_two_sided_mean": s["uptime_two_sided_mean"],
                       "feasible": feasible})
        print(json.dumps({"phase": phase, "baseline": name, "trial": trial.number, "params": params,
                          "pnl_mean": round(s["pnl_mean"], 4), "fills": round(s["fills_mean"], 1),
                          "uptime2": round(s["uptime_two_sided_mean"], 3), "feasible": feasible,
                          "wall_s": round(res["wall_s"], 2)}), flush=True)
        return s["pnl_mean"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({k: bl.DEFAULTS[name][k] for k in space})
    study.optimize(objective, n_trials=n_trials)
    feas = [t for t in trials if t["feasible"]]
    best = max(feas, key=lambda r: r["pnl_mean"]) if feas else None
    entry = {"phase": phase, "constraints": cons, "objective": OBJECTIVE,
             "best_params": best["params"] if best else None,
             "best_trial": best["trial"] if best else None,
             "val_pnl_mean": best["pnl_mean"] if best else None,
             "best_fills_mean": best["fills_mean"] if best else None,
             "best_uptime_two_sided_mean": best["uptime_two_sided_mean"] if best else None,
             "n_feasible": len(feas), "untuned_params": bl.DEFAULTS[name],
             "untuned_pnl_mean": trials[0]["pnl_mean"], "untuned_feasible": trials[0]["feasible"],
             "n_trials": len(trials), "sampler": "TPE" + (" (Trial.set_constraint)" if cons else ""),
             "sampler_seed": seed, "search_space": space, "tune_wall_s": round(time.time() - t0, 1),
             "env_sha256": ev.env_sha, "code_sha256": ev.code}
    if not feas:
        entry["closest_infeasible"] = min(trials, key=lambda r: max(violations(r, cons)))
    locked_update(out, [name], entry)
    return entry


def evaluate_variants(ev: Evaluator, name: str, params: dict, phase: str) -> dict:
    out = {}
    for v in VARIANTS:
        res = ev(name, params, "variant", phase, fill=v["fill_model"], lat=v["latency"], env=v["env"])
        tag = f'{v["fill_model"]}/L{v["latency"]}' + "".join(f"/{k}={x}" for k, x in v["env"].items())
        out[tag] = res["summary"]
        s = res["summary"]
        print(json.dumps({"phase": phase, "baseline": name, "variant": tag, "pnl_mean": round(s["pnl_mean"], 4),
                          "fills": round(s["fills_mean"], 1)}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["sym_touch", "as_logit", "fair_value", "fv_smart"])
    ap.add_argument("--phases", nargs="+", default=["primary"], choices=list(PHASES))
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="TPE sampler seed (active phase: seed + 1)")
    ap.add_argument("--ledger", default=str(HERE / "trials.jsonl"))
    ap.add_argument("--out-dir", default=str(HERE), help="where tuned*.json go")
    ap.add_argument("--eval-out", default=str(HERE / "eval_val.json"))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-quote", action="store_true", help="also evaluate no_quote (once + variants, primary)")
    ap.add_argument("--split", type=int, default=VAL, choices=[0, VAL],
                    help="1 = val (the protocol); 0 = train, only for pipeline dry runs with other outputs")
    ap.add_argument("--env-dir", default=None, help="env snapshot (runner.make_snapshot) to run")
    ap.add_argument("--make-snapshot", default=None, help="copy the current env to this dir and exit")
    ap.add_argument("--mark-pre-freeze", action="store_true", help="tag pre-freeze ledger rows and exit")
    ap.add_argument("--allow-unfrozen", action="store_true")
    args = ap.parse_args()
    if args.make_snapshot:
        from runner import make_snapshot
        print(json.dumps(make_snapshot(args.make_snapshot)["files"], indent=1))
        return
    if args.mark_pre_freeze:
        print(f"tagged {mark_pre_freeze(Path(args.ledger))} rows pre_freeze")
        return
    if args.split != VAL and Path(args.ledger).resolve() == (HERE / "trials.jsonl").resolve():
        raise SystemExit("--split 0 is a dry run: pass a scratch --ledger/--out-dir/--eval-out")

    ed = Path(args.env_dir) if args.env_dir else None
    mm_h = (ed or PM) / "mm.h"
    if hashlib.sha256(mm_h.read_bytes()).hexdigest() != FROZEN_MM_H and not args.allow_unfrozen:
        raise SystemExit(f"{mm_h} is not the frozen env (DECISIONS.md D1.4: sha256 {FROZEN_MM_H[:12]}...)")
    spec = load_spec(feat_meta=(ed / "feat_meta.json") if ed else None,
                     env_md=(ed / "ENV.md") if ed else None, mm_h=mm_h, env_params=core_params(mm_h))
    print(f"obs spec: {spec.source}", flush=True)
    ev = Evaluator(spec, Path(args.ledger), args.seed, args.split, ed)
    out_dir = Path(args.out_dir)
    for phase in args.phases:
        out = out_dir / PHASES[phase]["out"]
        seed = args.seed + (1 if phase == "active" else 0)
        names = list(args.baselines)
        if args.no_quote and phase == "primary" and "no_quote" not in names:
            names = ["no_quote"] + names
        for name in names:
            if name == "no_quote":
                if not args.eval_only:
                    res = ev(name, {}, "tune", phase, trial=0)
                    s = res["summary"]
                    locked_update(out, [name], {"phase": phase, "best_params": {}, "val_pnl_mean": s["pnl_mean"],
                                                "n_trials": 1, "objective": OBJECTIVE, "env_sha256": ev.env_sha})
                params = {}
            elif args.eval_only:
                params = json.loads(out.read_text())[name]["best_params"]
            else:
                params = tune_one(ev, name, args.trials, seed, out, phase)["best_params"]
            if not args.no_eval and params is not None:
                locked_update(Path(args.eval_out), [phase, name],
                              {"params": params, "variants": evaluate_variants(ev, name, params, phase)})


if __name__ == "__main__":
    main()
