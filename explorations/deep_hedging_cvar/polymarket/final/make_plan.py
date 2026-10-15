#!/usr/bin/env python3
"""Write TEST_PLAN.json, the pre-registered plan of the one test run (EVALUATION.md section 5.2).

    python final/make_plan.py                        # -> final/TEST_PLAN.json + plan_sha256.txt
    python final/make_plan.py --best-baseline fv_smart --puffer5 puffer5/selected.json

Reads (paths relative to --root, default polymarket/):
  baselines/tuned.json   tuned parameter set per baseline (no_quote is always added), as tune.py
                         writes it ({name: {"best_params": {...}, "val_pnl_mean": x, ...}}); also
                         {name: {"params": ..., "val": {"total_pnl": x}}} or {name: {...params...}}.
  train/selected.json    the selected PPO config and its seed checkpoints, as train/select.py writes
                         it ({"selected": {"cfg", "cfg_hash", "runs": [{"seed", "ckpt",
                         "ckpt_sha256"}]}}; checkpoint hashes are verified), or {"config",
                         "seeds": [{"seed", "checkpoint"}]}. Default path: train/selected.json, else
                         the only train/runs/sweeps/*/selected.json; --selected overrides.
                         >= 5 seeds required; checkpoint obs_size must equal the env's.
  puffer5/selected.json  optional 5.0 checkpoints: {"hidden", "layers", "seeds": [{"seed", "checkpoint"}]}
  **/trials.jsonl        every ledger under root (final/ excluded): N = all val evaluations.
  data/build/            file hashes; meta.json test_rows (row range + sha256 of the test rows).
  mm.h                   (backend cenv) the env parameters, frozen in the plan: MM_DEFAULTS (the
                         training env) plus --env-params overrides.

The best tuned baseline (the comparator of the primary endpoint) is fixed here: --best-baseline,
else the tuned baseline with the highest val total P&L recorded in tuned.json.

Refuses to overwrite an existing plan (or plan_sha256.txt). Prints sha256(TEST_PLAN.json) and
stores it in plan_sha256.txt next to the plan; record it in eval/EVALUATION.md's amendment log
and in a commit message before `run_test.py --split 2` (run_test checks both).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUILD, CODE_ROOT, FINAL, HEADLINE_ID, MARKOUT_HORIZONS, MID_CLOSE_STC,  # noqa: E402
                    PLAN_SCHEMA, PlanError, S, SPLIT_NAMES, Dataset, code_hashes, count_trials,
                    git_info, make_variants, resolve, utcnow, write_json)
from reduce import FLAG_DOCS, MAX_FLAG_SHARE, MAX_INVENTORY_SHARE  # noqa: E402

META_KEYS = {"val", "val_metrics", "metrics", "config_sha256", "params_sha256", "note", "notes",
             "trials", "n_trials", "score", "name", "selected", "best", "val_total_pnl", "val_pnl",
             "source", "study", "rank", "best_trial", "val_pnl_mean", "untuned_params",
             "untuned_pnl_mean", "sampler", "sampler_seed", "search_space", "objective", "tune_wall_s"}
VAL_KEYS = ("val_total_pnl", "total_pnl", "val_pnl", "pnl", "score")
PARAM_KEYS = ("params", "best_params", "tuned_params")


def _rel(p: Path, root: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(p)


def _val_metric(e: dict):
    """Val P&L of a tuned baseline (total, or the per-market mean: same ranking on one market set)."""
    for k in ("val_total_pnl", "val_pnl", "val_pnl_mean"):
        if isinstance(e.get(k), (int, float)):
            return float(e[k])
    for sub in ("val", "val_metrics", "metrics"):
        d = e.get(sub)
        if isinstance(d, dict):
            for k in VAL_KEYS:
                if isinstance(d.get(k), (int, float)):
                    return float(d[k])
    if isinstance(e.get("score"), (int, float)):
        return float(e["score"])
    return None


def parse_tuned(obj) -> list[dict]:
    if isinstance(obj, dict) and "baselines" in obj:
        obj = obj["baselines"]
    items = []
    def params_of(e):
        for k in PARAM_KEYS:
            if isinstance(e.get(k), dict):
                return e[k]
        return {k: v for k, v in e.items() if k not in META_KEYS}

    if isinstance(obj, list):
        for e in obj:
            params = params_of(e)
            items.append(dict(name=e["name"], params=params, val_metric=_val_metric(e),
                              config_sha256=e.get("config_sha256")))
    elif isinstance(obj, dict):
        for name, e in obj.items():
            if not isinstance(e, dict):
                raise PlanError(f"tuned.json: entry {name!r} is not an object")
            params = params_of(e)
            items.append(dict(name=name, params=params, val_metric=_val_metric(e),
                              config_sha256=e.get("config_sha256")))
    else:
        raise PlanError("tuned.json: unrecognised shape")
    return items


def baseline_methods(root: Path, code_root: Path) -> tuple[list[dict], dict]:
    p = root / "baselines" / "tuned.json"
    if not p.exists():
        raise PlanError(f"{p} missing: baselines must be tuned on val first")
    items = parse_tuned(json.loads(p.read_text()))
    try:
        import baselines as bl
        known = set(bl.FACTORIES)
    except Exception as e:  # noqa: BLE001
        raise PlanError(f"cannot import baselines/baselines.py: {e!r}")
    out = [dict(id="baseline/no_quote", group="baseline", kind="baseline", name="no_quote", params={},
                params_sha256=S.sha256_json({}), val_metric=0.0, tuned=False,
                note="sanity anchor (P&L 0 + reward-only check); not a tuned baseline")]
    for it in items:
        if it["name"] not in known:
            raise PlanError(f"tuned.json names unknown baseline {it['name']!r}; known {sorted(known)}")
        if it["name"] == "no_quote":
            continue
        out.append(dict(id=f"baseline/{it['name']}", group="baseline", kind="baseline", name=it["name"],
                        params=it["params"], params_sha256=S.sha256_json(it["params"]),
                        val_metric=it["val_metric"], tuned=True,
                        ledger_config_sha256=it.get("config_sha256")))
    return out, dict(path=_rel(p, root), sha256=S.sha256_file(p))


def _seed_entries(obj: dict) -> list[dict]:
    if isinstance(obj.get("selected"), dict):             # train/select.py document
        sel = obj["selected"]
        return [dict(seed=r["seed"], checkpoint=r["ckpt"], expected_sha256=r.get("ckpt_sha256"))
                for r in sel.get("runs", [])]
    seeds = obj.get("seeds")
    cks = obj.get("checkpoints")
    out = []
    if isinstance(seeds, list) and seeds and isinstance(seeds[0], dict):
        for e in seeds:
            out.append(dict(e))
    elif isinstance(cks, dict):
        out = [dict(seed=int(k), checkpoint=v) for k, v in cks.items()]
    elif isinstance(cks, list):
        for i, c in enumerate(cks):
            if isinstance(c, dict):
                out.append(dict(c))
            else:
                s = seeds[i] if isinstance(seeds, list) and i < len(seeds) else i
                out.append(dict(seed=s, checkpoint=c))
    else:
        raise PlanError("selected.json: need 'seeds' [{seed, checkpoint}] or 'checkpoints'")
    for e in out:
        if "checkpoint" not in e and "path" in e:
            e["checkpoint"] = e.pop("path")
    return out


def rl_methods(sel_path: Path, root: Path, group: str, default_kind: str, min_seeds: int):
    obj = json.loads(sel_path.read_text())
    entries = _seed_entries(obj)
    if len(entries) < min_seeds:
        raise PlanError(f"{sel_path}: {len(entries)} seeds < {min_seeds} (EVALUATION.md section 4)")
    kind0 = obj.get("kind", default_kind)
    out = []
    for e in entries:
        kind = e.get("kind", kind0)
        m = dict(id=f"{group}/seed{e['seed']}", group=group, kind=kind, seed=e["seed"])
        if kind == "callable":
            m["target"] = e["target"]
            m["standin"] = True
        else:
            ck = Path(e["checkpoint"])
            cands = [ck] if ck.is_absolute() else [sel_path.parent / ck, root / ck, root / "train" / ck]
            hit = next((c for c in cands if c.exists()), None)
            if hit is None:
                raise PlanError(f"checkpoint {ck} of seed {e['seed']} not found (looked in {cands})")
            m["path"] = _rel(hit, root)
            m["sha256"] = S.sha256_file(hit)
            if e.get("expected_sha256") and e["expected_sha256"] != m["sha256"]:
                raise PlanError(f"checkpoint {hit} sha256 differs from the one recorded at selection")
        if kind == "puffer5":
            m["hidden"] = int(e.get("hidden", obj.get("hidden", 128)))
            m["layers"] = int(e.get("layers", obj.get("layers", 4)))
            for k in ("obs_size", "align"):
                if k in e or k in obj:
                    m[k] = int(e.get(k, obj.get(k)))
        out.append(m)
    sel = obj.get("selected") if isinstance(obj.get("selected"), dict) else {}
    cfg = obj.get("config", obj.get("cfg", sel.get("cfg")))
    info = dict(path=_rel(sel_path, root), sha256=S.sha256_file(sel_path), config=cfg,
                config_sha256=obj.get("config_sha256") or sel.get("cfg_hash")
                or (S.sha256_json(cfg) if cfg is not None else None),
                seeds=[m["seed"] for m in out], n_seeds=len(out),
                selection_rule=obj.get("selection_rule", obj.get("rule")),
                val_seed_mean=sel.get("mean"), val_seed_std=sel.get("std"),
                ledger_trials_at_selection=obj.get("n_trials_ledger"),
                standin=any(m.get("standin") for m in out))
    return out, info


def _find_selected(root: Path, selected) -> Path:
    if selected:
        p = Path(selected)
        if not p.exists():
            raise PlanError(f"--selected {p} not found")
        return p
    p = root / "train" / "selected.json"
    if p.exists():
        return p
    sweeps = sorted((root / "train" / "runs" / "sweeps").glob("*/selected.json"))
    if len(sweeps) == 1:
        return sweeps[0]
    if sweeps:
        raise PlanError(f"several selected.json under train/runs/sweeps ({[str(s) for s in sweeps]}); "
                        "pass --selected")
    raise PlanError(f"{p} missing: the selected PPO config and its seeds are needed (--selected PATH)")


def _check_obs_size(methods, root, obs):
    """Fail at plan time if a checkpoint was trained on another observation layout."""
    for m in methods:
        if m["kind"] not in ("ppo3", "puffer5"):
            continue
        path = resolve(m["path"], root, CODE_ROOT)
        if m["kind"] == "ppo3":
            try:
                import torch
            except ImportError:
                continue
            n = int(torch.load(path, map_location="cpu", weights_only=False)["obs_size"])
        else:
            import numpy as np
            import puffernet_mm as p5
            n = m.get("obs_size") or p5.infer_obs_size(np.fromfile(path, "<f4").size, m["hidden"], m["layers"])
        if int(n) != int(obs):
            raise PlanError(f"{m['id']}: checkpoint obs_size {n} != env obs size {obs}")


def build_plan(root=CODE_ROOT, build_dir=BUILD, backend="cenv", backend_options=None,
               best_baseline=None, puffer5=None, min_seeds=5, allow_dirty=False,
               code_root=CODE_ROOT, n_boot=10_000, selected=None, env_params=None) -> dict:
    root, build_dir, code_root = Path(root), Path(build_dir), Path(code_root)
    methods, tuned_info = baseline_methods(root, code_root)
    tuned = [m for m in methods if m["tuned"]]
    if not tuned:
        raise PlanError("tuned.json holds no tuned baseline")
    if best_baseline:
        bid = best_baseline if best_baseline.startswith("baseline/") else f"baseline/{best_baseline}"
        if bid not in {m["id"] for m in tuned}:
            raise PlanError(f"--best-baseline {best_baseline!r} is not a tuned baseline")
        best = dict(id=bid, rule="named on the command line (--best-baseline)")
    else:
        scored = [m for m in tuned if m["val_metric"] is not None]
        if len(scored) != len(tuned):
            raise PlanError("tuned.json lacks val P&L for some baselines; pass --best-baseline")
        top = max(scored, key=lambda m: m["val_metric"])
        best = dict(id=top["id"], rule="highest val total P&L (central fills, 1 s) in tuned.json",
                    val_metric=top["val_metric"])
    sel = _find_selected(root, selected)
    rl, rl_info = rl_methods(sel, root, "rl", "ppo3", min_seeds)
    groups = {"baseline": dict(n=len(methods), source=tuned_info), "rl": rl_info}
    methods += rl
    p5 = Path(puffer5) if puffer5 else (root / "puffer5" / "selected.json")
    if p5.exists():
        rl5, rl5_info = rl_methods(p5, root, "rl5", "puffer5", min_seeds)
        methods += rl5
        groups["rl5"] = rl5_info
    elif puffer5:
        raise PlanError(f"--puffer5 {puffer5} not found")

    backend_options = dict(backend_options or {})
    if backend == "cenv":
        import cenv
        try:
            core = cenv.Core(code_root)
        except FileNotFoundError as e:
            raise PlanError(f"C env sources missing: {e}")
        params = dict(core.defaults)
        bad = set(env_params or {}) - set(params)
        if bad:
            raise PlanError(f"unknown env parameters {sorted(bad)}")
        params.update({k: float(v) for k, v in (env_params or {}).items()})
        backend_options["env_params"] = params
        backend_options["obs_size"] = core.obs
        _check_obs_size(methods, root, core.obs)
    ds = Dataset(build_dir)
    test_idx = ds.split_markets(2)
    lens = ds.markets["contract_len_s"][test_idx]
    data = dict(build_dir=_rel(build_dir, code_root), synthetic=ds.synthetic,
                file_sha256=ds.file_hashes(), test_rows=dict(ds.meta["test_rows"]),
                test_markets=dict(n=int(test_idx.size), indices=[int(i) for i in test_idx],
                                  slugs=[ds.slug(int(i)) for i in test_idx],
                                  by_length={str(int(L)): int((lens == L).sum()) for L in sorted(set(lens.tolist()))}),
                val_markets=dict(n=int(ds.split_markets(1).size)),
                note="test_rows.sha256 is copied from meta.json; run_test.py recomputes it from "
                     "ticks.bin before running (as data/load.py verify does)")
    if not data["test_rows"].get("sha256"):
        raise PlanError("meta.json test_rows has no sha256")

    git = git_info(code_root)
    code = code_hashes(code_root)
    dirty_code = sorted(set(git["dirty_files"]) & set(code)) if git["dirty"] else []
    if git["commit"] is None:
        dirty_code = sorted(code)                     # not a git checkout: nothing is frozen by git
    if dirty_code and not allow_dirty:
        raise PlanError(f"uncommitted evaluation code {dirty_code[:10]}: commit first (or --allow-dirty)")

    variants = make_variants()
    ledger = count_trials(root)
    rl_ids = [m["id"] for m in methods if m["group"] == "rl"]
    plan = dict(
        schema=PLAN_SCHEMA,
        created_utc=utcnow(),
        protocol=dict(path="eval/EVALUATION.md",
                      sha256=S.sha256_file(code_root / "eval" / "EVALUATION.md")
                      if (code_root / "eval" / "EVALUATION.md").exists() else None),
        test_split=dict(id=2, name=SPLIT_NAMES[2], dates="2026-08-24 UTC",
                        dry_run_split=dict(id=1, name=SPLIT_NAMES[1])),
        git=git, allow_dirty=bool(allow_dirty), dirty_code_at_plan_time=dirty_code,
        code_sha256=code,
        artifact_root=_rel(root, code_root),
        data=data,
        backend=dict(name=backend, options=backend_options or {},
                     of_record=backend == "cenv",
                     note="cenv = the C env (polymarket/env.py over mm.h); pyreplay is a stand-in, "
                          "never a number of record"),
        execution=dict(deterministic_actions=True, start_offset=0, episodes="every market of the split once",
                       run_once=True),
        methods=methods,
        groups=groups,
        primary_group="rl",
        best_baseline=best,
        variants=variants,
        headline_variant=HEADLINE_ID,
        metrics=[
            "P&L per day and per market (mean, median, distribution), with and without rewards+rebates",
            "P&L by contract length and coin",
            "hourly MTM Sharpe (per hour, not annualised) with block-bootstrap CI",
            "CVaR95 of per-market P&L (stats.cvar)",
            "max |inventory| per market and max combined portfolio delta (sum inv x dp/dlnS, USD)",
            "fill rate (fills per quoted market-minute) and our share of taker volume",
            "quote uptime (two-sided; one-sided separately) over live seconds",
            "attribution (stats.attribute_pnl): spread capture, adverse selection (+60 s), inventory "
            "carry, resolution, fees, rebates, rewards; markouts at 1/5/10/30/60 s",
            "exploit-flag table (stats.flag_pnl_share)", "concentration (stats.concentration)",
            "fill/latency sensitivity grid, size sweep, mirror test", "seed mean, min/max, std",
            "trial count N and deflated Sharpe (stats.deflated_sharpe)",
            "test-day vs train/val realised vol, trade counts, spread (T9)"],
        units=dict(pnl="USDC", mtm_grid="per-second inventory marked to the mark (feat.bin `mark`: book "
                   "mid if two-sided and valid, else the fair value), to the settlement value on the "
                   "closing row; summed over all live markets into zero-filled UTC bins",
                   bins_seconds=dict(primary=3600, secondary=900)),
        bootstrap=dict(kind="stationary", method="percentile", alpha=0.05, n_boot=int(n_boot),
                       seed=20260824, block_length="Politis-White (stats.optimal_block_length) on the "
                       "series, floored at 2 h (2 bins at 1 h, 8 bins at 15 min)", min_block_hours=2),
        attribution=dict(horizon_s=60, markout_horizons_s=list(MARKOUT_HORIZONS),
                         mid_close_seconds_to_close=MID_CLOSE_STC,
                         identity="components sum to total; checked against the env's cash accounting"),
        exploit_detectors=dict(flags=FLAG_DOCS, max_single_flag_share=MAX_FLAG_SHARE,
                               invalid_if="any single flag > 25% of total fill P&L, or fill P&L "
                                          "excluding all flagged fills <= 0 while the total is > 0"),
        directional=dict(max_inventory_share=MAX_INVENTORY_SHARE,
                         rule="directional if inventory/resolution P&L > 0 and > 50% of total P&L "
                              "(total > 0) or > 50% of the positive components",
                         mirror_test="mirrored (Up<->Down) minus original hourly P&L, paired block-"
                                     "bootstrap 95% CI; 'unchanged' if it contains 0"),
        concentration=dict(ks=[1, 3, 5, 10], ex_top_markets=5, ex_top_hours=3),
        primary_endpoint=dict(
            description="test-day total P&L of the selected RL config (seed mean) minus the best tuned "
                        "baseline, central fills, 1 s latency, rewards+rebates included",
            group="rl", methods=rl_ids, comparator=best["id"], variant=HEADLINE_ID,
            statistic="sum over hourly bins of (seed-mean RL MTM P&L - baseline MTM P&L)",
            ci="95% stationary block bootstrap over hourly bins spanning all markets "
               "(stats.paired_difference_ci, statistic='total')"),
        decision_rule=dict(success_requires_all=[
            dict(id="ci_lower_gt_0", text="primary endpoint 95% CI lower bound > 0"),
            dict(id="pessimistic_1s_gt_0", text="point estimate > 0 under pessimistic fills at 1 s",
                 variant="pessimistic_lat1_x1"),
            dict(id="central_2s_gt_0", text="point estimate > 0 under central fills at 2 s",
                 variant="central_lat2_x1"),
            dict(id="not_directional", text="RL (seed mean) not classified directional (T6)"),
            dict(id="no_exploit_flag", text="no invalidating exploit flag (T5), seed-pooled fills"),
            dict(id="ex_top_positive", text="RL seed-mean P&L excluding the top 5 markets and excluding "
                                            "the top 3 hours both > 0 (T7)")],
            note="everything else is secondary and descriptive; 0 s latency is never the headline"),
        ledger=ledger,
        seeds_rule="selected config only, >= 5 seeds, reported as mean with min/max/std; never the best seed",
        n_runs=len(methods) * len(variants),
    )
    return plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(CODE_ROOT), help="artifact root (polymarket/)")
    ap.add_argument("--build-dir", default=str(BUILD))
    ap.add_argument("--out", default=str(FINAL / "TEST_PLAN.json"))
    ap.add_argument("--backend", default="cenv", choices=["cenv", "pyreplay"])
    ap.add_argument("--backend-options", default="{}", help="JSON object")
    ap.add_argument("--best-baseline", default=None)
    ap.add_argument("--puffer5", default=None, help="selected.json of 5.0 checkpoints (optional)")
    ap.add_argument("--selected", default=None, help="train/select.py output (default: train/selected.json, "
                                                     "else the only train/runs/sweeps/*/selected.json)")
    ap.add_argument("--env-params", default="{}", help="JSON overrides of the C env parameters "
                                                       "(default: mm.h MM_DEFAULTS, as in training)")
    ap.add_argument("--min-seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--allow-dirty", action="store_true", help="plan with uncommitted code (recorded)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    hp = out.parent / "plan_sha256.txt"
    for p in (out, hp):
        if p.exists():
            print(f"refusing to overwrite {p}: a plan is written once (delete it deliberately and "
                  "record why in the amendment log)", file=sys.stderr)
            return 2
    try:
        plan = build_plan(a.root, a.build_dir, a.backend, json.loads(a.backend_options), a.best_baseline,
                          a.puffer5, a.min_seeds, a.allow_dirty, n_boot=a.n_boot, selected=a.selected,
                          env_params=json.loads(a.env_params))
    except PlanError as e:
        print(f"make_plan: {e}", file=sys.stderr)
        return 1
    write_json(out, plan)
    h = S.sha256_file(out)
    hp.write_text(f"{h}  {out.name}\n")
    n_rl = sum(m["group"] == "rl" for m in plan["methods"])
    print(f"wrote {out}: {len(plan['methods'])} methods ({n_rl} RL seeds), {len(plan['variants'])} variants, "
          f"{plan['n_runs']} runs; best baseline {plan['best_baseline']['id']}; "
          f"ledger N = {plan['ledger']['n_trials']} val trials ({plan['ledger']['n_distinct_configs']} configs)")
    print(f"TEST_PLAN.json sha256 = {h}")
    print(f"stored in {hp}. Before `MM_FINAL=1 run_test.py --split 2`: add it to the amendment log of "
          "eval/EVALUATION.md and to a commit message (run_test.py checks both).")
    if plan["groups"]["rl"].get("standin"):
        print("WARNING: the RL methods are stand-ins (callable), not trained checkpoints", file=sys.stderr)
    if os.environ.get("MM_FINAL"):
        print("note: MM_FINAL is set; make_plan does not need it", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
