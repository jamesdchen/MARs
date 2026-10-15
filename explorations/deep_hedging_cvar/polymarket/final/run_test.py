#!/usr/bin/env python3
"""Run the pre-registered plan once (EVALUATION.md section 5.3-5.4).

    python final/run_test.py                     # dry run on val (split 1, the default)
    MM_FINAL=1 python final/run_test.py --split 2   # THE test run: once

Before anything is evaluated (the data checks run on a private copy of the frozen files in
results/.snapshot/, which is what the env then reads, so a rebuild of data/build/ during the run
cannot change or crash it):
  1. sha256(TEST_PLAN.json) == plan_sha256.txt                              (both splits)
  2. split 2 only: MM_FINAL=1; the plan hash appears in eval/EVALUATION.md's amendment log
     and in a git commit message (--skip-registration-check REASON overrides, recorded)
  3. data: every file hash in the plan matches data/build; split 2 only: the sha256 of the test
     rows of ticks.bin is recomputed (as data/load.py does) and must equal meta.json and the
     plan; the test markets must be exactly the plan's list
  4. code: every source hash in the plan matches (--accept-code-change REASON overrides, recorded)
  5. run once: if results/runs.jsonl already holds a run of this split, refuse unless
     --force-rerun REASON (the reason is recorded in the run record and in the results)
  6. split 2 needs an env of record (backend cenv) unless the dataset is synthetic (tests)

Then every method x variant of the plan runs over every market of the split once,
deterministic actions, start offset 0, and results/<split>/ receives
  markets.parquet, minutes.parquet, fills.parquet, portfolio.parquet   (reduce.py)
  runs.json            per-run aggregates (attribution, exploit flags, directional numbers)
  test_report.json     (split 2) / dryrun_report.json (split 1): plan hash, checks, timings, reruns
and results/runs.jsonl gets a started / completed / crashed record. The mirror variant runs on
a mirrored copy of the snapshot (mirror.py). With the C env (cenv.py) every market is also
replayed in the C harness and in reference.py; their agreement is recorded per market and in
the report (`replay`). Completed (method, variant)
shards are kept in results/<split>/shards/ until the end, so a crash restart (which needs
--force-rerun REASON) can --resume them.

Dry-run-only options (split 1): --methods/--variants/--max-markets subsets, --results-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CODE_ROOT, FINAL, RESULTS_SCHEMA, SPLIT_NAMES, PlanError, Dataset,  # noqa: E402
                    append_jsonl, clean_json, code_hashes, git_info, git_log_contains, load_plan,
                    read_jsonl, resolve, utcnow, write_json)
import adapters  # noqa: E402
from reduce import RunAccumulator  # noqa: E402

TABLES = ("markets", "minutes", "fills", "portfolio")


class RunRefused(RuntimeError):
    pass


def _shard_name(method, variant):
    return f"{method.replace('/', '__')}@{variant}"


def preflight(plan, plan_hash, split, args, ds) -> dict:
    """All checks before the first step. Returns the check record (raises RunRefused)."""
    chk = preflight_meta(plan, plan_hash, split, args, ds.synthetic)
    chk.update(preflight_data(plan, split, args, ds))
    return chk


def preflight_meta(plan, plan_hash, split, args, synthetic) -> dict:
    """Checks that read no data: MM_FINAL, dry-run options, registration, env of record."""
    chk = dict(plan_sha256=plan_hash, split=split)
    if split == 2:
        if os.environ.get("MM_FINAL") != "1":
            raise RunRefused("split 2 (test) requires MM_FINAL=1 in the environment")
        if args.methods or args.variants or args.max_markets:
            raise RunRefused("--methods/--variants/--max-markets are dry-run options; the test runs the full plan")
        reg = dict(amendment_log=None, commit_message=None, skipped=args.skip_registration_check)
        if not args.skip_registration_check:
            ev = CODE_ROOT / "eval" / "EVALUATION.md"
            reg["amendment_log"] = ev.exists() and plan_hash in ev.read_text()
            reg["commit_message"] = git_log_contains(plan_hash, CODE_ROOT)
            if not (reg["amendment_log"] and reg["commit_message"]):
                raise RunRefused(f"plan hash {plan_hash} not registered (EVALUATION.md amendment log: "
                                 f"{reg['amendment_log']}, git commit message: {reg['commit_message']}); "
                                 "register it or pass --skip-registration-check REASON")
        chk["registration"] = reg
        if not plan["backend"].get("of_record") and not synthetic:
            raise RunRefused(f"backend {plan['backend']['name']!r} is a stand-in: no test run on real data")
    elif split != 1:
        raise RunRefused("--split must be 1 (val dry run) or 2 (test)")
    return chk


def preflight_data(plan, split, args, ds) -> dict:
    """Hashes of the data actually used (the private snapshot), test rows, code."""
    chk = {}
    want = plan["data"]["file_sha256"]
    have = ds.file_hashes()
    bad = {f: dict(plan=want[f], now=have.get(f)) for f in want if have.get(f) != want[f]}
    if bad:
        raise RunRefused(f"data files changed since the plan: {sorted(bad)}")
    chk["data_files"] = dict(ok=True, n=len(want))
    tr = plan["data"]["test_rows"]
    if ds.meta["test_rows"]["sha256"] != tr["sha256"] or ds.meta["test_rows"]["row_lo"] != tr["row_lo"] \
            or ds.meta["test_rows"]["row_hi"] != tr["row_hi"]:
        raise RunRefused("meta.json test_rows differ from the plan")
    if split == 2:
        t0 = time.time()
        h = ds.test_rows_sha256()
        if h != tr["sha256"]:
            raise RunRefused(f"test rows sha256 {h} != expected {tr['sha256']}: test data changed")
        idx = [int(i) for i in ds.split_markets(2)]
        if idx != plan["data"]["test_markets"]["indices"]:
            raise RunRefused("test markets differ from the plan's list")
        chk["test_rows"] = dict(ok=True, sha256=h, seconds=round(time.time() - t0, 2))
    else:
        chk["test_rows"] = dict(ok=None, note="dry run: test rows not read")

    # code
    now = code_hashes(CODE_ROOT)
    plan_code = plan["code_sha256"]
    diff = sorted(f for f in set(plan_code) | set(now) if plan_code.get(f) != now.get(f))
    chk["code"] = dict(ok=not diff, changed=diff, accepted_reason=args.accept_code_change)
    if diff and not args.accept_code_change:
        raise RunRefused(f"evaluation code changed since the plan: {diff[:10]} "
                         "(--accept-code-change REASON if the change touches no evaluation logic)")

    return chk


def run(args) -> int:
    plan_path = Path(args.plan)
    plan, plan_hash = load_plan(plan_path)
    split = int(args.split)
    sname = SPLIT_NAMES[split]
    results_root = Path(args.results_dir)
    out_dir = results_root / sname
    ledger = results_root / "runs.jsonl"
    build_dir = resolve(plan["data"]["build_dir"], CODE_ROOT)
    root = resolve(plan.get("artifact_root", "."), CODE_ROOT)
    prior = [r for r in read_jsonl(ledger)[0] if r.get("split") == split]
    if prior and not args.force_rerun:
        raise RunRefused(f"{len(prior)} earlier run record(s) of split {split} in {ledger} "
                         f"(last: {prior[-1].get('status')} at {prior[-1].get('started_utc')}); the plan runs "
                         "once: --force-rerun REASON to run again (recorded)")
    if args.resume and not args.force_rerun:
        raise RunRefused("--resume restarts a crashed run and needs --force-rerun REASON")
    chk = preflight_meta(plan, plan_hash, split, args, bool(plan["data"].get("synthetic")))
    # private copy of the frozen data: hashed and used byte for byte, immune to rebuilds
    snap_root = results_root / ".snapshot"
    snap = snap_root / sname
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True)
    for f in plan["data"]["file_sha256"]:
        shutil.copyfile(build_dir / f, snap / f)
    ds = Dataset(snap)
    chk.update(preflight_data(plan, split, args, ds))
    chk["snapshot"] = dict(source=str(build_dir), n_files=len(plan["data"]["file_sha256"]))

    methods = plan["methods"]
    variants = plan["variants"]
    if args.methods:
        methods = [m for m in methods if m["id"] in args.methods or m["group"] in args.methods]
    if args.variants:
        variants = [v for v in variants if v["id"] in args.variants]
    markets = ds.split_markets(split)
    if args.max_markets:
        markets = markets[: args.max_markets]
    if not methods or not variants or not markets.size:
        raise RunRefused("nothing to run")

    run_id = f"{sname}-{utcnow().replace(':', '')}"
    rec = dict(run_id=run_id, split=split, split_name=sname, plan_sha256=plan_hash, plan_path=str(plan_path),
               status="started", started_utc=utcnow(), force_rerun_reason=args.force_rerun,
               resume=bool(args.resume), prior_runs=[r.get("run_id") for r in prior],
               git=git_info(CODE_ROOT), backend=plan["backend"]["name"], argv=sys.argv,
               mm_final=os.environ.get("MM_FINAL"), checks=chk,
               subset=dict(methods=args.methods, variants=args.variants, max_markets=args.max_markets))
    append_jsonl(ledger, rec)
    t_start = time.time()
    shards = out_dir / "shards"
    try:
        if shards.exists() and not args.resume:
            shutil.rmtree(shards)
        shards.mkdir(parents=True, exist_ok=True)
        opts = dict(plan["backend"].get("options") or {}, work_dir=str(snap_root / "work"))
        backend = adapters.make_backend(plan["backend"]["name"], ds, opts)
        mirror = {}

        def mirrored():
            """Backend + dataset on the mirrored copy of the snapshot (built once, hashed)."""
            if not mirror:
                import mirror as mirror_mod
                mdir = snap_root / f"{sname}_mirror"
                if mdir.exists():
                    shutil.rmtree(mdir)
                chk["mirror_data"] = mirror_mod.mirror_build(snap, mdir)
                mds = Dataset(mdir)
                mirror.update(ds=mds, backend=adapters.make_backend(plan["backend"]["name"], mds, opts))
            return mirror["ds"], mirror["backend"]
        ctx = dict(spec=lambda: backend.obs_spec(split), obs_size=None, root=root)
        ctx["obs_size"] = backend.obs_size(split)
        t_range = ds.split_time_range(split, 3600)
        aggs, timings = [], []
        n_runs = len(methods) * len(variants)
        i = 0
        for m in methods:
            policy = None
            for v in variants:
                i += 1
                name = _shard_name(m["id"], v["id"])
                done = shards / f"{name}.agg.json"
                if args.resume and done.exists():
                    a = json.loads(done.read_text())
                    if a.get("plan_sha256") == plan_hash:
                        aggs.append(a["agg"])
                        timings.append(a["timing"])
                        print(f"[{i}/{n_runs}] {m['id']} {v['id']}: resumed")
                        continue
                if policy is None:
                    policy = adapters.make_policy(m, ctx)
                t0 = time.time()
                if v.get("mirror"):
                    mds, mbe = mirrored()
                    traces = mbe.run(policy, markets, dict(v, mirror=False), split, allow_test=(split == 2))
                    acc = RunAccumulator(mds, split, m["id"], v, t_range, mirror_in_data=True)
                else:
                    traces = backend.run(policy, markets, v, split, allow_test=(split == 2))
                    acc = RunAccumulator(ds, split, m["id"], v, t_range)
                for tr in traces:
                    acc.add(tr)
                res = acc.finish()
                for t in TABLES:
                    res[t].to_parquet(shards / f"{name}.{t}.parquet", index=False)
                tm = dict(method=m["id"], variant=v["id"], seconds=round(time.time() - t0, 2))
                write_json(done, clean_json(dict(plan_sha256=plan_hash, agg=res["agg"], timing=tm)))
                aggs.append(clean_json(res["agg"]))
                timings.append(tm)
                rp = res["agg"].get("replay")
                warn = f"  REPLAY MISMATCH in {rp['n_not_ok']} markets" if rp and rp["n_not_ok"] else ""
                print(f"[{i}/{n_runs}] {m['id']} {v['id']}: P&L {res['agg']['pnl']:+.2f} "
                      f"fills {res['agg']['n_fills']} ({tm['seconds']:.1f}s){warn}", flush=True)
                if split == 1 and not args.no_dryrun_ledger:
                    append_jsonl(out_dir / "dryrun_trials.jsonl", dict(
                        run_id=run_id, split=1, kind="final_dry_run", method=m["id"], variant=v["id"],
                        plan_sha256=plan_hash, pnl=res["agg"]["pnl"],
                        note="post-freeze dry run of the final plan; not used for selection"))
        # consolidate
        for t in TABLES:
            keep = {(_shard_name(m["id"], v["id"])) for m in methods for v in variants}
            parts = [pd.read_parquet(shards / f"{k}.{t}.parquet") for k in sorted(keep)
                     if (shards / f"{k}.{t}.parquet").exists()]
            df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            df.to_parquet(out_dir / f"{t}.parquet", index=False)
        write_json(out_dir / "runs.json", clean_json(dict(plan_sha256=plan_hash, runs=aggs)))
        rps = [a["replay"] for a in aggs if a.get("replay")]
        report = dict(
            schema=RESULTS_SCHEMA, run_id=run_id, split=split, split_name=sname, dry_run=split != 2,
            plan_sha256=plan_hash, plan_path=str(plan_path), created_utc=utcnow(),
            backend=dict(name=plan["backend"]["name"], of_record=bool(getattr(backend, "of_record", False))),
            dataset=dict(build_dir=str(build_dir), synthetic=ds.synthetic), checks=chk,
            t_range=list(t_range), n_markets=int(markets.size), markets=[int(x) for x in markets],
            methods=[m["id"] for m in methods], variants=[v["id"] for v in variants],
            complete_plan=(len(methods) == len(plan["methods"]) and len(variants) == len(plan["variants"])
                           and markets.size == ds.split_markets(split).size),
            reruns=[dict(reason=args.force_rerun, prior_runs=rec["prior_runs"], resume=bool(args.resume))]
            if args.force_rerun else [],
            replay=dict(checked=bool(rps), failures=[dict(method=a["method"], variant=a["variant"], **a["replay"])
                                                     for a in aggs if a.get("replay") and a["replay"]["n_not_ok"]],
                        state_max_abs_diff=max((r["state_max_abs_diff"] for r in rps), default=None),
                        log_pnl_max_abs_diff=max((r["log_pnl_max_abs_diff"] for r in rps), default=None)),
            timings=timings, seconds=round(time.time() - t_start, 1),
            headline={a["method"]: a["pnl"] for a in aggs if a["variant"] == plan["headline_variant"]})
        write_json(out_dir / ("test_report.json" if split == 2 else "dryrun_report.json"), clean_json(report))
        shutil.rmtree(shards)
        if not args.keep_snapshot:
            shutil.rmtree(snap_root, ignore_errors=True)
    except BaseException as e:
        append_jsonl(ledger, dict(rec, status="crashed", finished_utc=utcnow(),
                                  error=repr(e), traceback=traceback.format_exc()[-4000:]))
        raise
    append_jsonl(ledger, dict(rec, status="completed", finished_utc=utcnow(),
                              seconds=round(time.time() - t_start, 1)))
    print(f"done: {len(aggs)} runs in {time.time() - t_start:.0f}s -> {out_dir}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=str(FINAL / "TEST_PLAN.json"))
    ap.add_argument("--split", type=int, default=1, choices=[1, 2], help="1 = val dry run (default), 2 = test")
    ap.add_argument("--results-dir", default=str(FINAL / "results"))
    ap.add_argument("--force-rerun", default=None, metavar="REASON")
    ap.add_argument("--resume", action="store_true", help="reuse completed shards after a crash")
    ap.add_argument("--accept-code-change", default=None, metavar="REASON")
    ap.add_argument("--skip-registration-check", default=None, metavar="REASON")
    ap.add_argument("--methods", nargs="*", default=None, help="dry run only: method ids or groups")
    ap.add_argument("--variants", nargs="*", default=None, help="dry run only: variant ids")
    ap.add_argument("--max-markets", type=int, default=None, help="dry run only")
    ap.add_argument("--no-dryrun-ledger", action="store_true")
    ap.add_argument("--keep-snapshot", action="store_true", help="keep results/.snapshot (data copy)")
    a = ap.parse_args(argv)
    try:
        return run(a)
    except (RunRefused, PlanError) as e:
        print(f"run_test: REFUSED: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
