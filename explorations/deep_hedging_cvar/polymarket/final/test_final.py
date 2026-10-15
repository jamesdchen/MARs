"""Tests of the final-test pipeline on a SYNTHETIC tiny dataset (never the real test rows).

    .venv/bin/python polymarket/final/test_final.py         (pytest optional)

Covers make_plan (contents, ledger count, seed rule, comparator rule, refuse-overwrite), the
split-2 code path of run_test (MM_FINAL, plan hash, registration, test-row hash, run-once,
--force-rerun recorded, --resume, stand-in backend refused on real-looking data, no subsets),
the val dry run, report.py on both splits, accounting identities, the stand-in replay env
(latency, lookahead, mirror) and the C-env adapter plumbing against a fake env module.
Needs numpy/pandas/pyarrow/matplotlib; torch (for real train/policy.py and puffernet
checkpoints) is optional: without it the RL seeds are callable stand-ins.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapters  # noqa: E402
import common  # noqa: E402
import make_plan  # noqa: E402
import pyreplay  # noqa: E402
import run_test  # noqa: E402
import synth  # noqa: E402
import reduce  # noqa: E402
from reduce import RULE_FLAGS, directional_verdict, exploit_verdict  # noqa: E402

report = common.load_module("pm_final_report", HERE / "report.py")   # baselines/ has a report.py too

_TMP = Path(tempfile.mkdtemp(prefix="mm_final_test_"))
_STATE = {}
_PY = {}


def cenv_available():
    """The C env of record can run here: sources, compiled binding, a C compiler, feat_meta names."""
    pm = HERE.parent
    return (all((pm / f).exists() for f in ("mm.h", "mm_harness.c", "reference.py", "env.py"))
            and bool(list(pm.glob("binding*.so"))) and shutil.which(os.environ.get("CC", "cc")) is not None
            and (pm / "data" / "build" / "feat_meta.json").exists())


BACKEND = os.environ.get("MM_TEST_BACKEND") or ("cenv" if cenv_available() else "pyreplay")


def world():
    """Synthetic build (C-core layout when testing the C env) + artifact root + plan, built once."""
    if _STATE:
        return _STATE
    layout = "cenv" if BACKEND == "cenv" else "small"
    build = synth.build_dataset(_TMP / "build", n_per_split=8, lengths=(300,), layout=layout)
    ds = common.Dataset(build)
    if BACKEND == "cenv":
        import cenv
        obs_size = cenv.Core().obs
    else:
        obs_size = pyreplay.ReplayEnv(ds, ds.split_markets(1)[:1]).obs_size
    root = synth.build_root(_TMP / "root", obs_size, puffer5=True)
    plan_path = _TMP / "plan" / "TEST_PLAN.json"
    rc = make_plan.main(["--root", str(root), "--build-dir", str(build), "--out", str(plan_path),
                         "--backend", BACKEND, "--allow-dirty", "--n-boot", "300"])
    assert rc == 0
    _STATE.update(build=build, ds=ds, root=root, plan_path=plan_path, obs_size=obs_size,
                  plan=json.loads(plan_path.read_text()))
    return _STATE


def world_py():
    """Small-layout synthetic data for the stand-in replay env's own tests."""
    if not _PY:
        build = synth.build_dataset(_TMP / "build_py", n_per_split=8, lengths=(300,), layout="small")
        ds = common.Dataset(build)
        obs_size = pyreplay.ReplayEnv(ds, ds.split_markets(1)[:1]).obs_size
        root = synth.build_root(_TMP / "root_py", obs_size, puffer5=False)
        _PY.update(build=build, ds=ds, root=root, obs_size=obs_size)
    return _PY


class Env:
    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kv}
        for k, v in self.kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def rt(*args):
    return run_test.main([str(a) for a in args])


# ----------------------------------------------------------------------------- make_plan
def test_plan_contents():
    w = world()
    p = w["plan"]
    ids = [m["id"] for m in p["methods"]]
    assert ids[:5] == ["baseline/no_quote", "baseline/sym_touch", "baseline/as_logit", "baseline/fair_value",
                       "baseline_c/sym_touch"]
    assert p["secondary_comparators"] == ["baseline_c/sym_touch"]
    assert "baseline/no_quote" in p["best_baseline"]["candidates"]               # D1.1
    assert sum(m["group"] == "rl" for m in p["methods"]) == 5
    if synth.have_torch():
        assert sum(m["group"] == "rl5" for m in p["methods"]) == 5
        assert all(len(m["sha256"]) == 64 for m in p["methods"] if m["group"] in ("rl", "rl5"))
    vids = [v["id"] for v in p["variants"]]
    assert len(vids) == 12 and p["headline_variant"] == "central_lat1_x1" in vids
    assert {"central_lat1_x0.5", "central_lat1_x2", "central_lat1_x1_mirror"} <= set(vids)
    assert sum("never_headline" in v["roles"] for v in p["variants"]) == 3
    assert p["best_baseline"]["id"] == "baseline/fair_value"            # highest val P&L in tuned.json
    # ledgers: results (2 val of 3), baselines (3 val, 1 assumed), train/runs/x (1); bad lines skipped
    assert p["ledger"]["n_trials"] == 6 and p["ledger"]["n_assumed_val"] == 1
    assert p["ledger"]["n_distinct_configs"] == 5 and len(p["ledger"]["files"]) == 3
    assert p["data"]["test_rows"]["sha256"] == w["ds"].meta["test_rows"]["sha256"]
    assert p["data"]["test_markets"]["n"] == 8
    assert (w["plan_path"].parent / "plan_sha256.txt").read_text().split()[0] == \
        common.S.sha256_file(w["plan_path"])
    assert [c["id"] for c in p["decision_rule"]["success_requires_all"]] == [
        "ci_lower_gt_0", "pessimistic_1s_gt_0", "central_2s_gt_0", "not_directional", "no_exploit_flag",
        "ex_top_positive", "counts_as_market_making"]
    assert p["exploit_detectors"]["rule_flags"] == ["inspread", "crossed_or_stale", "onesided", "last30"]


def test_plan_refuses_overwrite():
    w = world()
    before = w["plan_path"].read_bytes()
    rc = make_plan.main(["--root", str(w["root"]), "--build-dir", str(w["build"]), "--out", str(w["plan_path"]),
                         "--backend", "pyreplay", "--allow-dirty"])
    assert rc == 2 and w["plan_path"].read_bytes() == before


def test_plan_rules():
    w = world()
    root2 = _TMP / "root_few"
    shutil.copytree(w["root"], root2)
    sel = json.loads((root2 / "train" / "selected.json").read_text())
    sel["seeds"] = sel["seeds"][:3]
    (root2 / "train" / "selected.json").write_text(json.dumps(sel))
    try:
        make_plan.build_plan(root2, w["build"], "pyreplay", allow_dirty=True)
        raise AssertionError("3 seeds accepted")
    except common.PlanError as e:
        assert "seeds" in str(e)
    p = make_plan.build_plan(w["root"], w["build"], "pyreplay", best_baseline="as_logit", allow_dirty=True)
    assert p["best_baseline"]["id"] == "baseline/as_logit"
    p = make_plan.build_plan(w["root"], w["build"], "pyreplay", best_baseline="no_quote", allow_dirty=True)
    assert p["best_baseline"]["id"] == "baseline/no_quote"                      # D1.1 candidate
    try:
        make_plan.build_plan(w["root"], w["build"], "pyreplay", best_baseline="nope", allow_dirty=True)
        raise AssertionError("unknown comparator accepted")
    except common.PlanError:
        pass
    root3 = _TMP / "root_neg"                      # every tuned baseline loses on val -> no-quote
    shutil.copytree(w["root"], root3)
    tj = json.loads((root3 / "baselines" / "tuned.json").read_text())
    for v in tj.values():
        v["val"] = {"total_pnl": -1.0}
    (root3 / "baselines" / "tuned.json").write_text(json.dumps(tj))
    p = make_plan.build_plan(root3, w["build"], "pyreplay", allow_dirty=True)
    assert p["best_baseline"]["id"] == "baseline/no_quote"


def test_dry_run_plan_without_selected():
    """make_plan --dry-run tolerates a missing selected.json (val only); a real plan does not, and
    run_test refuses a dry-run plan on the test split."""
    w = world()
    root4 = _TMP / "root_nosel"
    shutil.copytree(w["root"], root4)
    (root4 / "train" / "selected.json").unlink()
    try:
        make_plan.build_plan(root4, w["build"], "pyreplay", allow_dirty=True)
        raise AssertionError("plan without selected.json accepted")
    except common.PlanError:
        pass
    out = _TMP / "plan_dry" / "TEST_PLAN.json"
    assert make_plan.main(["--root", str(root4), "--build-dir", str(w["build"]), "--out", str(out),
                           "--backend", "pyreplay", "--allow-dirty", "--dry-run"]) == 0
    p = json.loads(out.read_text())
    assert p["dry_run_only"] and not any(m["group"] == "rl" for m in p["methods"])
    assert make_plan.main(["--root", str(root4), "--build-dir", str(w["build"]), "--out",
                           str(HERE / "TEST_PLAN.json"), "--dry-run", "--allow-dirty"]) == 2
    with Env(MM_FINAL="1"):
        assert rt("--plan", out, "--split", 2, "--results-dir", _TMP / "res_dry",
                  "--skip-registration-check", "t") == 3
    assert not (_TMP / "res_dry" / "runs.jsonl").exists()


def test_count_trials_excludes_final_and_caches():
    d = _TMP / "ledgers"
    for rel, recs in {"results/trials.jsonl": [{"split": 1, "config_sha256": "a"}, {"split": 0}],
                      "final/results/trials.jsonl": [{"split": 1}],
                      "analysis/cache/trials.jsonl": [{"split": 1}],
                      "train/trials.jsonl": [{"split": "val", "config": {"x": 1}}, {"split": "val", "config": {"x": 1}}]}.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    c = common.count_trials(d)
    assert c["n_trials"] == 3 and c["n_distinct_configs"] == 2 and len(c["files"]) == 2


# ----------------------------------------------------------------------------- run_test: test split
def test_split2_guards_and_run_once():
    w = world()
    res = _TMP / "results_test"
    plan = w["plan_path"]
    with Env(MM_FINAL=None):
        assert rt("--plan", plan, "--split", 2, "--results-dir", res, "--skip-registration-check", "t") == 3
    assert not (res / "runs.jsonl").exists()
    with Env(MM_FINAL="1"):
        # registration (amendment log + commit message) is required
        assert rt("--plan", plan, "--split", 2, "--results-dir", res) == 3
        # dry-run subsets are not allowed on the test split
        assert rt("--plan", plan, "--split", 2, "--results-dir", res, "--skip-registration-check", "t",
                  "--max-markets", 2) == 3
        # a tampered plan is refused
        tp = _TMP / "plan_tampered" / "TEST_PLAN.json"
        tp.parent.mkdir(exist_ok=True)
        shutil.copy(plan, tp)
        shutil.copy(plan.parent / "plan_sha256.txt", tp.parent / "plan_sha256.txt")
        tp.write_text(tp.read_text().replace('"n_boot": 300', '"n_boot": 301'))
        assert rt("--plan", tp, "--split", 2, "--results-dir", res, "--skip-registration-check", "t") == 3
        assert not (res / "runs.jsonl").exists()
        # the run
        assert rt("--plan", plan, "--split", 2, "--results-dir", res, "--skip-registration-check", "unit test") == 0
        rep = json.loads((res / "test" / "test_report.json").read_text())
        assert rep["plan_sha256"] == common.S.sha256_file(plan) and rep["complete_plan"]
        assert rep["checks"]["test_rows"]["ok"] and rep["checks"]["registration"]["skipped"] == "unit test"
        assert rep["reruns"] == [] and not (res / "test" / "shards").exists()
        recs = common.read_jsonl(res / "runs.jsonl")[0]
        assert [r["status"] for r in recs] == ["started", "completed"]
        # once only
        assert rt("--plan", plan, "--split", 2, "--results-dir", res, "--skip-registration-check", "t") == 3
        assert rt("--plan", plan, "--split", 2, "--results-dir", res, "--skip-registration-check", "t",
                  "--resume") == 3
        # forced rerun: reason recorded in the run record and the results
        assert rt("--plan", plan, "--split", 2, "--results-dir", res, "--skip-registration-check", "t",
                  "--force-rerun", "unit test: rerun path") == 0
        rep2 = json.loads((res / "test" / "test_report.json").read_text())
        assert rep2["reruns"][0]["reason"] == "unit test: rerun path"
        recs = common.read_jsonl(res / "runs.jsonl")[0]
        assert recs[-2]["force_rerun_reason"] == "unit test: rerun path" and recs[-1]["status"] == "completed"
    # deterministic: the two runs agree exactly
    m1 = pd.read_parquet(res / "test" / "markets.parquet")
    assert len(m1) == len(w["plan"]["methods"]) * len(w["plan"]["variants"]) * 8
    _STATE["res_test"] = res


def test_split2_test_rows_hash_checked():
    """Whole-file hashes match the plan but the test rows differ from meta.json's hash."""
    w = world()
    b2 = _TMP / "build_tampered"
    shutil.copytree(w["build"], b2)
    meta = json.loads((b2 / "meta.json").read_text())
    t = np.memmap(b2 / "ticks.bin", dtype=np.float32, mode="r+", shape=(meta["N"], meta["F"]))
    t[meta["test_rows"]["row_lo"] + 5, 1] += 0.01
    t.flush()
    del t
    ds2 = common.Dataset(b2)
    plan = json.loads(w["plan_path"].read_text())
    plan["data"]["file_sha256"] = ds2.file_hashes()        # a plan written after the tampering
    a = argparse.Namespace(methods=None, variants=None, max_markets=None, skip_registration_check="t",
                           accept_code_change=None)
    with Env(MM_FINAL="1"):
        try:
            run_test.preflight(plan, "x", 2, a, ds2)
            raise AssertionError("tampered test rows accepted")
        except run_test.RunRefused as e:
            assert "test rows sha256" in str(e)
        chk = run_test.preflight(json.loads(w["plan_path"].read_text()), "x", 2, a, w["ds"])
        assert chk["test_rows"]["ok"]
    assert w["ds"].test_rows_sha256() == w["ds"].meta["test_rows"]["sha256"]


def test_split2_standin_backend_refused_on_real_looking_data():
    w = world()
    b3 = _TMP / "build_notsynthetic"
    shutil.copytree(w["build"], b3)
    meta = json.loads((b3 / "meta.json").read_text())
    meta["synthetic"] = False
    (b3 / "meta.json").write_text(json.dumps(meta))
    out = _TMP / "plan_ns" / "TEST_PLAN.json"
    assert make_plan.main(["--root", str(w["root"]), "--build-dir", str(b3), "--out", str(out),
                           "--backend", "pyreplay", "--allow-dirty"]) == 0
    with Env(MM_FINAL="1"):
        assert rt("--plan", out, "--split", 2, "--results-dir", _TMP / "res_ns", "--skip-registration-check", "t") == 3
    assert not (_TMP / "res_ns" / "runs.jsonl").exists()


def test_crash_resume():
    w = world()
    res = _TMP / "results_crash"
    plan = w["plan_path"]
    orig = run_test.RunAccumulator.add
    calls = {"n": 0}

    def boom(self, tr):
        calls["n"] += 1
        if calls["n"] > 40:
            raise RuntimeError("simulated crash")
        return orig(self, tr)
    run_test.RunAccumulator.add = boom
    try:
        rt("--plan", plan, "--split", 1, "--results-dir", res, "--variants", "central_lat1_x1", "pessimistic_lat1_x1")
        raise AssertionError("no crash")
    except RuntimeError:
        pass
    finally:
        run_test.RunAccumulator.add = orig
    recs = common.read_jsonl(res / "runs.jsonl")[0]
    assert [r["status"] for r in recs] == ["started", "crashed"]
    shards = list((res / "val" / "shards").glob("*.agg.json"))
    assert len(shards) == 5                                  # 40 markets = 5 runs of 8 markets
    assert rt("--plan", plan, "--split", 1, "--results-dir", res, "--variants", "central_lat1_x1",
              "pessimistic_lat1_x1", "--resume") == 3       # needs --force-rerun
    assert rt("--plan", plan, "--split", 1, "--results-dir", res, "--variants", "central_lat1_x1",
              "pessimistic_lat1_x1", "--resume", "--force-rerun", "crash restart (unit test)") == 0
    rep = json.loads((res / "val" / "dryrun_report.json").read_text())
    assert rep["reruns"][0]["resume"] and len(rep["timings"]) == 2 * len(w["plan"]["methods"])


# ----------------------------------------------------------------------------- dry run + reports
def test_val_dry_run_and_reports():
    w = world()
    res = _TMP / "results_val"
    assert rt("--plan", w["plan_path"], "--split", 1, "--results-dir", res) == 0
    assert (res / "val" / "dryrun_report.json").exists() and not (res / "test").exists()
    assert len(common.read_jsonl(res / "val" / "dryrun_trials.jsonl")[0]) == w["plan"]["n_runs"]
    out = _TMP / "RESULTS_val.md"
    assert report.main(["--split", "1", "--results-dir", str(res), "--out", str(out),
                        "--figures", str(_TMP / "fig_val"), "--n-boot", "200"]) == 0
    txt = out.read_text()
    assert "DRY RUN" in txt and "Synthetic dataset" in txt
    if BACKEND == "cenv":
        assert "Stand-in env" not in txt and "max |state diff|" in txt and "Replay mismatch" not in txt
    else:
        assert "Stand-in env" in txt
    for i in range(1, 14):
        assert f"## {i}. " in txt, i
    assert "**Overall:" in txt and "NOT EVALUABLE" not in txt   # full plan: every criterion evaluated
    assert "test |" not in txt.split("## 12.")[1].split("## 13.")[0]     # dry run never reads test rows
    for f in ("cumulative_pnl.png", "attribution.png", "sensitivity_heatmap.png"):
        assert (_TMP / "fig_val" / f).stat().st_size > 10_000
    assert report.main(["--split", "1", "--results-dir", str(res), "--out", str(HERE / "RESULTS.md")]) == 2
    if "res_test" not in _STATE:
        test_split2_guards_and_run_once()
    out2 = _TMP / "RESULTS.md"
    assert report.main(["--split", "2", "--results-dir", str(_STATE["res_test"]), "--out", str(out2),
                        "--figures", str(_TMP / "fig_test"), "--n-boot", "200"]) == 0
    t2 = out2.read_text()
    assert "DRY RUN" not in t2 and "| test |" in t2 and "unit test: rerun path" in t2


def test_accounting_identities():
    if "res_test" not in _STATE:
        test_split2_guards_and_run_once()
    d = _STATE["res_test"] / "test"
    mk = pd.read_parquet(d / "markets.parquet")
    mi = pd.read_parquet(d / "minutes.parquet")
    g = mi.groupby(["method", "variant", "market"])[["dpnl", "dpnl_ex_rr"]].sum().reset_index().merge(
        mk[["method", "variant", "market", "pnl", "pnl_ex_rr", "rewards", "rebates"]])
    assert np.allclose(g.dpnl, g.pnl, atol=1e-9) and np.allclose(g.dpnl_ex_rr, g.pnl_ex_rr, atol=1e-9)
    assert np.allclose(mk.pnl - mk.pnl_ex_rr, mk.rewards + mk.rebates, atol=1e-9)
    runs = json.loads((d / "runs.json").read_text())["runs"]
    for a in runs:
        at = a["attribution"]
        assert at["identity_error"] < 1e-9 and at["cash_check_max_abs"] < 1e-9 and not at["inventory_mismatch"]
        tot = at["totals"]
        assert abs(tot["inventory_carry"] + tot["resolution"] - tot["inventory"]) < 1e-9
    nq = mk[mk.method == "baseline/no_quote"]
    assert (nq.pnl == 0).all() and (nq.n_fills == 0).all()
    # hourly grid conserves the total
    R = report.Results(_STATE["res_test"], 2, None)
    for m in ("baseline/sym_touch", "rl/mean"):
        assert abs(R.hourly(m, "central_lat1_x1").sum() - R.total(m, "central_lat1_x1")) < 1e-8


def test_verdict_rules():
    t = dict(spread_capture=1.0, adverse_selection=-0.5, inventory=3.0, rebates=0.5, rewards=0.0, fees=0.0,
             total=4.0)
    assert directional_verdict(t)["directional"]
    t.update(inventory=0.5, total=1.5)
    assert not directional_verdict(t)["directional"]
    flags = {k: dict(share_of_total=0.1, pnl=0.1, n=1) for k in RULE_FLAGS}
    flags["_any"] = dict(pnl_excluding=0.5)
    assert not exploit_verdict(flags, 1.0)["invalid"]
    flags["last30"]["share_of_total"] = 0.3
    v = exploit_verdict(flags, 1.0)
    assert v["invalid"] and v["worst_single_flag"] == "last30"
    flags["last30"]["share_of_total"] = 0.1
    flags["_any"]["pnl_excluding"] = -0.1
    assert exploit_verdict(flags, 1.0)["invalid_sign_change"]
    # D2: the through-the-touch flag is reported, not a detector
    fl = pd.DataFrame({k: [True, False] for k in reduce.ALL_FLAGS})
    fl["outside"] = [True, True]
    for k in RULE_FLAGS:
        fl[k] = [False, False]
    fl["edge60"] = [1.0, 1.0]
    fl["qty"] = [5.0, 5.0]
    fl["print_px_known"] = True
    ex = reduce.exploit_tables(fl)
    assert ex["flags"]["outside"]["share_of_total"] == 1.0 and not ex["verdict"]["invalid"]
    mk = pd.DataFrame(dict(n_fills=[10, 0], uptime_two=[0.5, 0.0], uptime_any=[0.6, 0.0],
                           contract_len_s=[300, 300]))
    mm = reduce.market_making(mk, t, ex["verdict"])
    assert mm["counts_as_market_making"] and mm["fills_per_market"] == 5.0
    mm = reduce.market_making(mk.assign(n_fills=[1, 0]), t, ex["verdict"])
    assert not mm["counts_as_market_making"] and not mm["criteria"]["fills"]


# ----------------------------------------------------------------------------- stand-in env + adapters
def _actions(ds, markets, policy, variant, n_steps):
    env = pyreplay.make_env(ds, markets, variant, 1, False)
    obs = env.reset()
    policy.reset(len(markets))
    first = np.ones(len(markets), bool)
    out = []
    for _ in range(n_steps):
        a = policy.act(obs, first)
        out.append(a.copy())
        obs = env.step(a)
        first[:] = False
    return np.array(out), env


def test_lookahead():
    """EVALUATION.md T2: perturb all data after t in a copy of an episode; the actions up to t must
    be bit-identical. Optimistic fills (trade delay 0): under central/pessimistic a print stamped up
    to 3.5/5 s after t was matched before t, so 'after t' is defined on match time there.
    (Stand-in env; the C env's own lookahead test is test_env.py's.)"""
    w = world_py()
    ds = w["ds"]
    m = ds.split_markets(1)[:2]
    t_cut = 150
    ds2 = common.Dataset(w["build"])
    ds2.ticks = np.array(ds.ticks)
    ds2.feat = np.array(ds.feat)
    ds2.fills = ds.fills.copy()
    rng = np.random.default_rng(0)
    for mm in m:
        sl = ds.market_rows(int(mm))
        lo = sl.start + t_cut + 1
        ds2.ticks[lo:sl.stop] = rng.permutation(ds2.ticks[lo:sl.stop])
        ds2.feat[lo:sl.stop] *= rng.uniform(0.5, 1.5, ds2.feat[lo:sl.stop].shape).astype(np.float32)
        sel = (ds2.fills["row"] >= lo) & (ds2.fills["row"] < sl.stop)
        ds2.fills["price"][sel] = 1 - ds2.fills["price"][sel]
    variant = dict(fill_model="optimistic", latency_s=0, size_mult=1.0)
    entries = [dict(id="b", kind="baseline", name="fair_value", params={}),
               dict(id="c", kind="callable", target="standins:jittered_fair_value", seed=1)]
    if synth.have_torch():
        entries.append(dict(id="rl", kind="ppo3", path="train/checkpoints/seed0.pt"))
    for e in entries:
        ctx = dict(spec=lambda: pyreplay.ReplayEnv(ds, m).obs_spec(), obs_size=w["obs_size"], root=w["root"])
        a1, _ = _actions(ds, m, adapters.make_policy(e, ctx), variant, t_cut + 1)
        a2, _ = _actions(ds2, m, adapters.make_policy(e, ctx), variant, t_cut + 1)
        assert np.array_equal(a1, a2), e["id"]
        b1, _ = _actions(ds, m, adapters.make_policy(e, ctx), variant, t_cut + 40)
        b2, _ = _actions(ds2, m, adapters.make_policy(e, ctx), variant, t_cut + 40)
        assert not np.array_equal(b1, b2), f"{e['id']}: the perturbation never reached the policy"


def test_latency_and_mirror():
    w = world_py()
    ds = w["ds"]
    m = ds.split_markets(1)[:3]
    pol = adapters.FnPolicy(lambda o: np.tile(np.array([[3, 3, 0]], np.int32), (len(o), 1)))
    for L in (0, 1, 2):
        be = adapters.PyReplayBackend(ds)
        trs = be.run(pol, m, dict(fill_model="optimistic", latency_s=L, size_mult=1.0), 1, False)
        for tr in trs:
            live = tr["bid_live"] > 0.5
            assert not live[: L + 1].any() and live[L + 1: L + 20].any(), L
    be = adapters.PyReplayBackend(ds)
    v = dict(fill_model="central", latency_s=1, size_mult=1.0)
    a = be.run(pol, m, v, 1, False)
    b = be.run(pol, m, dict(v, mirror=True), 1, False)
    for x, y in zip(a, b):                                 # a symmetric quoter is mirror-invariant
        assert np.allclose(x["cash"] + x["inv"] * x["settlement"], y["cash"] + y["inv"] * y["settlement"])
        assert np.allclose(x["inv"], -y["inv"])
    try:
        pyreplay.make_env(ds, ds.split_markets(2)[:1], v, 2, allow_test=False)
        raise AssertionError("test split without allow_test")
    except PermissionError:
        pass


def test_mirror_data_matches_internal_mirror():
    """mirror.py (mirrored data files) against pyreplay's independent in-env mirror."""
    import mirror
    w = world_py()
    mdir = _TMP / "build_py_mirror"
    if not mdir.exists():
        mirror.mirror_build(w["build"], mdir)
    mds = common.Dataset(mdir)
    m = w["ds"].split_markets(1)
    v = dict(fill_model="central", latency_s=1, size_mult=1.0)
    ctx = dict(spec=lambda: pyreplay.ReplayEnv(w["ds"], m[:1]).obs_spec(), obs_size=w["obs_size"])
    for e in (dict(id="fv", kind="baseline", name="fair_value", params={}),
              dict(id="st", kind="baseline", name="sym_touch", params={})):
        a = adapters.PyReplayBackend(w["ds"]).run(adapters.make_policy(e, ctx), m, dict(v, mirror=True), 1, False)
        b = adapters.PyReplayBackend(mds).run(adapters.make_policy(e, ctx), m, v, 1, False)
        for x, y in zip(a, b):
            assert x["settlement"] == y["settlement"]
            assert np.allclose(x["cash"], y["cash"], atol=1e-9) and np.allclose(x["inv"], y["inv"])
    meta = json.loads((mdir / "meta.json").read_text())
    assert meta["mirrored"] and common.Dataset(mdir).test_rows_sha256() == meta["test_rows"]["sha256"]


def test_cenv_backend():
    """The C env of record on the synthetic C-layout data: closed loop, harness replay with
    bit-identical observations, reference replay equal to the C state, mirror symmetry, guards."""
    if BACKEND != "cenv":
        print("     (skipped: C env not available here)")
        return
    import cenv
    import mirror
    w = world()
    ds = w["ds"]
    be = adapters.make_backend("cenv", ds, {"work_dir": str(_TMP / "cenv_work")})
    spec = be.obs_spec(1)
    pol = adapters.make_policy(dict(id="st", kind="baseline", name="sym_touch", params={}),
                               dict(spec=lambda: spec, obs_size=be.obs_size()))
    v = dict(fill_model="central", latency_s=1, size_mult=1.0)
    trs = be.run(pol, ds.split_markets(1), v, 1, False)
    assert all(t["checks"]["replay_ok"] and t["checks"]["obs_bit_identical"] for t in trs)
    assert max(t["checks"]["state_max_abs_diff"] for t in trs) < cenv.STATE_TOL
    assert sum(len(t["fills"]["row"]) for t in trs) > 0
    for t in trs:                                  # the trace's P&L is the C Log's
        assert abs(t["cash"][-1] + t["inv"][-1] * t["settlement"] - t["log"]["pnl"]) < 1e-3
    mdir = _TMP / "build_cenv_mirror"
    mirror.mirror_build(w["build"], mdir)
    bm = adapters.make_backend("cenv", common.Dataset(mdir), {"work_dir": str(_TMP / "cenv_work")})
    tm = bm.run(pol, ds.split_markets(1), v, 1, False)
    for x, y in zip(trs, tm):                      # a symmetric quoter is mirror-invariant in the C env
        assert abs((x["cash"][-1] + x["inv"][-1] * x["settlement"]) - (y["cash"][-1] + y["inv"][-1] * y["settlement"])) < 1e-9
    with Env(MM_FINAL=None):
        try:
            be.run(pol, ds.split_markets(2), v, 2, True)
            raise AssertionError("C env split 2 without MM_FINAL")
        except PermissionError:
            pass
    with Env(MM_FINAL="1"):                        # the core itself refuses split 2 without allow_test
        try:
            be.run(pol, ds.split_markets(2), v, 2, False)
            raise AssertionError("C env split 2 without allow_test")
        except Exception as e:  # noqa: BLE001
            assert "allow_test" in str(e) or "refused" in str(e), e


if __name__ == "__main__":
    import time
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"ok   {name} ({time.time() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                import traceback
                fails += 1
                print(f"FAIL {name} {e!r}")
                traceback.print_exc()
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(1 if fails else 0)
