# Worked Example: HPC Delegation for a Tier 2 Sweep

This directory walks through how the `experiment-runner` agent integrates with
[`claude-hpc`](https://github.com/jamesdchen/claude-hpc) for a Tier 2
experiment whose grid is large enough to delegate to a cluster.

**Status:** illustrative. Shapes here match the contract in
[`docs/hpc/integration-reference.md`](../../../docs/hpc/integration-reference.md)
at the pinned upstream commit (`ec041c6`). Re-sync if the pin moves.

## The hypothesis

> Percentile bootstrap 95% confidence intervals achieve nominal coverage for
> skewed distributions at small sample sizes.

A clean, falsifiable claim. Each task is independent (different seed × n ×
distribution combination), so it parallelizes trivially.

## The sweep

- `distribution ∈ {normal, exponential, lognormal}` (3)
- `sample_size ∈ {30, 100, 300}` (3)
- `seed ∈ range(50)` (50)
- **Total: 3 × 3 × 50 = 450 tasks**

At ~30 seconds per task (500 bootstrap iterations of ~1000 resamples), serial
local execution would take ~3.75 hours. The MARs decision rule (delegate when
grid > `experiment.hpc.delegate_when_tasks_over`, default 8) tells the agent
to delegate.

## Files

```
hpc-delegation/
├── README.md          # this file
├── meta.json          # MARs experiment metadata
├── pyproject.toml     # what ExperimentEnvironment.create scaffolds for tier-2
├── .hpc/
│   └── tasks.py       # claude-hpc reads this for fan-out (total + resolve)
└── scripts/
    └── run.py         # per-task executor (the cluster dispatches this)
```

`results/` and `_aggregated/` are created at runtime; not committed here.

## Per-task contract (read first)

The cluster dispatcher invokes `scripts/run.py` once per task. For each task it
sets:

- `RESULT_DIR` — where this task's `metrics.json` must land. The executor calls
  `write_metrics(dict)` (no arguments needed) and `metrics_io` reads
  `RESULT_DIR` from env. **Do not pass `result_dir=` from MARs.**
- `HPC_KW_DISTRIBUTION`, `HPC_KW_SAMPLE_SIZE`, `HPC_KW_SEED` — one env var per
  key in the dict that `.hpc/tasks.py:resolve(i)` returned. `read_kw_env()`
  strips the prefix and lowercases.

These are dispatcher-controlled — MARs's spawn env must NOT pre-set them.

Inside the executor, only `claude_hpc.mapreduce.metrics_io` and
`claude_hpc.executor_cli` are stable imports from the upstream package.

## Expected agent workflow

With `experiment.hpc.enabled = true` and `default_cluster = hoffman2`, the
`experiment-runner` agent walks the steps documented in the Cluster Execution
section of [`agents/experiment-runner.md`](../../../agents/experiment-runner.md):

1. **Preflight** the cluster:
   ```bash
   uv run hpc-agent preflight --cluster hoffman2
   ```
   Expect `{"ok": true, "data": {"all_ok": true, ...}}`. If `ssh_auth_sock` is
   false, re-load your key into ssh-agent and rerun.

2. **Verify the tasks module** loads cleanly and the grid size matches the
   hypothesis:
   ```bash
   uv run python -c 'from claude_hpc import load_tasks_module, tasks_path, compute_cmd_sha; m = load_tasks_module(tasks_path(".")); print("total=", m.total(), "cmd_sha=", compute_cmd_sha(m))'
   ```
   Expect `total= 450` and a 64-char hex `cmd_sha`.

3. **Dedup pre-check** — has this exact `tasks.py` already been submitted?
   ```bash
   uv run hpc-agent find-prior-run --cmd-sha <sha>
   ```
   If `data.run_id` is returned, skip submit and resume monitoring on that
   run_id.

4. **Build the submit spec** (`spec.json`) — `run_id` is **omitted** (claude-hpc emits it), and `profile`/`job_name` are overlaid from `meta.json::experiment_id` via the `mars_hpc` adapter (replaces the cleaved-out `hpc-agent submit --from-meta`):
   ```bash
   cat > base-spec.json <<'JSON'
   {
     "cluster": "hoffman2",
     "ssh_target": "user@hoffman2.idre.ucla.edu",
     "remote_path": "/u/scratch/user/run-007-bootstrap-coverage",
     "total_tasks": 450
   }
   JSON
   uv run python -m mars_hpc build-spec "$PWD" base-spec.json > spec.json
   ```

5. **Canary the first task** before launching the full array (recommended for
   any new experiment shape):
   ```bash
   uv run hpc-agent verify-canary --canary-run-id <id> --wait-budget-sec 600
   ```
   Block-poll one task end-to-end. If it succeeds, fan out the rest; if it
   fails, fix the executor and re-canary.

6. **Submit**:
   ```bash
   uv run hpc-agent submit --spec spec.json
   ```
   Record `data.run_id` (claude-hpc emits it; typical shape is
   `<profile>-<utc_ts>-<cmd_sha8>`). Also record `data.deduped` — if true, a
   prior run with the same submit identity exists.

7. **Monitor** (use `monitor-summary` for the human-readable view; `status`
   for the raw envelope):
   ```bash
   uv run hpc-agent monitor-summary --run-id <run_id>
   ```
   Backoff 30s → 60s → 120s while `lifecycle_state == in_flight`. On
   `data.preempted_count > 0`, selectively resubmit:
   ```bash
   uv run hpc-agent resubmit --run-id <run_id> --task-ids <ids> --category preempted
   ```
   On `lifecycle_state == failed`, get the clustered failure fingerprints:
   ```bash
   uv run hpc-agent failures --run-id <run_id>
   ```

8. **Aggregate** per wave, then verify completeness:
   ```bash
   uv run hpc-agent aggregate --run-id <run_id> --wave 0
   uv run hpc-agent verify-aggregation-complete --run-id <run_id> --combiner-dir _aggregated/<run_id>/
   ```
   Only after `verify-aggregation-complete` returns `ok: true` should the
   agent declare the experiment done.

9. **Assemble `results/metrics.json`** in MARs's canonical schema — see the
   example below.

## Expected `results/metrics.json` after aggregation

```json
{
  "experiment_id": "run-007-bootstrap-coverage",
  "timestamp": "2026-05-15T16:30:00Z",
  "seed": 42,
  "models": {
    "percentile_bootstrap_ci": {
      "coverage_by_distribution_and_n": {
        "normal":      {"30": 0.94, "100": 0.95, "300": 0.95},
        "exponential": {"30": 0.86, "100": 0.92, "300": 0.94},
        "lognormal":   {"30": 0.78, "100": 0.88, "300": 0.93}
      },
      "n_seeds": 50,
      "nominal_coverage": 0.95
    }
  },
  "rankings": [
    {"metric": "coverage_at_n_30", "best_distribution": "normal", "worst_distribution": "lognormal"}
  ],
  "statistical_tests": []
}
```

## Disabling HPC delegation

Toggle `experiment.hpc.enabled` to `false` (the default) and the same agent
runs `uv run python scripts/run.py` for each task locally, materializing the
HPC_KW_* env vars in-process. Same `.hpc/tasks.py`, same executor — the
cluster path is opt-in.

## Why both thresholds matter

The decision rule has both a task-count threshold AND a walltime threshold
because a small but slow grid still benefits from the cluster:

- 4 tasks × 6 hours each → delegate (long walltime per task).
- 200 tasks × 5 seconds each → run locally (cluster scheduling overhead
  dominates).

Tune via `experiment.hpc.delegate_when_tasks_over` and
`experiment.hpc.delegate_when_walltime_minutes_over`.

## Alternative kwarg-passing pattern (`FLAGS` + argparse)

This example uses `read_kw_env()` because it works without declaring a flag
schema. The other supported pattern declares each axis as a CLI flag in
`tasks.py`:

```python
from claude_hpc.executor_cli import flag, generic_args

FLAGS = {
    "scripts.run": [
        *generic_args(),
        flag("distribution", type=str),
        flag("sample-size", type=int),
        flag("seed", type=int),
    ],
}
```

The executor then uses
`claude_hpc.executor_cli.build_parser_from_flags(FLAGS["scripts.run"])`
instead of `read_kw_env()`. Pick whichever style fits the executor.
