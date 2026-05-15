# Worked Example: HPC Delegation for a Tier 2 Sweep

This directory walks through how the `experiment-runner` agent integrates with
[`claude-hpc`](https://github.com/jamesdchen/claude-hpc) for a Tier 2
experiment whose grid is large enough to delegate to a cluster.

**Status:** illustrative. The shapes of `meta.json`, `.hpc/tasks.py`,
`pyproject.toml`, and the `hpc-agent` JSON envelopes are faithful to what's
documented in [`docs/hpc/integration-reference.md`](../../../docs/hpc/integration-reference.md).
The exact CLI invocation of the executor (`scripts/run.py`) follows the
convention documented upstream — if claude-hpc's wire format changes,
re-sync the integration reference and re-check this example.

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

At ~30 seconds per task on a single CPU (500 bootstrap iterations of 1000
resamples), running locally would take ~3.75 hours serial. The MARs decision
rule (delegate when grid > `experiment.hpc.delegate_when_tasks_over`,
default 8) tells the agent to delegate.

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

## Expected agent workflow

With `experiment.hpc.enabled = true` and `default_cluster = hoffman2`, the
`experiment-runner` agent walks the steps documented in
[`agents/experiment-runner.md`](../../../agents/experiment-runner.md)
("Cluster Execution (Optional)"):

1. **Preflight** the cluster:
   ```bash
   uv run hpc-agent preflight --cluster hoffman2
   ```
   Expect `{"ok": true, "data": {"all_ok": true, ...}}`. If `ssh_auth_sock` is
   false, the operator hasn't set `SSH_AUTH_SOCK` in their shell — re-run
   after `ssh-add -l` confirms a key is loaded.

2. **Verify the tasks module** loads cleanly and the grid size matches the
   hypothesis (3 × 3 × 50 = 450):
   ```bash
   uv run python -c 'from claude_hpc import load_tasks_module, tasks_path, compute_cmd_sha; m = load_tasks_module(tasks_path(".")); print("total=", m.total(), "cmd_sha=", compute_cmd_sha(m)[:8])'
   ```

3. **Build the submit spec** (`spec.json`):
   ```json
   {
     "profile": "run-007-bootstrap-coverage",
     "cluster": "hoffman2",
     "ssh_target": "user@hoffman2.idre.ucla.edu",
     "remote_path": "/u/scratch/user/run-007-bootstrap-coverage",
     "job_name": "run-007-bootstrap-coverage",
     "run_id": "run-007-bootstrap-coverage-20260515T140000Z-a1b2c3d4",
     "job_ids": [],
     "total_tasks": 450
   }
   ```
   `run_id` is `{experiment_id}-{utc_ts}-{cmd_sha[:8]}` — same `.hpc/tasks.py`
   re-submit produces the same `cmd_sha`, and `submit` dedupes on the full
   `run_id`.

4. **Submit**:
   ```bash
   uv run hpc-agent submit --spec spec.json
   ```
   Returns either `data.deduped: true` (existing run; skip to polling) or
   `data.deduped: false` with fresh `data.job_ids`.

5. **Poll status** with backoff 30s → 60s → 120s:
   ```bash
   uv run hpc-agent status --run-id <run_id>
   ```
   Watch for `data.lifecycle_state ∈ {in_flight, complete, failed,
   timeout, abandoned}` and `data.preempted_count`. If non-zero
   `preempted_count`, selectively resubmit just those task ids:
   ```bash
   uv run hpc-agent resubmit --run-id <run_id> --task-ids <ids> --category preempted
   ```

6. **Aggregate** per wave:
   ```bash
   uv run hpc-agent aggregate --run-id <run_id> --wave 0
   ```
   Per-task outputs land at `_aggregated/<run_id>/`.

7. **Assemble `results/metrics.json`** in MARs's canonical schema — see the
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

Toggle `experiment.hpc.enabled` to `false` (or leave it default) and the same
agent runs `uv run python scripts/run.py` for each task locally, with the
task-axis loop materialized in-process. Same `.hpc/tasks.py`, same executor —
the cluster path is opt-in.

## When NOT to delegate

The decision rule has both a task-count threshold AND a walltime threshold
because a small but slow grid still benefits from the cluster:

- 4 tasks × 6 hours each → delegate (long walltime per task).
- 200 tasks × 5 seconds each → run locally (cluster scheduling overhead
  dominates).

Adjust `experiment.hpc.delegate_when_tasks_over` and
`experiment.hpc.delegate_when_walltime_minutes_over` per project.
