# claude-hpc Integration Reference (Vendored)

> **Vendored from** <https://github.com/jamesdchen/claude-hpc>
> **Source:** integration contract @ commit `9c0e184` on branch `claude/post-mars-cleanup-iRQMr`
> **Synced:** 2026-05-15
> **Re-sync** when upgrading the pinned `claude-hpc` range in
> `src/paper/experiments/environment.ts`.

This file is the contract MARs's `experiment-runner` agent and tier-2
experiment scaffolding depend on. The runtime source-of-truth for the agent
is the "Cluster Execution (Optional)" section in
[`agents/experiment-runner.md`](../../agents/experiment-runner.md); this
document is the human-readable reference behind it.

---

## What changed at `9c0e184` (the cleavage)

claude-hpc was previously a more MARs-aware tool. At this commit it stopped
knowing about MARs's experiment shape. The following surfaces **moved out of
claude-hpc and into MARs** (specifically into the `mars_hpc.py` adapter that
the scaffolder writes into each tier-2 experiment dir):

| Removed from claude-hpc | Owned by MARs (in `mars_hpc.py`) |
|---|---|
| `claude_hpc.state.discover.detect_mars_tier(...)` (auto-detected probe/run from path layout) | `detect_experiment_tier(experiment_dir)` |
| `claude_hpc.state.discover.read_meta_json(...)` | `read_meta_json(experiment_dir)` |
| `hpc-agent discover` envelope's `data.meta` block (experiment_id/seed/purpose/tier) | `discover_with_meta(experiment_dir)` wraps the CLI and re-adds it |
| `hpc-agent submit --from-meta` (overlay experiment_id onto profile/job_name) | `build_submit_spec(experiment_dir, base_spec)` |
| Auto-narrowing the executor scan to `scripts/` when meta.json was present | `discover_with_meta` passes `search_dirs=["scripts"]` to the Python API for tier-2 |

Why the split: claude-hpc parallelizes whatever the caller hands it. It has
no business knowing about probe-vs-run tiers, `experiment_id` semantics, or
the src-is-modules convention — those are MARs's contracts. The two halves
still interlock cleanly through `hpc-agent`'s JSON envelope and the per-run
sidecar; that contract is unchanged.

**Known upstream gap (file as feature request):** the
`hpc-agent discover` CLI does **not** expose `--search-dirs` at `9c0e184`,
even though `claude_hpc.state.discover.discover_executors(root, search_dirs=...)`
accepts the override. `mars_hpc.discover_with_meta` imports the Python API
directly to apply the override; once the CLI flag ships upstream, switch the
adapter to the CLI for fewer cross-package imports.

## Setup Steps

The maintainer needs three changes:

1. **Add dependency**: `claude-hpc` is included in the tier-2 `pyproject.toml`
   written by `src/paper/experiments/environment.ts`. It's installed from a
   pinned git+ URL (commit `ec041c6`) because claude-hpc is not on PyPI yet —
   when it publishes, switch the pin in environment.ts to `claude-hpc>=X,<Y`
   and re-sync this document.
2. **Update agent prompt**: the cluster-execution section from upstream
   `docs/workflows/mars/experiment-runner.snippet.md` is appended verbatim
   to `agents/experiment-runner.md`.
3. **Configure spawning**: SSH credentials are forwarded explicitly when
   spawning the agent via `Bun.spawn` in `runInEnv`.

## Critical Environment Variables

### MARs-controlled (forwarded by `runInEnv` in `src/paper/experiments/environment.ts`)

| Variable               | Default                                                | Why |
|------------------------|--------------------------------------------------------|-----|
| `SSH_AUTH_SOCK`        | parent env                                             | Without it, every cluster call hangs on auth — single most common spawn failure. |
| `SSH_AGENT_PID`        | parent env                                             | Pair with `SSH_AUTH_SOCK`. |
| `HPC_JOURNAL_DIR`      | `~/.mars/hpc/<experiment-name>/`                       | Per-experiment journal so concurrent MARs runs don't share state. |
| `HPC_CLUSTERS_CONFIG`  | parent env (operator sets it)                          | Path to `clusters.yaml`. |
| `HPC_SSH_TIMEOUT_SEC`  | parent env, claude-hpc default 60                      | Raise to ~120 for flaky login nodes. |
| `HPC_TELEMETRY_SINK`   | parent env, claude-hpc default `none`                  | Set to `stderr-jsonl` to capture telemetry into MARs's log stream. |

### Dispatcher-controlled (DO NOT set in MARs's spawn env)

The cluster-side job dispatcher sets these per-task before invoking the
executor. Setting them in the parent env that MARs forwards will confuse
the dispatcher's per-task scope.

| Variable        | Set by                                  | Read by                                |
|-----------------|------------------------------------------|----------------------------------------|
| `RESULT_DIR`    | dispatcher                               | `metrics_io.write_metrics()` (default arg) |
| `HPC_KW_*`      | dispatcher (from `tasks.resolve(i)`)     | `metrics_io.read_kw_env()`             |
| `LOCAL_DATA_DIR`| dispatcher (when `nfs_data_dir` is set)  | executor (optional)                    |

### Executor import boundary

Inside any executor that ships to the cluster, only these claude-hpc names
are stable imports:

- `claude_hpc.mapreduce.metrics_io.write_metrics`
- `claude_hpc.mapreduce.metrics_io.read_kw_env`
- `claude_hpc.executor_cli.flag`
- `claude_hpc.executor_cli.generic_args`
- `claude_hpc.executor_cli.gpu_args`
- `claude_hpc.executor_cli.build_parser_from_flags`

Anything else (e.g., `claude_hpc.runner.*`, `claude_hpc.mapreduce.reduce.*`)
is internal and may break across releases.

### `write_metrics` — actual signature

```python
def write_metrics(metrics: dict, *, result_dir: str | None = None) -> str
```

- `metrics` is a positional `dict`.
- `result_dir` is keyword-only; defaults to reading the `RESULT_DIR` env var.
- Include `"n_samples"` in the dict for weighted aggregation (defaults to 1).
- Atomic write (tempfile + fsync + rename).

The dispatcher sets `RESULT_DIR` per task — executors call
`write_metrics(d)` with no `result_dir` argument and it just works.

## Error Code → Retry Policy

| Error Code                    | Category | Retry Safe | Action |
|-------------------------------|----------|------------|--------|
| `ssh_unreachable`             | network  | ✓          | Halt and prompt; run preflight after operator fixes |
| `scheduler_throttled`         | cluster  | ✓          | Backoff: 1s → 2s → 4s (max 4 retries) |
| `cluster_timeout`             | cluster  | ✓          | Backoff: 4s → 8s → 16s (max 3 retries) |
| `combiner_failed`             | cluster  | ✓          | Single retry; surface if persistent |
| `preempted`                   | cluster  | ✓          | Resubmit immediately (not a failure) |
| `cluster_partially_degraded`  | cluster  | ✓          | Continue polling; inspect `partial_errors` array |
| `remote_command_failed`       | cluster  | ✗          | Surface with `stderr_tail`; no auto-retry |
| `spec_invalid`                | user     | ✗          | Surface; agent must regenerate spec |
| `executor_not_found`          | user     | ✗          | Surface; check executor path |
| `cluster_unknown`             | user     | ✗          | Run `clusters list` to recover |
| `config_invalid`              | user     | ✗          | Surface; fix clusters.yaml |
| `outputs_missing`             | user     | ✗          | Surface; inspect logs |
| `journal_corrupt`             | internal | ✗          | Surface; investigate `$HPC_JOURNAL_DIR` |
| `schema_incompat`             | internal | ✗          | Pin compatible versions |
| `internal`                    | internal | ✗          | Bug report |

**Exit codes**: 0 = success, 1 = user error, 2 = cluster/network, 3 = internal.

## The `.hpc/tasks.py` Boundary

MARs writes this file; claude-hpc imports it. It must expose two callables:

```python
def total() -> int:
    """How many tasks this experiment fans out into."""

def resolve(task_id: int) -> dict:
    """Return the kwargs for task #i."""
```

Example for a hyperparameter sweep with `{lr: [0.01, 0.001], seed: [42, 1337]}`:

```python
import itertools
_TASKS = [
    {"lr": lr, "seed": seed}
    for lr, seed in itertools.product([0.01, 0.001], [42, 1337])
]
def total() -> int:
    return len(_TASKS)
def resolve(i: int) -> dict:
    return _TASKS[i]
```

Verify locally with:

```bash
python -c 'from claude_hpc import load_tasks_module, tasks_path; m = load_tasks_module(tasks_path(".")); print("total=", m.total(), "sample=", m.resolve(0))'
```

## Troubleshooting Silent Hangs

Run this from the same environment MARs uses:

```bash
uv run hpc-agent preflight --cluster <your_cluster>
```

Check the returned `data.checks[]` for:

- `ssh_auth_sock`: if false, ensure `ssh-agent` is running with your key
  loaded in the shell that launches MARs.
- `cluster_tcp_22`: if false, the cluster is offline or the hostname is wrong
  (operator config issue).

Defense-in-depth: the `status`, `aggregate`, and `reconcile` subcommands fail
fast with `error_code: "ssh_unreachable"` (exit 2) when `SSH_AUTH_SOCK` is
unset.

## MARs's Dependency Surface on claude-hpc

Anyone refactoring claude-hpc should check this list before deleting, renaming,
or restructuring. These are the contract surfaces MARs actively depends on at
the pinned commit; breaking any of them breaks MARs.

### Console script

- `hpc-agent` — entry point in `pyproject.toml`. MARs invokes via `uv run hpc-agent ...`.

### `hpc-agent` subcommands invoked by MARs

`preflight`, `find-prior-run`, `submit` (with `--dry-run`), `verify-canary`,
`monitor-summary`, `status`, `failures`, `logs`, `resubmit`, `aggregate`,
`verify-aggregation-complete`, `reconcile`, `best-submit-window`, `clusters list`.

Flags MARs passes: `--cluster`, `--cmd-sha`, `--spec`, `--canary-run-id`,
`--wait-budget-sec`, `--run-id`, `--task-ids`, `--category`, `--wave`,
`--combiner-dir`, `--all-failed`, `--lines`, `--profile`, `--within-hours`.

### JSON envelope shape

`{"ok": bool, "idempotent": bool, "data": {...}, "partial_errors": [...]}`.
Exit codes: `0` success, `1` user error, `2` cluster/network, `3` internal.

### `data.lifecycle_state` values

`in_flight`, `complete`, `failed`, `timeout`, `abandoned`.

### `data` fields MARs reads

`all_ok`, `checks[]`, `run_id`, `job_ids`, `deduped`, `preempted_count`,
`preempted_task_ids`, `last_status`, `failures[]` (with `fingerprint`,
`count`, `category`, `sample_logs`).

### Error codes MARs branches on

`ssh_unreachable`, `scheduler_throttled`, `cluster_timeout`, `combiner_failed`,
`preempted`, `cluster_partially_degraded`, `remote_command_failed`,
`spec_invalid`, `executor_not_found`, `cluster_unknown`, `config_invalid`,
`outputs_missing`, `journal_corrupt`, `schema_incompat`.

### Submit-spec JSON fields MARs writes

`profile`, `cluster`, `ssh_target`, `remote_path`, `job_name`, `total_tasks`.
MARs does NOT supply `run_id` — claude-hpc emits it in the response.

### Python imports MARs uses

Public package surface (must remain in `__all__` or equivalent):
- `claude_hpc.load_tasks_module`
- `claude_hpc.tasks_path`
- `claude_hpc.compute_cmd_sha`

Executor-side stable imports (the documented import boundary):
- `claude_hpc.mapreduce.metrics_io.write_metrics`
- `claude_hpc.mapreduce.metrics_io.read_kw_env`
- `claude_hpc.executor_cli.flag`
- `claude_hpc.executor_cli.generic_args`
- `claude_hpc.executor_cli.gpu_args`
- `claude_hpc.executor_cli.build_parser_from_flags`

MARs does NOT import `_PACKAGE_ROOT` or any other leading-underscore name —
the canonical `tasks_example.py` is located via `claude_hpc.__path__` + rglob.

### Env vars MARs forwards (caller-side contract)

`HPC_CLUSTERS_CONFIG`, `HPC_JOURNAL_DIR`, `HPC_SSH_TIMEOUT_SEC`,
`HPC_TELEMETRY_SINK`, `SSH_AUTH_SOCK`, `SSH_AGENT_PID`.

### Env vars MARs avoids (dispatcher-controlled)

`RESULT_DIR`, `HPC_KW_*`, `LOCAL_DATA_DIR`.

### `clusters.yaml` fields MARs's template references

`scheduler` (required), `host`, `user`, `scratch`, `modules`, `conda_source`,
`conda_envs`, `gpu_types`, `default_partition`, `account`, `gpu_constraint`,
`constraints.{max_array_size, max_walltime, max_concurrent_jobs, est_spin_up}`,
`cold_start_mem_buffer`, `nfs_data_dir`, `walltime_arbitrage`,
`auto_daisy_chain`, `max_walltime_sec`, `max_node_mem_mb`, `gpu_queues`,
`excluded_gpu_queue_prefixes`.

### Convention guarantees

- `.hpc/tasks.py` shape: `total() -> int` and `resolve(i: int) -> dict`.
- Idempotency-skip on resubmit when `result_dir/metrics.json` exists with
  non-zero size.
- `cmd_sha` is the dedup key (not `run_id`).
- Python ≥ 3.10.

## Key Design Points

- **Tier-1 probes stay local** — unchanged `uv run python probe.py`.
- Tier-2 runs exceeding local capacity delegate to `hpc-agent`; otherwise
  stay local.
- The agent decides per-run whether to delegate (opt-in, gated by
  `experiment.hpc.enabled`).
- No directory restructuring, no changes to `meta.json`, no changes to
  `results/metrics.json` schema.
- claude-hpc cannot kill cluster jobs by design (`scancel` / `qdel` are
  denied). If MARs decides a run is bad, stop polling and let it expire.
