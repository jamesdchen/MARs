# claude-hpc Integration Reference (Vendored)

> **Vendored from** <https://github.com/jamesdchen/claude-hpc>
> **Source:** `docs/workflows/mars-integration.md` @ `ec041c6399adc17c0f96d2fd10c5478aea30d7f2`
> **Synced:** 2026-05-15
> **Re-sync** when upgrading the pinned `claude-hpc` range in
> `src/paper/experiments/environment.ts`.

This file is the contract MARs's `experiment-runner` agent and tier-2
experiment scaffolding depend on. The runtime source-of-truth for the agent
is the "Cluster Execution (Optional)" section in
[`agents/experiment-runner.md`](../../agents/experiment-runner.md); this
document is the human-readable reference behind it.

---

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
