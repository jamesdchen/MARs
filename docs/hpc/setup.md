# HPC (Cluster) Execution Setup

MARs delegates large Tier 2 runs to a cluster via
[`claude-hpc`](https://github.com/jamesdchen/claude-hpc), a Python orchestrator
that submits and monitors array-batch jobs on SGE/SLURM. The
`experiment-runner` agent invokes `hpc-agent` from its `Bash` tool when a
grid exceeds local capacity.

This guide covers the one-time operator setup. The integration contract that
the agent follows is vendored at
[`docs/hpc/integration-reference.md`](integration-reference.md).

## Prerequisites

- SSH access to your cluster login node, with key-based auth configured.
- `ssh` and `rsync` on `PATH`.
- `ssh-agent` running with your key loaded. Confirm with:
  ```bash
  ssh-add -l           # should list your key
  echo $SSH_AUTH_SOCK  # should be a non-empty socket path
  ```
  If `SSH_AUTH_SOCK` is empty in the shell that launches MARs, every
  cluster call will hang on auth — this is the single most common spawn
  failure.
- `uv` installed (already required by MARs).

## Install `claude-hpc`

Each Tier 2 experiment auto-installs `claude-hpc` into its own `uv`
environment, pinned to a specific upstream commit via a git+ URL in
`src/paper/experiments/environment.ts` (claude-hpc isn't on PyPI yet). You
don't need a global install for the experiment-runner agent to work, but
the cluster scaffolding will hit GitHub at first `uv sync` for each
experiment.

For interactive use of the upstream Claude Code slash commands
(`/preflight`, `/submit-hpc`, `/monitor-hpc`, `/aggregate-hpc`,
`/campaign-hpc`) outside MARs:

```bash
uv tool install claude-hpc
```

## Configure your clusters

1. Copy the template:
   ```bash
   cp docs/hpc/clusters.yaml.example ~/.claude-hpc/clusters.yaml
   ```
2. Edit it — set `host`, `user`, `scheduler` (`sge` or `slurm`), `scratch`,
   `modules`, `conda_source`, and any optional `conda_envs` / `gpu_types`.
3. Point `claude-hpc` at it:
   ```bash
   export HPC_CLUSTERS_CONFIG=$HOME/.claude-hpc/clusters.yaml
   ```
   Add this to your shell profile so MARs inherits it. Do **not** commit
   your edited `clusters.yaml` to MARs — it contains your username and
   site-specific paths.

## Enable HPC in MARs config

Turn the HPC delegation on and pick a default cluster:

```bash
cpaper config set experiment.hpc.enabled true
cpaper config set experiment.hpc.default_cluster <name>     # e.g. hoffman2
```

Optional thresholds (defaults shown):

```bash
cpaper config set experiment.hpc.delegate_when_tasks_over 8
cpaper config set experiment.hpc.delegate_when_walltime_minutes_over 30
```

The agent checks these before delegating. With `enabled=false` (default), every
Tier 2 run stays local — same behavior as before this integration.

## Smoke test

Inside any Tier 2 experiment directory (or any uv project with `claude-hpc`
installed):

```bash
uv run hpc-agent preflight --cluster <name>
```

Expected: `{"ok": true, "data": {"all_ok": true, ...}}`. If `data.all_ok` is
false, inspect `data.checks[]` — the most actionable failures are
`ssh_auth_sock` (re-load your key into ssh-agent) and `cluster_tcp_22`
(cluster offline or hostname wrong in `clusters.yaml`).

## What MARs forwards automatically

When the experiment-runner runs `hpc-agent` inside an experiment via
`runInEnv`, MARs's `Bun.spawn` env block explicitly forwards:

| Variable               | Source / default                                          |
|------------------------|-----------------------------------------------------------|
| `SSH_AUTH_SOCK`        | parent env                                                |
| `SSH_AGENT_PID`        | parent env                                                |
| `HPC_JOURNAL_DIR`      | `$HPC_JOURNAL_DIR` if set, else `~/.mars/hpc/<exp-name>/` |
| `HPC_CLUSTERS_CONFIG`  | parent env (you set it)                                   |
| `PATH`, `HOME`, etc.   | parent env                                                |

The per-experiment default for `HPC_JOURNAL_DIR` keeps concurrent MARs runs
from sharing claude-hpc state.

## Further reading

- [`docs/hpc/integration-reference.md`](integration-reference.md) — full
  contract: env vars, error_code → retry policy, `.hpc/tasks.py` shape,
  troubleshooting.
- [`agents/experiment-runner.md`](../../agents/experiment-runner.md) — the
  prompt the agent runs against (Cluster Execution section).
- Upstream: <https://github.com/jamesdchen/claude-hpc> — primitive catalog
  and slash-command reference.
