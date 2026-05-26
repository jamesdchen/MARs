# HPC (Cluster) Execution Setup

MARs delegates large Tier 2 runs to a cluster via
[`hpc-agent`](https://github.com/jamesdchen/hpc-agent), a Python orchestrator
that submits and monitors array-batch jobs on SGE/SLURM. The
`experiment-runner` agent invokes `hpc-agent`'s **workflow skills**
(`hpc-submit`, `hpc-status`, `hpc-aggregate`, `hpc-campaign`) via the
`Skill` tool when a grid exceeds local capacity, falling back to `Bash`
for setup-time primitives (`setup`, `describe`, `capabilities`). The
workflow skills are installed under `~/.claude/skills/` by `hpc-agent
setup --cluster <name>` and auto-discovered by the Kode substrate.

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

## Install `hpc-agent`

Each Tier 2 experiment auto-installs `hpc-agent` into its own `uv`
environment, pinned to a specific upstream commit via a git+ URL in
`src/paper/experiments/environment.ts` (hpc-agent isn't on PyPI yet). You
don't need a global install for the MARs experiment-runner to work, but
the first `uv sync` for each tier-2 experiment will hit GitHub. Python 3.10
is the floor; MARs scaffolds with 3.11 by default.

Optional: install with the forecasting extra for higher-quality queue-wait
predictions (`best-submit-window --backend des`, `predict-queue-wait`):

```bash
uv pip install 'hpc-agent[forecasting] @ git+https://github.com/jamesdchen/hpc-agent.git'
```

Without the extra, the predictor falls back to a diurnal moving-average
baseline — the calls still succeed.

**About upstream slash commands.** The hpc-agent repo ships its
`/submit-hpc`, `/monitor-hpc`, `/aggregate-hpc`, `/campaign-hpc` slashes
as a thin **interview** layer that, after collecting user input, hands
off to the same workflow skills (`hpc-submit`, `hpc-status`,
`hpc-aggregate`, `hpc-campaign`) the MARs experiment-runner invokes
directly via the Skill tool in `mode: "autonomous"`. The MARs
experiment-runner does NOT depend on the slashes — only on the workflow
skills, which `hpc-agent setup` installs under `~/.claude/skills/`. The
slashes are useful for interactive operator-driven work outside MARs.

## Configure your clusters

1. Copy the template:
   ```bash
   cp docs/hpc/clusters.yaml.example ~/.hpc-agent/clusters.yaml
   ```
2. Edit it — set `host`, `user`, `scheduler` (`sge` or `slurm`), `scratch`,
   `modules`, `conda_source`, and any optional `conda_envs` / `gpu_types`.
3. Point `hpc-agent` at it:
   ```bash
   export HPC_CLUSTERS_CONFIG=$HOME/.hpc-agent/clusters.yaml
   ```
   Add this to your shell profile so MARs inherits it. Do **not** commit
   your edited `clusters.yaml` to MARs — it contains your username and
   site-specific paths. MARs's `.gitignore` excludes `clusters.yaml`
   anywhere in the tree (the `.example` template is exempted) but the
   safest practice is to keep the edited file outside the repo.

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

Inside any Tier 2 experiment directory (or any uv project with `hpc-agent`
installed):

```bash
uv run hpc-agent setup --cluster <name>
```

`setup` composes `preflight` *and* writes a 24h cache marker that
subsequent submits skip on — so a passing `setup` doubles as the smoke
test. Expected: `{"ok": true, "data": {"all_ok": true, ...}}`. If
`data.all_ok` is false, inspect `data.checks[]` — the most actionable
failures are `ssh_auth_sock` (re-load your key into ssh-agent) and
`cluster_tcp_22` (cluster offline or hostname wrong in `clusters.yaml`).

### Verify the skills got installed

`setup` also writes the seven workflow + sub-skill files into
`~/.claude/skills/` so the Kode substrate auto-discovers them and exposes
them to the MARs experiment-runner via the Skill tool. Verify after setup
completes:

```bash
ls ~/.claude/skills/hpc-submit/SKILL.md \
   ~/.claude/skills/hpc-status/SKILL.md \
   ~/.claude/skills/hpc-aggregate/SKILL.md \
   ~/.claude/skills/hpc-campaign/SKILL.md \
   ~/.claude/skills/hpc-classify-axis/SKILL.md \
   ~/.claude/skills/hpc-wrap-entry-point/SKILL.md \
   ~/.claude/skills/hpc-build-executor/SKILL.md
```

If any file is missing, the Kode Skill mechanism won't surface the
workflow skills to `experiment-runner` and Tier 2 delegation will fall
back to error paths — this is the most common silent failure mode.
Re-run `setup --cluster <name>` (or `setup --cluster <name> --force` if
the cache marker is stale) to reinstall.

### Optional: cron-driven queue-wait forecasting

For users who want LightGBM-residual queue-wait forecasting populated
automatically, install hpc-agent's cron entry once:

```bash
uv run hpc-agent setup --cluster <name> --install-cron
```

This schedules periodic queue probes whose history feeds
`best-submit-window` / `predict-queue-wait`. Full docs in
hpc-agent-pro; without it those predictors fall back to the diurnal-MA
baseline.

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
from sharing hpc-agent state.

## Further reading

- [`docs/hpc/integration-reference.md`](integration-reference.md) — full
  contract: env vars, error_code → retry policy, `.hpc/tasks.py` shape,
  troubleshooting.
- [`agents/experiment-runner.md`](../../agents/experiment-runner.md) — the
  prompt the agent runs against (Cluster Execution section).
- Upstream: <https://github.com/jamesdchen/hpc-agent> — primitive catalog
  and slash-command reference.
