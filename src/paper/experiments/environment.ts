import { mkdirSync } from 'node:fs'
import { basename, join } from 'node:path'

/**
 * Slugify a name for use as a directory/experiment identifier.
 */
export function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[\s_]+/g, '-')
    .replace(/[^a-z0-9-]/g, '')
    .replace(/[-]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/**
 * Run a command and return trimmed stdout, or fallback on failure.
 */
async function runCmdSafe(
  args: string[],
  cwd: string,
  fallback = 'unknown',
): Promise<string> {
  try {
    const proc = Bun.spawn(args, {
      cwd,
      stdout: 'pipe',
      stderr: 'pipe',
    })
    const out = await new Response(proc.stdout).text()
    await proc.exited
    return out.trim() || fallback
  } catch {
    return fallback
  }
}

/**
 * MARs-side helper code scaffolded into each tier-2 experiment dir.
 *
 * Split by concern (so .hpc/ only holds HPC-shaped code):
 *   - meta_utils.py at the experiment root: pure MARs helpers (tier
 *     detection from path layout, meta.json reading). No claude-hpc dep.
 *   - .hpc/mars_spec.py: HPC adapters that bridge to claude-hpc post-
 *     cleavage at commit 9c0e184. Imports from meta_utils.
 *
 * Keep both in sync with docs/hpc/integration-reference.md
 * ("What changed at 9c0e184").
 */
const META_UTILS_PY = `"""meta_utils — read MARs's meta.json and detect tier from path layout.

Pure MARs-side helpers; no claude-hpc dependency. Used by .hpc/mars_spec.py
and (potentially) other MARs-side scaffolding.
"""

from __future__ import annotations

import json
import pathlib


def detect_experiment_tier(experiment_dir):
    """1 for /probes/probe-*/probe.py, 2 for /runs/run-*/scripts/, else None."""
    exp = pathlib.Path(experiment_dir).resolve()
    parent = exp.parent.name
    if parent == "probes" and exp.name.startswith("probe-") and (exp / "probe.py").exists():
        return 1
    if parent == "runs" and exp.name.startswith("run-") and (exp / "scripts").is_dir():
        return 2
    return None


def read_meta_json(experiment_dir):
    """Parsed meta.json or None on missing/unreadable/invalid/non-dict. Never raises."""
    path = pathlib.Path(experiment_dir) / "meta.json"
    try:
        with path.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
`

const MARS_SPEC_PY = `"""mars_spec — MARs-side adapters that bridge to claude-hpc.

Replaces logic cleaved out of claude-hpc at commit 9c0e184:
- the data.meta enrichment on \`hpc-agent discover\`
- the \`--from-meta\` overlay for \`hpc-agent submit\`
- the auto search_dirs=["scripts"] narrowing for tier-2 runs

Imports from <experiment-root>/meta_utils.py for the MARs-shaped concerns
(tier detection, meta.json reading) that aren't actually HPC-shaped.

Invoked as a script from the experiment root:
    python .hpc/mars_spec.py discover <experiment-dir>
    python .hpc/mars_spec.py build-spec <experiment-dir> <base-spec.json>
"""

from __future__ import annotations

import json
import pathlib
import sys

# meta_utils.py lives at the experiment root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from meta_utils import detect_experiment_tier, read_meta_json  # noqa: E402


def discover_with_meta(experiment_dir):
    """Wrap \`hpc-agent discover\` and re-add the data.meta block client-side.

    For tier-2, narrows the scan to scripts/ via the Python API (the upstream
    CLI does not yet expose --search-dirs; the Python API does).
    """
    from claude_hpc.state.discover import discover_executors

    exp = pathlib.Path(experiment_dir).resolve()
    tier = detect_experiment_tier(exp)
    if tier == 2:
        executors = discover_executors(exp, search_dirs=["scripts"])
    else:
        executors = discover_executors(exp)
    envelope = {"ok": True, "data": {"executors": executors}}
    meta = read_meta_json(exp)
    if meta is not None:
        envelope["data"]["meta"] = {
            "experiment_id": meta.get("experiment_id"),
            "seed": meta.get("seed"),
            "purpose": meta.get("purpose"),
            "tier": tier,
        }
    return envelope


def build_submit_spec(experiment_dir, base_spec):
    """Overlay meta.json's experiment_id onto profile/job_name. Replaces --from-meta."""
    spec = dict(base_spec)
    meta = read_meta_json(experiment_dir)
    if meta is not None:
        exp_id = meta.get("experiment_id")
        if exp_id:
            spec.setdefault("profile", exp_id)
            spec.setdefault("job_name", exp_id)
    return spec


def _main(argv):
    cmd, exp_dir, *rest = argv[1:]
    if cmd == "discover":
        print(json.dumps(discover_with_meta(exp_dir)))
    elif cmd == "build-spec":
        base = json.loads(pathlib.Path(rest[0]).read_text())
        print(json.dumps(build_submit_spec(exp_dir, base), indent=2))
    else:
        print(f"unknown: {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
`

/**
 * Manages tiered experiment directory structures and uv-based Python environments.
 *
 * This is the NEW experiments module (plural `experiments/`).
 * The OLD `experiment/environment.ts` (singular) handles Docker/venv isolation detection.
 */
export class ExperimentEnvironment {
  constructor(private projectDir: string) {}

  /**
   * Create an experiment directory with tier-appropriate structure.
   *
   * Tier 1 (probes): lightweight, just results/
   * Tier 2 (full runs): src/, tests/, configs/, scripts/, results/ with subdirs
   */
  async create(experimentDir: string, tier: 1 | 2): Promise<void> {
    // Create directories
    if (tier === 1) {
      mkdirSync(join(experimentDir, 'results'), { recursive: true })
    } else {
      for (const dir of ['src', 'tests', 'configs', 'scripts']) {
        mkdirSync(join(experimentDir, dir), { recursive: true })
      }
      for (const sub of ['figures', 'tables', 'logs', 'statistical_tests']) {
        mkdirSync(join(experimentDir, 'results', sub), { recursive: true })
      }
    }

    // Generate pyproject.toml
    const name = basename(experimentDir)
    const tier1Deps = ['numpy', 'pandas', 'scipy']
    const tier2Deps = [
      ...tier1Deps,
      'matplotlib',
      'pytest',
      'ruff',
      'claude-hpc @ git+https://github.com/jamesdchen/claude-hpc.git@9c0e184',
    ]
    const deps = tier === 1 ? tier1Deps : tier2Deps
    const depsStr = deps.map(d => `    "${d}",`).join('\n')

    const pyproject = `[project]
name = "${name}"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
${depsStr}
]
`
    await Bun.write(join(experimentDir, 'pyproject.toml'), pyproject)

    // Tier 2 only: ship the MARs-side helpers that bridge to claude-hpc.
    // Split by concern (so .hpc/ only holds HPC-shaped code):
    //   meta_utils.py at the experiment root — pure MARs (tier detection,
    //     meta.json reading); imported by mars_spec.py.
    //   .hpc/mars_spec.py — HPC adapters (build_submit_spec,
    //     discover_with_meta) replacing logic cleaved out of claude-hpc
    //     at commit 9c0e184.
    if (tier === 2) {
      await Bun.write(join(experimentDir, 'meta_utils.py'), META_UTILS_PY)
      mkdirSync(join(experimentDir, '.hpc'), { recursive: true })
      await Bun.write(
        join(experimentDir, '.hpc', 'mars_spec.py'),
        MARS_SPEC_PY,
      )
    }

    // Run uv sync (best-effort)
    try {
      const proc = Bun.spawn(['uv', 'sync'], {
        cwd: experimentDir,
        stdout: 'pipe',
        stderr: 'pipe',
      })

      const timeout = setTimeout(() => {
        proc.kill()
      }, 30_000)

      await proc.exited
      clearTimeout(timeout)
    } catch {
      console.warn('uv sync failed or uv not found; skipping venv setup')
    }

    // Write env_snapshot.json
    const pythonRaw = await runCmdSafe(
      ['uv', 'run', 'python', '--version'],
      experimentDir,
    )
    const uvRaw = await runCmdSafe(['uv', '--version'], experimentDir)

    const snapshot = {
      python_version: pythonRaw.replace(/^Python\s*/i, '') || 'unknown',
      uv_version: uvRaw.replace(/^uv\s*/i, '') || 'unknown',
      platform: process.platform,
      arch: process.arch,
      created_at: new Date().toISOString(),
    }

    await Bun.write(
      join(experimentDir, 'env_snapshot.json'),
      JSON.stringify(snapshot, null, 2) + '\n',
    )
  }

  /**
   * Run a command inside the experiment's uv environment.
   */
  async runInEnv(
    experimentDir: string,
    command: string,
    timeoutMs = 300_000,
  ): Promise<{ exitCode: number; stdout: string; stderr: string }> {
    const proc = Bun.spawn(['bash', '-c', `uv run ${command}`], {
      cwd: experimentDir,
      stdout: 'pipe',
      stderr: 'pipe',
      env: {
        ...process.env,
        PYTHONHASHSEED: '42',
        // Explicit forwards for hpc-agent. Missing SSH_AUTH_SOCK is the most
        // common cluster-call failure (every call hangs on auth). Telemetry
        // sink defaults to "none" upstream; sending to stderr lets MARs's
        // log capture pick up claude-hpc's structured events.
        SSH_AUTH_SOCK: process.env.SSH_AUTH_SOCK ?? '',
        SSH_AGENT_PID: process.env.SSH_AGENT_PID ?? '',
        HPC_JOURNAL_DIR:
          process.env.HPC_JOURNAL_DIR ??
          join(
            process.env.HOME ?? '.',
            '.mars',
            'hpc',
            basename(experimentDir),
          ),
        HPC_SSH_TIMEOUT_SEC: process.env.HPC_SSH_TIMEOUT_SEC ?? '120',
        HPC_TELEMETRY_SINK:
          process.env.HPC_TELEMETRY_SINK ?? 'stderr-jsonl',
      },
    })

    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      proc.kill()
    }, timeoutMs)

    const [stdout, stderr] = await Promise.all([
      new Response(proc.stdout).text(),
      new Response(proc.stderr).text(),
    ])

    const exitCode = await proc.exited
    clearTimeout(timer)

    return {
      exitCode: timedOut ? 124 : exitCode,
      stdout,
      stderr,
    }
  }
}
