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
      'claude-hpc>=0.1,<1.0',
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
        // Explicit forwards for hpc-agent: missing SSH_AUTH_SOCK is the most
        // common cluster-call failure (every call hangs on auth).
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
