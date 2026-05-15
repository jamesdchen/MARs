import { describe, it, expect } from 'bun:test'
import { mkdtempSync, existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import {
  ExperimentEnvironment,
  slugify,
} from '../../src/paper/experiments/index'

function makeTmpDir(): string {
  return mkdtempSync(join(tmpdir(), 'exp-env-test-'))
}

// ── slugify ────────────────────────────────────────────────────────

describe('slugify', () => {
  it('lowercases and replaces spaces with hyphens', () => {
    expect(slugify('Hello World')).toBe('hello-world')
  })

  it('replaces underscores with hyphens', () => {
    expect(slugify('foo_bar')).toBe('foo-bar')
  })

  it('strips special characters and trims hyphens', () => {
    expect(slugify('GARCH Sanity Check!!!')).toBe('garch-sanity-check')
  })

  it('returns empty string for empty input', () => {
    expect(slugify('')).toBe('')
  })

  it('collapses multiple hyphens', () => {
    expect(slugify('a---b')).toBe('a-b')
  })

  it('handles mixed separators', () => {
    expect(slugify('one_two three__four')).toBe('one-two-three-four')
  })
})

// ── Tier 1 create ──────────────────────────────────────────────────

describe('ExperimentEnvironment.create tier 1', () => {
  it('creates results/ directory', async () => {
    const tmp = makeTmpDir()
    const expDir = join(tmp, 'probe-001')
    const env = new ExperimentEnvironment(tmp)
    await env.create(expDir, 1)

    expect(existsSync(join(expDir, 'results'))).toBe(true)
  })

  it('does NOT create tier-2 directories', async () => {
    const tmp = makeTmpDir()
    const expDir = join(tmp, 'probe-002')
    const env = new ExperimentEnvironment(tmp)
    await env.create(expDir, 1)

    expect(existsSync(join(expDir, 'src'))).toBe(false)
    expect(existsSync(join(expDir, 'tests'))).toBe(false)
    expect(existsSync(join(expDir, 'configs'))).toBe(false)
    expect(existsSync(join(expDir, 'scripts'))).toBe(false)
  })

  it('generates pyproject.toml with tier-1 deps only', async () => {
    const tmp = makeTmpDir()
    const expDir = join(tmp, 'probe-003')
    const env = new ExperimentEnvironment(tmp)
    await env.create(expDir, 1)

    const content = readFileSync(join(expDir, 'pyproject.toml'), 'utf-8')
    expect(content).toContain('"numpy"')
    expect(content).toContain('"pandas"')
    expect(content).toContain('"scipy"')
    expect(content).not.toContain('"matplotlib"')
    expect(content).not.toContain('"pytest"')
    expect(content).not.toContain('"ruff"')
    expect(content).not.toContain('claude-hpc')
  })

  it('writes env_snapshot.json', async () => {
    const tmp = makeTmpDir()
    const expDir = join(tmp, 'probe-004')
    const env = new ExperimentEnvironment(tmp)
    await env.create(expDir, 1)

    expect(existsSync(join(expDir, 'env_snapshot.json'))).toBe(true)
    const snap = JSON.parse(
      readFileSync(join(expDir, 'env_snapshot.json'), 'utf-8'),
    )
    expect(snap).toHaveProperty('python_version')
    expect(snap).toHaveProperty('uv_version')
    expect(snap).toHaveProperty('platform')
    expect(snap).toHaveProperty('arch')
  })
})

// ── Tier 2 create ──────────────────────────────────────────────────

describe('ExperimentEnvironment.create tier 2', () => {
  it('creates all tier-2 directories', async () => {
    const tmp = makeTmpDir()
    const expDir = join(tmp, 'run-001')
    const env = new ExperimentEnvironment(tmp)
    await env.create(expDir, 2)

    for (const dir of ['src', 'tests', 'configs', 'scripts']) {
      expect(existsSync(join(expDir, dir))).toBe(true)
    }
    for (const sub of ['figures', 'tables', 'logs', 'statistical_tests']) {
      expect(existsSync(join(expDir, 'results', sub))).toBe(true)
    }
  })

  it('generates pyproject.toml with tier-2 deps', async () => {
    const tmp = makeTmpDir()
    const expDir = join(tmp, 'run-002')
    const env = new ExperimentEnvironment(tmp)
    await env.create(expDir, 2)

    const content = readFileSync(join(expDir, 'pyproject.toml'), 'utf-8')
    expect(content).toContain('"numpy"')
    expect(content).toContain('"matplotlib"')
    expect(content).toContain('"pytest"')
    expect(content).toContain('"ruff"')
    expect(content).toMatch(/"claude-hpc @ git\+https:\/\/github\.com\/jamesdchen\/claude-hpc\.git@[0-9a-f]+"/)
    expect(content).toContain('requires-python = ">=3.11"')
  })

  it('writes mars_hpc.py adapter for tier-2 only', async () => {
    const tmp = makeTmpDir()
    const tier2Dir = join(tmp, 'run-003')
    const tier1Dir = join(tmp, 'probe-003')
    const env = new ExperimentEnvironment(tmp)
    await env.create(tier2Dir, 2)
    await env.create(tier1Dir, 1)

    expect(existsSync(join(tier2Dir, 'mars_hpc.py'))).toBe(true)
    expect(existsSync(join(tier1Dir, 'mars_hpc.py'))).toBe(false)

    const adapter = readFileSync(join(tier2Dir, 'mars_hpc.py'), 'utf-8')
    expect(adapter).toContain('def detect_experiment_tier(')
    expect(adapter).toContain('def read_meta_json(')
    expect(adapter).toContain('def discover_with_meta(')
    expect(adapter).toContain('def build_submit_spec(')
  })
})

// ── pyproject.toml content ─────────────────────────────────────────

describe('pyproject.toml content', () => {
  it('sets project name from directory basename', async () => {
    const tmp = makeTmpDir()
    const expDir = join(tmp, 'my-experiment')
    const env = new ExperimentEnvironment(tmp)
    await env.create(expDir, 1)

    const content = readFileSync(join(expDir, 'pyproject.toml'), 'utf-8')
    expect(content).toContain('name = "my-experiment"')
  })

  it('has valid [project] header', async () => {
    const tmp = makeTmpDir()
    const expDir = join(tmp, 'valid-check')
    const env = new ExperimentEnvironment(tmp)
    await env.create(expDir, 1)

    const content = readFileSync(join(expDir, 'pyproject.toml'), 'utf-8')
    expect(content).toContain('[project]')
  })
})
