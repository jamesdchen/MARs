import { existsSync, readFileSync } from 'fs'
import { join } from 'path'
import os from 'os'

// ── Research Stance ──

export type ResearchStance = 'exploratory' | 'standard'

// ── Model Assignments ──

export interface ModelAssignments {
  research: string
  reasoning: string
  reasoning_deep: string
  coding: string
  writing: string
  review: string
  quick: string
}

export const DEFAULT_MODEL_ASSIGNMENTS: ModelAssignments = {
  research: 'anthropic:claude-opus-4-6',
  reasoning: 'openai:gpt-5.4',
  reasoning_deep: 'openai:gpt-5.4-pro',
  coding: 'anthropic:claude-opus-4-6',
  writing: 'anthropic:claude-opus-4-6',
  review: 'openai:gpt-5.4',
  quick: 'anthropic:claude-opus-4-6',
}

let _cachedAssignments: ModelAssignments | null = null

/**
 * Get model assignments, loading from config if available.
 * Merges user config over defaults so partially-configured setups work.
 */
export function getModelAssignments(): ModelAssignments {
  if (_cachedAssignments) return _cachedAssignments

  try {
    const configPath = join(os.homedir(), '.claude-paper', 'config.json')
    if (existsSync(configPath)) {
      const config = JSON.parse(readFileSync(configPath, 'utf-8'))
      if (config.models && typeof config.models === 'object') {
        _cachedAssignments = {
          ...DEFAULT_MODEL_ASSIGNMENTS,
          ...config.models,
        }
        return _cachedAssignments
      }
    }
  } catch {
    // Fall through to defaults
  }

  _cachedAssignments = DEFAULT_MODEL_ASSIGNMENTS
  return _cachedAssignments
}

// ── Project State (lightweight, for /paper init; cognitive state is in ResearchState) ──

export interface ProjectState {
  id: string
  name: string
  topic: string
  created_at: string
  updated_at: string

  model_assignments: ModelAssignments

  artifacts: {
    literature_db: string
    selected_proposal: string | null
    experiment_code: string | null
    results_dir: string | null
    paper_tex: string | null
    compiled_pdf: string | null
  }
}

// ── Project Config ──

export interface ProjectConfig {
  model_assignments: ModelAssignments
  orchestrator?: {
    rigor_level?: 1 | 2 | 3
  }
  paper: {
    template: string
    compiler: 'pdflatex' | 'xelatex' | 'lualatex'
    language: string
    max_pages: number
    target_venue: string
  }
  literature: {
    sources: string[]
    arxiv_categories: string[]
    max_papers: number
    year_from: number
    citation_threshold: number
  }
  experiment: {
    python_version: string
    gpu_required: boolean
    max_runtime_hours: number
    auto_retry_on_error: boolean
    max_retries: number
    hpc?: {
      enabled: boolean
      default_cluster: string
      delegate_when_tasks_over: number
      delegate_when_walltime_minutes_over: number
    }
  }
  proposals: {
    count: number
    include_feasibility: boolean
    auto_novelty_check: boolean
  }
  review: {
    num_reviewers: number
    max_revision_rounds: number
    acceptance_threshold: number
    auto_accept: boolean
  }
  budget: {
    total_usd: number
    warn_at_percent: number
  }
  auto_mode: boolean
}

// ── Checkpoint ──

export interface CheckpointData {
  label: string
  timestamp: string
  state_snapshot: ProjectState
  metadata: Record<string, unknown>
}

// ── History ──

export interface HistoryEntry {
  timestamp: string
  action: string
  details: string
}
