import React, { useState, useMemo, useEffect } from 'react'
import { Box, Text, useInput } from 'ink'
import TextInput from 'ink-text-input'
import {
  loadConfig,
  saveConfig,
  getDefaultConfig,
  getNestedValue,
  setNestedValue,
} from '../../paper/config-io'
import { useFullscreenDimensions } from './FullscreenLayout'
import { getTheme } from '@utils/theme'

// ── Types ─────────────────────────────────────────────

type FieldType = 'string' | 'number' | 'boolean' | 'enum' | 'masked_string'

interface FieldDef {
  key: string
  label: string
  type: FieldType
  description: string
  options?: string[] // for enum
}

interface SectionDef {
  name: string
  fields: FieldDef[]
}

// ── Section Schema ────────────────────────────────────

const MODEL_ROLE_DESCRIPTIONS: Record<string, string> = {
  research: 'Builder, Arbiter, deep research, PDF extraction, investigator',
  reasoning: 'Math proofs, formal verification (reasoning tokens)',
  coding: 'Experiment code, system tasks, agent execution',
  writing: 'LaTeX fragments, paper assembly, revision handling',
  review: 'Skeptic phase, peer review, novelty checks',
  quick: 'State enrichment, evidence extraction, lightweight tasks',
}

const MODEL_ROLES = [
  'research',
  'reasoning',
  'coding',
  'writing',
  'review',
  'quick',
]

const SECTIONS: SectionDef[] = [
  {
    name: 'Models',
    fields: MODEL_ROLES.map(role => ({
      key: `models.${role}`,
      label: role,
      type: 'string' as FieldType,
      description: MODEL_ROLE_DESCRIPTIONS[role] ?? '',
    })),
  },
  {
    name: 'Advanced',
    fields: [], // handled specially
  },
  {
    name: 'Paper',
    fields: [
      {
        key: 'paper.template',
        label: 'template',
        type: 'enum',
        description: 'LaTeX template for paper output',
        options: ['neurips', 'icml', 'aaai', 'acl', 'jfe', 'custom'],
      },
      {
        key: 'paper.compiler',
        label: 'compiler',
        type: 'enum',
        description: 'LaTeX compiler to use',
        options: ['pdflatex', 'xelatex', 'lualatex'],
      },
      {
        key: 'paper.language',
        label: 'language',
        type: 'enum',
        description: 'Primary paper language',
        options: ['english', 'chinese'],
      },
      {
        key: 'paper.max_pages',
        label: 'max_pages',
        type: 'number',
        description: 'Maximum page count for paper',
      },
    ],
  },
  {
    name: 'Review',
    fields: [
      {
        key: 'review.num_reviewers',
        label: 'num_reviewers',
        type: 'number',
        description: 'Number of parallel reviewers',
      },
      {
        key: 'review.max_rounds',
        label: 'max_rounds',
        type: 'number',
        description: 'Maximum revision rounds',
      },
      {
        key: 'review.strength',
        label: 'strength',
        type: 'enum',
        description: 'Review stringency level',
        options: ['lenient', 'standard', 'strict'],
      },
      {
        key: 'review.acceptance_threshold',
        label: 'threshold',
        type: 'number',
        description: 'Minimum score for acceptance (1-10)',
      },
      {
        key: 'review.ground_in_literature',
        label: 'ground_in_literature',
        type: 'boolean',
        description: 'Ground reviews in latest literature',
      },
      {
        key: 'review.check_novelty',
        label: 'check_novelty',
        type: 'boolean',
        description: 'Run novelty checks during review',
      },
      {
        key: 'review.auto_accept',
        label: 'auto_accept',
        type: 'boolean',
        description: 'Auto-accept papers above threshold',
      },
    ],
  },
  {
    name: 'Proposals',
    fields: [
      {
        key: 'proposals.count',
        label: 'count',
        type: 'number',
        description: 'Number of proposals to generate',
      },
      {
        key: 'proposals.detail_level',
        label: 'detail_level',
        type: 'enum',
        description: 'Proposal detail level',
        options: ['brief', 'standard', 'full'],
      },
      {
        key: 'proposals.include_feasibility',
        label: 'include_feasibility',
        type: 'boolean',
        description: 'Include feasibility analysis',
      },
      {
        key: 'proposals.include_risk',
        label: 'include_risk',
        type: 'boolean',
        description: 'Include risk assessment',
      },
      {
        key: 'proposals.include_timeline',
        label: 'include_timeline',
        type: 'boolean',
        description: 'Include estimated timeline',
      },
      {
        key: 'proposals.auto_novelty_check',
        label: 'auto_novelty_check',
        type: 'boolean',
        description: 'Automatically check novelty for proposals',
      },
    ],
  },
  {
    name: 'Experiment',
    fields: [
      {
        key: 'experiment.python_version',
        label: 'python_version',
        type: 'string',
        description: 'Python version for experiment environments',
      },
      {
        key: 'experiment.max_runtime_hours',
        label: 'max_runtime_hours',
        type: 'number',
        description: 'Maximum experiment runtime in hours',
      },
      {
        key: 'experiment.max_retries',
        label: 'max_retries',
        type: 'number',
        description: 'Maximum retry attempts on error',
      },
      {
        key: 'experiment.gpu_required',
        label: 'gpu_required',
        type: 'boolean',
        description: 'Require GPU for experiments',
      },
      {
        key: 'experiment.auto_retry_on_error',
        label: 'auto_retry_on_error',
        type: 'boolean',
        description: 'Automatically retry failed experiments',
      },
      {
        key: 'experiment.prefer_docker',
        label: 'prefer_docker',
        type: 'boolean',
        description: 'Prefer Docker for experiment isolation',
      },
      {
        key: 'experiment.hpc.enabled',
        label: 'hpc.enabled',
        type: 'boolean',
        description: 'Allow tier-2 runs to delegate to an HPC cluster via claude-hpc',
      },
      {
        key: 'experiment.hpc.default_cluster',
        label: 'hpc.default_cluster',
        type: 'string',
        description: 'Cluster name from clusters.yaml (e.g., hoffman2)',
      },
      {
        key: 'experiment.hpc.delegate_when_tasks_over',
        label: 'hpc.delegate_when_tasks_over',
        type: 'number',
        description: 'Delegate to HPC when grid size exceeds this many tasks',
      },
      {
        key: 'experiment.hpc.delegate_when_walltime_minutes_over',
        label: 'hpc.delegate_when_walltime_minutes_over',
        type: 'number',
        description: 'Delegate to HPC when estimated walltime exceeds this many minutes',
      },
    ],
  },
]

// Advanced fields per model
const ADVANCED_FIELDS: FieldDef[] = [
  {
    key: 'thinking_effort',
    label: 'thinking_effort',
    type: 'enum',
    description: 'Reasoning effort level',
    options: ['low', 'medium', 'high', 'max'],
  },
  {
    key: 'temperature',
    label: 'temperature',
    type: 'number',
    description: 'Sampling temperature (0-2)',
  },
  {
    key: 'max_output_tokens',
    label: 'max_output_tokens',
    type: 'number',
    description: 'Maximum output tokens per request',
  },
  {
    key: 'base_url',
    label: 'base_url',
    type: 'string',
    description: 'Custom API base URL',
  },
  {
    key: 'api_key',
    label: 'api_key',
    type: 'masked_string',
    description: 'API key override for this model',
  },
]

// ── Component ─────────────────────────────────────────

export interface SettingsPanelProps {
  onDone: (result?: string) => void
}

export function SettingsPanel({ onDone }: SettingsPanelProps): React.ReactNode {
  const theme = getTheme()
  const { contentHeight, contentWidth } = useFullscreenDimensions()

  const [activeSection, setActiveSection] = useState(0)
  const [cursor, setCursor] = useState(0)
  const [editing, setEditing] = useState(false)
  const [editValue, setEditValue] = useState('')
  const [config, setConfig] = useState(() => loadConfig())
  const [saveFlash, setSaveFlash] = useState<string | null>(null)

  // Advanced section state
  const [advancedExpanded, setAdvancedExpanded] = useState(false)
  const [advancedRoleIdx, setAdvancedRoleIdx] = useState(0)

  const section = SECTIONS[activeSection]
  const isAdvanced = section.name === 'Advanced'

  // Compute visible fields based on section + advanced state
  const visibleFields = useMemo((): FieldDef[] => {
    if (!isAdvanced) return section.fields
    if (!advancedExpanded) {
      return MODEL_ROLES.map(role => ({
        key: `models.${role}`,
        label: role,
        type: 'string' as FieldType,
        description: `Advanced settings for ${role} model`,
      }))
    }
    return ADVANCED_FIELDS
  }, [isAdvanced, advancedExpanded, section])

  // Get the current advanced model spec for config key mapping
  const advancedModelSpec = useMemo((): string => {
    if (!isAdvanced || !advancedExpanded) return ''
    const modelKey = `models.${MODEL_ROLES[advancedRoleIdx]}`
    return (getNestedValue(config, modelKey) as string) ?? ''
  }, [isAdvanced, advancedExpanded, advancedRoleIdx, config])

  // Resolve full config key for a field
  function resolveKey(field: FieldDef): string {
    if (isAdvanced && advancedExpanded) {
      return `advanced_models.${advancedModelSpec}.${field.key}`
    }
    return field.key
  }

  function getFieldValue(field: FieldDef): any {
    return getNestedValue(config, resolveKey(field))
  }

  function updateField(field: FieldDef, value: any): void {
    const newConfig = JSON.parse(JSON.stringify(config))
    setNestedValue(newConfig, resolveKey(field), value)
    setConfig(newConfig)
    saveConfig(newConfig)
    flashSaved()
  }

  function flashSaved(): void {
    setSaveFlash('Saved')
    setTimeout(() => setSaveFlash(null), 1000)
  }

  // ── Navigation input ──
  useInput(
    (input, key) => {
      if (editing) return

      // Section switching
      if (key.leftArrow || (key.tab && key.shift)) {
        setActiveSection(i => (i - 1 + SECTIONS.length) % SECTIONS.length)
        setCursor(0)
        setAdvancedExpanded(false)
        return
      }
      if (key.rightArrow || (key.tab && !key.shift)) {
        setActiveSection(i => (i + 1) % SECTIONS.length)
        setCursor(0)
        setAdvancedExpanded(false)
        return
      }

      // Cursor movement
      if (key.upArrow) {
        setCursor(c => Math.max(0, c - 1))
        return
      }
      if (key.downArrow) {
        setCursor(c => Math.min(visibleFields.length - 1, c + 1))
        return
      }

      // Enter
      if (key.return && visibleFields.length > 0) {
        const field = visibleFields[cursor]

        // Advanced: role list → expand
        if (isAdvanced && !advancedExpanded) {
          setAdvancedRoleIdx(cursor)
          setAdvancedExpanded(true)
          setCursor(0)
          return
        }

        if (field.type === 'boolean') {
          const current = getFieldValue(field)
          updateField(field, !current)
          return
        }

        if (field.type === 'enum' && field.options) {
          const current = getFieldValue(field) ?? field.options[0]
          const idx = field.options.indexOf(String(current))
          const next = field.options[(idx + 1) % field.options.length]
          updateField(field, next)
          return
        }

        // String, number, masked_string → enter edit mode
        const current = getFieldValue(field)
        setEditValue(
          field.type === 'masked_string' ? '' : String(current ?? ''),
        )
        setEditing(true)
        return
      }

      // Esc
      if (key.escape) {
        if (isAdvanced && advancedExpanded) {
          setAdvancedExpanded(false)
          setCursor(advancedRoleIdx)
          return
        }
        onDone()
        return
      }
    },
    { isActive: !editing },
  )

  // ── Editing input ──
  function handleEditSubmit(value: string): void {
    const field = visibleFields[cursor]
    let parsed: any = value
    if (field.type === 'number') {
      const num = Number(value)
      if (isNaN(num)) {
        setEditing(false)
        return
      }
      parsed = num
    }
    updateField(field, parsed)
    setEditing(false)
  }

  function handleEditCancel(): void {
    setEditing(false)
  }

  // ── Render helpers ──

  function renderTabBar(): React.ReactNode {
    return (
      <Box marginBottom={1}>
        {SECTIONS.map((sec, i) => (
          <Box key={sec.name} marginRight={2}>
            <Text
              bold={i === activeSection}
              color={i === activeSection ? theme.primary : undefined}
              dimColor={i !== activeSection}
            >
              {i === activeSection ? `[${sec.name}]` : sec.name}
            </Text>
          </Box>
        ))}
      </Box>
    )
  }

  function formatValue(
    field: FieldDef,
    value: any,
    isFocused: boolean,
  ): React.ReactNode {
    if (field.type === 'boolean') {
      return value ? (
        <Text color={theme.success}>[ON]</Text>
      ) : (
        <Text dimColor>[OFF]</Text>
      )
    }

    if (field.type === 'enum' && isFocused) {
      return (
        <Text color={theme.primary}>
          {'< '}
          {String(value ?? field.options?.[0] ?? '')}
          {' >'}
        </Text>
      )
    }

    if (field.type === 'masked_string') {
      const str = String(value ?? '')
      if (!str) return <Text dimColor>(not set)</Text>
      const masked = '****' + str.slice(-4)
      return <Text color={isFocused ? theme.primary : undefined}>{masked}</Text>
    }

    if (value === undefined || value === null || value === '') {
      return <Text dimColor>(not set)</Text>
    }

    return (
      <Text color={isFocused ? theme.primary : undefined}>{String(value)}</Text>
    )
  }

  function renderFieldList(): React.ReactNode {
    if (visibleFields.length === 0) {
      return <Text dimColor>No fields in this section</Text>
    }

    // For advanced expanded, show a breadcrumb header
    const header =
      isAdvanced && advancedExpanded ? (
        <Box marginBottom={1}>
          <Text dimColor>
            Advanced {'>'} {MODEL_ROLES[advancedRoleIdx]} (
            {advancedModelSpec || 'not set'})
          </Text>
        </Box>
      ) : null

    const maxLabelLen = Math.max(...visibleFields.map(f => f.label.length), 8)

    return (
      <Box flexDirection="column">
        {header}
        {visibleFields.map((field, i) => {
          const isFocused = i === cursor
          const value = getFieldValue(field)
          const isEditingThis = editing && isFocused

          // For advanced role list, show model string + indicator
          const displayValue =
            isAdvanced && !advancedExpanded
              ? getNestedValue(config, field.key)
              : value

          const hasAdvancedConfig =
            isAdvanced &&
            !advancedExpanded &&
            getNestedValue(
              config,
              `advanced_models.${getNestedValue(config, field.key)}`,
            )

          return (
            <Box key={field.key}>
              <Text color={isFocused ? theme.primary : undefined}>
                {isFocused ? '> ' : '  '}
              </Text>
              <Text
                bold={isFocused}
                color={isFocused ? theme.primary : undefined}
              >
                {field.label.padEnd(maxLabelLen)}
              </Text>
              <Text> </Text>
              {isEditingThis ? (
                <Box>
                  <TextInput
                    value={editValue}
                    onChange={setEditValue}
                    onSubmit={handleEditSubmit}
                  />
                </Box>
              ) : (
                <Box>
                  {formatValue(
                    isAdvanced && !advancedExpanded
                      ? { ...field, type: 'string' }
                      : field,
                    displayValue,
                    isFocused,
                  )}
                  {hasAdvancedConfig && <Text dimColor> *</Text>}
                  {isFocused && saveFlash && (
                    <Text color={theme.success}> {saveFlash}</Text>
                  )}
                </Box>
              )}
            </Box>
          )
        })}
      </Box>
    )
  }

  function renderDescription(): React.ReactNode {
    if (visibleFields.length === 0) return null
    const field = visibleFields[cursor]
    if (!field) return null

    // For model fields, show role description
    const desc =
      !isAdvanced && activeSection === 0 && MODEL_ROLE_DESCRIPTIONS[field.label]
        ? MODEL_ROLE_DESCRIPTIONS[field.label]
        : field.description

    return (
      <Box marginTop={1}>
        <Text dimColor>{desc}</Text>
      </Box>
    )
  }

  // Handle Esc during editing via useInput since TextInput doesn't support onCancel
  useInput(
    (_input, key) => {
      if (key.escape) {
        handleEditCancel()
      }
    },
    { isActive: editing },
  )

  return (
    <Box flexDirection="column" height={contentHeight}>
      {renderTabBar()}
      <Box flexDirection="column" flexGrow={1}>
        {renderFieldList()}
      </Box>
      {renderDescription()}
    </Box>
  )
}
