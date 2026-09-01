/**
 * The shape `agentdoc diagnose --json` emits (schema_version 2).
 *
 * Kept deliberately close to the Python side (`agentdoc/report/json_export.py`)
 * so a report can be dropped in with no transformation step. Anything the
 * dashboard derives - agent rosters, handoff edges, per-agent failure counts -
 * is computed in `derive.ts` rather than expected from the file.
 */

export type MastCategory =
  | 'system_design_issues'
  | 'inter_agent_misalignment'
  | 'task_verification'

export interface ToolCall {
  name: string
  call_id: string | null
  args: Record<string, unknown>
  result: string | null
  error: string | null
}

export interface Turn {
  step: number
  role: 'system' | 'human' | 'agent' | 'tool'
  agent: string | null
  content: string | null
  tool_calls: ToolCall[]
  timestamp: string | null
  parent_step: number | null
  handoff_to: string | null
  metadata: Record<string, unknown>
}

export interface FlaggedFailure {
  failure_mode: string
  category: MastCategory
  turn_indices: number[]
  justification: string
  confidence: number
}

export interface AgentDocReport {
  schema_version: number
  model: string | null
  source_framework: string | null
  trace_turn_count: number
  total_failures: number
  narrative: string
  category_counts: { category: MastCategory; count: number }[]
  ranked_failure_modes: { failure_mode: string; count: number }[]
  flagged_failures: FlaggedFailure[]
  turns: Turn[]
}

export const CATEGORY_LABEL: Record<MastCategory, string> = {
  system_design_issues: 'System Design Issues',
  inter_agent_misalignment: 'Inter-Agent Misalignment',
  task_verification: 'Task Verification',
}

export const CATEGORY_SHORT: Record<MastCategory, string> = {
  system_design_issues: 'System Design',
  inter_agent_misalignment: 'Misalignment',
  task_verification: 'Verification',
}

export const CATEGORY_COLOR: Record<MastCategory, string> = {
  system_design_issues: 'var(--color-design)',
  inter_agent_misalignment: 'var(--color-align)',
  task_verification: 'var(--color-verify)',
}

export const CATEGORY_ORDER: MastCategory[] = [
  'system_design_issues',
  'inter_agent_misalignment',
  'task_verification',
]

/** Human-readable names for the MAST modes, keyed by paper ID. */
export const FAILURE_MODE_NAME: Record<string, string> = {
  'FM-1.1': 'Disobey task specification',
  'FM-1.2': 'Disobey role specification',
  'FM-1.3': 'Step repetition',
  'FM-1.4': 'Loss of conversation history',
  'FM-1.5': 'Unaware of termination conditions',
  'FM-2.1': 'Conversation reset',
  'FM-2.2': 'Fail to ask for clarification',
  'FM-2.3': 'Task derailment',
  'FM-2.4': 'Information withholding',
  'FM-2.5': "Ignored other agent's input",
  'FM-2.6': 'Reasoning-action mismatch',
  'FM-3.1': 'Premature termination',
  'FM-3.2': 'No or incomplete verification',
  'FM-3.3': 'Incorrect verification',
}

export function failureModeName(id: string): string {
  return FAILURE_MODE_NAME[id] ?? id
}

export class ReportValidationError extends Error {}

/**
 * Validate an arbitrary parsed JSON value as an AgentDoc report.
 *
 * Checks the fields the dashboard actually reads, and reports the first
 * problem in the interface's own vocabulary - a message a person can act on,
 * not a schema dump. A file missing `turns` is called out specifically
 * because it is the common case: a report from schema v1, before turns were
 * exported.
 */
export function validateReport(data: unknown): AgentDocReport {
  if (data === null || typeof data !== 'object' || Array.isArray(data)) {
    throw new ReportValidationError('This file contains JSON, but not an object.')
  }

  const d = data as Record<string, unknown>

  if (!('flagged_failures' in d) || !('turns' in d)) {
    if ('flagged_failures' in d && !('turns' in d)) {
      throw new ReportValidationError(
        'This report has no turns. Re-export it with a current agentdoc build to see the graph and timeline.',
      )
    }
    throw new ReportValidationError("This doesn't look like an AgentDoc report.")
  }

  if (!Array.isArray(d.turns) || !Array.isArray(d.flagged_failures)) {
    throw new ReportValidationError(
      'This report has turns and failures, but not in the expected format.',
    )
  }

  return {
    schema_version: typeof d.schema_version === 'number' ? d.schema_version : 0,
    model: typeof d.model === 'string' ? d.model : null,
    source_framework:
      typeof d.source_framework === 'string' ? d.source_framework : null,
    trace_turn_count:
      typeof d.trace_turn_count === 'number'
        ? d.trace_turn_count
        : (d.turns as unknown[]).length,
    total_failures:
      typeof d.total_failures === 'number'
        ? d.total_failures
        : (d.flagged_failures as unknown[]).length,
    narrative: typeof d.narrative === 'string' ? d.narrative : '',
    category_counts: Array.isArray(d.category_counts)
      ? (d.category_counts as AgentDocReport['category_counts'])
      : [],
    ranked_failure_modes: Array.isArray(d.ranked_failure_modes)
      ? (d.ranked_failure_modes as AgentDocReport['ranked_failure_modes'])
      : [],
    flagged_failures: d.flagged_failures as FlaggedFailure[],
    turns: d.turns as Turn[],
  }
}
