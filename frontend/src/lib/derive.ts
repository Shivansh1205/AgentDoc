/**
 * Everything the dashboard shows that isn't literally in the report file.
 *
 * The report gives turns and failures; the panels need agents, edges, tool
 * satellites and per-agent failure counts. Deriving them here (once, memoized
 * by the caller) keeps the views themselves declarative, and means an
 * uploaded report and the sample data go through identical code.
 */

import type { AgentDocReport, MastCategory, Turn } from './schema'
import { CATEGORY_ORDER } from './schema'

export interface AgentSummary {
  name: string
  turnCount: number
  failureCount: number
  /** Worst category affecting this agent, by MAST order; null when healthy. */
  category: MastCategory | null
  steps: number[]
}

export interface GraphEdge {
  source: string
  target: string
  /** True when the trace declared the handoff rather than us inferring it. */
  explicit: boolean
  category: MastCategory | null
  /** Preview of what the source agent said when control moved. */
  preview: string
}

export interface DerivedTrace {
  agents: AgentSummary[]
  edges: GraphEdge[]
  /** Failure categories affecting each step, for timeline block coloring. */
  categoryByStep: Map<number, MastCategory>
  toolCallsByAgent: Map<string, string[]>
  totalToolCalls: number
  handoffCount: number
  averageConfidence: number
  topFailureMode: string | null
}

function worstCategory(categories: MastCategory[]): MastCategory | null {
  for (const candidate of CATEGORY_ORDER) {
    if (categories.includes(candidate)) return candidate
  }
  return null
}

function previewOf(turn: Turn): string {
  const text = turn.content?.trim()
  if (text) return text.length > 140 ? `${text.slice(0, 140)}...` : text
  if (turn.tool_calls.length) {
    return turn.tool_calls.map((call) => call.name).join(', ')
  }
  return 'No message content'
}

export function deriveTrace(report: AgentDocReport): DerivedTrace {
  const { turns, flagged_failures: failures } = report

  // Which categories hit which step. A step can be named by several failures.
  const categoriesByStep = new Map<number, MastCategory[]>()
  for (const failure of failures) {
    for (const step of failure.turn_indices) {
      const list = categoriesByStep.get(step) ?? []
      list.push(failure.category)
      categoriesByStep.set(step, list)
    }
  }

  const categoryByStep = new Map<number, MastCategory>()
  for (const [step, categories] of categoriesByStep) {
    const worst = worstCategory(categories)
    if (worst) categoryByStep.set(step, worst)
  }

  // Agents, in first-appearance order so the roster reads like the trace runs.
  const agentTurns = turns.filter((turn) => turn.agent)
  const order: string[] = []
  const byAgent = new Map<string, Turn[]>()
  for (const turn of agentTurns) {
    const name = turn.agent as string
    if (!byAgent.has(name)) {
      byAgent.set(name, [])
      order.push(name)
    }
    byAgent.get(name)!.push(turn)
  }

  const agents: AgentSummary[] = order.map((name) => {
    const owned = byAgent.get(name)!
    const steps = owned.map((turn) => turn.step)
    const hits: MastCategory[] = []
    for (const step of steps) {
      const stepCategories = categoriesByStep.get(step)
      if (stepCategories) hits.push(...stepCategories)
    }
    return {
      name,
      turnCount: owned.length,
      failureCount: steps.filter((step) => categoriesByStep.has(step)).length,
      category: worstCategory(hits),
      steps,
    }
  })

  // Edges. An explicit `handoff_to` is authoritative; otherwise fall back to
  // inferring a handoff wherever the acting agent changes between turns. The
  // two strategies are never mixed, matching how the Python HTML report
  // decides - a trace that declares any handoff is trusted to declare them all.
  const hasExplicit = agentTurns.some((turn) => turn.handoff_to)
  const edges: GraphEdge[] = []
  const seen = new Set<string>()

  const pushEdge = (
    source: string,
    target: string,
    explicit: boolean,
    turn: Turn,
  ) => {
    if (source === target) return
    const key = `${source}->${target}`
    if (seen.has(key)) return
    seen.add(key)
    edges.push({
      source,
      target,
      explicit,
      category: categoryByStep.get(turn.step) ?? null,
      preview: previewOf(turn),
    })
  }

  if (hasExplicit) {
    for (const turn of agentTurns) {
      if (turn.handoff_to && turn.agent) {
        pushEdge(turn.agent, turn.handoff_to, true, turn)
      }
    }
  } else {
    for (let i = 1; i < agentTurns.length; i += 1) {
      const previous = agentTurns[i - 1]
      const current = agentTurns[i]
      if (previous.agent && current.agent && previous.agent !== current.agent) {
        pushEdge(previous.agent, current.agent, false, previous)
      }
    }
  }

  const toolCallsByAgent = new Map<string, string[]>()
  let totalToolCalls = 0
  for (const turn of agentTurns) {
    if (!turn.agent || !turn.tool_calls.length) continue
    const names = toolCallsByAgent.get(turn.agent) ?? []
    for (const call of turn.tool_calls) {
      totalToolCalls += 1
      if (!names.includes(call.name)) names.push(call.name)
    }
    toolCallsByAgent.set(turn.agent, names)
  }

  const averageConfidence = failures.length
    ? failures.reduce((sum, failure) => sum + failure.confidence, 0) /
      failures.length
    : 0

  return {
    agents,
    edges,
    categoryByStep,
    toolCallsByAgent,
    totalToolCalls,
    handoffCount: agentTurns.filter((turn) => turn.handoff_to).length,
    averageConfidence,
    topFailureMode: report.ranked_failure_modes[0]?.failure_mode ?? null,
  }
}
