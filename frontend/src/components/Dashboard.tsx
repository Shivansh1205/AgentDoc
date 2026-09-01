import { useEffect, useMemo, useState } from 'react'
import { NetworkBackdrop } from './NetworkBackdrop'
import { AgentGraph } from './AgentGraph'
import { Timeline } from './Timeline'
import { deriveTrace } from '../lib/derive'
import {
  CATEGORY_COLOR,
  CATEGORY_ORDER,
  CATEGORY_SHORT,
  failureModeName,
  type AgentDocReport,
} from '../lib/schema'
import { cn } from '../lib/utils'

/** Counts up to `value` on mount; the stat row reads as instruments coming live. */
function useCountUp(value: number, duration = 400, delay = 0) {
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    let raf = 0
    const timer = window.setTimeout(() => {
      const start = performance.now()
      const step = (now: number) => {
        const progress = Math.min((now - start) / duration, 1)
        setDisplay(value * (1 - Math.pow(1 - progress, 3)))
        if (progress < 1) raf = requestAnimationFrame(step)
      }
      raf = requestAnimationFrame(step)
    }, delay)
    return () => {
      window.clearTimeout(timer)
      cancelAnimationFrame(raf)
    }
  }, [value, duration, delay])
  return display
}

function Stat({
  label,
  value,
  delay,
  decimals = 0,
  alert = false,
}: {
  label: string
  value: number
  delay: number
  decimals?: number
  alert?: boolean
}) {
  const display = useCountUp(value, 400, delay)
  return (
    <div
      className="rounded border border-line bg-surface/60 px-3 py-1.5 backdrop-blur-sm"
      style={
        alert && value > 0
          ? {
              boxShadow: '0 1px 6px -1px rgba(220,38,38,0.28)',
              borderColor: 'rgba(220,38,38,0.45)',
            }
          : undefined
      }
    >
      <div className="font-mono text-[10px] tracking-wide text-muted uppercase">
        {label}
      </div>
      <div
        className={cn('font-mono text-lg leading-tight', alert && value > 0 ? 'text-align' : 'text-ink')}
      >
        {display.toFixed(decimals)}
      </div>
    </div>
  )
}

function Ring({
  category,
  count,
  total,
  active,
  onClick,
}: {
  category: (typeof CATEGORY_ORDER)[number]
  count: number
  total: number
  active: boolean
  onClick: () => void
}) {
  const circumference = 2 * Math.PI * 18
  const filled = total > 0 ? (count / total) * circumference : 0
  const color = CATEGORY_COLOR[category]

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'flex flex-col items-center gap-1 rounded p-1.5 transition-all duration-200',
        active ? 'bg-raised' : 'hover:bg-raised/50',
      )}
    >
      <svg width="44" height="44" viewBox="0 0 44 44">
        <circle cx="22" cy="22" r="18" fill="none" stroke="var(--color-line)" strokeWidth="3" />
        <circle
          cx="22"
          cy="22"
          r="18"
          fill="none"
          stroke={color}
          strokeWidth="3"
          strokeDasharray={`${filled} ${circumference}`}
          strokeLinecap="round"
          transform="rotate(-90 22 22)"
          style={{ transition: 'stroke-dasharray 600ms ease-out' }}
        />
        <text
          x="22"
          y="26"
          textAnchor="middle"
          className="font-mono text-xs"
          fill={count > 0 ? color : 'var(--color-faint)'}
        >
          {count}
        </text>
      </svg>
      <span className="font-mono text-[9px] leading-tight text-muted">
        {CATEGORY_SHORT[category]}
      </span>
    </button>
  )
}

interface DashboardProps {
  report: AgentDocReport
  onBack: () => void
}

export function Dashboard({ report, onBack }: DashboardProps) {
  const derived = useMemo(() => deriveTrace(report), [report])
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [selectedStep, setSelectedStep] = useState<number | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)

  // One selection at a time: picking an agent clears a step pick and vice
  // versa, so the filtered panels never disagree about what is selected.
  const pickAgent = (agent: string | null) => {
    setSelectedAgent(agent)
    setSelectedStep(null)
  }
  const pickStep = (step: number | null) => {
    setSelectedStep(step)
    setSelectedAgent(null)
  }

  const visibleFailures = report.flagged_failures.filter((failure) => {
    if (selectedCategory && failure.category !== selectedCategory) return false
    if (selectedStep !== null) return failure.turn_indices.includes(selectedStep)
    if (selectedAgent) {
      const agent = derived.agents.find((candidate) => candidate.name === selectedAgent)
      return failure.turn_indices.some((step) => agent?.steps.includes(step))
    }
    return true
  })

  const filterLabel =
    selectedStep !== null
      ? `step ${selectedStep}`
      : selectedAgent
        ? selectedAgent
        : selectedCategory
          ? CATEGORY_SHORT[selectedCategory as keyof typeof CATEGORY_SHORT]
          : null

  return (
    <div className="relative flex h-screen flex-col overflow-hidden">
      <NetworkBackdrop density={0.5} />

      <header className="relative z-10 flex shrink-0 items-center gap-4 border-b border-line px-5 py-2.5">
        <button
          type="button"
          onClick={onBack}
          className="font-mono text-base font-bold text-signal text-glow transition-opacity hover:opacity-75"
          title="Back to upload"
        >
          AgentDoc
        </button>

        <div className="ml-4 flex gap-2">
          <Stat label="Turns" value={report.trace_turn_count} delay={0} />
          <Stat label="Agents" value={derived.agents.length} delay={50} />
          <Stat label="Failures" value={report.total_failures} delay={100} alert />
          <Stat
            label="Avg conf"
            value={derived.averageConfidence}
            delay={150}
            decimals={2}
          />
        </div>

        <div className="ml-auto flex gap-2 font-mono text-[10px]">
          <span className="rounded border border-line bg-surface px-2 py-1 text-muted">
            {report.source_framework ?? 'unknown'}
          </span>
          <span className="rounded border border-line bg-surface px-2 py-1 text-muted">
            {report.model ?? 'unknown model'}
          </span>
        </div>
      </header>

      <div className="relative z-10 grid min-h-0 flex-1 grid-cols-[minmax(200px,22%)_1fr_minmax(280px,28%)] gap-3 p-3">
        {/* Left: roster + category rings */}
        <aside className="flex min-h-0 flex-col gap-3">
          <section className="flex min-h-0 flex-col rounded border border-line bg-surface/50 p-3 backdrop-blur-sm">
            <h2 className="font-mono text-xs tracking-wide text-muted uppercase">
              Agent Overview
            </h2>
            <div className="mt-2 min-h-0 flex-1 space-y-1.5 overflow-y-auto">
              {derived.agents.map((agent, index) => {
                const color = agent.category
                  ? CATEGORY_COLOR[agent.category]
                  : 'var(--color-signal)'
                const isSelected = selectedAgent === agent.name
                return (
                  <button
                    key={agent.name}
                    type="button"
                    onClick={() => pickAgent(isSelected ? null : agent.name)}
                    className={cn(
                      'w-full animate-slide-in-left rounded border p-2 text-left transition-all duration-200',
                      isSelected ? 'bg-raised' : 'bg-void/60 hover:bg-raised/70',
                    )}
                    style={{
                      animationDelay: `${index * 60}ms`,
                      borderColor: isSelected ? color : 'var(--color-line)',
                      opacity: selectedAgent && !isSelected ? 0.4 : 1,
                    }}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs" style={{ color }}>
                        {agent.name}
                      </span>
                      {agent.failureCount > 0 && (
                        <span
                          className="rounded-full px-1.5 font-mono text-[9px]"
                          style={{
                            background: `color-mix(in srgb, ${color} 20%, transparent)`,
                            color,
                          }}
                        >
                          {agent.failureCount}
                        </span>
                      )}
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <div className="h-1 flex-1 overflow-hidden rounded-full bg-raised">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${(agent.turnCount / Math.max(...derived.agents.map((a) => a.turnCount))) * 100}%`,
                            background: color,
                            opacity: 0.55,
                          }}
                        />
                      </div>
                      <span className="shrink-0 font-mono text-[9px] text-faint">
                        {agent.turnCount} turns
                      </span>
                    </div>
                  </button>
                )
              })}
            </div>
          </section>

          <section className="shrink-0 rounded border border-line bg-surface/50 p-3 backdrop-blur-sm">
            <h2 className="font-mono text-xs tracking-wide text-muted uppercase">
              MAST Breakdown
            </h2>
            <div className="mt-2 flex justify-between">
              {CATEGORY_ORDER.map((category) => {
                const count =
                  report.category_counts.find((c) => c.category === category)?.count ?? 0
                return (
                  <Ring
                    key={category}
                    category={category}
                    count={count}
                    total={Math.max(report.total_failures, 1)}
                    active={selectedCategory === category}
                    onClick={() =>
                      setSelectedCategory(selectedCategory === category ? null : category)
                    }
                  />
                )
              })}
            </div>
          </section>
        </aside>

        {/* Center: graph + timeline */}
        <main className="grid min-h-0 grid-rows-[1fr_minmax(140px,32%)] gap-3">
          <section className="min-h-0 rounded border border-line bg-surface/50 p-3 backdrop-blur-sm">
            <h2 className="font-mono text-xs tracking-wide text-muted uppercase">
              Agent Interaction Graph
            </h2>
            <div className="h-[calc(100%-1.25rem)] animate-enter-tool">
              <AgentGraph
                derived={derived}
                selectedAgent={selectedAgent}
                onSelectAgent={pickAgent}
              />
            </div>
          </section>

          <section className="min-h-0 rounded border border-line bg-surface/50 p-3 backdrop-blur-sm">
            <Timeline
              report={report}
              derived={derived}
              selectedAgent={selectedAgent}
              selectedStep={selectedStep}
              onSelectStep={pickStep}
            />
          </section>
        </main>

        {/* Right: failure detail + summary */}
        <aside className="flex min-h-0 flex-col gap-3">
          <section className="flex min-h-0 flex-1 flex-col rounded border border-line bg-surface/50 p-3 backdrop-blur-sm">
            <div className="flex items-baseline justify-between">
              <h2 className="font-mono text-xs tracking-wide text-muted uppercase">
                Failure Detail
              </h2>
              {filterLabel && (
                <button
                  type="button"
                  onClick={() => {
                    setSelectedAgent(null)
                    setSelectedStep(null)
                    setSelectedCategory(null)
                  }}
                  className="font-mono text-[10px] text-signal hover:underline"
                >
                  {filterLabel} · clear
                </button>
              )}
            </div>

            <div className="mt-2 min-h-0 flex-1 space-y-2 overflow-y-auto">
              {visibleFailures.length === 0 ? (
                <p className="pt-6 text-center text-xs text-faint">
                  {report.total_failures === 0
                    ? 'No failures flagged in this trace.'
                    : 'No failures match this selection.'}
                </p>
              ) : (
                visibleFailures.map((failure, index) => {
                  const color = CATEGORY_COLOR[failure.category]
                  const isOpen = expanded === index
                  return (
                    <article
                      key={`${failure.failure_mode}-${index}`}
                      className="animate-rise rounded border border-line bg-void/60 p-2.5"
                      style={{
                        animationDelay: `${index * 50}ms`,
                        borderLeft: `2px solid ${color}`,
                      }}
                    >
                      <div className="flex items-baseline gap-2">
                        <span className="font-mono text-[11px]" style={{ color }}>
                          {failure.failure_mode}
                        </span>
                        <span className="text-[11px] text-ink">
                          {failureModeName(failure.failure_mode)}
                        </span>
                      </div>

                      <div className="mt-1.5 flex items-center gap-2">
                        <div className="h-1 flex-1 overflow-hidden rounded-full bg-raised">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${failure.confidence * 100}%`,
                              background: color,
                            }}
                          />
                        </div>
                        <span className="font-mono text-[9px] text-muted">
                          {failure.confidence.toFixed(2)}
                        </span>
                      </div>

                      <button
                        type="button"
                        onClick={() => setExpanded(isOpen ? null : index)}
                        className="mt-1.5 w-full text-left text-[11px] leading-relaxed text-muted hover:text-ink"
                      >
                        <span className={isOpen ? '' : 'line-clamp-2'}>
                          {failure.justification}
                        </span>
                      </button>

                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {failure.turn_indices.map((step) => (
                          <button
                            key={step}
                            type="button"
                            onClick={() => pickStep(step)}
                            className={cn(
                              'rounded border px-1.5 font-mono text-[9px] transition-colors',
                              selectedStep === step
                                ? 'border-signal text-signal'
                                : 'border-line text-faint hover:text-muted',
                            )}
                          >
                            step {step}
                          </button>
                        ))}
                      </div>
                    </article>
                  )
                })
              )}
            </div>
          </section>

          <section className="shrink-0 rounded border border-line bg-surface/50 p-3 backdrop-blur-sm">
            <h2 className="font-mono text-xs tracking-wide text-muted uppercase">
              Trace Summary
            </h2>
            <p className="mt-2 text-[11px] leading-relaxed text-muted">
              {report.narrative}
            </p>
            <dl className="mt-2.5 grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-[10px]">
              <dt className="text-faint">Tool calls</dt>
              <dd className="text-right text-ink">{derived.totalToolCalls}</dd>
              <dt className="text-faint">Handoffs</dt>
              <dd className="text-right text-ink">{derived.handoffCount}</dd>
              <dt className="text-faint">Avg confidence</dt>
              <dd className="text-right text-ink">
                {derived.averageConfidence.toFixed(2)}
              </dd>
              <dt className="text-faint">Top mode</dt>
              <dd className="text-right text-ink">{derived.topFailureMode ?? '--'}</dd>
            </dl>
          </section>
        </aside>
      </div>
    </div>
  )
}
