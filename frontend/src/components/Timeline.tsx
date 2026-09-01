import { useCallback, useRef, useState } from 'react'
import type { DerivedTrace } from '../lib/derive'
import { CATEGORY_COLOR, type AgentDocReport } from '../lib/schema'
import { cn } from '../lib/utils'

/**
 * Swimlane view of the trace: one lane per agent, one block per turn, block
 * width proportional to how much that turn actually said.
 *
 * The playhead is draggable and scrubs through the trace in step order,
 * which is the one control here that reveals sequence - the graph shows who
 * talks to whom, this shows when.
 */

interface TimelineProps {
  report: AgentDocReport
  derived: DerivedTrace
  selectedAgent: string | null
  selectedStep: number | null
  onSelectStep: (step: number | null) => void
}

export function Timeline({
  report,
  derived,
  selectedAgent,
  selectedStep,
  onSelectStep,
}: TimelineProps) {
  const [playhead, setPlayhead] = useState<number | null>(null)
  const [hover, setHover] = useState<{ x: number; y: number; label: string } | null>(
    null,
  )
  const trackRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const agentTurns = report.turns.filter((turn) => turn.agent)
  const maxStep = report.turns.length - 1

  const scrub = useCallback(
    (clientX: number) => {
      const track = trackRef.current
      if (!track) return
      const rect = track.getBoundingClientRect()
      // Skip the label gutter so a click lands on the step under the cursor;
      // must match the playhead's own offset below or the two disagree.
      const gutter = 16 * 6.5
      const usable = Math.max(rect.width - gutter, 1)
      const ratio = Math.min(Math.max((clientX - rect.left - gutter) / usable, 0), 1)
      setPlayhead(Math.round(ratio * maxStep))
    },
    [maxStep],
  )

  const startDrag = (event: React.MouseEvent) => {
    dragging.current = true
    scrub(event.clientX)
    const move = (e: MouseEvent) => dragging.current && scrub(e.clientX)
    const up = () => {
      dragging.current = false
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-baseline justify-between">
        <h2 className="font-mono text-xs tracking-wide text-muted uppercase">
          Interaction Timeline
        </h2>
        <span className="font-mono text-[10px] text-faint">
          {playhead === null ? 'drag to scrub' : `step ${playhead}`}
        </span>
      </div>

      <div
        ref={trackRef}
        onMouseDown={startDrag}
        className="relative mt-2 flex flex-1 cursor-ew-resize flex-col justify-center gap-1.5 overflow-hidden"
      >
        {derived.agents.map((agent, laneIndex) => (
          <div key={agent.name} className="flex items-center gap-2">
            <span
              className={cn(
                'w-24 shrink-0 truncate text-right font-mono text-[10px] transition-opacity duration-200',
                selectedAgent && selectedAgent !== agent.name
                  ? 'text-faint opacity-40'
                  : 'text-muted',
              )}
            >
              {agent.name}
            </span>
            <div className="relative flex h-6 flex-1 items-center gap-0.5">
              {agentTurns
                .filter((turn) => turn.agent === agent.name)
                .map((turn) => {
                  const category = derived.categoryByStep.get(turn.step)
                  const length = (turn.content ?? '').length
                  const width = Math.max(28, Math.min(160, 28 + length / 4))
                  const isPast = playhead !== null && turn.step <= playhead
                  const isSelected = selectedStep === turn.step
                  const isDim =
                    (selectedAgent && selectedAgent !== agent.name) ||
                    (playhead !== null && !isPast)

                  return (
                    <button
                      key={turn.step}
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation()
                        onSelectStep(isSelected ? null : turn.step)
                      }}
                      onMouseEnter={(event) =>
                        setHover({
                          x: event.clientX,
                          y: event.clientY,
                          label:
                            (turn.content ?? '').slice(0, 120) ||
                            turn.tool_calls.map((c) => c.name).join(', ') ||
                            `step ${turn.step}`,
                        })
                      }
                      onMouseLeave={() => setHover(null)}
                      className="h-5 shrink-0 rounded-sm border transition-all duration-200 animate-slide-in-left"
                      style={{
                        width,
                        animationDelay: `${laneIndex * 60 + turn.step * 30}ms`,
                        borderColor: category
                          ? CATEGORY_COLOR[category]
                          : 'var(--color-signal)',
                        background: category
                          ? `color-mix(in srgb, ${CATEGORY_COLOR[category]} 22%, transparent)`
                          : 'color-mix(in srgb, var(--color-signal) 16%, transparent)',
                        opacity: isDim ? 0.28 : 1,
                        boxShadow: isSelected
                          ? `0 0 0 1px ${category ? CATEGORY_COLOR[category] : 'var(--color-signal)'}`
                          : undefined,
                      }}
                      aria-label={`Turn ${turn.step} by ${agent.name}`}
                    />
                  )
                })}
            </div>
          </div>
        ))}

        {playhead !== null && (
          // Positioned over the lane area only (the 6rem label gutter plus the
          // 0.5rem gap sit to its left), so the head lines up with the blocks
          // it is scrubbing rather than the panel edge.
          <div
            className="pointer-events-none absolute top-0 bottom-0 w-px bg-signal"
            style={{
              left: `calc(6.5rem + (100% - 6.5rem) * ${playhead / Math.max(maxStep, 1)})`,
              boxShadow: '0 0 4px rgba(4,120,87,0.5)',
            }}
          />
        )}
      </div>

      {hover && (
        <div
          className="pointer-events-none fixed z-50 max-w-xs rounded border border-line bg-void/95 px-2 py-1 text-[10px] leading-relaxed text-ink"
          style={{ left: hover.x + 12, top: hover.y - 8 }}
        >
          {hover.label}
        </div>
      )}
    </div>
  )
}
