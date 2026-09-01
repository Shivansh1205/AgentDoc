import { useEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { DerivedTrace } from '../lib/derive'
import { CATEGORY_COLOR } from '../lib/schema'

/**
 * The agent interaction graph: a d3-force simulation over agents and the
 * handoffs between them, with each agent's tools as satellites.
 *
 * Nodes start clustered at the centre and settle outward over roughly two
 * seconds, so the panel reads as a system coming online rather than a static
 * diagram appearing.
 */

interface NodeDatum extends d3.SimulationNodeDatum {
  id: string
  kind: 'agent' | 'tool'
  turnCount: number
  failureCount: number
  color: string
  owner?: string
}

interface LinkDatum extends d3.SimulationLinkDatum<NodeDatum> {
  source: string | NodeDatum
  target: string | NodeDatum
  kind: 'handoff' | 'tool'
  explicit: boolean
  color: string
  preview: string
}

interface AgentGraphProps {
  derived: DerivedTrace
  selectedAgent: string | null
  onSelectAgent: (agent: string | null) => void
}

export function AgentGraph({ derived, selectedAgent, onSelectAgent }: AgentGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [size, setSize] = useState({ width: 640, height: 340 })
  const [hover, setHover] = useState<{ x: number; y: number; label: string } | null>(
    null,
  )
  const [tick, setTick] = useState(0)
  const nodesRef = useRef<NodeDatum[]>([])
  const linksRef = useRef<LinkDatum[]>([])

  const { nodes, links } = useMemo(() => {
    const agentNodes: NodeDatum[] = derived.agents.map((agent) => ({
      id: agent.name,
      kind: 'agent',
      turnCount: agent.turnCount,
      failureCount: agent.failureCount,
      color: agent.category ? CATEGORY_COLOR[agent.category] : 'var(--color-signal)',
    }))

    const toolNodes: NodeDatum[] = []
    const toolLinks: LinkDatum[] = []
    for (const [owner, tools] of derived.toolCallsByAgent) {
      for (const tool of tools) {
        const id = `${owner}::${tool}`
        toolNodes.push({
          id,
          kind: 'tool',
          turnCount: 0,
          failureCount: 0,
          color: 'var(--color-muted)',
          owner,
        })
        toolLinks.push({
          source: owner,
          target: id,
          kind: 'tool',
          explicit: false,
          color: 'var(--color-faint)',
          preview: tool,
        })
      }
    }

    const handoffLinks: LinkDatum[] = derived.edges.map((edge) => ({
      source: edge.source,
      target: edge.target,
      kind: 'handoff',
      explicit: edge.explicit,
      color: edge.category ? CATEGORY_COLOR[edge.category] : 'var(--color-signal)',
      preview: edge.preview,
    }))

    return {
      nodes: [...agentNodes, ...toolNodes],
      links: [...handoffLinks, ...toolLinks],
    }
  }, [derived])

  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      if (width > 0 && height > 0) setSize({ width, height })
    })
    observer.observe(svg)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const { width, height } = size
    // Start clustered so the settle reads as the system assembling itself.
    const seeded: NodeDatum[] = nodes.map((node) => ({
      ...node,
      x: width / 2 + (Math.random() - 0.5) * 30,
      y: height / 2 + (Math.random() - 0.5) * 30,
    }))
    const seededLinks: LinkDatum[] = links.map((link) => ({ ...link }))

    const simulation = d3
      .forceSimulation<NodeDatum>(seeded)
      .force(
        'link',
        d3
          .forceLink<NodeDatum, LinkDatum>(seededLinks)
          .id((node) => node.id)
          .distance((link) => (link.kind === 'tool' ? 46 : 130))
          .strength((link) => (link.kind === 'tool' ? 0.55 : 0.32)),
      )
      .force(
        'charge',
        d3.forceManyBody<NodeDatum>().strength((node) =>
          node.kind === 'tool' ? -70 : -420,
        ),
      )
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force(
        'collide',
        d3.forceCollide<NodeDatum>().radius((node) => (node.kind === 'tool' ? 14 : 42)),
      )
      // Gentle pull toward the middle on both axes: without it a small graph
      // drifts to one side and leaves half the panel empty.
      .force('x', d3.forceX(width / 2).strength(0.06))
      .force('y', d3.forceY(height / 2).strength(0.09))

    nodesRef.current = seeded
    linksRef.current = seededLinks
    simulation.on('tick', () => setTick((value) => value + 1))

    return () => {
      simulation.stop()
    }
  }, [nodes, links, size])

  const radius = (node: NodeDatum) =>
    node.kind === 'tool' ? 3.5 : 15 + Math.min(node.turnCount, 6) * 3.5

  const dimmed = (id: string, owner?: string) =>
    selectedAgent !== null && selectedAgent !== id && selectedAgent !== owner

  return (
    <div className="relative h-full w-full">
      <svg ref={svgRef} className="h-full w-full" data-tick={tick}>
        <defs>
          <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="1" fill="var(--color-line)" opacity="0.9" />
          </pattern>
          <marker
            id="arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="5"
            markerHeight="5"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
          </marker>
        </defs>

        <rect width="100%" height="100%" fill="url(#dots)" />

        {linksRef.current.map((link, index) => {
          const source = link.source as NodeDatum
          const target = link.target as NodeDatum
          if (!source?.x || !target?.x) return null
          const isDim =
            dimmed(source.id, source.owner) && dimmed(target.id, target.owner)
          return (
            <line
              key={`link-${index}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke={link.color}
              strokeWidth={link.kind === 'tool' ? 1 : link.explicit ? 2 : 1.4}
              strokeDasharray={link.kind === 'tool' ? '2 3' : '5 4'}
              strokeOpacity={isDim ? 0.12 : link.kind === 'tool' ? 0.4 : 0.8}
              markerEnd={link.kind === 'handoff' ? 'url(#arrow)' : undefined}
              color={link.color}
              className={link.kind === 'handoff' ? 'animate-edge-flow' : undefined}
              style={{ transition: 'stroke-opacity 200ms ease-out' }}
              onMouseEnter={(event) =>
                setHover({
                  x: event.clientX,
                  y: event.clientY,
                  label: link.kind === 'tool' ? link.preview : `→ ${link.preview}`,
                })
              }
              onMouseLeave={() => setHover(null)}
            />
          )
        })}

        {nodesRef.current.map((node) => {
          if (node.x === undefined || node.y === undefined) return null
          const isDim = dimmed(node.id, node.owner)
          const isSelected = selectedAgent === node.id
          const faulted = node.failureCount > 0

          return (
            <g
              key={node.id}
              transform={`translate(${node.x},${node.y})`}
              opacity={isDim ? 0.35 : 1}
              style={{ transition: 'opacity 200ms ease-out', cursor: node.kind === 'agent' ? 'pointer' : 'default' }}
              onClick={() =>
                node.kind === 'agent' &&
                onSelectAgent(selectedAgent === node.id ? null : node.id)
              }
              onMouseEnter={(event) =>
                setHover({
                  x: event.clientX,
                  y: event.clientY,
                  label:
                    node.kind === 'tool'
                      ? node.id.split('::')[1]
                      : `${node.id} · ${node.turnCount} turns · ${node.failureCount} failures`,
                })
              }
              onMouseLeave={() => setHover(null)}
            >
              <circle
                r={radius(node)}
                fill={node.kind === 'tool' ? 'var(--color-raised)' : 'transparent'}
                stroke={node.color}
                strokeWidth={isSelected ? 2.5 : faulted ? 2 : 1.2}
                className={faulted ? 'animate-fault-pulse' : undefined}
                style={
                  {
                    ['--pulse-color' as string]: node.color,
                    transition: 'stroke-width 150ms ease-out',
                  } as React.CSSProperties
                }
              />
              {node.kind === 'agent' && (
                <text
                  y={radius(node) + 14}
                  textAnchor="middle"
                  className="font-mono text-[10px] select-none"
                  fill={isSelected ? node.color : 'var(--color-muted)'}
                >
                  {node.id}
                </text>
              )}
            </g>
          )
        })}
      </svg>

      {hover && (
        <div
          className="pointer-events-none fixed z-50 rounded border border-line bg-void/95 px-2 py-1 font-mono text-[10px] text-ink"
          style={{ left: hover.x + 12, top: hover.y - 8 }}
        >
          {hover.label}
        </div>
      )}
    </div>
  )
}
