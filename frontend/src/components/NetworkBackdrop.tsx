import { useEffect, useRef } from 'react'

/**
 * The ambient backdrop shared by both views.
 *
 * Not generic drifting particles: nodes here periodically fire a pulse along
 * a link to a neighbour, so the backdrop reads as agents passing control -
 * the exact thing AgentDoc diagnoses. It stays at very low opacity and never
 * competes with the foreground; it is atmosphere, not decoration.
 *
 * Honors prefers-reduced-motion by rendering one static frame.
 */

interface Node {
  x: number
  y: number
  vx: number
  vy: number
}

interface Pulse {
  from: number
  to: number
  t: number
  speed: number
}

export function NetworkBackdrop({ density = 1 }: { density?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const LINK_DISTANCE = 150
    let nodes: Node[] = []
    let pulses: Pulse[] = []
    let frame = 0
    let raf = 0

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const { clientWidth: w, clientHeight: h } = canvas
      canvas.width = w * dpr
      canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      const count = Math.round(((w * h) / 26000) * density)
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.14,
        vy: (Math.random() - 0.5) * 0.14,
      }))
      pulses = []
    }

    const draw = () => {
      const { clientWidth: w, clientHeight: h } = canvas
      ctx.clearRect(0, 0, w, h)

      if (!reduced) {
        for (const node of nodes) {
          node.x += node.vx
          node.y += node.vy
          if (node.x < 0 || node.x > w) node.vx *= -1
          if (node.y < 0 || node.y > h) node.vy *= -1
        }
      }

      // Links between near neighbours, fading with distance.
      ctx.lineWidth = 1
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const dx = nodes[i].x - nodes[j].x
          const dy = nodes[i].y - nodes[j].y
          const dist = Math.hypot(dx, dy)
          if (dist > LINK_DISTANCE) continue
          ctx.strokeStyle = `rgba(4,120,87,${0.13 * (1 - dist / LINK_DISTANCE)})`
          ctx.beginPath()
          ctx.moveTo(nodes[i].x, nodes[i].y)
          ctx.lineTo(nodes[j].x, nodes[j].y)
          ctx.stroke()
        }
      }

      for (const node of nodes) {
        ctx.fillStyle = 'rgba(4,120,87,0.3)'
        ctx.beginPath()
        ctx.arc(node.x, node.y, 1.1, 0, Math.PI * 2)
        ctx.fill()
      }

      // A handoff fires along an existing link every so often.
      if (!reduced) {
        frame += 1
        if (frame % 34 === 0 && nodes.length > 1 && pulses.length < 5) {
          const from = Math.floor(Math.random() * nodes.length)
          const candidates: number[] = []
          for (let j = 0; j < nodes.length; j += 1) {
            if (j === from) continue
            const dist = Math.hypot(nodes[from].x - nodes[j].x, nodes[from].y - nodes[j].y)
            if (dist < LINK_DISTANCE) candidates.push(j)
          }
          if (candidates.length) {
            pulses.push({
              from,
              to: candidates[Math.floor(Math.random() * candidates.length)],
              t: 0,
              speed: 0.014 + Math.random() * 0.012,
            })
          }
        }

        pulses = pulses.filter((pulse) => {
          pulse.t += pulse.speed
          if (pulse.t >= 1) return false
          const a = nodes[pulse.from]
          const b = nodes[pulse.to]
          if (!a || !b) return false
          const x = a.x + (b.x - a.x) * pulse.t
          const y = a.y + (b.y - a.y) * pulse.t
          const fade = Math.sin(pulse.t * Math.PI)
          ctx.fillStyle = `rgba(4,120,87,${0.7 * fade})`
          ctx.beginPath()
          ctx.arc(x, y, 1.9, 0, Math.PI * 2)
          ctx.fill()
          return true
        })
      }

      raf = requestAnimationFrame(draw)
    }

    resize()
    draw()
    window.addEventListener('resize', resize)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [density])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 h-full w-full"
    />
  )
}
