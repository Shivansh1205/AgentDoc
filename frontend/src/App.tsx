import { useState } from 'react'
import { Landing } from './components/Landing'
import { Dashboard } from './components/Dashboard'
import type { AgentDocReport } from './lib/schema'

/**
 * Two views, one piece of state. Routing is deliberately not a router: there
 * is no URL to share (the report lives only in this tab's memory), so a
 * router would imply a permanence the app does not have.
 *
 * The crossfade holds both views for the length of the transition so the
 * dashboard appears to come up through the landing page rather than replace
 * it.
 */

type View = 'landing' | 'dashboard'

export default function App() {
  const [view, setView] = useState<View>('landing')
  const [report, setReport] = useState<AgentDocReport | null>(null)
  const [leaving, setLeaving] = useState(false)

  const enterDashboard = (loaded: AgentDocReport) => {
    setReport(loaded)
    setLeaving(true)
    window.setTimeout(() => {
      setView('dashboard')
      setLeaving(false)
    }, 220)
  }

  const backToUpload = () => {
    setLeaving(true)
    window.setTimeout(() => {
      setView('landing')
      setLeaving(false)
    }, 220)
  }

  return (
    <div
      className="h-full transition-all duration-200 ease-out"
      style={{
        opacity: leaving ? 0 : 1,
        transform: leaving
          ? `scale(${view === 'landing' ? 1.02 : 0.99})`
          : 'scale(1)',
      }}
    >
      {view === 'landing' || !report ? (
        <Landing onDiagnose={enterDashboard} />
      ) : (
        <Dashboard report={report} onBack={backToUpload} />
      )}
    </div>
  )
}
