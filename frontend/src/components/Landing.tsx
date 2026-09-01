import { useCallback, useRef, useState } from 'react'
import { AlertCircle, Check, GitBranch, Upload } from 'lucide-react'
import { NetworkBackdrop } from './NetworkBackdrop'
import { SAMPLE_REPORT } from '../lib/sample'
import {
  CATEGORY_ORDER,
  CATEGORY_COLOR,
  CATEGORY_LABEL,
  ReportValidationError,
  validateReport,
  type AgentDocReport,
} from '../lib/schema'
import { cn } from '../lib/utils'

const CATEGORY_BLURB: Record<string, string> = {
  system_design_issues:
    'Agents that ignore their brief, repeat finished work, or miss the stop condition.',
  inter_agent_misalignment:
    'Agents that talk past each other, withhold findings, or drift off task.',
  task_verification:
    'Work closed out unchecked, or checked so loosely the error survives.',
}

interface LandingProps {
  onDiagnose: (report: AgentDocReport, label: string) => void
}

export function Landing({ onDiagnose }: LandingProps) {
  const [repo, setRepo] = useState('')
  const [dragging, setDragging] = useState(false)
  const [loaded, setLoaded] = useState<{ report: AgentDocReport; name: string } | null>(
    null,
  )
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const readFile = useCallback(async (file: File) => {
    setError(null)
    try {
      const text = await file.text()
      let parsed: unknown
      try {
        parsed = JSON.parse(text)
      } catch {
        throw new ReportValidationError(
          `${file.name} isn't valid JSON. Check the file exported cleanly.`,
        )
      }
      setLoaded({ report: validateReport(parsed), name: file.name })
    } catch (err) {
      setLoaded(null)
      setError(
        err instanceof ReportValidationError
          ? err.message
          : "This doesn't look like an AgentDoc report.",
      )
    }
  }, [])

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()
      setDragging(false)
      const file = event.dataTransfer.files?.[0]
      if (file) void readFile(file)
    },
    [readFile],
  )

  const useSample = useCallback(() => {
    setError(null)
    setLoaded({ report: SAMPLE_REPORT, name: 'sample-trace.json' })
  }, [])

  const submit = () => {
    if (loaded) onDiagnose(loaded.report, loaded.name)
  }

  return (
    <div className="relative min-h-screen overflow-y-auto">
      <NetworkBackdrop />

      <div className="relative mx-auto flex min-h-screen max-w-3xl flex-col px-8 py-16">
        <header className="animate-rise" style={{ animationDelay: '0ms' }}>
          <h1 className="font-mono text-2xl font-bold tracking-tight text-signal text-glow">
            AgentDoc
          </h1>
          <p
            className="mt-2 animate-rise text-sm text-muted"
            style={{ animationDelay: '120ms' }}
          >
            Diagnose why your multi-agent LLM system failed
          </p>
        </header>

        <main
          className="mt-12 animate-rise rounded-lg border border-line bg-surface/70 p-7 backdrop-blur-sm"
          style={{
            animationDelay: '240ms',
            ['--glow-color' as string]: 'var(--color-signal)',
          }}
        >
          <label className="block">
            <span className="font-mono text-xs tracking-wide text-muted uppercase">
              Repository <span className="normal-case">(optional, for context)</span>
            </span>
            <input
              type="text"
              value={repo}
              onChange={(event) => setRepo(event.target.value)}
              placeholder="github.com/yourname/yourrepo"
              className="mt-2 w-full rounded border border-line bg-void px-3 py-2 font-mono text-sm text-ink placeholder:text-faint focus:border-signal focus:outline-none"
            />
          </label>

          <div className="mt-6">
            <span className="font-mono text-xs tracking-wide text-muted uppercase">
              Trace file
            </span>

            <div
              onDragOver={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  inputRef.current?.click()
                }
              }}
              role="button"
              tabIndex={0}
              aria-label="Choose a trace file"
              className={cn(
                'mt-2 cursor-pointer rounded border border-dashed px-6 py-8 text-center transition-all duration-200',
                dragging
                  ? 'border-signal bg-signal/5'
                  : loaded
                    ? 'border-signal/40 bg-signal/[0.03]'
                    : 'border-line hover:border-muted hover:bg-raised/40',
              )}
              style={
                dragging
                  ? { boxShadow: '0 0 0 1px var(--color-signal), 0 2px 10px -2px rgba(4,120,87,0.3)' }
                  : undefined
              }
            >
              {loaded ? (
                <div className="flex items-center justify-center gap-2">
                  <Check size={15} className="text-signal" />
                  <span className="font-mono text-sm text-ink">{loaded.name}</span>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <Upload size={18} className="text-faint" />
                  <span className="font-mono text-sm text-muted">
                    Drop a trace file, or click to browse
                  </span>
                </div>
              )}
            </div>

            <input
              ref={inputRef}
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void readFile(file)
                event.target.value = ''
              }}
            />

            <p className="mt-3 text-xs leading-relaxed text-muted">
              A LangGraph execution trace, exported as JSON. Don't have one?{' '}
              <button
                type="button"
                onClick={useSample}
                className="text-signal underline decoration-signal/40 underline-offset-2 transition-colors hover:decoration-signal"
              >
                Try a sample trace
              </button>
            </p>

            <p className="mt-2 font-mono text-xs text-faint">
              Run{' '}
              <span className="text-muted">agentdoc diagnose --json</span> locally to
              export a trace file, then upload it here.
            </p>

            {error && (
              <div
                role="alert"
                className="mt-4 flex items-start gap-2 rounded border border-align/30 bg-align/5 px-3 py-2"
              >
                <AlertCircle size={14} className="mt-0.5 shrink-0 text-align" />
                <p className="text-xs leading-relaxed text-ink">
                  {error}{' '}
                  <button
                    type="button"
                    onClick={useSample}
                    className="text-signal underline underline-offset-2"
                  >
                    Try the sample instead
                  </button>
                </p>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={submit}
            disabled={!loaded}
            className={cn(
              'mt-7 w-full rounded py-2.5 font-mono text-sm font-medium transition-all duration-200',
              loaded
                ? 'bg-signal text-void hover:bg-signal-bright'
                : 'cursor-not-allowed border border-line bg-raised text-faint',
            )}
            style={
              loaded
                ? { boxShadow: '0 2px 12px -3px rgba(4,120,87,0.45)' }
                : undefined
            }
          >
            Diagnose
          </button>
        </main>

        <section
          className="mt-8 grid animate-rise grid-cols-3 gap-3"
          style={{ animationDelay: '380ms' }}
        >
          {CATEGORY_ORDER.map((category) => (
            <div
              key={category}
              className="rounded border border-line bg-surface/60 py-3 pr-3 pl-3 backdrop-blur-sm"
              style={{ borderLeft: `2px solid ${CATEGORY_COLOR[category]}` }}
            >
              <h2
                className="font-mono text-xs font-medium"
                style={{ color: CATEGORY_COLOR[category] }}
              >
                {CATEGORY_LABEL[category]}
              </h2>
              <p className="mt-1.5 text-xs leading-relaxed text-muted">
                {CATEGORY_BLURB[category]}
              </p>
            </div>
          ))}
        </section>

        <footer className="mt-auto pt-10">
          <p className="flex items-center gap-1.5 text-xs text-faint">
            <GitBranch size={12} />
            Free and open source. Powered by the MAST failure taxonomy (Cemri et al.,{' '}
            <span className="font-mono">arXiv:2503.13657</span>).
          </p>
        </footer>
      </div>
    </div>
  )
}
