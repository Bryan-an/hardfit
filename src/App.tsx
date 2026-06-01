import { lazy, Suspense, useMemo, useState } from 'react'
import type { FitAllResult } from './engine/index'
import { DISTRIBUTIONS, MIN_FIT_SAMPLE_SIZE } from './engine/index'
import { runFitAll } from './engineClient'
import { buildHistogramPdf } from './ui/buildHistogramPdf'
import { CHART_LAYOUT } from './ui/chart-constants'
import { DataInput } from './ui/DataInput'
import { ResultsTable } from './ui/ResultsTable'

// Lazy-load the chart so Plotly (~1.4 MB) is split into an on-demand chunk fetched only after a
// fit completes, keeping the initial JS bundle small. `PlotlyChart` is a named export, so map it
// to `default` for `React.lazy` (which expects `{ default: ComponentType }`).
const PlotlyChart = lazy(() => import('./ui/PlotlyChart').then((m) => ({ default: m.PlotlyChart })))

const TOO_FEW_VALUES_MESSAGE = `Please provide at least ${MIN_FIT_SAMPLE_SIZE} numeric values.`
/** Shown while the on-demand Plotly chart chunk is being fetched/parsed. */
const CHART_LOADING_MESSAGE = 'Rendering chart…'

export default function App() {
  const [data, setData] = useState<number[]>([])
  const [result, setResult] = useState<FitAllResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function onData(d: number[]) {
    setError(null)
    setResult(null)
    if (d.length < MIN_FIT_SAMPLE_SIZE) {
      setError(TOO_FEW_VALUES_MESSAGE)
      setData([])
      return
    }
    setData(d)
    setBusy(true)
    try {
      setResult(await runFitAll(d))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const traces = useMemo(() => {
    if (!result || result.ranked.length === 0 || data.length === 0) return null
    const best = result.ranked[0]
    if (!best) return null
    const dist = DISTRIBUTIONS.find((d) => d.name === best.name)
    return dist ? buildHistogramPdf(data, dist, best.params) : null
  }, [result, data])

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 p-6 flex flex-col gap-6 max-w-3xl mx-auto">
      <header>
        <h1 className="text-3xl font-bold">HardFit</h1>
        <p className="text-slate-600">
          Fit your data to probability distributions, in your browser.
        </p>
      </header>

      <DataInput onData={onData} />

      {error && (
        <p role="alert" className="text-red-600">
          {error}
        </p>
      )}
      {busy && <p>Fitting…</p>}

      {result && result.ranked.length > 0 && (
        <section className="flex flex-col gap-4">
          <h2 className="text-xl font-semibold">Ranked fits (by AICc)</h2>
          <ResultsTable ranked={result.ranked} />
          {traces && (
            <Suspense fallback={<p>{CHART_LOADING_MESSAGE}</p>}>
              <PlotlyChart data={traces} layout={CHART_LAYOUT} />
            </Suspense>
          )}
          {result.failures.length > 0 && (
            <p className="text-sm text-slate-500">
              Not applicable: {result.failures.map((f) => f.label).join(', ')} (data outside
              support).
            </p>
          )}
        </section>
      )}
    </main>
  )
}
