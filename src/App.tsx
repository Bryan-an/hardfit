import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { DEFAULT_BOOTSTRAP_B, DEFAULT_CI_ALPHA } from './engine/constants'
import type { BootstrapResult, FitAllResult } from './engine/index'
import { BOOTSTRAP_TOP_K, DISTRIBUTIONS, MIN_FIT_SAMPLE_SIZE } from './engine/index'
import { cancelBootstrap, runBootstrap, runFitAll } from './engineClient'
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
/** Fixed master seed for the parametric bootstrap so confidence intervals are reproducible
 *  across runs of the same data (each top-k fit derives a distinct stream seed from this). */
const DEFAULT_BOOTSTRAP_SEED = 0x48_46_69_74 // "HFit"
/** Name of the engine's cooperative-cancellation rejection (a plain `Error` across Comlink,
 *  so it is matched by `name` rather than `instanceof`). */
const BOOTSTRAP_CANCELLED_NAME = 'BootstrapCancelledError'
/** Progress fraction (0–1) formatted as a whole-number percent for the indicator. */
const formatPercent = (fraction: number): string => `${Math.round(fraction * 100)}%`

export default function App() {
  const [data, setData] = useState<number[]>([])
  const [result, setResult] = useState<FitAllResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [bootstrap, setBootstrap] = useState<BootstrapResult | null>(null)
  // Progress fraction (0–1) while the bootstrap runs; null when idle/settled.
  const [bootstrapProgress, setBootstrapProgress] = useState<number | null>(null)

  async function onData(d: number[]) {
    setError(null)
    setResult(null)
    // Drop any prior CIs/progress so stale intervals never linger over fresh data; the
    // in-flight bootstrap (if any) is cancelled + superseded by the effect's cleanup below.
    setBootstrap(null)
    setBootstrapProgress(null)
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

  // Auto-run the top-k parametric bootstrap in the background once a fit resolves. The
  // `active` closure flag — flipped false by the cleanup on a new fit OR on unmount — gates
  // every setState so a superseded/late run can never overwrite current state, and also
  // cooperatively cancels the worker. Keyed on [result, data]: `onData` nulls `result`
  // before awaiting the fit, so this effect only launches on the real-result render.
  useEffect(() => {
    if (result === null || result.ranked.length === 0 || data.length === 0) return
    let active = true
    setBootstrapProgress(0)
    runBootstrap(
      data,
      result.ranked,
      {
        topK: BOOTSTRAP_TOP_K,
        B: DEFAULT_BOOTSTRAP_B,
        alpha: DEFAULT_CI_ALPHA,
        seed: DEFAULT_BOOTSTRAP_SEED,
      },
      (fraction) => {
        if (active) setBootstrapProgress(fraction)
      },
    )
      .then((r) => {
        if (!active) return
        setBootstrap(r)
        setBootstrapProgress(null)
      })
      .catch((e: unknown) => {
        if (!active) return
        setBootstrapProgress(null)
        // A user cancel resolves quietly (no error alert); anything else is logged but kept
        // off the fitAll error path so the primary fit/ranking stays visible.
        if (!(e instanceof Error && e.name === BOOTSTRAP_CANCELLED_NAME)) {
          console.error('Bootstrap failed:', e)
        }
      })
    return () => {
      active = false
      void cancelBootstrap()
    }
  }, [result, data])

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
          {bootstrapProgress !== null && (
            <div className="flex items-center gap-3 text-sm text-slate-600">
              <p role="status" aria-live="polite">
                Computing confidence intervals… {formatPercent(bootstrapProgress)}
              </p>
              <button
                type="button"
                onClick={() => void cancelBootstrap()}
                className="rounded border border-slate-300 px-2 py-0.5 text-slate-700 hover:bg-slate-100"
              >
                Cancel
              </button>
            </div>
          )}
          <ResultsTable ranked={result.ranked} {...(bootstrap ? { bootstrap } : {})} />
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
