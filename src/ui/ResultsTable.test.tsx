// @vitest-environment jsdom
// This React component test needs a DOM; the suite default is the fast `node` environment
// (see vite.config.ts), so this file opts into jsdom explicitly.
import '@testing-library/jest-dom/vitest'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { BootstrapResult } from '../engine/index'
import type { RankedFit } from '../engine/types'
import { ResultsTable } from './ResultsTable'

const ranked: RankedFit[] = [
  {
    name: 'gamma',
    label: 'Gamma',
    k: 2,
    params: { shape: 2, scale: 1.5, rate: 1 / 1.5 },
    logLik: -30,
    aic: 64,
    aicc: 64.5,
    bic: 67,
    ks: 0.08,
    // Gamma has no closed-form AD critical values -> diagnostic (p-value null).
    ad: { statistic: 0.512, pValue: null, method: 'diagnostic' },
    cvm: { statistic: 0.071, pValue: null, method: 'diagnostic' },
    chiSquared: { statistic: 3.214, pValue: 0.201, df: 2, bins: 5, method: 'table' },
    rank: 1,
    deltaAICc: 0,
    weight: 0.7,
  },
  {
    name: 'normal',
    label: 'Normal',
    k: 2,
    params: { mu: 3, sigma: 1.2 },
    logLik: -32,
    aic: 68,
    aicc: 68.5,
    bic: 71,
    ks: 0.12,
    // Normal AD uses the D'Agostino-Stephens closed-form p-value.
    ad: { statistic: 0.734, pValue: 0.412, method: 'closed-form' },
    cvm: { statistic: 0.098, pValue: null, method: 'diagnostic' },
    chiSquared: { statistic: 5.876, pValue: 0.053, df: 2, bins: 5, method: 'table' },
    rank: 2,
    deltaAICc: 4,
    weight: 0.3,
  },
]

// Bootstrap result for gamma (rank 1) only; normal (rank 2) intentionally has no entry so we
// can assert it keeps the M2.1 diagnostic display. Gamma's paramCIs include the redundant
// `scale` key (fit stores shape+scale+rate) — the component must omit it for gamma.
const bootstrap: BootstrapResult = {
  gamma: {
    seed: 1,
    paramCIs: {
      shape: { point: 2.34, percentile: [1.79, 3.05], bca: [1.81, 3.02], method: 'bca' },
      scale: { point: 0.5, percentile: [0.41, 0.62], bca: [0.42, 0.61], method: 'bca' },
      rate: { point: 2.0, percentile: [1.64, 2.44], bca: [1.66, 2.41], method: 'bca' },
    },
    gofPValues: { ks: 0.652, ad: 0.587, cvm: 0.604 },
  },
}

describe('ResultsTable', () => {
  it('renders a row per fit, best first, with label + AICc', () => {
    render(<ResultsTable ranked={ranked} />)
    const rows = screen.getAllByRole('row')
    expect(rows.length).toBe(1 + ranked.length) // header + 2
    expect(screen.getByText('Gamma')).toBeInTheDocument()
    expect(screen.getByText('Normal')).toBeInTheDocument()
  })

  it('renders the Anderson-Darling, Cramér-von Mises and Chi-Squared headers', () => {
    render(<ResultsTable ranked={ranked} />)
    expect(screen.getByRole('columnheader', { name: 'AD' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'CvM' })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'χ²' })).toBeInTheDocument()
  })

  it('shows each GoF statistic formatted to the diagnostic precision', () => {
    render(<ResultsTable ranked={ranked} />)
    expect(screen.getByText('0.734')).toBeInTheDocument() // normal AD A²
    expect(screen.getByText('0.071')).toBeInTheDocument() // gamma CvM W²
    expect(screen.getByText('3.214')).toBeInTheDocument() // gamma χ²
  })

  it('shows a numeric p-value when available and a diagnostic marker when not', () => {
    render(<ResultsTable ranked={ranked} />)
    // Normal AD is closed-form: its p-value is rendered.
    expect(screen.getByText('p=0.412')).toBeInTheDocument()
    // Gamma AD is diagnostic-only: a "diag." marker stands in for the p-value.
    expect(screen.getAllByText('diag.').length).toBeGreaterThan(0)
  })

  it('renders parameter point estimates with their bootstrap CI for a bootstrapped row', () => {
    render(<ResultsTable ranked={ranked} bootstrap={bootstrap} />)
    // Gamma is bootstrapped: shape + rate show point + the [lo, hi] BCa interval.
    expect(screen.getByText('shape')).toBeInTheDocument()
    expect(screen.getByText('2.340')).toBeInTheDocument() // shape point
    expect(screen.getByText('[1.810, 3.020]')).toBeInTheDocument() // shape BCa interval
    expect(screen.getByText('rate')).toBeInTheDocument()
    expect(screen.getByText('[1.660, 2.410]')).toBeInTheDocument() // rate BCa interval
  })

  it('omits the redundant gamma scale parameter, keeping the locked shape+rate convention', () => {
    render(<ResultsTable ranked={ranked} bootstrap={bootstrap} />)
    expect(screen.queryByText('scale')).not.toBeInTheDocument()
    // The scale interval [0.420, 0.610] must not leak into the params cell.
    expect(screen.queryByText('[0.420, 0.610]')).not.toBeInTheDocument()
  })

  it('replaces the GoF marker with the bootstrap p-value (method label) when present', () => {
    render(<ResultsTable ranked={ranked} bootstrap={bootstrap} />)
    // Gamma AD/CvM (diagnostic-only in M2.1) now show the rigorous bootstrap p-value.
    expect(screen.getByText('p=0.587')).toBeInTheDocument() // gamma AD bootstrap p
    expect(screen.getByText('p=0.604')).toBeInTheDocument() // gamma CvM bootstrap p
    // KS gains a bootstrap p secondary it lacks in the M2.1 display.
    expect(screen.getByText('p=0.652')).toBeInTheDocument() // gamma KS bootstrap p
    // The method is surfaced via the cell title (same pattern as M2.1's `method` tooltip).
    expect(screen.getAllByTitle('bootstrap').length).toBeGreaterThanOrEqual(3)
  })

  it('keeps the M2.1 diagnostic display for a row without a bootstrap entry', () => {
    render(<ResultsTable ranked={ranked} bootstrap={bootstrap} />)
    // Normal has no bootstrap entry: its closed-form AD p-value and the diagnostic marker
    // for its CvM are unchanged.
    expect(screen.getByText('p=0.412')).toBeInTheDocument()
    expect(screen.getAllByText('diag.').length).toBeGreaterThan(0)
  })

  it('keeps the M2.1 display (no params column) while the bootstrap is still computing', () => {
    // No bootstrap prop yet (loading): the table renders exactly as in M2.1.
    render(<ResultsTable ranked={ranked} />)
    expect(screen.queryByRole('columnheader', { name: /Parameters/ })).not.toBeInTheDocument()
    expect(screen.getAllByRole('row').length).toBe(1 + ranked.length)
  })
})
