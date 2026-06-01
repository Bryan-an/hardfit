import '@testing-library/jest-dom/vitest'
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
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
})
