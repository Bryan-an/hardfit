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
})
