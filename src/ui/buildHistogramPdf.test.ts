import { describe, expect, it } from 'vitest'
import { normal } from '../engine/distributions/normal'
import { buildHistogramPdf } from './buildHistogramPdf'

const GRID_POINTS = 64

describe('buildHistogramPdf', () => {
  it('returns a histogram trace (probability density) + a fitted-PDF line trace', () => {
    const data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    const traces = buildHistogramPdf(data, normal, normal.fit(data), GRID_POINTS)
    expect(traces).toHaveLength(2)

    const [histogram, line] = traces
    if (!histogram || !line) throw new Error('expected two traces')

    expect(histogram.type).toBe('histogram')
    expect((histogram as { histnorm?: string }).histnorm).toBe('probability density')

    expect(line.type).toBe('scatter')
    const scatter = line as { x: number[]; y: number[] }
    expect(scatter.x).toHaveLength(GRID_POINTS)
    expect(scatter.y).toHaveLength(GRID_POINTS)
    expect(scatter.y.every((v) => v >= 0 && Number.isFinite(v))).toBe(true)
  })
})
