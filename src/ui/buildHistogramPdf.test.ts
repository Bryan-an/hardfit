import { describe, expect, it } from 'vitest'
import { normal } from '../engine/distributions/normal'
import { poisson } from '../engine/distributions/poisson'
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

  it('draws integer PMF stems (markers), not a continuous grid, for a discrete fit', () => {
    const counts = [0, 1, 1, 2, 2, 2, 3, 3, 4]
    const traces = buildHistogramPdf(counts, poisson, poisson.fit(counts), GRID_POINTS)
    const [, overlay] = traces
    if (!overlay) throw new Error('expected an overlay trace')
    const stems = overlay as { x: number[]; y: number[]; mode?: string }
    // Markers (stems), not a connected line; x are the INTEGER support points in the data range.
    expect(stems.mode).toBe('markers')
    expect(stems.x).toEqual([0, 1, 2, 3, 4])
    // y are PMF masses = exp(logpmf) at each integer, all in (0,1].
    expect(stems.y.every((v) => v > 0 && v <= 1)).toBe(true)
  })
})
