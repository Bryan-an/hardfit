import type { Data } from 'plotly.js'
import type { Distribution, FittedParams } from '../engine/types'
import {
  GRID_PADDING_FACTOR,
  HISTNORM_DENSITY,
  HISTOGRAM_COLOR,
  HISTOGRAM_OPACITY,
  PDF_GRID_POINTS,
  PDF_LINE_COLOR,
  PDF_LINE_WIDTH,
  SCATTER_MODE_LINES,
  TRACE_TYPE_HISTOGRAM,
  TRACE_TYPE_SCATTER,
} from './chart-constants'

/** Build [histogram(density), fitted-PDF line] traces for the data + a fitted distribution. */
export function buildHistogramPdf(
  data: readonly number[],
  dist: Distribution,
  params: FittedParams,
  gridPoints = PDF_GRID_POINTS,
): Data[] {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of data) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  } // plain loop, not Math.min(...data) — spreading a large array overflows the stack
  const span = hi - lo || 1
  const x0 = lo - span * GRID_PADDING_FACTOR
  const x1 = hi + span * GRID_PADDING_FACTOR
  const xGrid: number[] = []
  const yGrid: number[] = []
  for (let i = 0; i < gridPoints; i++) {
    const x = x0 + ((x1 - x0) * i) / (gridPoints - 1)
    xGrid.push(x)
    yGrid.push(Math.exp(dist.logpdf(x, params))) // pdf = exp(logpdf)
  }
  return [
    {
      type: TRACE_TYPE_HISTOGRAM,
      x: data as number[],
      histnorm: HISTNORM_DENSITY,
      name: 'data',
      opacity: HISTOGRAM_OPACITY,
      marker: { color: HISTOGRAM_COLOR },
    },
    {
      type: TRACE_TYPE_SCATTER,
      mode: SCATTER_MODE_LINES,
      x: xGrid,
      y: yGrid,
      name: `${dist.label} fit`,
      line: { color: PDF_LINE_COLOR, width: PDF_LINE_WIDTH },
    },
  ]
}
