import type { Layout } from 'plotly.js'

/** Single source of truth for every chart literal (colors, sizes, grid resolution, layout).
 *  Keeps `buildHistogramPdf`, `PlotlyChart`, and `App` free of magic strings/numbers. */

/** Fill color for the data histogram bars. */
export const HISTOGRAM_COLOR = '#7aa6ff'
/** Stroke color for the fitted-PDF overlay line. */
export const PDF_LINE_COLOR = '#ff5d5d'
/** Stroke width (px) for the fitted-PDF overlay line. */
export const PDF_LINE_WIDTH = 2
/** Opacity of the histogram bars so the PDF line stays visible through them. */
export const HISTOGRAM_OPACITY = 0.6

/** Rendered chart height in CSS pixels. */
export const CHART_HEIGHT_PX = 400
/** Number of x-grid points used to draw the smooth fitted-PDF line. */
export const PDF_GRID_POINTS = 256
/** Fraction of the data span to pad the PDF grid beyond [min, max] on each side. */
export const GRID_PADDING_FACTOR = 0.05
/** Gap between histogram bars (0 = touching). */
export const HISTOGRAM_BAR_GAP = 0.02

// `as const` is load-bearing on these: the fields target Plotly string-literal unions, so a
// widened `string` would not be assignable to `Data` / `Layout`.
/** Plotly `histnorm` value that turns bar heights into a probability density. */
export const HISTNORM_DENSITY = 'probability density' as const
/** Plotly trace `type` for the data histogram. */
export const TRACE_TYPE_HISTOGRAM = 'histogram' as const
/** Plotly trace `type` for the fitted-PDF line. */
export const TRACE_TYPE_SCATTER = 'scatter' as const
/** Plotly scatter `mode` for a connected line (no markers). */
export const SCATTER_MODE_LINES = 'lines' as const

/** Axis titles for the fit chart. */
export const X_AXIS_TITLE = 'value'
export const Y_AXIS_TITLE = 'density'

/** Stable layout reference for the fit chart. Defined once at module scope so `PlotlyChart`'s
 *  effect does not re-run `Plotly.react`/`purge` on every parent render (a fresh inline object
 *  would change identity each time). */
export const CHART_LAYOUT: Partial<Layout> = {
  bargap: HISTOGRAM_BAR_GAP,
  showlegend: true,
  xaxis: { title: { text: X_AXIS_TITLE } },
  yaxis: { title: { text: Y_AXIS_TITLE } },
}
