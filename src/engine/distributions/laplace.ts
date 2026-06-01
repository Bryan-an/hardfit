import cdf from '@stdlib/stats-base-dists-laplace-cdf'
import logpdf from '@stdlib/stats-base-dists-laplace-logpdf'
import quantile from '@stdlib/stats-base-dists-laplace-quantile'
import { mean } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter laplace MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Divisor for the midpoint average of the two middle order statistics (even-n median). */
const EVEN_MEDIAN_DIVISOR = 2

/**
 * Laplace's fitted parameters (data scale): `mu` = location (median), `b` = SCALE (diversity).
 * `@stdlib`'s laplace functions take `(x, mu, b)` with `b` the scale, so this matches them
 * directly — no rate/scale slot-trap as in gamma/weibull; `b` passes straight through.
 *
 * `FittedParams` is the engine-wide DTO (`Record<string, number>`) that crosses the
 * Comlink worker boundary, so it stays untyped. Inside this module we narrow `p` back
 * to the slots `@stdlib` expects with ONE assertion per density function — this is the
 * standard pattern every distribution copies. The cast is for READABILITY only (no
 * `?? Number.NaN` per arg) and to keep the `Distribution` interface and the registry
 * array generic-free. It does NOT validate slots at runtime: the assertion erases at
 * compile time, so a mismatched params object yields NaN — which `fitAll`'s
 * `Number.isFinite(ll)` guard turns into a reported failure rather than a crash.
 * Must be a `type` alias, not an `interface`: an interface lacks the implicit index
 * signature, so `p as LaplaceParams` would not compile without an `as unknown` step.
 */
type LaplaceParams = { mu: number; b: number }

// params: { mu = location (median), b = scale (diversity) }  (data scale)
// @stdlib laplace uses (x, mu, b) with b the SCALE — logpdf/cdf pass p.b directly.
export const laplace: Distribution = {
  name: DistributionName.Laplace,
  label: 'Laplace',
  k: 2,
  kind: 'continuous',
  fit(data): LaplaceParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`laplace: need n >= ${MIN_SAMPLE_SIZE}`)
    // Sort a COPY (numeric comparator — default sort is lexicographic) to keep `data` immutable.
    const sorted = [...data].sort((a, b) => a - b)
    const n = sorted.length
    const mid = Math.floor(n / EVEN_MEDIAN_DIVISOR)
    // `noUncheckedIndexedAccess` is on: narrow the order-stat reads (mid and mid-1 are valid
    // indices for n >= MIN_SAMPLE_SIZE, so the NaN fallback is unreachable).
    const upper = sorted[mid] ?? Number.NaN
    const lower = sorted[mid - 1] ?? Number.NaN
    // MLE location = median: middle order stat (odd n) or midpoint average of the two (even n).
    const mu = n % EVEN_MEDIAN_DIVISOR === 0 ? (lower + upper) / EVEN_MEDIAN_DIVISOR : upper
    // MLE scale = mean absolute deviation about the median.
    const b = mean(data.map((x) => Math.abs(x - mu)))
    if (!(b > 0)) throw new Error('laplace: degenerate (zero scale)')
    return { mu, b }
  },
  logpdf(x: number, p: FittedParams): number {
    const { mu, b } = p as LaplaceParams
    return logpdf(x, mu, b)
  },
  cdf(x: number, p: FittedParams): number {
    const { mu, b } = p as LaplaceParams
    return cdf(x, mu, b)
  },
  quantile(prob: number, p: FittedParams): number {
    const { mu, b } = p as LaplaceParams
    return quantile(prob, mu, b)
  },
}
