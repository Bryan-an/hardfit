import digamma from '@stdlib/math-base-special-digamma'
import trigamma from '@stdlib/math-base-special-trigamma'
import cdf from '@stdlib/stats-base-dists-chi-cdf'
import logpdf from '@stdlib/stats-base-dists-chi-logpdf'
import quantile from '@stdlib/stats-base-dists-chi-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { meanLog } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 1-parameter chi MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 1
/** The chi score equation works in a = k/2; this halves the seed and doubles the root back to k. */
const HALF = 0.5
/** Right-hand constant of the score: target = 2*meanLog(x) - ln 2. */
const LN2 = Math.log(2)

/**
 * Chi's fitted parameter: `k` = degrees of freedom (the single SHAPE; scale is fixed at 1).
 * `@stdlib`'s chi functions take `(x, k)` directly, so `logpdf`/`cdf`/`quantile` pass `k` as-is.
 * THE TRAP: `@stdlib` ships both `stats-base-dists-chi-*` AND `stats-base-dists-chisquare-*` —
 * DIFFERENT distributions (chi is the sqrt of chi-squared). This module imports from `-chi-`; the
 * convention-guard test in chi.test.ts compares logpdf(1,{k:3}) against the elementary chi closed
 * form (≈ −0.726, vs chi-squared's ≈ −1.419) to catch a wrong-package import. The MLE is found by
 * 1-D Newton on a = k/2: with target = 2·meanLog(x) − ln2, solve g(a) = digamma(a) − target = 0,
 * which is bijective (g'(a) = trigamma(a) > 0) so a unique finite root exists for any positive
 * sample — hence the ONLY degeneracy is non-positive data (no zero-log-variance case as in gamma).
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
 * signature, so `p as ChiParams` would not compile without an `as unknown` step.
 */
type ChiParams = { k: number }

// params: { k = degrees of freedom }  (single SHAPE; scale fixed at 1).
// @stdlib chi uses (x, k) — NOT chisquare; logpdf/cdf/quantile pass p.k directly.
export const chi: Distribution = {
  name: DistributionName.Chi,
  label: 'Chi',
  k: 1,
  kind: 'continuous',
  fit(data): ChiParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`chi: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('chi requires all x > 0')
    // Score target: psi(k/2) = 2*mean(ln x) - ln 2. Solve in a = k/2 (g monotone increasing).
    const target = 2 * meanLog(data) - LN2
    // Seed from the moment E[X^2] = k, so k0 = mean(x^2) and a0 = k0/2. Loop-summed (no indexing).
    let sumSq = 0
    for (const v of data) sumSq += v * v
    const k0 = sumSq / data.length
    let a = k0 * HALF
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      const g = digamma(a) - target
      const gp = trigamma(a) // > 0, so g is strictly increasing with a unique root
      const step = g / gp
      const next = a - step
      if (!Number.isFinite(next) || next <= 0) {
        a = a * HALF // damp toward positivity; loop continues
        continue
      }
      a = next
      if (Math.abs(step) < NEWTON_REL_TOL * a) break
    }
    return { k: 2 * a }
  },
  logpdf(x: number, p: FittedParams): number {
    const { k } = p as ChiParams
    return logpdf(x, k)
  },
  cdf(x: number, p: FittedParams): number {
    const { k } = p as ChiParams
    return cdf(x, k)
  },
  quantile(prob: number, p: FittedParams): number {
    const { k } = p as ChiParams
    return quantile(prob, k)
  },
}
