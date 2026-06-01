import digamma from '@stdlib/math-base-special-digamma'
import trigamma from '@stdlib/math-base-special-trigamma'
import cdf from '@stdlib/stats-base-dists-chisquare-cdf'
import logpdf from '@stdlib/stats-base-dists-chisquare-logpdf'
import quantile from '@stdlib/stats-base-dists-chisquare-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { mean, meanLog } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 1-parameter chi-squared MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 1
/** ln 2 appears in the chi-squared score g(k) = ln 2 + psi(k/2) - mean(ln x); named so the
 *  Newton loop carries no bare literal. */
const LN_TWO = Math.log(2)
/** Halving factor: the score and its derivative are functions of k/2 (chi-squared = gamma with
 *  shape k/2, scale 2), and both the Newton damping step and the half-argument use this. */
const HALF = 0.5

/**
 * Chi-squared's fitted parameter: `df` = degrees of freedom (the single shape; chi-squared is
 * gamma with shape df/2 and scale 2). `@stdlib`'s chisquare functions take `(x, k=df)` for
 * logpdf/cdf and `(p, k=df)` for quantile — probability FIRST in quantile, so `logpdf`/`cdf` pass
 * `df` as the SECOND argument while `quantile` passes the probability first and `df` second. The
 * MLE has no closed form: solve the score g(k) = ln 2 + psi(k/2) - mean(ln x) = 0 by 1-D Newton.
 * Because g is strictly increasing in k (g'(k) = (1/2) psi'(k/2) > 0), the root is unique.
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
 * signature, so `p as ChiSquaredParams` would not compile without an `as unknown` step.
 */
type ChiSquaredParams = { df: number }

// params: { df = degrees of freedom }  (single shape; chi-squared = gamma(shape=df/2, scale=2)).
// @stdlib chisquare uses (x, k=df) for logpdf/cdf and (p, k=df) for quantile (probability FIRST).
export const chisquare: Distribution = {
  name: DistributionName.ChiSquared,
  label: 'Chi-Squared',
  k: 1,
  kind: 'continuous',
  fit(data): ChiSquaredParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`chisquare: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('chisquare requires all x > 0')
    const meanLn = meanLog(data)
    if (!Number.isFinite(meanLn)) throw new Error('chisquare: degenerate (non-finite mean log)')
    // Seed df0 = sample mean, since E[chi2(k)] = k.
    let df = mean(data)
    // 1-D Newton on the score g(k) = ln 2 + psi(k/2) - mean(ln x), monotone increasing -> unique root.
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      const g = LN_TWO + digamma(df * HALF) - meanLn
      const gp = HALF * trigamma(df * HALF) // > 0 (g strictly increasing)
      const step = g / gp
      const next = df - step
      if (!Number.isFinite(next) || next <= 0) {
        df = df * HALF // damp toward positivity; loop continues
        continue
      }
      df = next
      if (Math.abs(step) < NEWTON_REL_TOL * df) break
    }
    return { df }
  },
  logpdf(x: number, p: FittedParams): number {
    const { df } = p as ChiSquaredParams
    return logpdf(x, df)
  },
  cdf(x: number, p: FittedParams): number {
    const { df } = p as ChiSquaredParams
    return cdf(x, df)
  },
  quantile(prob: number, p: FittedParams): number {
    const { df } = p as ChiSquaredParams
    return quantile(prob, df) // probability FIRST, then df
  },
}
