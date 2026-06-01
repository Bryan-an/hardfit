import cdf from '@stdlib/stats-base-dists-discrete-uniform-cdf'
import logpmf from '@stdlib/stats-base-dists-discrete-uniform-logpmf'
import quantile from '@stdlib/stats-base-dists-discrete-uniform-quantile'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter discrete-uniform MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2

/**
 * Discrete Uniform's fitted parameters: `a` = min integer support, `b` = max integer support,
 * INCLUSIVE — the distribution puts equal mass on the n = b - a + 1 integers {a, a+1, …, b}.
 * `@stdlib`'s discrete-uniform functions take `(x, a, b)` with these exact endpoints, so
 * `logpdf`/`cdf`/`quantile` pass `a`, `b` as-is (no transform). The MLE is the boundary
 * estimator (NOT a smooth interior optimum): â = min(data), b̂ = max(data).
 *
 * Discrete Uniform legitimately covers NEGATIVE integers (e.g. a = -3, b = 3), so `fit`
 * deliberately has NO `v < 0` support guard — the only data guard is integrality, and the
 * only parameter guard is non-degeneracy (b > a).
 *
 * Because this is a DISCRETE distribution, `logpdf` carries log-PMF mass: it delegates to
 * `@stdlib`'s `logpmf` (named -logpmf, NOT -logpdf, for discrete packages). The
 * `Distribution` interface method stays named `logpdf`; `selection.logLik` sums it correctly.
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
 * signature, so `p as DiscreteUniformParams` would not compile without an `as unknown` step.
 */
type DiscreteUniformParams = { a: number; b: number }

// params: { a = min support, b = max support }  (INCLUSIVE integer endpoints; n = b - a + 1).
// @stdlib discrete-uniform uses (x, a, b) — logpdf/cdf/quantile pass a, b directly.
export const discreteUniform: Distribution = {
  name: DistributionName.DiscreteUniform,
  label: 'Discrete Uniform',
  k: 2,
  kind: 'discrete',
  fit(data): DiscreteUniformParams {
    if (data.length < MIN_SAMPLE_SIZE)
      throw new Error(`discrete-uniform: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => !Number.isInteger(v)))
      throw new Error('discrete-uniform requires integer counts')
    // Single-pass min/max: Math.min(...data) would stack-overflow on large arrays. The
    // MIN_SAMPLE_SIZE check above guarantees the sentinels get overwritten by real data.
    let a = Number.POSITIVE_INFINITY
    let b = Number.NEGATIVE_INFINITY
    for (const v of data) {
      if (v < a) a = v
      if (v > b) b = v
    }
    if (!(b > a)) throw new Error('discrete-uniform: degenerate (max <= min)')
    return { a, b }
  },
  logpdf(x: number, p: FittedParams): number {
    const { a, b } = p as DiscreteUniformParams
    return logpmf(x, a, b) // log-PMF mass for the discrete logLik sum
  },
  cdf(x: number, p: FittedParams): number {
    const { a, b } = p as DiscreteUniformParams
    return cdf(x, a, b)
  },
  quantile(prob: number, p: FittedParams): number {
    const { a, b } = p as DiscreteUniformParams
    return quantile(prob, a, b)
  },
  support(p: FittedParams): { min: number; max: number } {
    const { a, b } = p as DiscreteUniformParams
    return { min: a, max: b }
  },
}
