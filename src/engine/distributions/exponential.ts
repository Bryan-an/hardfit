import cdf from '@stdlib/stats-base-dists-exponential-cdf'
import logpdf from '@stdlib/stats-base-dists-exponential-logpdf'
import quantile from '@stdlib/stats-base-dists-exponential-quantile'
import { mean } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 1-parameter exponential MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 1

/**
 * Exponential's fitted parameter: `rate` = lambda (RATE convention; mean = 1/rate).
 * `@stdlib`'s exponential functions take the rate, so this matches them directly.
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
 * signature, so `p as ExponentialParams` would not compile without an `as unknown` step.
 */
type ExponentialParams = { rate: number }

// params: { rate = lambda }  (RATE convention; mean = 1/rate)
export const exponential: Distribution = {
  name: DistributionName.Exponential,
  label: 'Exponential',
  k: 1,
  kind: 'continuous',
  fit(data): ExponentialParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`exponential: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v < 0)) throw new Error('exponential requires all x >= 0')
    const m = mean(data)
    if (!(m > 0)) throw new Error('exponential: degenerate (mean <= 0)')
    return { rate: 1 / m }
  },
  logpdf(x: number, p: FittedParams): number {
    const { rate } = p as ExponentialParams
    return logpdf(x, rate)
  },
  cdf(x: number, p: FittedParams): number {
    const { rate } = p as ExponentialParams
    return cdf(x, rate)
  },
  quantile(prob: number, p: FittedParams): number {
    const { rate } = p as ExponentialParams
    return quantile(prob, rate)
  },
}
