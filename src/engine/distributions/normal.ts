import cdf from '@stdlib/stats-base-dists-normal-cdf'
import logpdf from '@stdlib/stats-base-dists-normal-logpdf'
import { mean, populationVariance } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter normal MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2

/**
 * Normal's fitted parameters (data scale): `mu` = mean, `sigma` = standard deviation.
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
 * signature, so `p as NormalParams` would not compile without an `as unknown` step.
 */
type NormalParams = { mu: number; sigma: number }

// params: { mu = mean, sigma = standard deviation }  (data scale)
export const normal: Distribution = {
  name: DistributionName.Normal,
  label: 'Normal',
  k: 2,
  fit(data): NormalParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`normal: need n >= ${MIN_SAMPLE_SIZE}`)
    const mu = mean(data)
    const sigma = Math.sqrt(populationVariance(data, mu))
    if (!(sigma > 0)) throw new Error('normal: degenerate (zero variance)')
    return { mu, sigma }
  },
  logpdf(x: number, p: FittedParams): number {
    const { mu, sigma } = p as NormalParams
    return logpdf(x, mu, sigma)
  },
  cdf(x: number, p: FittedParams): number {
    const { mu, sigma } = p as NormalParams
    return cdf(x, mu, sigma)
  },
}
