import cdf from '@stdlib/stats-base-dists-lognormal-cdf'
import logpdf from '@stdlib/stats-base-dists-lognormal-logpdf'
import { mean, populationVariance } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter lognormal MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2

/**
 * Lognormal's fitted parameters: `mu` and `sigma` are the mean & std-dev of ln(X)
 * (LOG scale, NOT data scale). The MLE is just a Normal fit on the logs of the data.
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
 * signature, so `p as LognormalParams` would not compile without an `as unknown` step.
 */
type LognormalParams = { mu: number; sigma: number }

// params: { mu, sigma } = mean & std-dev of ln(X)  (LOG scale, NOT data scale)
export const lognormal: Distribution = {
  name: DistributionName.Lognormal,
  label: 'Lognormal',
  k: 2,
  fit(data): LognormalParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`lognormal: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('lognormal requires all x > 0')
    const logs = data.map((v) => Math.log(v))
    const mu = mean(logs)
    const sigma = Math.sqrt(populationVariance(logs, mu))
    if (!(sigma > 0)) throw new Error('lognormal: degenerate (constant logs)')
    return { mu, sigma }
  },
  // @stdlib's lognormal-logpdf includes the −ln(x) Jacobian, so Σ logpdf is on the data
  // scale and directly comparable to the other distributions' log-likelihoods.
  logpdf(x: number, p: FittedParams): number {
    const { mu, sigma } = p as LognormalParams
    return logpdf(x, mu, sigma)
  },
  cdf(x: number, p: FittedParams): number {
    const { mu, sigma } = p as LognormalParams
    return cdf(x, mu, sigma)
  },
}
