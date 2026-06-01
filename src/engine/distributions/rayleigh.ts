import cdf from '@stdlib/stats-base-dists-rayleigh-cdf'
import logpdf from '@stdlib/stats-base-dists-rayleigh-logpdf'
import quantile from '@stdlib/stats-base-dists-rayleigh-quantile'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 1-parameter rayleigh MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 1
/** MLE denominator factor: sigma = sqrt(sum x^2 / (2n)). */
const VARIANCE_FACTOR = 2

/**
 * Rayleigh's fitted parameter: `sigma` = SCALE. `@stdlib`'s rayleigh functions take the scale
 * directly (NO rate inversion), so `logpdf`/`cdf`/`quantile` pass `sigma` as-is. The MLE is
 * closed-form and unique: sigma = sqrt(sum x_i^2 / (2n)).
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
 * signature, so `p as RayleighParams` would not compile without an `as unknown` step.
 */
type RayleighParams = { sigma: number }

// params: { sigma = scale }  (SCALE convention; @stdlib takes sigma directly, no inversion).
export const rayleigh: Distribution = {
  name: DistributionName.Rayleigh,
  label: 'Rayleigh',
  k: 1,
  kind: 'continuous',
  fit(data): RayleighParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`rayleigh: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v < 0)) throw new Error('rayleigh requires all x >= 0')
    let sumSq = 0
    for (const v of data) sumSq += v * v
    const sigma = Math.sqrt(sumSq / (VARIANCE_FACTOR * data.length))
    if (!(sigma > 0)) throw new Error('rayleigh: degenerate (sum of squares is zero)')
    return { sigma }
  },
  logpdf(x: number, p: FittedParams): number {
    const { sigma } = p as RayleighParams
    return logpdf(x, sigma)
  },
  cdf(x: number, p: FittedParams): number {
    const { sigma } = p as RayleighParams
    return cdf(x, sigma)
  },
  quantile(prob: number, p: FittedParams): number {
    const { sigma } = p as RayleighParams
    return quantile(prob, sigma)
  },
}
