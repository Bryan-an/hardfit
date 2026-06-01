import cdf from '@stdlib/stats-base-dists-frechet-cdf'
import logpdf from '@stdlib/stats-base-dists-frechet-logpdf'
import quantile from '@stdlib/stats-base-dists-frechet-quantile'
import { type Distribution, DistributionName, type FittedParams } from '../types'
import { weibull } from './weibull'

/** Fewest observations a 2-parameter frechet MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Fixed location parameter (m). Frechet here is the 2-parameter EV Type II with origin at 0;
 *  `m` is NOT estimated or stored, but @stdlib's frechet functions take it as a 4th arg, so
 *  every call passes this literal in the location slot. */
const LOCATION = 0

/**
 * Frechet's fitted parameters: `shape` (alpha) and `scale` (s); the location m is fixed at 0
 * and is NOT stored. THE TRAP: `@stdlib`'s frechet functions take `(x, alpha=shape, s=SCALE,
 * m=location)` — `s` is the SCALE, NOT a rate, so `logpdf`/`cdf`/`quantile` pass `scale`
 * directly and the literal `LOCATION` (0) in the 4th slot. The MLE uses the reciprocal-Weibull
 * identity: X ~ Frechet(alpha, s) iff 1/X ~ Weibull(shape=alpha, scale=1/s). So `fit` reciprocates
 * the data, delegates to `weibull.fit`, and maps back as shape = w.shape, scale = 1 / w.scale.
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
 * signature, so `p as FrechetParams` would not compile without an `as unknown` step.
 */
type FrechetParams = { shape: number; scale: number }

// params: { shape = alpha, scale = s }  (s is SCALE, not rate; location m fixed at 0, not stored).
// @stdlib frechet uses (x, alpha=shape, s=SCALE, m=location) — logpdf/cdf/quantile pass p.scale and LOCATION.
export const frechet: Distribution = {
  name: DistributionName.Frechet,
  label: 'Fréchet',
  k: 2,
  kind: 'continuous',
  fit(data): FrechetParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`frechet: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('frechet requires all x > 0')
    if (data.every((v) => v === data[0])) throw new Error('frechet: degenerate (all values equal)')
    // Reciprocal-Weibull identity: X ~ Frechet(alpha, s) iff 1/X ~ Weibull(shape=alpha, scale=1/s).
    const reciprocals = data.map((x) => 1 / x)
    const w = weibull.fit(reciprocals) as { shape: number; scale: number }
    return { shape: w.shape, scale: 1 / w.scale }
  },
  logpdf(x: number, p: FittedParams): number {
    const { shape, scale } = p as FrechetParams
    return logpdf(x, shape, scale, LOCATION) // s = SCALE, m = 0
  },
  cdf(x: number, p: FittedParams): number {
    const { shape, scale } = p as FrechetParams
    return cdf(x, shape, scale, LOCATION) // s = SCALE, m = 0
  },
  quantile(prob: number, p: FittedParams): number {
    const { shape, scale } = p as FrechetParams
    return quantile(prob, shape, scale, LOCATION) // s = SCALE, m = 0
  },
}
