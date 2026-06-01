import cdf from '@stdlib/stats-base-dists-invgamma-cdf'
import logpdf from '@stdlib/stats-base-dists-invgamma-logpdf'
import quantile from '@stdlib/stats-base-dists-invgamma-quantile'
import { type Distribution, DistributionName, type FittedParams } from '../types'
import { gamma } from './gamma'

/** Fewest observations a 2-parameter inverse-gamma MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2

/**
 * Inverse-gamma (Pearson V) fitted parameters: `shape` (alpha) and `scale` (beta). THE TRAP:
 * `@stdlib`'s invgamma functions take `(x, alpha=shape, beta=SCALE)` where `beta` is the SCALE
 * passed DIRECTLY — the OPPOSITE of `@stdlib`'s gamma, whose `beta` slot is a RATE. So
 * `logpdf`/`cdf`/`quantile` pass `scale` straight into the `beta` slot, no inversion. The MLE
 * uses the reciprocal-gamma identity: X ~ InvGamma(alpha, beta) iff 1/X ~ Gamma(shape=alpha,
 * rate=beta). So `fit` reciprocates the data, delegates to `gamma.fit`, and maps back as
 * shape = w.shape, scale = w.RATE. CRITICAL: the invgamma scale is the gamma fit's RATE
 * (w.rate = 1/w.scale), NOT w.scale — taking w.scale would silently flip the scale.
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
 * signature, so `p as InvGammaParams` would not compile without an `as unknown` step.
 */
type InvGammaParams = { shape: number; scale: number }

// params: { shape = alpha, scale = beta }  (beta is SCALE, passed DIRECTLY — NOT a rate; opposite of @stdlib gamma).
// @stdlib invgamma uses (x, alpha=shape, beta=SCALE) — logpdf/cdf/quantile pass p.scale into the beta slot, no inversion.
export const invgamma: Distribution = {
  name: DistributionName.InvGamma,
  label: 'Inverse Gamma',
  k: 2,
  kind: 'continuous',
  fit(data): InvGammaParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`invgamma: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('invgamma requires all x > 0')
    // Reciprocal-gamma identity: X ~ InvGamma(alpha, beta) iff 1/X ~ Gamma(shape=alpha, rate=beta).
    // gamma.fit will throw on degenerate reciprocals (zero log-variance).
    const w = gamma.fit(data.map((x) => 1 / x)) as { shape: number; scale: number; rate: number }
    // invgamma scale = gamma fit's RATE (w.rate = 1/w.scale), NOT w.scale.
    return { shape: w.shape, scale: w.rate }
  },
  logpdf(x: number, p: FittedParams): number {
    const { shape, scale } = p as InvGammaParams
    return logpdf(x, shape, scale) // beta = SCALE, passed directly
  },
  cdf(x: number, p: FittedParams): number {
    const { shape, scale } = p as InvGammaParams
    return cdf(x, shape, scale) // beta = SCALE, passed directly
  },
  quantile(prob: number, p: FittedParams): number {
    const { shape, scale } = p as InvGammaParams
    return quantile(prob, shape, scale) // beta = SCALE, same slot-trap as cdf
  },
}
