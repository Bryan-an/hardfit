import digamma from '@stdlib/math-base-special-digamma'
import trigamma from '@stdlib/math-base-special-trigamma'
import cdf from '@stdlib/stats-base-dists-gamma-cdf'
import logpdf from '@stdlib/stats-base-dists-gamma-logpdf'
import quantile from '@stdlib/stats-base-dists-gamma-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { mean, meanLog } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter gamma MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2

/**
 * Gamma's fitted parameters: `shape` (k), `scale` (theta = 1/rate), and `rate` (beta).
 * Carried together so `scale = 1/rate` and `mean = shape*scale` are both available without
 * re-deriving. THE TRAP: `@stdlib`'s gamma functions take `(x, alpha=shape, beta=RATE)`, so
 * `logpdf`/`cdf` MUST pass `rate`, NOT `scale`. Passing scale silently yields a wrong-but-
 * finite density (no crash, self-consistency still holds) — the convention-guard tests in
 * gamma.test.ts compare against the rate-parameterized closed form to catch exactly that swap.
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
 * signature, so `p as GammaParams` would not compile without an `as unknown` step.
 */
type GammaParams = { shape: number; scale: number; rate: number }

// params: { shape, scale, rate }  where scale = 1/rate, mean = shape*scale.
// @stdlib gamma uses (x, alpha=shape, beta=RATE) — logpdf/cdf pass p.rate, not p.scale.
export const gamma: Distribution = {
  name: DistributionName.Gamma,
  label: 'Gamma',
  k: 2,
  kind: 'continuous',
  fit(data): GammaParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`gamma: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('gamma requires all x > 0')
    const m = mean(data)
    const s = Math.log(m) - meanLog(data) // >= 0 by Jensen; the MLE target for ln k - psi(k)
    if (!(s > 0)) throw new Error('gamma: degenerate (zero log-variance)')
    // Minka (2002) closed-form shape seed, then 1-D Newton on g(k) = ln k - psi(k) - s.
    let k = (3 - s + Math.sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s)
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      const g = Math.log(k) - digamma(k) - s
      const gp = 1 / k - trigamma(k) // < 0
      const step = g / gp
      const next = k - step
      if (!Number.isFinite(next) || next <= 0) {
        k = k / 2 // damp toward positivity; loop continues
        continue
      }
      k = next
      if (Math.abs(step) < NEWTON_REL_TOL * k) break
    }
    const scale = m / k
    return { shape: k, scale, rate: 1 / scale }
  },
  logpdf(x: number, p: FittedParams): number {
    const { shape, rate } = p as GammaParams
    return logpdf(x, shape, rate) // beta = RATE, not scale
  },
  cdf(x: number, p: FittedParams): number {
    const { shape, rate } = p as GammaParams
    return cdf(x, shape, rate) // beta = RATE, not scale
  },
  quantile(prob: number, p: FittedParams): number {
    const { shape, rate } = p as GammaParams
    return quantile(prob, shape, rate) // beta = RATE, same slot-trap as cdf
  },
}
