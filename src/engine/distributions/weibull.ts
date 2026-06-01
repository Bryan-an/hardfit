import cdf from '@stdlib/stats-base-dists-weibull-cdf'
import logpdf from '@stdlib/stats-base-dists-weibull-logpdf'
import quantile from '@stdlib/stats-base-dists-weibull-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { meanLog } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter weibull MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2

/**
 * Weibull's fitted parameters: `shape` (k) and `scale` (lambda). THE TRAP: `@stdlib`'s weibull
 * functions take `(x, k=shape, lambda=SCALE)` — lambda is the SCALE, NOT a rate, so `logpdf`/`cdf`
 * pass `scale` directly. Passing a rate (1/scale) would silently yield a wrong-but-finite density
 * (no crash, self-consistency still holds) — the convention-guard test in weibull.test.ts compares
 * against the elementary scale-parameterized closed form to catch exactly that slot-swap.
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
 * signature, so `p as WeibullParams` would not compile without an `as unknown` step.
 */
type WeibullParams = { shape: number; scale: number }

// params: { shape = k, scale = lambda }  (lambda is SCALE, not rate).
// @stdlib weibull uses (x, k=shape, lambda=SCALE) — logpdf/cdf pass p.scale, not a rate.
export const weibull: Distribution = {
  name: DistributionName.Weibull,
  label: 'Weibull',
  k: 2,
  kind: 'continuous',
  fit(data): WeibullParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`weibull: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('weibull requires all x > 0')
    const n = data.length
    const meanLn = meanLog(data)
    // Pair each x with ln(x) once so the Newton loop iterates with for..of (no indexed access).
    const obs = data.map((x) => ({ x, lnx: Math.log(x) }))
    // Menon (1963) shape seed: k0 = (pi/sqrt6) / sd(ln x).
    let sdLnSq = 0
    for (const { lnx } of obs) sdLnSq += (lnx - meanLn) * (lnx - meanLn)
    const sdLn = Math.sqrt(sdLnSq / n)
    let k = sdLn > 0 ? Math.PI / Math.sqrt(6) / sdLn : 1
    // 1-D Newton on the shape score g(k) = 1/k + mean(ln x) - (sum x^k ln x)/(sum x^k).
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      let S0 = 0
      let S1 = 0
      let S2 = 0
      for (const { x, lnx } of obs) {
        const xk = x ** k
        S0 += xk
        S1 += xk * lnx
        S2 += xk * lnx * lnx
      }
      const g = 1 / k + meanLn - S1 / S0
      const gp = -1 / (k * k) - (S2 * S0 - S1 * S1) / (S0 * S0) // < 0 (g monotone decreasing)
      const step = g / gp
      const next = k - step
      if (!Number.isFinite(next) || next <= 0) {
        k = k / 2 // damp toward positivity; loop continues
        continue
      }
      k = next
      if (Math.abs(step) < NEWTON_REL_TOL * k) break
    }
    let S0 = 0
    for (const { x } of obs) S0 += x ** k
    const scale = (S0 / n) ** (1 / k)
    return { shape: k, scale }
  },
  logpdf(x: number, p: FittedParams): number {
    const { shape, scale } = p as WeibullParams
    return logpdf(x, shape, scale) // lambda = SCALE, not rate
  },
  cdf(x: number, p: FittedParams): number {
    const { shape, scale } = p as WeibullParams
    return cdf(x, shape, scale) // lambda = SCALE, not rate
  },
  quantile(prob: number, p: FittedParams): number {
    const { shape, scale } = p as WeibullParams
    return quantile(prob, shape, scale) // lambda = SCALE, not rate
  },
}
