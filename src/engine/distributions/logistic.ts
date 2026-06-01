import cdf from '@stdlib/stats-base-dists-logistic-cdf'
import logpdf from '@stdlib/stats-base-dists-logistic-logpdf'
import quantile from '@stdlib/stats-base-dists-logistic-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { mean, populationVariance } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter logistic MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Method-of-moments seed scale: sd(x)·sqrt(3)/pi (logistic variance = pi²s²/3 ⇒ s = sd·√3/π). */
const MOM_SCALE_FACTOR = Math.sqrt(3) / Math.PI
/** Initial (undamped) Newton step length; halved while the scale would go non-positive. */
const FULL_STEP = 1

/**
 * Logistic's fitted parameters: `mu` = location, `s` = SCALE. THE TRAP: `@stdlib`'s logistic
 * functions take `(x, mu, s=SCALE)` — `s` is the scale (variance = π²s²/3), NOT a rate, so
 * `logpdf`/`cdf`/`quantile` pass `s` directly. Passing a rate (1/s) would silently yield a
 * wrong-but-finite density (no crash, self-consistency still holds) — the convention-guard test
 * in logistic.test.ts compares against the elementary scale-parameterized closed form to catch it.
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
 * signature, so `p as LogisticParams` would not compile without an `as unknown` step.
 */
type LogisticParams = { mu: number; s: number }

// params: { mu = location, s = SCALE }  (s is SCALE, not rate; variance = pi²s²/3).
// @stdlib logistic uses (x, mu, s=SCALE) — logpdf/cdf/quantile pass p.s directly.
export const logistic: Distribution = {
  name: DistributionName.Logistic,
  label: 'Logistic',
  k: 2,
  kind: 'continuous',
  fit(data): LogisticParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`logistic: need n >= ${MIN_SAMPLE_SIZE}`)
    const n = data.length
    const xbar = mean(data)
    const variance = populationVariance(data, xbar)
    if (!(variance > 0)) throw new Error('logistic: degenerate (zero variance)')
    // Method-of-moments seed: mu0 = mean, s0 = sd·√3/π (matches the logistic variance π²s²/3).
    let mu = xbar
    let s = Math.sqrt(variance) * MOM_SCALE_FACTOR
    // 2-D Newton on the score G = [Σ(2F-1), Σz(2F-1) - n] with F=1/(1+e^{-z}), z=(x-mu)/s.
    // The log-concave logistic likelihood has a unique maximum, so Newton from the MoM seed
    // converges; we DAMP the step so the scale stays strictly positive.
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      let a = 0 // Σ(2F-1)
      let b = 0 // Σ z(2F-1)
      let sw = 0 // Σ w           where w = F(1-F) = dF/dz
      let swz = 0 // Σ wz
      let swz2 = 0 // Σ wz²
      for (const x of data) {
        const z = (x - mu) / s
        const f = 1 / (1 + Math.exp(-z))
        const w = f * (1 - f)
        const r = 2 * f - 1
        a += r
        b += z * r
        sw += w
        swz += w * z
        swz2 += w * z * z
      }
      const g0 = a
      const g1 = b - n
      // Jacobian of G w.r.t. (mu, s); the shared 1/s factor cancels in the linear solve.
      const j00 = -(2 / s) * sw
      const j01 = -(2 / s) * swz
      const j10 = -(1 / s) * (a + 2 * swz)
      const j11 = -(1 / s) * (b + 2 * swz2)
      const det = j00 * j11 - j01 * j10
      const stepMu = (j11 * g0 - j01 * g1) / det
      const stepS = (-j10 * g0 + j00 * g1) / det
      if (!Number.isFinite(stepMu) || !Number.isFinite(stepS)) break
      // Damp so the scale stays > 0: halve the step until s - step·stepS is positive.
      let step = FULL_STEP
      while (s - step * stepS <= 0) step /= 2
      mu -= step * stepMu
      s -= step * stepS
      if (Math.abs(step * stepMu) < NEWTON_REL_TOL * Math.abs(mu) + NEWTON_REL_TOL) {
        if (Math.abs(step * stepS) < NEWTON_REL_TOL * s) break
      }
    }
    return { mu, s }
  },
  logpdf(x: number, p: FittedParams): number {
    const { mu, s } = p as LogisticParams
    return logpdf(x, mu, s) // s = SCALE, not rate
  },
  cdf(x: number, p: FittedParams): number {
    const { mu, s } = p as LogisticParams
    return cdf(x, mu, s) // s = SCALE, not rate
  },
  quantile(prob: number, p: FittedParams): number {
    const { mu, s } = p as LogisticParams
    return quantile(prob, mu, s) // s = SCALE, same slot-trap as cdf
  },
}
