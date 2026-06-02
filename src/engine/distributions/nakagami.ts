import digamma from '@stdlib/math-base-special-digamma'
import gammaln from '@stdlib/math-base-special-gammaln'
import trigamma from '@stdlib/math-base-special-trigamma'
import gammaCdf from '@stdlib/stats-base-dists-gamma-cdf'
import gammaQuantile from '@stdlib/stats-base-dists-gamma-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { mean, meanLog } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter Nakagami MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Minka (2002) closed-form shape-seed coefficients, reused VERBATIM from gamma.ts's 1-D Newton
 *  (Nakagami's `m` reduces EXACTLY to the gamma shape MLE on x²). The seed is
 *  m0 = (SEED_C3 − s + √((s − SEED_C3)² + SEED_C24·s)) / (SEED_C12·s). Named to keep the
 *  no-magic-literals rule satisfied while staying byte-aligned with the gamma derivation. */
const SEED_C3 = 3
const SEED_C24 = 24
const SEED_C12 = 12
/** Density algebra constants (no-magic-literals rule). The Nakagami log-density is
 *    ln 2 + m·ln m − lnΓ(m) − m·ln Ω + (2m − 1)·ln x − m·x²/Ω.
 *  TWO scales the `m` in the `2m − 1` exponent of the `x` power; ONE is the `−1` in that same
 *  exponent. (`ln 2` is `Math.LN2`; `m·x²` squares x via `x*x`.) */
const TWO = 2
const ONE = 1

/**
 * Nakagami-m fitted parameters: `m` = shape / fading figure (> 0; m = 1 is Rayleigh, larger m → less
 * fading / lighter tail) and `Omega` = the spread / second moment E[x²] (> 0). HardFit's `(m, Omega)`
 * is NOT scipy's parameterization — scipy.stats.nakagami uses `(nu = m, loc=0, scale = √Ω)`, so the
 * Mode-B fixture freeze is `nakagami(m, 0, sqrt(Omega))` and `Omega = scale²` (scipy's scale multiplies
 * x; Ω is a SECOND-moment quantity — forgetting to square the scale makes Ω off by a √).
 *
 * THE GAMMA-ON-x² REDUCTION (load-bearing): if X ~ Nakagami(m, Ω) then X² ~ Gamma(shape = m,
 * scale = Ω/m). So the cdf is `gammaCdf(x², m, RATE)` and the quantile is `√gammaQuantile(p, m, RATE)`,
 * and the MLE of `m` is EXACTLY the gamma shape MLE on the squared data (the same 1-D Newton gamma.ts
 * uses). `@stdlib`'s gamma functions take `(x, alpha = shape, beta = RATE)`, and the natural scale here
 * is Ω/m, so the RATE is its INVERSE = `m/Ω` — passed to cdf/quantile/sampler EVERYWHERE. Passing Ω/m
 * (the scale) as the rate is wrong-but-finite (no crash); the rate-vs-scale convention-guard test in
 * nakagami.test.ts pins the correct `m/Ω` rate and asserts it differs from the inverted one.
 *
 * `FittedParams` is the engine-wide DTO (`Record<string, number>`) that crosses the Comlink worker
 * boundary, so it stays untyped. Inside this module we narrow `p` back to the slots the density expects
 * with ONE assertion per function — the standard pattern every distribution copies. The cast is for
 * READABILITY only; it erases at compile time, so a mismatched params object yields NaN — which
 * `fitAll`'s `Number.isFinite(ll)` guard turns into a reported failure rather than a crash. Must be a
 * `type` alias, not an `interface` (an interface lacks the implicit index signature, so
 * `p as NakagamiParams` would not compile without an `as unknown` step).
 */
type NakagamiParams = { m: number; Omega: number }

/** Hand-composed Nakagami log-density at (x; m, Omega); −Infinity outside the open support x > 0.
 *  VERIFIED to match `scipy.nakagami.logpdf` to machine precision. The single building block of the
 *  public `logpdf`. */
function nakagamiLogpdf(x: number, m: number, Omega: number): number {
  if (!(x > 0)) return Number.NEGATIVE_INFINITY
  return (
    Math.LN2 +
    m * Math.log(m) -
    gammaln(m) -
    m * Math.log(Omega) +
    (TWO * m - ONE) * Math.log(x) -
    (m * x * x) / Omega
  )
}

// params: { m = shape/fading (> 0), Omega = spread E[x²] (> 0) }.
// Reduces to Gamma(shape=m, scale=Ω/m) on x²: @stdlib gamma takes (x, alpha=shape, beta=RATE),
// and the rate is m/Ω (the INVERSE of the natural scale Ω/m) — cdf/quantile/sampler pass m/Ω.
export const nakagami: Distribution = {
  name: DistributionName.Nakagami,
  label: 'Nakagami',
  k: 2,
  kind: 'continuous',
  fit(data): NakagamiParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`nakagami: need n >= ${MIN_SAMPLE_SIZE}`)
    for (const x of data) {
      if (!(x > 0)) throw new Error('nakagami: data must be strictly positive (x > 0)')
    }
    // Ω is the closed-form MLE of the second moment: Ω = mean(x²).
    const Omega = mean(data.map((x) => x * x))
    // m reduces EXACTLY to the gamma shape MLE on x²: the target is s = ln(mean x²) − mean(ln x²).
    // Since ln(x²) = 2·ln x, mean(ln x²) = 2·meanLog(x), so s = ln Ω − 2·meanLog(data). >= 0 by Jensen.
    const s = Math.log(Omega) - TWO * meanLog(data)
    if (!(s > 0)) throw new Error('nakagami: degenerate (zero log-variance)')
    // Minka (2002) closed-form shape seed, then the SAME 1-D Newton gamma.ts uses, on
    // g(m) = ln m − ψ(m) − s, g'(m) = 1/m − ψ'(m). Do NOT clamp m >= 0.5: a floor drops the LL below
    // scipy on low-m (heavy-fading) data and fails the parity gate. The unconstrained Newton is left free.
    let m = (SEED_C3 - s + Math.sqrt((s - SEED_C3) * (s - SEED_C3) + SEED_C24 * s)) / (SEED_C12 * s)
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      const g = Math.log(m) - digamma(m) - s
      const gp = ONE / m - trigamma(m) // < 0
      const step = g / gp
      const next = m - step
      if (!Number.isFinite(next) || next <= 0) {
        m = m / TWO // damp toward positivity; loop continues
        continue
      }
      m = next
      if (Math.abs(step) < NEWTON_REL_TOL * m) break
    }
    return { m, Omega }
  },
  logpdf(x: number, p: FittedParams): number {
    const { m, Omega } = p as NakagamiParams
    return nakagamiLogpdf(x, m, Omega)
  },
  cdf(x: number, p: FittedParams): number {
    const { m, Omega } = p as NakagamiParams
    if (!(x > 0)) return 0 // below the open support x > 0
    return gammaCdf(x * x, m, m / Omega) // x² ~ Gamma(shape=m, RATE=m/Ω); beta = RATE, not scale
  },
  quantile(prob: number, p: FittedParams): number {
    const { m, Omega } = p as NakagamiParams
    return Math.sqrt(gammaQuantile(prob, m, m / Omega)) // invert via the squared-data gamma; RATE=m/Ω
  },
}
