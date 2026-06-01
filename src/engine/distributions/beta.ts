import digamma from '@stdlib/math-base-special-digamma'
import trigamma from '@stdlib/math-base-special-trigamma'
import cdf from '@stdlib/stats-base-dists-beta-cdf'
import logpdf from '@stdlib/stats-base-dists-beta-logpdf'
import quantile from '@stdlib/stats-base-dists-beta-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { mean, populationVariance } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter beta MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Method-of-moments fallback shapes when the moment seed is non-finite/non-positive: the
 *  uniform on (0,1) is Beta(1, 1), a neutral, always-valid starting point for Newton. */
const FALLBACK_SHAPE = 1
/** Max times the Newton step is halved to keep both shapes positive. A FINITE step shrinks to ~0
 *  within ~60 halvings (2^60 ≈ 1e18), so this bounds the backtracking loop — without it a
 *  non-finite step (det → 0 ⇒ ±Infinity, and Infinity/2 stays Infinity) would spin forever. */
const MAX_STEP_HALVINGS = 60

/**
 * Beta's fitted parameters: `alpha` (first shape) and `beta` (second shape) — BOTH PURE SHAPES,
 * no scale/rate. `@stdlib`'s beta functions take `(x, alpha, beta)`, so `logpdf`/`cdf`/`quantile`
 * pass `alpha` then `beta` in that order. The ONLY slot trap is the alpha-before-beta ordering:
 * swapping them yields a wrong-but-finite density (no crash, self-consistency still holds) — the
 * convention-guard test in beta.test.ts uses ASYMMETRIC shapes (alpha=2, beta=5) at an asymmetric
 * x to catch exactly that swap. The MLE is a 2-D Newton on the strictly concave log-likelihood, so
 * the stationary point is the unique global maximum.
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
 * signature, so `p as BetaParams` would not compile without an `as unknown` step.
 */
type BetaParams = { alpha: number; beta: number }

// params: { alpha = first shape, beta = second shape }  (BOTH pure shapes, no scale/rate).
// @stdlib beta uses (x, alpha, beta) — logpdf/cdf/quantile pass p.alpha then p.beta, in that order.
export const beta: Distribution = {
  name: DistributionName.Beta,
  label: 'Beta',
  k: 2,
  kind: 'continuous',
  fit(data): BetaParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`beta: need n >= ${MIN_SAMPLE_SIZE}`)
    // Single positive predicate so NaN/±Infinity are rejected too (NaN <= 0 and NaN >= 1 are both
    // false, so the negated `v <= 0 || v >= 1` form would let a NaN observation slip through).
    if (data.some((v) => !(v > 0 && v < 1))) throw new Error('beta requires all 0 < x < 1')
    // Degenerate (no-spread) guard via the data RANGE, which is float-exact for identical values.
    // (A variance check is not: for constant data like [0.4, 0.4, 0.4] the computed mean carries a
    // rounding residual, so populationVariance returns ~1e-32 > 0 and the MLE would not converge.)
    let lo = data[0] ?? Number.NaN
    let hi = lo
    for (const x of data) {
      if (x < lo) lo = x
      if (x > hi) hi = x
    }
    if (!(hi > lo)) throw new Error('beta: degenerate (zero variance)')
    const m = mean(data)
    const v = populationVariance(data, m)
    // Sufficient statistics for the beta MLE: L1 = mean(ln x), L2 = mean(ln(1 - x)).
    let sumLn = 0
    let sumLn1m = 0
    for (const x of data) {
      sumLn += Math.log(x)
      sumLn1m += Math.log(1 - x)
    }
    const l1 = sumLn / data.length
    const l2 = sumLn1m / data.length
    // Method-of-moments seed from (m, v) in (0,1); fall back to Beta(1,1) if it is non-finite/non-positive.
    let a = m * ((m * (1 - m)) / v - 1)
    let b = (a * (1 - m)) / m
    if (!(Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0)) {
      a = FALLBACK_SHAPE
      b = FALLBACK_SHAPE
    }
    // 2-D Newton on the (strictly concave) mean log-likelihood. Scores g1/g2 and the constant
    // per-observation Hessian (H11, H22, H12) give the full step = -H^{-1} g; HALVE the step until
    // both shapes stay positive (terminates because a,b > 0 and the deltas shrink toward 0).
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      const dab = digamma(a + b)
      const g1 = dab - digamma(a) + l1
      const g2 = dab - digamma(b) + l2
      const tab = trigamma(a + b)
      const h11 = tab - trigamma(a)
      const h22 = tab - trigamma(b)
      const h12 = tab
      const det = h11 * h22 - h12 * h12
      let da = -(h22 * g1 - h12 * g2) / det
      let db = -(-h12 * g1 + h11 * g2) / det
      // A non-finite step (det → 0, or a digamma/trigamma overflow) cannot improve the estimate
      // and — critically — would make the positivity backtrack below spin forever (Infinity/2 is
      // still Infinity). Stop with the current, still-valid (a, b).
      if (!(Number.isFinite(da) && Number.isFinite(db))) break
      let na = a + da
      let nb = b + db
      // Backtrack so both shapes stay positive; bounded so it always terminates (the finite step
      // shrinks to ~0 within MAX_STEP_HALVINGS, at which point na→a>0, nb→b>0).
      for (let h = 0; (na <= 0 || nb <= 0) && h < MAX_STEP_HALVINGS; h++) {
        da /= 2
        db /= 2
        na = a + da
        nb = b + db
      }
      if (na <= 0 || nb <= 0) break // no positive step found; keep the current estimate
      a = na
      b = nb
      if (Math.abs(da) < NEWTON_REL_TOL * a && Math.abs(db) < NEWTON_REL_TOL * b) break
    }
    return { alpha: a, beta: b }
  },
  logpdf(x: number, p: FittedParams): number {
    const { alpha, beta: b } = p as BetaParams
    return logpdf(x, alpha, b) // (x, alpha, beta) — alpha before beta
  },
  cdf(x: number, p: FittedParams): number {
    const { alpha, beta: b } = p as BetaParams
    return cdf(x, alpha, b) // (x, alpha, beta) — alpha before beta
  },
  quantile(prob: number, p: FittedParams): number {
    const { alpha, beta: b } = p as BetaParams
    return quantile(prob, alpha, b) // (prob, alpha, beta) — same slot order as cdf
  },
}
