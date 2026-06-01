import digamma from '@stdlib/math-base-special-digamma'
import trigamma from '@stdlib/math-base-special-trigamma'
import cdf from '@stdlib/stats-base-dists-betaprime-cdf'
import logpdf from '@stdlib/stats-base-dists-betaprime-logpdf'
import quantile from '@stdlib/stats-base-dists-betaprime-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { mean, populationVariance } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter betaprime MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Robust fallback for either shape when the method-of-moments seed is degenerate. */
const FALLBACK_SHAPE = 2
/** Smallest seed value for `beta`: the moment formulas below require b > 2 to be finite,
 *  so a degenerate MoM result is clamped just above that boundary. */
const MIN_BETA_SEED = 2

/**
 * Beta Prime's (Pearson Type VI) fitted parameters: `alpha` and `beta`, BOTH pure SHAPES.
 * There is no rate/scale trap here — `@stdlib`'s betaprime functions take `(x, alpha, beta)`
 * with both as shapes, so `logpdf`/`cdf`/`quantile` pass them straight through. The ONLY slot
 * hazard is ORDER: alpha must precede beta. A swap is silent at the symmetric point x=1 (where
 * x^(alpha-1) is 1 and (1+x)^(-alpha-beta) depends only on alpha+beta, and B(a,b)=B(b,a)), so
 * the convention guard in betaprime.test.ts is written at x != 1 to actually discriminate order.
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
 * signature, so `p as BetaPrimeParams` would not compile without an `as unknown` step.
 */
type BetaPrimeParams = { alpha: number; beta: number }

// params: { alpha, beta }  — BOTH pure shapes (no rate/scale inversion).
// @stdlib betaprime uses (x, alpha, beta); the only hazard is the alpha-before-beta order.
export const betaprime: Distribution = {
  name: DistributionName.BetaPrime,
  label: 'Beta Prime',
  k: 2,
  kind: 'continuous',
  fit(data): BetaPrimeParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`betaprime: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('betaprime requires all x > 0')
    // Sufficient statistics for the betaprime log-likelihood. The per-observation log-density is
    // (alpha-1) ln x + (-alpha-beta) ln(1+x) - lnB(alpha,beta); averaging its score over the sample
    // depends on the data ONLY through L1 = mean(ln x) and L2 = mean(ln(1+x)).
    let sumLog = 0
    let sumLog1p = 0
    for (const x of data) {
      sumLog += Math.log(x)
      sumLog1p += Math.log(1 + x)
    }
    const n = data.length
    const L1 = sumLog / n
    const L2 = sumLog1p / n
    // Method-of-moments seed. For beta > 2, betaprime mean m = alpha/(beta-1) and variance reduces
    // to v = m(m+1)/(beta-2); inverting gives beta = 2 + m(m+1)/v and alpha = m(beta-1). Fall back
    // to the robust (2, 2) seed when the data give a degenerate (non-finite / out-of-support) guess.
    const m = mean(data)
    const v = populationVariance(data, m)
    let beta = MIN_BETA_SEED + (m * (m + 1)) / v
    if (!(beta > MIN_BETA_SEED) || !Number.isFinite(beta)) beta = MIN_BETA_SEED
    let alpha = m * (beta - 1)
    if (!(alpha > 0) || !Number.isFinite(alpha)) alpha = FALLBACK_SHAPE
    // 2-D Newton on the strictly-concave log-likelihood (unique maximum). Per-observation score:
    //   g_a = L1 - L2 - (psi(alpha) - psi(alpha+beta))
    //   g_b =     -L2 - (psi(beta)  - psi(alpha+beta))
    // (The L1 - L2 in g_a is the d/dalpha of (alpha-1)ln x + (-alpha-beta)ln(1+x).) Hessian:
    //   H_aa = -(psi'(alpha) - psi'(alpha+beta));  H_bb = -(psi'(beta) - psi'(alpha+beta));
    //   H_ab = H_ba = psi'(alpha+beta).  Solve [da,db] = -H^{-1}[g_a,g_b] via the 2x2 inverse
    //   (1/det)[[H_bb,-H_ab],[-H_ab,H_aa]], det = H_aa*H_bb - H_ab^2 (n cancels in the linear solve).
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      const psiAB = digamma(alpha + beta)
      const triAB = trigamma(alpha + beta)
      const gA = L1 - L2 - (digamma(alpha) - psiAB)
      const gB = -L2 - (digamma(beta) - psiAB)
      const hAA = -(trigamma(alpha) - triAB)
      const hBB = -(trigamma(beta) - triAB)
      const hAB = triAB
      const det = hAA * hBB - hAB * hAB
      const dA = -((hBB * gA - hAB * gB) / det)
      const dB = -((-hAB * gA + hAA * gB) / det)
      // Damped step: HALVE until the update keeps both shapes strictly positive (in-support).
      let step = 1
      let nextAlpha = alpha + dA
      let nextBeta = beta + dB
      while ((nextAlpha <= 0 || nextBeta <= 0) && step > NEWTON_REL_TOL) {
        step /= 2
        nextAlpha = alpha + step * dA
        nextBeta = beta + step * dB
      }
      const appliedA = step * dA
      const appliedB = step * dB
      alpha = nextAlpha
      beta = nextBeta
      if (!Number.isFinite(alpha) || !Number.isFinite(beta)) {
        throw new Error('betaprime: degenerate (Newton iteration diverged)')
      }
      if (Math.abs(appliedA) < NEWTON_REL_TOL * alpha && Math.abs(appliedB) < NEWTON_REL_TOL * beta)
        break
    }
    return { alpha, beta }
  },
  logpdf(x: number, p: FittedParams): number {
    const { alpha, beta } = p as BetaPrimeParams
    return logpdf(x, alpha, beta)
  },
  cdf(x: number, p: FittedParams): number {
    const { alpha, beta } = p as BetaPrimeParams
    return cdf(x, alpha, beta)
  },
  quantile(prob: number, p: FittedParams): number {
    const { alpha, beta } = p as BetaPrimeParams
    return quantile(prob, alpha, beta)
  },
}
