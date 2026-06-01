import cdf from '@stdlib/stats-base-dists-gumbel-cdf'
import logpdf from '@stdlib/stats-base-dists-gumbel-logpdf'
import quantile from '@stdlib/stats-base-dists-gumbel-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { mean, populationVariance } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter gumbel MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Method-of-moments scale seed factor: beta0 = sd(x)·sqrt(6)/pi (Gumbel sd = beta·pi/sqrt6). */
const SCALE_SEED_FACTOR = Math.sqrt(6) / Math.PI

/**
 * Gumbel's fitted parameters: `mu` = location, `beta` = SCALE (the MAX / Extreme-Value-Type-I
 * convention, right-skewed). THE TRAP: `@stdlib`'s gumbel functions take `(x, mu, beta=SCALE)` —
 * beta is the SCALE, NOT a rate, so `logpdf`/`cdf`/`quantile` pass `beta` directly. `@stdlib`'s
 * CDF is exp(-exp(-(x-mu)/beta)), i.e. the MAX (not MIN) parameterization. Passing a rate (1/beta)
 * would silently yield a wrong-but-finite density (no crash, self-consistency still holds) — the
 * convention-guard test in gumbel.test.ts compares against the elementary closed form to catch it.
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
 * signature, so `p as GumbelParams` would not compile without an `as unknown` step.
 */
type GumbelParams = { mu: number; beta: number }

/**
 * NUMERICAL STABILITY (load-bearing): e^{-x/beta} can overflow for large x or small beta, so we
 * factor out the max exponent. With z = -x/beta and zmax = max(z), let w = e^{z - zmax} ∈ (0, 1].
 * The score and its derivative only need the ratios S1/S0 = (sum x e^{-x/beta})/(sum e^{-x/beta})
 * and S2/S0, which are e^{zmax}-invariant: ratio1 = (sum x w)/(sum w), ratio2 = (sum x² w)/(sum w).
 * meanW + zmax recover the location MLE: mu = -beta·(ln(meanW) + zmax). Returns all four at once
 * so the Newton loop and the post-loop mu step share one pass.
 */
function shiftedAccumulators(
  data: readonly number[],
  beta: number,
): { ratio1: number; ratio2: number; meanW: number; zmax: number } {
  let zmax = Number.NEGATIVE_INFINITY
  for (const x of data) {
    const z = -x / beta
    if (z > zmax) zmax = z
  }
  let sumW = 0 // sum w        (= S0 · e^{-zmax})
  let sumXW = 0 // sum x w     (= S1 · e^{-zmax})
  let sumX2W = 0 // sum x² w   (= S2 · e^{-zmax})
  for (const x of data) {
    const w = Math.exp(-x / beta - zmax)
    sumW += w
    sumXW += x * w
    sumX2W += x * x * w
  }
  return { ratio1: sumXW / sumW, ratio2: sumX2W / sumW, meanW: sumW / data.length, zmax }
}

// params: { mu = location, beta = SCALE }  (MAX / Extreme-Value-Type-I convention).
// @stdlib gumbel uses (x, mu, beta=SCALE) — logpdf/cdf/quantile pass p.beta, not a rate.
export const gumbel: Distribution = {
  name: DistributionName.Gumbel,
  label: 'Gumbel (max)',
  k: 2,
  kind: 'continuous',
  fit(data): GumbelParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`gumbel: need n >= ${MIN_SAMPLE_SIZE}`)
    const xbar = mean(data)
    const sd = Math.sqrt(populationVariance(data, xbar))
    if (!(sd > 0)) throw new Error('gumbel: degenerate (zero spread)')
    // Profiled 1-D MLE: solve the score g(beta) = xbar - beta - (sum x e^{-x/beta})/(sum e^{-x/beta})
    // for beta (monotone decreasing), then recover mu in closed form. Method-of-moments scale seed.
    let beta = sd * SCALE_SEED_FACTOR
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      const acc = shiftedAccumulators(data, beta)
      const g = xbar - beta - acc.ratio1
      // g'(beta) = -1 - (S2 S0 - S1^2)/(beta^2 S0^2) = -1 - (ratio2 - ratio1^2)/beta^2  (< 0).
      const gp = -1 - (acc.ratio2 - acc.ratio1 * acc.ratio1) / (beta * beta)
      const step = g / gp
      const next = beta - step
      if (!Number.isFinite(next) || next <= 0) {
        beta = beta / 2 // damp toward positivity; loop continues
        continue
      }
      beta = next
      if (Math.abs(step) < NEWTON_REL_TOL * beta) break
    }
    // mu MLE given the converged beta: sum e^{-(x-mu)/beta} = n  =>  mu = -beta·(ln(meanW) + zmax).
    const { meanW, zmax } = shiftedAccumulators(data, beta)
    const mu = -beta * (Math.log(meanW) + zmax)
    return { mu, beta }
  },
  logpdf(x: number, p: FittedParams): number {
    const { mu, beta } = p as GumbelParams
    return logpdf(x, mu, beta) // beta = SCALE, not rate
  },
  cdf(x: number, p: FittedParams): number {
    const { mu, beta } = p as GumbelParams
    return cdf(x, mu, beta) // beta = SCALE, not rate
  },
  quantile(prob: number, p: FittedParams): number {
    const { mu, beta } = p as GumbelParams
    return quantile(prob, mu, beta) // beta = SCALE, same slot-trap as cdf
  },
}
