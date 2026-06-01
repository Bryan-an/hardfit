import cdf from '@stdlib/stats-base-dists-cosine-cdf'
import logpdf from '@stdlib/stats-base-dists-cosine-logpdf'
import quantile from '@stdlib/stats-base-dists-cosine-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { mean, populationVariance } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter cosine MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Method-of-moments scale factor: the raised cosine on [mu-s, mu+s] has variance
 *  s^2 (1/3 - 2/pi^2), so sd = s * sqrt(pi^2/3 - 2)/pi ⇒ s = pi * sd / sqrt(pi^2/3 - 2).
 *  (The pi factor matters: sqrt(pi^2/3 - 2) = pi * sqrt(1/3 - 2/pi^2), so dropping it makes the
 *  MoM seed pi-times too small.) This MoM scale seeds the optimizer; the hard support barrier is
 *  enforced separately by widening past max|x - mu0|. */
const MOM_SCALE_DENOM = Math.sqrt((Math.PI * Math.PI) / 3 - 2)
/** Seed-widening factor applied to max|x - mu0|: the support [mu0-s0, mu0+s0] MUST contain
 *  every observation or the seed log-likelihood is -inf, so s0 is forced strictly above the
 *  largest deviation. 1.05 leaves a 5% margin so the boundary points stay interior at the seed. */
const SEED_WIDEN = 1.05
/** Step-halving shrink factor used to retreat a Newton step back inside the hard barrier. */
const HALVING = 0.5
/** Max step-halvings before a Newton coordinate update is abandoned for this sweep. */
const MAX_HALVINGS = 60

/**
 * Cosine's (raised-cosine) fitted parameters: `mu` = LOCATION (the center/median) and `s` = SCALE
 * (> 0). The density is f(x) = (1/(2s))(1 + cos(pi*(x-mu)/s)) on the BOUNDED support [mu-s, mu+s]
 * and 0 outside. `@stdlib`'s cosine functions take `(x, mu, s)` with `s` the SCALE passed DIRECTLY
 * (no pi factor in the module — the pi appears only inside the cosine argument), so
 * `logpdf`/`cdf`/`quantile` forward `mu` and `s` as-is.
 *
 * The MLE has NO closed form and is CONSTRAINED: the log-likelihood is -inf unless s > max|x - mu|,
 * so every optimizer step is step-halved back inside that hard barrier (see `fit`). The optimizer
 * is coordinate ascent — a 1-D Newton update of mu given s, then of s given mu — re-checking the
 * barrier on each update; both per-coordinate second derivatives are negative (the LL is concave in
 * each coordinate on the feasible region), so each Newton step climbs.
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
 * signature, so `p as CosineParams` would not compile without an `as unknown` step.
 */
type CosineParams = { mu: number; s: number }

/** Largest absolute deviation of the data from `mu`; the hard support half-width must exceed it. */
function maxAbsDev(data: readonly number[], mu: number): number {
  let m = 0
  for (const x of data) {
    const d = Math.abs(x - mu)
    if (d > m) m = d
  }
  return m
}

/** tan(theta/2) = sin(theta)/(1 + cos(theta)); the recurring score factor. Finite on the OPEN
 *  support where 1 + cos(theta) > 0 (theta in (-pi, pi)); the optimizer never evaluates it AT the
 *  boundary because the barrier keeps every point strictly interior. */
function tanHalf(theta: number): number {
  return Math.sin(theta) / (1 + Math.cos(theta))
}

/** Score wrt mu: d/dmu = sum (pi/s) * tan(theta/2), theta = pi*(x-mu)/s. */
function scoreMu(data: readonly number[], mu: number, s: number): number {
  let g = 0
  for (const x of data) {
    const theta = (Math.PI * (x - mu)) / s
    g += (Math.PI / s) * tanHalf(theta)
  }
  return g
}

/** d(scoreMu)/dmu = -sum pi^2 / (s^2 (1 + cos theta)) < 0 (LL concave in mu on the support). */
function dScoreMu(data: readonly number[], mu: number, s: number): number {
  let g = 0
  for (const x of data) {
    const theta = (Math.PI * (x - mu)) / s
    g -= (Math.PI * Math.PI) / (s * s * (1 + Math.cos(theta)))
  }
  return g
}

/** Score wrt s: d/ds = -n/s + sum (pi*(x-mu)/s^2) * tan(theta/2). */
function scoreS(data: readonly number[], mu: number, s: number): number {
  let g = -data.length / s
  for (const x of data) {
    const u = x - mu
    const theta = (Math.PI * u) / s
    g += ((Math.PI * u) / (s * s)) * tanHalf(theta)
  }
  return g
}

/** d(scoreS)/ds = n/s^2 + sum [ -2 pi u / s^3 * tan(theta/2) - pi^2 u^2 / (s^4 (1 + cos theta)) ],
 *  using sec^2(theta/2) = 2 / (1 + cos theta). Negative at the feasible optimum (LL concave in s). */
function dScoreS(data: readonly number[], mu: number, s: number): number {
  let g = data.length / (s * s)
  for (const x of data) {
    const u = x - mu
    const theta = (Math.PI * u) / s
    g += ((-2 * Math.PI * u) / (s * s * s)) * tanHalf(theta)
    g -= (Math.PI * Math.PI * u * u) / (s * s * s * s * (1 + Math.cos(theta)))
  }
  return g
}

/** One barrier-safe 1-D Newton update of `mu` given `s`: step = scoreMu / dScoreMu, then halve the
 *  step until the new support [mu-s, mu+s] strictly contains every point (s > max|x - mu|). */
function updateMu(data: readonly number[], mu: number, s: number): number {
  const g = scoreMu(data, mu, s)
  const gp = dScoreMu(data, mu, s)
  if (!(gp < 0)) return mu // degenerate curvature: leave mu unchanged this sweep
  const step = g / gp
  let factor = 1
  for (let h = 0; h <= MAX_HALVINGS; h++) {
    const next = mu - step * factor
    if (maxAbsDev(data, next) < s) return next
    factor *= HALVING
  }
  return mu // never found a feasible step: keep mu
}

/** One barrier-safe 1-D Newton update of `s` given `mu`: step = scoreS / dScoreS, then halve the
 *  step until s stays both positive AND strictly above max|x - mu| (the hard support barrier). */
function updateS(data: readonly number[], mu: number, s: number): number {
  const g = scoreS(data, mu, s)
  const gp = dScoreS(data, mu, s)
  if (!(gp < 0)) return s // degenerate curvature: leave s unchanged this sweep
  const step = g / gp
  const floor = maxAbsDev(data, mu)
  let factor = 1
  for (let h = 0; h <= MAX_HALVINGS; h++) {
    const next = s - step * factor
    if (next > floor && next > 0) return next
    factor *= HALVING
  }
  return s // never found a feasible step: keep s
}

// params: { mu = location (center/median), s = SCALE (> 0) }; support is [mu-s, mu+s].
// @stdlib cosine uses (x, mu, s=SCALE) — logpdf/cdf/quantile pass mu and s DIRECTLY (no pi factor).
export const cosine: Distribution = {
  name: DistributionName.Cosine,
  label: 'Cosine',
  k: 2,
  kind: 'continuous',
  fit(data): CosineParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`cosine: need n >= ${MIN_SAMPLE_SIZE}`)
    const mu0 = mean(data)
    const sd = Math.sqrt(populationVariance(data, mu0))
    if (!(sd > 0)) throw new Error('cosine: degenerate (zero spread)')
    const momScale = (Math.PI * sd) / MOM_SCALE_DENOM
    // Widen the seed scale so [mu0-s0, mu0+s0] STRICTLY contains all data (LL is finite at the seed).
    let mu = mu0
    let s = Math.max(momScale, SEED_WIDEN * maxAbsDev(data, mu0))
    // Coordinate ascent: update mu given s, then s given mu, each a barrier-safe 1-D Newton step.
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      const nextMu = updateMu(data, mu, s)
      const nextS = updateS(data, nextMu, s)
      const dMu = nextMu - mu
      const dS = nextS - s
      mu = nextMu
      s = nextS
      if (Math.abs(dMu) < NEWTON_REL_TOL * s && Math.abs(dS) < NEWTON_REL_TOL * s) break
    }
    if (!(s > 0)) throw new Error('cosine: degenerate (non-positive scale)')
    return { mu, s }
  },
  logpdf(x: number, p: FittedParams): number {
    const { mu, s } = p as CosineParams
    return logpdf(x, mu, s) // s = SCALE, passed directly
  },
  cdf(x: number, p: FittedParams): number {
    const { mu, s } = p as CosineParams
    return cdf(x, mu, s) // s = SCALE, passed directly
  },
  quantile(prob: number, p: FittedParams): number {
    const { mu, s } = p as CosineParams
    return quantile(prob, mu, s) // s = SCALE, same slot as cdf
  },
}
