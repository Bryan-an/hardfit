import cdf from '@stdlib/stats-base-dists-cauchy-cdf'
import logpdf from '@stdlib/stats-base-dists-cauchy-logpdf'
import quantile from '@stdlib/stats-base-dists-cauchy-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter cauchy MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Quartile probabilities for the robust IQR-based seed (Cauchy has no finite moments,
 *  so mean/variance seeds are useless — order statistics are the only stable anchor). */
const MEDIAN_PROB = 0.5
const Q1_PROB = 0.25
const Q3_PROB = 0.75
/** Half the IQR (= (Q3−Q1)/2) is the closed-form Cauchy scale at the quartiles, hence the
 *  natural gamma seed; the 1/2 turns the full IQR into the per-side half-width gamma is. */
const HALF = 0.5

/**
 * Cauchy's fitted parameters: `x0` = location (the distribution's median) and `gamma` = SCALE
 * (the half-width at half-maximum, NOT a rate). `@stdlib`'s cauchy functions take `(x, x0, gamma)`
 * with gamma the scale, so `logpdf`/`cdf`/`quantile` pass `gamma` directly. The Cauchy has NO
 * finite mean or variance, so the MLE location is NOT the sample mean (nor exactly the median) —
 * it is found by an IRLS fixed point, defended against the likelihood's multimodality by a
 * multi-start over robust order-statistic seeds (see `fit`).
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
 * signature, so `p as CauchyParams` would not compile without an `as unknown` step.
 */
type CauchyParams = { x0: number; gamma: number }

/** Linearly-interpolated sample quantile of an ASCENDING-sorted array (numpy 'linear' / R type-7):
 *  h = (n−1)p; q = sorted[⌊h⌋] + (h−⌊h⌋)·(sorted[⌈h⌉] − sorted[⌊h⌋]). Index reads are guarded for
 *  `noUncheckedIndexedAccess`; a malformed (empty) array yields NaN rather than throwing. */
function sortedQuantile(sorted: readonly number[], prob: number): number {
  const h = (sorted.length - 1) * prob
  const lo = Math.floor(h)
  const hi = Math.ceil(h)
  const vLo = sorted[lo]
  const vHi = sorted[hi]
  if (vLo === undefined || vHi === undefined) return Number.NaN
  return vLo + (h - lo) * (vHi - vLo)
}

/** Total Cauchy log-likelihood at (x0, gamma): the score-maximization target and the basis for
 *  selecting the best multi-start result. Uses the same @stdlib logpdf the public API exposes. */
function logLik(data: readonly number[], x0: number, gamma: number): number {
  let ll = 0
  for (const x of data) ll += logpdf(x, x0, gamma)
  return ll
}

/**
 * IRLS fixed-point MLE from one (x0, gamma) seed. With weights w_i = 1/(gamma² + (x_i − x0)²):
 *   x0  ← (Σ w_i x_i) / (Σ w_i)
 *   γ²  ← (Σ w_i (x_i − x0)²) / (Σ w_i)
 * The gamma update IS the score solution: using the identity w_i (x_i − x0)² = 1 − γ² w_i, its
 * numerator is n − γ² Σw_i, so γ²_new = n/Σw_i − γ²; the fixed point γ²_new = γ² gives
 * 2γ² = n/Σw_i ⟺ γ² = n/(2 Σw_i), which is exactly ∂L/∂γ = n/γ − 2γ Σw_i = 0. So iterating to
 * convergence drives both scores to zero. Converges scale-relative (x0 may be ≈ 0, so a tolerance
 * relative to x0 would be ill-defined — both deltas are compared against gamma).
 */
function irls(data: readonly number[], seedX0: number, seedGamma: number): CauchyParams {
  let x0 = seedX0
  let gamma = seedGamma
  for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
    let sumW = 0
    let sumWX = 0
    for (const x of data) {
      const d = x - x0
      const w = 1 / (gamma * gamma + d * d)
      sumW += w
      sumWX += w * x
    }
    const nextX0 = sumWX / sumW
    let sumWD2 = 0
    for (const x of data) {
      const d = x - nextX0
      const w = 1 / (gamma * gamma + d * d)
      sumWD2 += w * d * d
    }
    const nextGamma = Math.sqrt(sumWD2 / sumW)
    const dX0 = nextX0 - x0
    const dGamma = nextGamma - gamma
    x0 = nextX0
    gamma = nextGamma
    if (Math.abs(dX0) < NEWTON_REL_TOL * gamma && Math.abs(dGamma) < NEWTON_REL_TOL * gamma) break
  }
  return { x0, gamma }
}

// params: { x0 = location (median), gamma = SCALE (half-width) }.
// @stdlib cauchy uses (x, x0, gamma=SCALE) — logpdf/cdf/quantile pass p.gamma, not a rate.
export const cauchy: Distribution = {
  name: DistributionName.Cauchy,
  label: 'Cauchy',
  k: 2,
  kind: 'continuous',
  fit(data): CauchyParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`cauchy: need n >= ${MIN_SAMPLE_SIZE}`)
    // The Cauchy MLE is UNBOUNDED (gamma -> 0, log-likelihood -> +infinity) when a STRICT majority
    // of the points share one exact value: those points pin the location there while the scale
    // collapses, and the LL rises without limit. The IQR seed below only catches such a tie when it
    // straddles both quartiles, so detect modal concentration directly — otherwise fit() returns a
    // near-degenerate tiny-gamma result whose large-but-finite LL would spuriously rank it best.
    const counts = new Map<number, number>()
    let maxCount = 0
    for (const x of data) {
      const c = (counts.get(x) ?? 0) + 1
      counts.set(x, c)
      if (c > maxCount) maxCount = c
    }
    if (maxCount * 2 > data.length) {
      throw new Error('cauchy: degenerate (a majority of values are identical; MLE unbounded)')
    }
    const sorted = [...data].sort((a, b) => a - b)
    const median = sortedQuantile(sorted, MEDIAN_PROB)
    const q1 = sortedQuantile(sorted, Q1_PROB)
    const q3 = sortedQuantile(sorted, Q3_PROB)
    const halfIQR = (q3 - q1) * HALF
    if (!(halfIQR > 0)) throw new Error('cauchy: degenerate (zero spread)')
    // Trimmed (middle-50%) mean: robust location anchor between Q1 and Q3.
    let trimSum = 0
    let trimCount = 0
    for (const x of sorted) {
      if (x >= q1 && x <= q3) {
        trimSum += x
        trimCount += 1
      }
    }
    const trimmedMean = trimCount > 0 ? trimSum / trimCount : median
    // MULTI-START defeats the Cauchy likelihood's multimodality: run IRLS from each robust seed
    // and keep the highest-LL result. The RAW seeds are folded into the candidate pool too, so
    // LL(fit) >= LL(best seed) holds by construction even if a run fails to fully converge — this
    // is what guarantees HardFit's LL >= scipy's single-start fit (the upstream parity gate).
    const seedX0s = [median, trimmedMean, median - halfIQR, median + halfIQR]
    let best: CauchyParams = { x0: median, gamma: halfIQR }
    let bestLL = Number.NEGATIVE_INFINITY
    for (const seedX0 of seedX0s) {
      for (const candidate of [{ x0: seedX0, gamma: halfIQR }, irls(data, seedX0, halfIQR)]) {
        const ll = logLik(data, candidate.x0, candidate.gamma)
        if (ll > bestLL) {
          bestLL = ll
          best = candidate
        }
      }
    }
    return best
  },
  logpdf(x: number, p: FittedParams): number {
    const { x0, gamma } = p as CauchyParams
    return logpdf(x, x0, gamma) // gamma = SCALE, not rate
  },
  cdf(x: number, p: FittedParams): number {
    const { x0, gamma } = p as CauchyParams
    return cdf(x, x0, gamma) // gamma = SCALE, not rate
  },
  quantile(prob: number, p: FittedParams): number {
    const { x0, gamma } = p as CauchyParams
    return quantile(prob, x0, gamma) // gamma = SCALE, same slot as cdf
  },
}
