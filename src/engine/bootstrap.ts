import normalCdf from '@stdlib/stats-base-dists-normal-cdf'
import normalQuantile from '@stdlib/stats-base-dists-normal-quantile'
import { BCA_JACKKNIFE_MAX_N, CHUNK } from './constants'
import { adStatistic, cramerVonMises, ksStatistic } from './gof'
import { mean } from './math'
import { makeSampler } from './sampling'
import type { Distribution, FittedParams } from './types'

/**
 * Confidence-interval math for HardFit's parametric bootstrap (U2): R type-7 percentile
 * intervals, the bias-corrected-and-accelerated (BCa) interval, and the jackknife pass
 * that supplies BCa's acceleration. The fused B-iteration loop that PRODUCES the
 * replicates fed in here lands in U3; this module is pure CI arithmetic.
 *
 * Φ (standard-normal CDF) and Φ⁻¹ (quantile) come from the same `@stdlib` packages the
 * distributions use, evaluated at (·, 0, 1).
 */

/** Standard-normal CDF, Φ(x). */
function phi(x: number): number {
  return normalCdf(x, 0, 1)
}

/** Standard-normal quantile, Φ⁻¹(p). */
function phiInv(p: number): number {
  return normalQuantile(p, 0, 1)
}

/**
 * R type-7 sample quantile by linear interpolation between order statistics:
 *   h = (n−1)p; lo = ⌊h⌋; hi = min(lo+1, n−1);
 *   q = sorted[lo] + (h − lo)·(sorted[hi] − sorted[lo]).
 * `sorted` MUST already be in ascending order. The `hi` clamp makes p=1 return the max
 * (h−lo=0 there, so the clamped term contributes nothing). Index reads are guarded for
 * `noUncheckedIndexedAccess`; a malformed (empty) array yields NaN rather than throwing.
 */
function quantileType7(sorted: readonly number[], p: number): number {
  const n = sorted.length
  const h = (n - 1) * p
  const lo = Math.floor(h)
  const hi = Math.min(lo + 1, n - 1)
  const vLo = sorted[lo]
  const vHi = sorted[hi]
  if (vLo === undefined || vHi === undefined) return Number.NaN
  return vLo + (h - lo) * (vHi - vLo)
}

/**
 * Percentile confidence interval at level `1 − alpha`: `[Q(alpha/2), Q(1 − alpha/2)]`
 * with R type-7 quantiles. Sorts a COPY internally (numeric comparator) so callers cannot
 * misuse it with unsorted input and the caller's array is never mutated.
 */
export function percentileCI(reps: readonly number[], alpha: number): [number, number] {
  const sorted = [...reps].sort((a, b) => a - b)
  return [quantileType7(sorted, alpha / 2), quantileType7(sorted, 1 - alpha / 2)]
}

/** Outcome of a BCa request: the interval and which method actually produced it. */
export interface BcaResult {
  ci: [number, number]
  method: 'bca' | 'percentile'
}

/**
 * Bias-corrected-and-accelerated (BCa) confidence interval for one scalar parameter
 * (Efron & Tibshirani 1993, ch. 14). The literal `6` and `1.5` (= 3/2) in the
 * acceleration, and the cube in `num`, are the published BCa formula — inlined, not
 * extracted to constants.
 *
 *  - z0 = Φ⁻¹(#{reps_b < thetaHat} / B)  (STRICT `<`, against the original θ̂).
 *  - EDGE GUARD: if `below ∈ {0, B}` then z0 = ±∞ → fall back to the percentile CI and
 *    flag `method:'percentile'`.
 *  - acceleration from the jackknife: m = mean(jackTheta);
 *    a = Σ(m − θ_i)³ / (6·[Σ(m − θ_i)²]^{3/2}); den=0 ⇒ a=0 (no acceleration).
 *  - adjusted percentiles: adj(p) = Φ( z0 + (z0 + z) / (1 − a·(z0 + z)) ), z = Φ⁻¹(p);
 *    CI = [Q7(a1), Q7(a2)] over the SORTED reps.
 *
 * INVARIANT: with z0=0 and a=0, adj(p) = Φ(Φ⁻¹(p)) = p, so BCa reduces to the percentile
 * CI (up to floating-point round-off in the Φ∘Φ⁻¹ composition).
 */
export function bcaCI(
  thetaHat: number,
  reps: readonly number[],
  jackTheta: readonly number[],
  alpha: number,
): BcaResult {
  const b = reps.length
  const sorted = [...reps].sort((a, b2) => a - b2)

  let below = 0
  for (const r of reps) if (r < thetaHat) below++ // STRICT `<`

  // z0 = ±∞ at the extremes → BCa is undefined; report the percentile CI instead.
  if (below === 0 || below === b) {
    return {
      ci: [quantileType7(sorted, alpha / 2), quantileType7(sorted, 1 - alpha / 2)],
      method: 'percentile',
    }
  }

  const z0 = phiInv(below / b)

  // Acceleration from the jackknife leave-one-out estimates.
  const m = mean(jackTheta)
  let num = 0
  let den = 0
  for (const t of jackTheta) {
    const d = m - t
    num += d * d * d
    den += d * d
  }
  const a = den === 0 ? 0 : num / (6 * den ** 1.5)

  // Adjusted BCa percentile for a nominal tail probability p.
  const adj = (p: number): number => {
    const z = phiInv(p)
    return phi(z0 + (z0 + z) / (1 - a * (z0 + z)))
  }
  const a1 = adj(alpha / 2)
  const a2 = adj(1 - alpha / 2)

  return {
    ci: [quantileType7(sorted, a1), quantileType7(sorted, a2)],
    method: 'bca',
  }
}

/**
 * Jackknife (leave-one-out) refits for the BCa acceleration: for each i in 0..n−1, refit
 * `dist` on `data` with observation i removed, and collect every fitted param value into a
 * per-param array. Returns one array of length n per parameter key.
 *
 * O(n) refits, each O(refit cost); for weibull (an O(n) Newton refit) this is O(n²)
 * overall, so the U3 caller caps n before calling this (BCa then falls back to percentile).
 */
export function jackknife(dist: Distribution, data: readonly number[]): Record<string, number[]> {
  const n = data.length
  const result: Record<string, number[]> = {}
  for (let i = 0; i < n; i++) {
    const without: number[] = []
    for (let j = 0; j < n; j++) {
      if (j === i) continue
      const v = data[j]
      if (v !== undefined) without.push(v)
    }
    const params: FittedParams = dist.fit(without)
    for (const key of Object.keys(params)) {
      const value = params[key]
      if (value === undefined) continue
      let arr = result[key]
      if (arr === undefined) {
        arr = []
        result[key] = arr
      }
      arr.push(value)
    }
  }
  return result
}

/** Confidence interval for one fitted parameter: the point estimate and both intervals. */
export interface ParamCI {
  point: number
  percentile: [number, number]
  bca: [number, number]
  method: 'bca' | 'percentile'
}

/** The fused-bootstrap result: per-parameter CIs plus the Lilliefors-correct GoF p-values. */
export interface BootstrapFitResult {
  seed: number
  paramCIs: Record<string, ParamCI>
  gofPValues: { ks: number; ad: number; cvm: number }
}

/** Knobs for {@link bootstrapFit}. `B` replicates, two-sided miscoverage `alpha`, one
 *  `seed` for the whole stream; optional cooperative `onChunk`/`isCancelled` hooks. */
export interface BootstrapFitOptions {
  B: number
  alpha: number
  seed: number
  onChunk?: (fraction: number) => void
  isCancelled?: () => boolean
}

/** Thrown when `isCancelled` reports a cancellation at a chunk boundary. */
export class BootstrapCancelledError extends Error {
  constructor() {
    super('bootstrapFit: cancelled')
    this.name = 'BootstrapCancelledError'
  }
}

/** A no-op yield to the event loop so a host worker can service messages between chunks. */
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0))
}

/**
 * The FUSED parametric-bootstrap loop (M2.2 U3): ONE pass over B replicates produces BOTH
 * the per-parameter confidence intervals (percentile + BCa) AND the Lilliefors-correct GoF
 * p-values for KS / Anderson–Darling / Cramér–von Mises. The only extra work is the single
 * jackknife pass that supplies BCa's acceleration (capped by `jackknifeMaxN`).
 *
 * Algorithm:
 *  1. Observed statistics S_obs at θ̂: KS/AD/CvM against `(x) => dist.cdf(x, fittedParams)`.
 *  2. Seed ONE sampler `draw` from `(dist.name, fittedParams, seed)` — never rebuilt inside
 *     the loop (rebuilding would replay identical samples, a silent statistical bug).
 *  3. For b = 0..B−1: draw an n-sample, REFIT it (a degenerate synthetic sample throws →
 *     `try/catch` skips that replicate), record each fitted param into `reps`, and for the
 *     refitted CDF count S*_b ≥ S_obs into the GoF tail counters.
 *  4. Jackknife (unless n exceeds the cap) → BCa per param; otherwise percentile only.
 *  5. p = (1 + #{S*_b ≥ S_obs}) / (B + 1) — the `+1` form (Davison & Hinkley); never 0.
 *
 * ASYNC + chunked: every CHUNK iterations it reports progress (`onChunk(b/B)`), polls
 * `isCancelled` (throwing {@link BootstrapCancelledError} when set), and yields a 0 ms timer
 * — these are the worker's cancellation/responsiveness checkpoints.
 */
export async function bootstrapFit(
  dist: Distribution,
  data: readonly number[],
  fittedParams: FittedParams,
  options: BootstrapFitOptions,
): Promise<BootstrapFitResult> {
  const { B, alpha, seed, onChunk, isCancelled } = options
  const n = data.length

  // 1. Observed statistics at the original fit.
  const cdfHat = (x: number): number => dist.cdf(x, fittedParams)
  const ksObs = ksStatistic(data, cdfHat)
  const adObs = adStatistic(data, cdfHat)
  const cvmObs = cramerVonMises(data, cdfHat)

  // 2. ONE seeded sampler stream for all B·n draws.
  const draw = makeSampler(dist.name, fittedParams, seed)

  // 3. Pre-initialize one reps array per fitted param (keeps reads index-safe and the
  //    all-refits-skipped case from feeding percentileCI an absent array).
  const reps: Record<string, number[]> = {}
  for (const key of Object.keys(fittedParams)) reps[key] = []

  let geKs = 0
  let geAd = 0
  let geCvm = 0

  for (let b = 0; b < B; b++) {
    if (b % CHUNK === 0) {
      onChunk?.(b / B)
      if (isCancelled?.()) throw new BootstrapCancelledError()
      await yieldToEventLoop()
    }

    const sample = Array.from({ length: n }, () => draw())

    let tb: FittedParams
    try {
      tb = dist.fit(sample)
    } catch {
      continue // degenerate synthetic sample → skip this replicate
    }

    for (const key of Object.keys(reps)) {
      const value = tb[key]
      if (value !== undefined) reps[key]?.push(value)
    }

    const cdfB = (x: number): number => dist.cdf(x, tb)
    if (ksStatistic(sample, cdfB) >= ksObs) geKs++
    if (adStatistic(sample, cdfB) >= adObs) geAd++
    if (cramerVonMises(sample, cdfB) >= cvmObs) geCvm++
  }

  // 4. Jackknife for BCa acceleration (skipped above the cap → percentile fallback).
  const jt = n <= BCA_JACKKNIFE_MAX_N ? jackknife(dist, data) : undefined

  const paramCIs: Record<string, ParamCI> = {}
  for (const [key, point] of Object.entries(fittedParams)) {
    const repArr = reps[key] ?? []
    const percentile = percentileCI(repArr, alpha)
    const jackTheta = jt?.[key]
    const bcaResult =
      jackTheta !== undefined
        ? bcaCI(point, repArr, jackTheta, alpha)
        : { ci: percentile, method: 'percentile' as const }
    paramCIs[key] = { point, percentile, bca: bcaResult.ci, method: bcaResult.method }
  }

  // 5. Lilliefors-correct bootstrap p-values: p = (1 + #{S* ≥ S_obs}) / (B + 1).
  const gofPValues = {
    ks: (1 + geKs) / (B + 1),
    ad: (1 + geAd) / (B + 1),
    cvm: (1 + geCvm) / (B + 1),
  }

  return { seed, paramCIs, gofPValues }
}
