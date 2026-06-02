import { describe, expect, it } from 'vitest'
import { bcaCI, bootstrapFit, jackknife, percentileCI } from './bootstrap'
import { BCA_JACKKNIFE_MAX_N } from './constants'
import { exponential } from './distributions/exponential'
import { normal } from './distributions/normal'
import { poisson } from './distributions/poisson'
import { studentT } from './distributions/student-t'
import { mean } from './math'
import { makeSampler } from './sampling'
import type { Distribution, FittedParams } from './types'

/** Two-sided CI level used throughout: a 95% CI (tails at 0.025 / 0.975). */
const ALPHA = 0.05
/** Float tolerance (decimal places) for the BCa↔percentile invariant: the Φ∘Φ⁻¹ round
 *  trip is exact only to ~1e-16, so endpoints match to many places but not bit-for-bit. */
const INVARIANT_PLACES = 10

describe('percentileCI (R type-7 linear interpolation)', () => {
  it('matches a hand-computed quantile pair for [1,2,3,4,5] at alpha=0.05', () => {
    // q(p): h=(n-1)p; lo=floor(h); v[lo]+(h-lo)(v[lo+1]-v[lo]).
    //   q(0.025): h=4·0.025=0.1 → 1 + 0.1·(2-1) = 1.1
    //   q(0.975): h=4·0.975=3.9 → 4 + 0.9·(5-4) = 4.9
    const [lo, hi] = percentileCI([1, 2, 3, 4, 5], ALPHA)
    expect(lo).toBeCloseTo(1.1, INVARIANT_PLACES)
    expect(hi).toBeCloseTo(4.9, INVARIANT_PLACES)
  })

  it('returns the exact endpoints when h lands on an integer index', () => {
    // n=5, alpha=0.5 → p=0.25: h=4·0.25=1 → exactly v[1]; p=0.75: h=3 → exactly v[3].
    const [lo, hi] = percentileCI([10, 20, 30, 40, 50], 0.5)
    expect(lo).toBe(20)
    expect(hi).toBe(40)
  })

  it('sorts a copy internally (unsorted input yields the same CI; input untouched)', () => {
    const input = [5, 1, 4, 2, 3]
    const [lo, hi] = percentileCI(input, ALPHA)
    expect(lo).toBeCloseTo(1.1, INVARIANT_PLACES)
    expect(hi).toBeCloseTo(4.9, INVARIANT_PLACES)
    expect(input).toEqual([5, 1, 4, 2, 3]) // not mutated
  })
})

describe('bcaCI', () => {
  /** Build reps where exactly half are STRICTLY below thetaHat → below/B=0.5 → z0=0. */
  function repsWithZeroZ0(thetaHat: number, half: number): number[] {
    const below = Array.from({ length: half }, (_, i) => thetaHat - 1 - i) // all < thetaHat
    const atOrAbove = Array.from({ length: half }, (_, i) => thetaHat + i) // none < thetaHat
    return [...below, ...atOrAbove]
  }

  it('INVARIANT: z0=0 (half reps below) + a=0 (constant jackTheta → den=0) ⇒ BCa = percentile', () => {
    const thetaHat = 100
    const reps = repsWithZeroZ0(thetaHat, 6) // 6 below, 6 at-or-above → below/12 = 0.5
    const jackTheta = Array(8).fill(42) // all equal → den=0 → a=0
    const { ci, method } = bcaCI(thetaHat, reps, jackTheta, ALPHA)
    const percentile = percentileCI(reps, ALPHA)
    expect(method).toBe('bca')
    expect(ci[0]).toBeCloseTo(percentile[0], INVARIANT_PLACES)
    expect(ci[1]).toBeCloseTo(percentile[1], INVARIANT_PLACES)
  })

  it('EDGE GUARD: all reps below thetaHat (below=B) ⇒ percentile fallback, finite', () => {
    const thetaHat = 100
    const reps = [10, 20, 30, 40, 50] // all strictly < thetaHat → below = B
    const jackTheta = [1, 2, 3, 4, 5] // would give a≠0, but the guard fires first
    const { ci, method } = bcaCI(thetaHat, reps, jackTheta, ALPHA)
    expect(method).toBe('percentile')
    expect(ci).toEqual(percentileCI(reps, ALPHA))
    expect(Number.isFinite(ci[0])).toBe(true)
    expect(Number.isFinite(ci[1])).toBe(true)
  })

  it('EDGE GUARD: no reps below thetaHat (below=0) ⇒ percentile fallback, finite', () => {
    const thetaHat = 0
    const reps = [10, 20, 30, 40, 50] // none < thetaHat → below = 0
    const jackTheta = [1, 2, 3, 4, 5]
    const { ci, method } = bcaCI(thetaHat, reps, jackTheta, ALPHA)
    expect(method).toBe('percentile')
    expect(ci).toEqual(percentileCI(reps, ALPHA))
    expect(Number.isFinite(ci[0])).toBe(true)
    expect(Number.isFinite(ci[1])).toBe(true)
  })

  it('ACCELERATION SIGN: jackTheta with num=Σ(m−θ_i)³ > 0 ⇒ a>0, both endpoints shift UP', () => {
    const thetaHat = 100
    const reps = repsWithZeroZ0(thetaHat, 50) // z0=0 so the shift is purely the acceleration
    // A long LEFT tail: a few values far below the mean make Σ(m−θ_i)³ > 0.
    const jackTheta = [-100, -100, 1, 1, 1, 1, 1, 1, 1, 1]
    const m = mean(jackTheta)
    let num = 0
    let den = 0
    for (const t of jackTheta) {
      num += (m - t) ** 3
      den += (m - t) ** 2
    }
    expect(num).toBeGreaterThan(0) // confirm the construction by direct computation
    expect(den).toBeGreaterThan(0)
    const a = num / (6 * den ** 1.5)
    expect(a).toBeGreaterThan(0)

    const { ci, method } = bcaCI(thetaHat, reps, jackTheta, ALPHA)
    const percentile = percentileCI(reps, ALPHA)
    expect(method).toBe('bca')
    // With z0=0 and a>0, both BCa percentiles exceed (α/2, 1−α/2) → endpoints move up.
    expect(ci[0]).toBeGreaterThan(percentile[0])
    expect(ci[1]).toBeGreaterThan(percentile[1])
  })
})

describe('jackknife', () => {
  it('leave-one-out normal means match hand computation; n estimates per param', () => {
    const data = [2, 4, 6, 8] // sum = 20
    const result = jackknife(normal, data)
    // normal.fit returns { mu, sigma }; expect one array of length n per key.
    expect(Object.keys(result).sort()).toEqual(['mu', 'sigma'])
    expect(result.mu).toHaveLength(data.length)
    expect(result.sigma).toHaveLength(data.length)
    // Leave-one-out means: (20 - x_i) / 3.
    const expectedMu = data.map((x) => (20 - x) / (data.length - 1))
    for (let i = 0; i < data.length; i++) {
      expect(result.mu?.[i]).toBeCloseTo(expectedMu[i] ?? Number.NaN, INVARIANT_PLACES)
    }
  })
})

describe('bootstrapFit (fused CIs + GoF p-values)', () => {
  /** Modest B for test speed; the production default (DEFAULT_BOOTSTRAP_B) is 999. */
  const B = 300
  /** A fixed seed makes every run below deterministic (single seeded sampler stream). */
  const SEED = 12345

  /** Deterministic dataset drawn FROM `dist` at `params` via the seeded sampler. */
  function simulate(dist: Distribution, params: FittedParams, n: number, seed: number): number[] {
    const draw = makeSampler(dist.name, params, seed)
    return Array.from({ length: n }, () => draw())
  }

  it('REPRODUCIBILITY: same seed ⇒ identical paramCIs + gofPValues', async () => {
    const data = simulate(normal, { mu: 5, sigma: 2 }, 60, 777)
    const fitted = normal.fit(data)
    const a = await bootstrapFit(normal, data, fitted, { B, alpha: ALPHA, seed: SEED })
    const b = await bootstrapFit(normal, data, fitted, { B, alpha: ALPHA, seed: SEED })
    expect(a).toEqual(b) // deep equality: every CI endpoint + p-value bit-identical
  })

  it('CI SANITY: percentile CI brackets the point estimate for a well-fit case', async () => {
    const data = simulate(normal, { mu: 5, sigma: 2 }, 60, 777)
    const fitted = normal.fit(data)
    const { paramCIs } = await bootstrapFit(normal, data, fitted, { B, alpha: ALPHA, seed: SEED })
    for (const ci of Object.values(paramCIs)) {
      expect(ci.percentile[0]).toBeLessThanOrEqual(ci.point)
      expect(ci.point).toBeLessThanOrEqual(ci.percentile[1])
    }
  })

  it('P-VALUE SANITY: good fit ⇒ high p, obviously-wrong fit ⇒ low p', async () => {
    // Good fit: normal data fitted by normal → bootstrap p should NOT be tiny.
    const normalData = simulate(normal, { mu: 10, sigma: 1 }, 80, 2024)
    const normalFit = normal.fit(normalData)
    const good = await bootstrapFit(normal, normalData, normalFit, { B, alpha: ALPHA, seed: SEED })

    // Bad fit: the SAME bell-shaped (all-positive) data forced into an exponential, whose
    // monotone-decreasing density cannot match a bell → large statistic → p ≈ 1/(B+1).
    const expFit = exponential.fit(normalData)
    const bad = await bootstrapFit(exponential, normalData, expFit, { B, alpha: ALPHA, seed: SEED })

    // Continuous fits always carry GoF p-values (only discrete fits null them out).
    const goodGof = good.gofPValues
    const badGof = bad.gofPValues
    if (goodGof === null || badGof === null) throw new Error('continuous fit must have gofPValues')
    // Direction-based bounds (each bootstrap GoF p is ~Uniform under a correct model):
    expect(goodGof.ks).toBeGreaterThan(0.1)
    expect(goodGof.ad).toBeGreaterThan(0.1)
    expect(goodGof.cvm).toBeGreaterThan(0.1)
    expect(badGof.ks).toBeLessThan(0.05)
    expect(badGof.ad).toBeLessThan(0.05)
    expect(badGof.cvm).toBeLessThan(0.05)
    expect(goodGof.ks).toBeGreaterThan(badGof.ks)
  })

  it('CANCELLATION: an isCancelled that returns true throws (checked at the first chunk)', async () => {
    const data = simulate(normal, { mu: 5, sigma: 2 }, 40, 777)
    const fitted = normal.fit(data)
    await expect(
      bootstrapFit(normal, data, fitted, {
        B,
        alpha: ALPHA,
        seed: SEED,
        isCancelled: () => true,
      }),
    ).rejects.toThrow(/cancel/i)
  })

  it('PROGRESS: onChunk is invoked with fractions in [0, 1)', async () => {
    const data = simulate(normal, { mu: 5, sigma: 2 }, 40, 777)
    const fitted = normal.fit(data)
    const fractions: number[] = []
    await bootstrapFit(normal, data, fitted, {
      B,
      alpha: ALPHA,
      seed: SEED,
      onChunk: (f) => fractions.push(f),
    })
    expect(fractions.length).toBeGreaterThan(0)
    expect(fractions[0]).toBe(0) // first chunk boundary is b=0
    for (const f of fractions) {
      expect(f).toBeGreaterThanOrEqual(0)
      expect(f).toBeLessThan(1)
    }
  })

  it('DEGENERATE REFIT: a fit that always throws is skipped (try/catch) and does not crash', async () => {
    // n above the jackknife cap forces the percentile-only path so the always-throwing
    // fit is exercised ONLY inside the per-replicate try/catch (the jackknife — which is
    // OUTSIDE the catch — is skipped, so its throwing fit never escapes). Every refit
    // throws immediately, so the 2001×B draws never materialize a sample of work.
    const data = simulate(normal, { mu: 5, sigma: 2 }, BCA_JACKKNIFE_MAX_N + 1, 777)
    const fitted = normal.fit(data)
    let refits = 0
    const alwaysThrows: Distribution = {
      ...normal,
      fit(): FittedParams {
        refits++
        throw new Error('degenerate synthetic sample')
      },
    }
    const result = await bootstrapFit(alwaysThrows, data, fitted, {
      B: 20,
      alpha: ALPHA,
      seed: SEED,
    })
    expect(refits).toBe(20) // every replicate attempted the (throwing) refit
    // All replicates skipped → empty reps → finite p-values, percentile-only CIs.
    expect(result.gofPValues).not.toBeNull()
    expect(result.gofPValues?.ks).toBeCloseTo(1 / (20 + 1), INVARIANT_PLACES)
    for (const ci of Object.values(result.paramCIs)) {
      expect(ci.method).toBe('percentile')
    }
  })

  it('EXPENSIVE FAMILY (student-t): jackknife skipped ⇒ percentile CIs, finite endpoints', async () => {
    // Student-t's MLE is the iterative Nelder–Mead fit (expensiveFit=true), so the parametric
    // bootstrap must SKIP the O(n) BCa jackknife (n leave-one-out NM refits would be an O(n·NM)
    // blowup) and report percentile CIs instead. n stays well under BCA_JACKKNIFE_MAX_N so this
    // verifies the expensiveFit gate, not the n-cap gate.
    const data = simulate(studentT, { loc: 0, scale: 1, df: 5 }, 60, 4242)
    const fitted = studentT.fit(data)
    const { paramCIs } = await bootstrapFit(studentT, data, fitted, { B, alpha: ALPHA, seed: SEED })
    for (const ci of Object.values(paramCIs)) {
      expect(ci.method).toBe('percentile') // jackknife skipped → percentile fallback
      expect(Number.isFinite(ci.percentile[0])).toBe(true)
      expect(Number.isFinite(ci.percentile[1])).toBe(true)
      // bca mirrors percentile when the jackknife is skipped.
      expect(Number.isFinite(ci.bca[0])).toBe(true)
      expect(Number.isFinite(ci.bca[1])).toBe(true)
    }
  })

  it('CHEAP FAMILY (normal): jackknife still runs ⇒ method=bca (unchanged)', async () => {
    // A closed-form (non-expensiveFit) family keeps the full BCa path: the jackknife runs (n is
    // under the cap) and the per-param method is 'bca'. This pins that CHANGE 1 left cheap
    // families' behavior untouched.
    const data = simulate(normal, { mu: 5, sigma: 2 }, 60, 777)
    const fitted = normal.fit(data)
    const { paramCIs } = await bootstrapFit(normal, data, fitted, { B, alpha: ALPHA, seed: SEED })
    for (const ci of Object.values(paramCIs)) {
      expect(ci.method).toBe('bca')
    }
  })

  it('discrete fit: param CIs are computed but GoF resampling is skipped (gofPValues null)', async () => {
    // The Batch C kind-branch: a discrete law's EDF statistics are invalid under ties, so the
    // bootstrap must NOT resample them (that would yield a spurious ~1/(B+1) p-value); it still
    // produces parameter confidence intervals from the refitted integer replicates.
    const counts = [0, 1, 2, 1, 3, 0, 2, 1, 4, 2, 1, 0, 3, 2, 1, 2, 0, 1, 3, 1, 2, 1, 0, 2, 1]
    const fitted = poisson.fit(counts)
    const result = await bootstrapFit(poisson, counts, fitted, { B: 200, alpha: ALPHA, seed: 7 })
    expect(result.gofPValues).toBeNull() // χ² keeps its table p-value; no EDF bootstrap
    const lambdaCI = result.paramCIs.lambda
    if (!lambdaCI) throw new Error('expected a lambda CI')
    // A finite, ordered interval bracketing the point estimate (λ̂ ≈ 1.44).
    expect(lambdaCI.bca[0]).toBeLessThanOrEqual(lambdaCI.point)
    expect(lambdaCI.point).toBeLessThanOrEqual(lambdaCI.bca[1])
    expect(Number.isFinite(lambdaCI.bca[0]) && Number.isFinite(lambdaCI.bca[1])).toBe(true)
  })
})

describe('quick-mode fit (relaxed NM caps for bootstrap replicate refits)', () => {
  /** Relative tolerance for quick-vs-full params: quick mode deliberately drops sub-1e-6 precision
   *  (fewer iterations, no restarts, looser fTol), so endpoints agree only to a few percent. */
  const QUICK_REL_TOL = 0.05

  function simulate(dist: Distribution, params: FittedParams, n: number, seed: number): number[] {
    const draw = makeSampler(dist.name, params, seed)
    return Array.from({ length: n }, () => draw())
  }

  it('student-t fit(data,{quick:true}) returns finite params close to fit(data)', () => {
    const data = simulate(studentT, { loc: 1, scale: 2, df: 6 }, 80, 9090)
    const full = studentT.fit(data)
    const quick = studentT.fit(data, { quick: true })
    for (const key of Object.keys(full)) {
      const f = full[key] ?? Number.NaN
      const q = quick[key] ?? Number.NaN
      expect(Number.isFinite(q)).toBe(true)
      // Relative agreement: |q − f| ≤ tol·max(|f|, 1) — loose because quick drops sub-1e-6 precision.
      expect(Math.abs(q - f)).toBeLessThanOrEqual(QUICK_REL_TOL * Math.max(Math.abs(f), 1))
    }
  })
})
