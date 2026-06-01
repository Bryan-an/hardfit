import { describe, expect, it } from 'vitest'
import { bcaCI, jackknife, percentileCI } from './bootstrap'
import { normal } from './distributions/normal'
import { mean } from './math'

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
