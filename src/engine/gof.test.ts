import { describe, expect, it } from 'vitest'
import { expectClose } from '../test/relClose'
import { adStatistic, chiSquaredGof, cramerVonMises, ksStatistic } from './gof'

describe('ksStatistic', () => {
  it('uniform sample vs uniform CDF — exact small case', () => {
    // sorted [0.25,0.5,0.75], F(x)=x. n=3.
    // D+ = max(1/3-0.25, 2/3-0.5, 3/3-0.75) = max(0.0833,0.1667,0.25)=0.25
    // D- = max(0.25-0, 0.5-1/3, 0.75-2/3) = max(0.25,0.1667,0.0833)=0.25
    expectClose(
      ksStatistic([0.5, 0.25, 0.75], (x) => x),
      0.25,
      1e-12,
    )
  })
  it('does not mutate the input array', () => {
    const data = [3, 1, 2]
    ksStatistic(data, (x) => x / 3)
    const first = data[0]
    if (first === undefined) throw new Error('expected a non-empty input array')
    expectClose(first, 3) // original order preserved
  })
  it('clamps tiny negative FP noise to 0 (perfect fit)', () => {
    const d = ksStatistic([1, 2, 3, 4], (x) => x / 4) // step exactly matches at integers
    expectClose(d, 0.25, 1e-12) // D- at i=1: F(1)-0 = 0.25
  })
})

describe('adStatistic', () => {
  it('uniform sample vs uniform CDF — golden raw A²', () => {
    // sorted u = [0.25,0.5,0.75], F(x)=x, n=3.
    // A² = -3 - (1/3)·[1·(ln.25+ln.25) + 3·(ln.5+ln.5) + 5·(ln.75+ln.75)]
    //    = 2·ln.25 + 6·ln.5 + 10·ln.75 reindexed → 0.269430843372420 (hand-expanded)
    expectClose(
      adStatistic([0.5, 0.25, 0.75], (x) => x),
      0.26943084337242,
      1e-12,
    )
  })
  it('four-point sample vs uniform CDF — golden raw A²', () => {
    // sorted u = [0.1,0.4,0.6,0.9], F(x)=x, n=4 → 0.171554508525624 (hand-expanded)
    expectClose(
      adStatistic([0.9, 0.1, 0.6, 0.4], (x) => x),
      0.171554508525624,
      1e-12,
    )
  })
  it('clamps F into the open interval to avoid ln(0) at the support edges', () => {
    // [1,2,3,4] with F(x)=x/4 → u_(4)=1.0; the 1−EPS upper clamp keeps ln(1−u) finite.
    const a2 = adStatistic([1, 2, 3, 4], (x) => x / 4)
    expect(Number.isFinite(a2)).toBe(true)
    expectClose(a2, 6.543395749070212, 1e-9)
  })
  it('stays finite when F hits both 0 and 1 (both clamp bounds exercised)', () => {
    const a2 = adStatistic([0, 0.5, 1], (x) => x)
    expect(Number.isFinite(a2)).toBe(true)
    expect(a2).toBeGreaterThanOrEqual(0) // A² is always non-negative
  })
  it('A² ≥ 0 across all golden cases', () => {
    expect(adStatistic([0.5, 0.25, 0.75], (x) => x)).toBeGreaterThanOrEqual(0)
    expect(adStatistic([0.9, 0.1, 0.6, 0.4], (x) => x)).toBeGreaterThanOrEqual(0)
  })
  it('returns NaN for an empty sample', () => {
    expect(Number.isNaN(adStatistic([], (x) => x))).toBe(true)
  })
  it('does not mutate the input array', () => {
    const data = [3, 1, 2]
    adStatistic(data, (x) => x / 3)
    expect(data).toEqual([3, 1, 2]) // original order preserved
  })
})

describe('cramerVonMises', () => {
  it('uniform sample vs uniform CDF — golden W²', () => {
    // sorted u = [0.25,0.5,0.75], F(x)=x, n=3. Plotting positions (2i+1)/(2n) = [1/6,1/2,5/6].
    // W² = 1/(12·3) + (0.25−1/6)² + 0 + (0.75−5/6)² = 1/36 + 2·(1/12)² = 0.0416̄ (scipy n·ω²).
    expectClose(
      cramerVonMises([0.5, 0.25, 0.75], (x) => x),
      0.041666666666666671,
      1e-12,
    )
  })
  it('four-point sample vs uniform CDF — golden W² matches 1/(12n)+Σ form', () => {
    // sorted u = [0.1,0.4,0.6,0.9], n=4 → 1/48 + Σ(u_i−(2i+1)/8)² = 0.0233… (scipy n·ω²)
    expectClose(
      cramerVonMises([0.9, 0.1, 0.6, 0.4], (x) => x),
      0.023333333333333,
      1e-12,
    )
  })
  it('clamps F into [0,1] at the support edges', () => {
    const w2 = cramerVonMises([1, 2, 3, 4], (x) => x / 4) // u_(4)=1.0 stays in range
    expect(Number.isFinite(w2)).toBe(true)
    expect(w2).toBeGreaterThanOrEqual(0)
  })
  it('returns NaN for an empty sample', () => {
    expect(Number.isNaN(cramerVonMises([], (x) => x))).toBe(true)
  })
  it('does not mutate the input array', () => {
    const data = [3, 1, 2]
    cramerVonMises(data, (x) => x / 3)
    expect(data).toEqual([3, 1, 2]) // original order preserved
  })
})

describe('chiSquaredGof', () => {
  // Identity quantile Q(p)=p models a Uniform(0,1) fit; edges are evenly spaced at j/k.
  const identityQuantile = (p: number): number => p

  it('binning math: k = max(2, floor(n/5)), expected = n/k, df = k − 1 − nParams', () => {
    // n=10 → k=max(2,floor(10/5))=2; expected=5; one edge at Q(1/2)=0.5; df=2−1−0=1.
    // 7 points below 0.5, 3 above (all off the edge): statistic = (7−5)²/5 + (3−5)²/5 = 1.6.
    const data = [0.05, 0.15, 0.25, 0.35, 0.4, 0.45, 0.48, 0.55, 0.65, 0.95]
    const r = chiSquaredGof(data, identityQuantile, 0)
    expect(r.bins).toBe(2)
    expect(r.df).toBe(1) // k − 1 − nParams = 2 − 1 − 0
    expectClose(r.statistic, 1.6, 1e-12)
    expectClose(r.pValue, 0.20590321073206, 1e-9) // 1 − chi2cdf(1.6, 1)
  })

  it('equiprobable edges are evenly spaced from the quantile', () => {
    // n=15 → k=floor(15/5)=3; edges at Q(1/3),Q(2/3) = 1/3, 2/3 (evenly spaced); expected=5.
    // 5 points in each third (all off the edges) → perfect equiprobable spread → statistic 0.
    const data = [
      0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.45, 0.5, 0.55, 0.6, 0.72, 0.78, 0.85, 0.9, 0.95,
    ]
    const r = chiSquaredGof(data, identityQuantile, 0)
    expect(r.bins).toBe(3)
    expect(r.df).toBe(2) // 3 − 1 − 0
    expectClose(r.statistic, 0, 1e-12) // O_j = E_j = 5 in every bin
    expectClose(r.pValue, 1, 1e-12) // chi2cdf(0, 2) = 0
  })

  it('df accounts for fitted parameters (nParams subtracted)', () => {
    // Same n=15 (k=3) but a 2-parameter fit → df = 3 − 1 − 2 = 0 → pValue NaN guard.
    const data = [
      0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 0.45, 0.5, 0.55, 0.6, 0.72, 0.78, 0.85, 0.9, 0.95,
    ]
    const r = chiSquaredGof(data, identityQuantile, 2)
    expect(r.df).toBe(0)
    expect(Number.isNaN(r.pValue)).toBe(true) // df < 1 guard
  })

  it('df < 1 returns NaN p-value for a small sample', () => {
    // n=4 → k=max(2,floor(4/5))=max(2,0)=2; df=2−1−2=−1 with a 2-param fit → NaN.
    const r = chiSquaredGof([0.2, 0.4, 0.6, 0.8], identityQuantile, 2)
    expect(r.bins).toBe(2)
    expect(r.df).toBe(-1)
    expect(Number.isNaN(r.pValue)).toBe(true)
  })

  it('bounded support is handled via quantile edges (no out-of-range edge)', () => {
    // A bounded fit on [10, 20]: Q(p) = 10 + 10p. n=10 → k=2, edge at Q(0.5)=15.
    // All edges stay inside [10,20] by construction; data binned correctly.
    const boundedQuantile = (p: number): number => 10 + 10 * p
    const data = [10.5, 11, 12, 13, 14, 16, 17, 18, 19, 19.5] // 5 below 15, 5 above
    const r = chiSquaredGof(data, boundedQuantile, 0)
    expect(r.bins).toBe(2)
    expectClose(r.statistic, 0, 1e-12) // perfectly split → 5 vs 5
  })

  it('a point exactly on an edge falls into the lower bin (strict x > edge)', () => {
    // n=10 → k=2, edge at 0.5. 5 points strictly below, the tie sits on 0.5, 4 strictly above.
    const data = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    const r = chiSquaredGof(data, identityQuantile, 0)
    // 0.5 is NOT > 0.5 → lower bin: 6 below (incl. tie), 4 above → (6−5)²/5 + (4−5)²/5 = 0.4.
    expectClose(r.statistic, 0.4, 1e-12)
  })
})
