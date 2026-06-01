import { describe, expect, it } from 'vitest'
import { fitAll } from './fitAll'
import { DistributionName } from './types'

// Deterministic positive sample (looks roughly gamma/lognormal-ish).
const sample = [
  2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5, 2.7, 3.1, 4.4,
]

describe('fitAll', () => {
  it('fits all 5 distributions, ranks by AICc, weights sum to 1', () => {
    const res = fitAll(sample)
    expect(res.n).toBe(sample.length)
    expect(res.ranked.length).toBe(5)
    expect(res.failures.length).toBe(0)
    const best = res.ranked[0]
    if (!best) throw new Error('expected at least one ranked fit')
    expect(best.rank).toBe(1)
    // ascending AICc
    const aiccs = res.ranked.map((r) => r.aicc)
    const sortedAiccs = [...aiccs].sort((a, b) => a - b)
    expect(aiccs).toEqual(sortedAiccs)
    const wsum = res.ranked.reduce((s, r) => s + r.weight, 0)
    expect(Math.abs(wsum - 1)).toBeLessThan(1e-9)
    // each ranked fit carries finite diagnostics
    for (const r of res.ranked) {
      expect(Number.isFinite(r.logLik)).toBe(true)
      expect(Number.isFinite(r.aic)).toBe(true)
      expect(r.ks).toBeGreaterThanOrEqual(0)
    }
  })
  it('reports failures (not crashes) when a distribution rejects the data', () => {
    const withNeg = [-1, 2, 3, 4, 5, 6]
    // Every positive-support distribution rejects the negative value: gamma/lognormal/weibull
    // need x > 0, and exponential needs x >= 0 (its fit() guards on negatives). Only normal
    // survives. (The plan's prose listed only gamma/lognormal/weibull, omitting exponential.)
    const res = fitAll(withNeg)
    const failed = res.failures.map((f) => f.name).sort()
    expect(failed).toEqual(
      [
        DistributionName.Exponential,
        DistributionName.Gamma,
        DistributionName.Lognormal,
        DistributionName.Weibull,
      ].sort(),
    )
    expect(res.ranked.some((r) => r.name === DistributionName.Normal)).toBe(true)
  })
  it('throws on too-small samples', () => expect(() => fitAll([1, 2])).toThrow())
})
