import { describe, expect, it } from 'vitest'
import { fitAll } from './fitAll'
import { DistributionName } from './types'

// Deterministic positive sample (looks roughly gamma/lognormal-ish).
const sample = [
  2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5, 2.7, 3.1, 4.4,
]

describe('fitAll', () => {
  it('fits 20 of 25 distributions on continuous positive data, ranks by AICc, weights sum to 1', () => {
    const res = fitAll(sample)
    expect(res.n).toBe(sample.length)
    // This sample is continuous (non-integer) and > 1. Beta needs 0 < x < 1, and the 4 discrete
    // families reject the non-integer values — so 5 fail and the other 20 rank (incl. Student-t).
    expect(res.ranked.length).toBe(20)
    expect(res.failures.map((f) => f.name).sort()).toEqual(
      [
        DistributionName.Beta,
        DistributionName.Poisson,
        DistributionName.Geometric,
        DistributionName.NegativeBinomial,
        DistributionName.DiscreteUniform,
      ].sort(),
    )
    const best = res.ranked[0]
    if (!best) throw new Error('expected at least one ranked fit')
    expect(best.rank).toBe(1)
    // ascending AICc
    const aiccs = res.ranked.map((r) => r.aicc)
    const sortedAiccs = [...aiccs].sort((a, b) => a - b)
    expect(aiccs).toEqual(sortedAiccs)
    const wsum = res.ranked.reduce((s, r) => s + r.weight, 0)
    expect(Math.abs(wsum - 1)).toBeLessThan(1e-9)
    // each ranked fit carries finite diagnostics + a populated GoF battery
    for (const r of res.ranked) {
      expect(Number.isFinite(r.logLik)).toBe(true)
      expect(Number.isFinite(r.aic)).toBe(true)
      expect(r.ks).toBeGreaterThanOrEqual(0)
      // AD raw A² is finite even when its p-value is diagnostic/null (e.g. gamma).
      expect(Number.isFinite(r.ad.statistic)).toBe(true)
      // CvM is diagnostic-only until the M2.2 bootstrap.
      expect(Number.isFinite(r.cvm.statistic)).toBe(true)
      expect(r.cvm.pValue).toBeNull()
      // χ² carries an integer df + bin count; its p-value is null when df < 1 (k=2 fits at n=18).
      expect(Number.isInteger(r.chiSquared.df)).toBe(true)
      expect(r.chiSquared.bins).toBeGreaterThanOrEqual(2)
      expect(Number.isFinite(r.chiSquared.statistic)).toBe(true)
    }
  })
  it('reports failures (not crashes) when a distribution rejects the data', () => {
    const withNeg = [-1, 2, 3, 4, 5, 6] // integers, one negative
    // Positive-support families reject the negative value: lognormal/gamma/weibull/pareto/frechet/
    // levy/chisquare/chi/invgamma/betaprime need x > 0, exponential/rayleigh need x >= 0, beta needs
    // 0 < x < 1, and the discrete poisson/geometric/negative-binomial need x >= 0. The real-support
    // families (normal/uniform/laplace/logistic/gumbel/cauchy/cosine) survive — AS DOES discrete
    // uniform, whose support is any contiguous integer range (here {-1,…,6}), so it fits.
    const res = fitAll(withNeg)
    const failed = res.failures.map((f) => f.name).sort()
    expect(failed).toEqual(
      [
        DistributionName.Beta,
        DistributionName.BetaPrime,
        DistributionName.Chi,
        DistributionName.ChiSquared,
        DistributionName.Exponential,
        DistributionName.Frechet,
        DistributionName.Gamma,
        DistributionName.Geometric,
        DistributionName.InvGamma,
        DistributionName.Levy,
        DistributionName.Lognormal,
        DistributionName.NegativeBinomial,
        DistributionName.Pareto,
        DistributionName.Poisson,
        DistributionName.Rayleigh,
        DistributionName.Weibull,
      ].sort(),
    )
    expect(res.ranked.some((r) => r.name === DistributionName.Normal)).toBe(true)
    expect(res.ranked.some((r) => r.name === DistributionName.DiscreteUniform)).toBe(true)
  })
  it('ranks discrete distributions on integer count data with χ²-only GoF (NaN EDF)', () => {
    // Overdispersed integer counts so Poisson/Geometric/Negative-Binomial/Discrete-Uniform all fit.
    const counts = [0, 1, 2, 5, 3, 0, 4, 2, 1, 6, 0, 3, 8, 1, 2, 4, 0, 5, 1, 3, 2, 7, 0, 1, 4]
    const res = fitAll(counts)
    const poisson = res.ranked.find((r) => r.name === DistributionName.Poisson)
    if (!poisson) throw new Error('expected Poisson to rank on integer data')
    // Discrete fits are χ²-only: the EDF columns carry NaN sentinels (render as `— diag.`).
    expect(Number.isNaN(poisson.ks)).toBe(true)
    expect(Number.isNaN(poisson.ad.statistic)).toBe(true)
    expect(Number.isNaN(poisson.cvm.statistic)).toBe(true)
    // The χ² IS the discrete test: a finite statistic + integer df/bins.
    expect(Number.isFinite(poisson.chiSquared.statistic)).toBe(true)
    expect(Number.isInteger(poisson.chiSquared.df)).toBe(true)
    // All four discrete families fit (and rank among finite-AICc results).
    for (const name of [
      DistributionName.Poisson,
      DistributionName.Geometric,
      DistributionName.NegativeBinomial,
      DistributionName.DiscreteUniform,
    ]) {
      expect(res.ranked.some((r) => r.name === name)).toBe(true)
    }
  })
  it('throws on too-small samples', () => expect(() => fitAll([1, 2])).toThrow())
})
