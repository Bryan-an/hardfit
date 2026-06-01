import { describe, expect, it } from 'vitest'
import { exponential } from './distributions/exponential'
import { normal } from './distributions/normal'
import { goodnessOfFit } from './goodnessOfFit'
import type { Distribution, FittedParams } from './types'
import { DistributionName } from './types'

// n = 18 deterministic positive sample (matches fitAll.test). At n=18, k = max(2, ⌊18/5⌋) = 3,
// so χ² df = k − 1 − nParams = 2 − nParams: k=2 fits (normal) → df 0 → null pValue;
// k=1 fits (exponential) → df 1 → finite pValue. This exercises BOTH sides of the NaN→null map.
const sample = [
  2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5, 2.7, 3.1, 4.4,
]

describe('goodnessOfFit — continuous routing', () => {
  it('returns the full battery (ks, ad, cvm, chiSquared) for a continuous distribution', () => {
    const params = normal.fit(sample)
    const gof = goodnessOfFit(normal, sample, params)
    // KS: raw diagnostic D, non-negative.
    expect(gof.ks).toBeGreaterThanOrEqual(0)
    expect(Number.isFinite(gof.ks)).toBe(true)
    // AD: raw A² finite; normal carries a closed-form p-value.
    expect(Number.isFinite(gof.ad.statistic)).toBe(true)
    expect(gof.ad.method).toBe('closed-form')
    expect(gof.ad.pValue).not.toBeNull()
    // CvM: diagnostic-only (pValue null) until M2.2 bootstrap.
    expect(Number.isFinite(gof.cvm.statistic)).toBe(true)
    expect(gof.cvm.pValue).toBeNull()
    expect(gof.cvm.method).toBe('diagnostic')
    // χ²: table method, df an integer, bins ≥ 2.
    expect(gof.chiSquared.method).toBe('table')
    expect(Number.isInteger(gof.chiSquared.df)).toBe(true)
    expect(gof.chiSquared.bins).toBeGreaterThanOrEqual(2)
    expect(Number.isFinite(gof.chiSquared.statistic)).toBe(true)
  })

  it('maps a NaN χ² p-value (df < 1) to null — k=2 fit at n=18 gives df 0', () => {
    const params = normal.fit(sample) // k = 2 → df = 2 − 2 = 0
    const gof = goodnessOfFit(normal, sample, params)
    expect(gof.chiSquared.df).toBe(0)
    expect(gof.chiSquared.pValue).toBeNull()
  })

  it('carries a finite χ² p-value when df ≥ 1 — k=1 exponential fit at n=18 gives df 1', () => {
    const params = exponential.fit(sample) // k = 1 → df = 2 − 1 = 1
    const gof = goodnessOfFit(exponential, sample, params)
    expect(gof.chiSquared.df).toBe(1)
    expect(typeof gof.chiSquared.pValue).toBe('number')
    expect(gof.chiSquared.pValue).not.toBeNull()
  })

  it('builds the cdf/quantile closures from the PASSED params (not a refit of data)', () => {
    const fitted = normal.fit(sample)
    const { mu, sigma } = fitted
    if (mu === undefined || sigma === undefined) throw new Error('normal fit missing mu/sigma')
    // Deliberately wrong params: the statistics must change, proving the closures bind to the
    // argument, not to a refit of `data`. (If goodnessOfFit ignored params and refit, both calls
    // would produce identical statistics.)
    const wrong: FittedParams = { mu: mu + 50, sigma }
    const atFit = goodnessOfFit(normal, sample, fitted)
    const atWrong = goodnessOfFit(normal, sample, wrong)
    expect(atWrong.ad.statistic).not.toBe(atFit.ad.statistic)
    expect(atWrong.cvm.statistic).not.toBe(atFit.cvm.statistic)
    expect(atWrong.ks).not.toBe(atFit.ks)
    // And repeating with the same params is deterministic.
    expect(goodnessOfFit(normal, sample, fitted).ad.statistic).toBe(atFit.ad.statistic)
  })
})

describe('goodnessOfFit — discrete routing', () => {
  // Minimal stub Distribution with kind:'discrete' to cover the routing fork. Real discrete
  // binning lands in M2.3 Batch C; for now the branch throws a clear not-implemented error.
  const discreteStub: Distribution = {
    name: 'poisson-stub',
    label: 'Poisson (stub)',
    k: 1,
    kind: 'discrete',
    fit(): FittedParams {
      return { lambda: 3 }
    },
    logpdf(): number {
      return Number.NaN
    },
    cdf(): number {
      return Number.NaN
    },
    quantile(): number {
      return Number.NaN
    },
  }

  it('routes discrete distributions to a clear not-implemented error', () => {
    const params = discreteStub.fit([])
    expect(() => goodnessOfFit(discreteStub, [1, 2, 3, 4, 5], params)).toThrow(
      /discrete GoF not yet implemented/,
    )
  })

  it('does not affect the continuous path (kind is the dispatch key)', () => {
    // Sanity: a real continuous distribution never hits the discrete branch.
    const params = normal.fit(sample)
    expect(() => goodnessOfFit(normal, sample, params)).not.toThrow()
    expect(normal.name).toBe(DistributionName.Normal)
  })
})
