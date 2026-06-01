import { describe, expect, it } from 'vitest'
import { DistributionName } from '../types'
import { DISTRIBUTIONS } from './index'

describe('DISTRIBUTIONS registry', () => {
  it('contains the M1 + M2.3 Batch A + Batch B + Batch C + Batch D distributions with unique names', () => {
    const names = DISTRIBUTIONS.map((d) => d.name)
    expect(names).toEqual([
      DistributionName.Normal,
      DistributionName.Lognormal,
      DistributionName.Exponential,
      DistributionName.Gamma,
      DistributionName.Weibull,
      DistributionName.Uniform,
      DistributionName.Rayleigh,
      DistributionName.Pareto,
      DistributionName.Laplace,
      DistributionName.Logistic,
      DistributionName.Gumbel,
      DistributionName.Cauchy,
      DistributionName.Frechet,
      DistributionName.Levy,
      DistributionName.ChiSquared,
      DistributionName.Chi,
      DistributionName.InvGamma,
      DistributionName.BetaPrime,
      DistributionName.Cosine,
      DistributionName.Beta,
      DistributionName.Poisson,
      DistributionName.Geometric,
      DistributionName.NegativeBinomial,
      DistributionName.DiscreteUniform,
      DistributionName.StudentT,
      DistributionName.FisherF,
    ])
    expect(new Set(names).size).toBe(names.length)
  })
  it('every entry exposes fit/logpdf/cdf/quantile, a valid kind, and a k of 1, 2, or 3', () => {
    for (const d of DISTRIBUTIONS) {
      expect(typeof d.fit).toBe('function')
      expect(typeof d.logpdf).toBe('function')
      expect(typeof d.cdf).toBe('function')
      expect(typeof d.quantile).toBe('function')
      expect(['continuous', 'discrete']).toContain(d.kind)
      // Discrete distributions must expose integer support bounds for the PMF-binned χ².
      if (d.kind === 'discrete') expect(typeof d.support).toBe('function')
      expect([1, 2, 3]).toContain(d.k)
    }
  })
})
