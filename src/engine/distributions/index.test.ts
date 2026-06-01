import { describe, expect, it } from 'vitest'
import { DistributionName } from '../types'
import { DISTRIBUTIONS } from './index'

describe('DISTRIBUTIONS registry', () => {
  it('contains the M1 + M2.3 Batch A distributions with unique names', () => {
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
    ])
    expect(new Set(names).size).toBe(names.length)
  })
  it('every entry exposes fit/logpdf/cdf/quantile, a continuous kind, and a k of 1 or 2', () => {
    for (const d of DISTRIBUTIONS) {
      expect(typeof d.fit).toBe('function')
      expect(typeof d.logpdf).toBe('function')
      expect(typeof d.cdf).toBe('function')
      expect(typeof d.quantile).toBe('function')
      expect(d.kind).toBe('continuous')
      expect([1, 2]).toContain(d.k)
    }
  })
})
