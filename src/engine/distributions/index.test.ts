import { describe, expect, it } from 'vitest'
import { DistributionName } from '../types'
import { DISTRIBUTIONS } from './index'

describe('DISTRIBUTIONS registry', () => {
  it('contains the 5 M1 distributions with unique names', () => {
    const names = DISTRIBUTIONS.map((d) => d.name)
    expect(names).toEqual([
      DistributionName.Normal,
      DistributionName.Lognormal,
      DistributionName.Exponential,
      DistributionName.Gamma,
      DistributionName.Weibull,
    ])
    expect(new Set(names).size).toBe(5)
  })
  it('every entry exposes fit/logpdf/cdf and a k of 1 or 2', () => {
    for (const d of DISTRIBUTIONS) {
      expect(typeof d.fit).toBe('function')
      expect(typeof d.logpdf).toBe('function')
      expect(typeof d.cdf).toBe('function')
      expect([1, 2]).toContain(d.k)
    }
  })
})
