import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { lognormal } from './lognormal'

/** Mirrors the (unexported) LognormalParams slot type used inside lognormal.ts; lets the
 *  test read fitted params by name without `?? NaN` noise. `lognormal.fit` returns the
 *  engine-wide `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected. */
type LognormalParams = { mu: number; sigma: number }

describe('lognormal', () => {
  it('fits Normal on ln(x): for data=[1, e, e^2] -> mu=1, sigma^2=2/3', () => {
    // ln -> [0,1,2], mean 1, population variance (÷3) = 2/3
    const p = lognormal.fit([1, Math.E, Math.E * Math.E]) as LognormalParams
    expectClose(p.mu, 1)
    expectClose(p.sigma, Math.sqrt(2 / 3))
  })
  it('rejects non-positive data', () => expect(() => lognormal.fit([1, 0, 2])).toThrow())
  it('cdf(median=e^mu)=0.5', () => expectClose(lognormal.cdf(Math.E, { mu: 1, sigma: 0.5 }), 0.5))
  it('k = 2', () => expect(lognormal.k).toBe(2))
})
