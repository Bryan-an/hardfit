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
  it('logpdf is on the DATA scale (includes the -ln x Jacobian)', () => {
    // CONVENTION GUARD: logpdf(x; mu, sigma) = -ln x - 0.5 ln(2π) - ln σ - 0.5((ln x - mu)/σ)^2.
    // At x=e, mu=0, sigma=1: -1 - 0.5 ln(2π) - 0 - 0.5 = -2.4189... (NOT -1.4189 without the Jacobian).
    // Without the Jacobian, lognormal's LL would be silently incomparable to the other 4 distributions.
    const expected = -1 - 0.5 * Math.log(2 * Math.PI) - 0.5
    expectClose(lognormal.logpdf(Math.E, { mu: 0, sigma: 1 }), expected, 1e-9)
  })
  it('cdf(median=e^mu)=0.5', () => expectClose(lognormal.cdf(Math.E, { mu: 1, sigma: 0.5 }), 0.5))
  it('k = 2', () => expect(lognormal.k).toBe(2))
})
