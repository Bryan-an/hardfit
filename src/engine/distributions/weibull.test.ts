import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { meanLog } from '../math'
import { weibull } from './weibull'

/** Mirrors the (unexported) WeibullParams slot type used inside weibull.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `weibull.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type WeibullParams = { shape: number; scale: number }

const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5]

describe('weibull', () => {
  it('shape solves the score: 1/k + meanLn - (sum x^k ln x)/(sum x^k) = 0', () => {
    const p = weibull.fit(sample) as WeibullParams
    const k = p.shape
    let S0 = 0
    let S1 = 0
    for (const x of sample) {
      const xk = x ** k
      S0 += xk
      S1 += xk * Math.log(x)
    }
    expectClose(1 / k + meanLog(sample) - S1 / S0, 0, 1e-7, 1e-7)
  })
  it('scale = (mean(x^k))^(1/k)', () => {
    const p = weibull.fit(sample) as WeibullParams
    let S0 = 0
    for (const x of sample) S0 += x ** p.shape
    expectClose(p.scale, (S0 / sample.length) ** (1 / p.shape), 1e-7)
  })
  it('cdf(scale) = 1 - 1/e', () =>
    expectClose(weibull.cdf(3, { shape: 2, scale: 3 }), 1 - Math.exp(-1)))
  it('logpdf passes SCALE to @stdlib: matches the elementary closed form', () => {
    // CONVENTION GUARD: weibull(shape k=2, scale lam=3): logf = ln(k) - ln(lam) + (k-1)ln(x/lam) - (x/lam)^k
    const expected = Math.log(2) - Math.log(3) + (2 - 1) * Math.log(2 / 3) - (2 / 3) ** 2
    expectClose(weibull.logpdf(2, { shape: 2, scale: 3 }), expected, 1e-9)
  })
  it('rejects non-positive data', () => expect(() => weibull.fit([1, 0])).toThrow())
  it('k = 2', () => expect(weibull.k).toBe(2))
})
