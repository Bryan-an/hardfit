import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { exponential } from './exponential'

/** Mirrors the (unexported) ExponentialParams slot type used inside exponential.ts; lets the
 *  test read fitted params by name without `?? NaN` noise. `exponential.fit` returns the
 *  engine-wide `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected. */
type ExponentialParams = { rate: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('exponential', () => {
  it('MLE rate = 1/mean', () => {
    const p = exponential.fit([1, 2, 3, 4]) as ExponentialParams // mean 2.5 -> rate 0.4
    expectClose(p.rate, 0.4)
  })
  it('rejects negative data', () => expect(() => exponential.fit([1, -1])).toThrow())
  it('cdf(x,rate) = 1 - e^{-rate x}', () =>
    expectClose(exponential.cdf(1, { rate: 2 }), 1 - Math.exp(-2)))
  it('logpdf passes RATE: = ln(rate) - rate*x', () =>
    expectClose(exponential.logpdf(2, { rate: 0.5 }), Math.log(0.5) - 0.5 * 2))
  it('quantile passes RATE: median = ln(2)/rate', () =>
    expectClose(exponential.quantile(0.5, { rate: 2 }), Math.log(2) / 2))
  it('quantile inverts cdf (round-trip)', () => {
    const p = { rate: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(exponential.cdf(exponential.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('k = 1', () => expect(exponential.k).toBe(1))
  it("kind = 'continuous'", () => expect(exponential.kind).toBe('continuous'))
})
