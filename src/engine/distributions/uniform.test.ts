import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { uniform } from './uniform'

/** Mirrors the (unexported) UniformParams slot type used inside uniform.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `uniform.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type UniformParams = { a: number; b: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('uniform', () => {
  it('MLE: a = min(data), b = max(data)', () => {
    const data = [1, 2, 3, 4]
    const p = uniform.fit(data) as UniformParams
    expect(p.a).toBe(1)
    expect(p.b).toBe(4)
  })
  it('logpdf passes (a, b) endpoints to @stdlib: matches the elementary density 1/(b-a)', () => {
    // CONVENTION GUARD: uniform(a=1, b=3) has constant density 1/(b-a)=1/2, so logf = -ln(b-a) = -ln 2.
    // Written out by hand (NOT via @stdlib) to catch a wrong arg slot (e.g. b read as a width).
    expectClose(uniform.logpdf(2, { a: 1, b: 3 }), -Math.log(3 - 1), 1e-9)
  })
  it('cdf(midpoint) = 0.5', () => expectClose(uniform.cdf(2, { a: 1, b: 3 }), 0.5))
  it('quantile(0.25; a=1,b=3) = 1.5', () =>
    expectClose(uniform.quantile(0.25, { a: 1, b: 3 }), 1.5))
  it('quantile inverts cdf (round-trip)', () => {
    const p = { a: 1, b: 3 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(uniform.cdf(uniform.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('rejects degenerate data (zero width)', () => expect(() => uniform.fit([2, 2, 2])).toThrow())
  it('k = 2', () => expect(uniform.k).toBe(2))
  it("kind = 'continuous'", () => expect(uniform.kind).toBe('continuous'))
})
