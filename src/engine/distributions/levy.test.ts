import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { levy } from './levy'

/** Mirrors the (unexported) LevyParams slot type used inside levy.ts; lets the test read fitted
 *  params by name without `?? NaN` noise. `levy.fit` returns the engine-wide `FittedParams`
 *  (Record<string, number>) DTO, so a narrowing cast is expected here. */
type LevyParams = { c: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('levy', () => {
  it('logpdf passes SCALE to @stdlib in the c slot (mu=0): matches the elementary closed form', () => {
    // CONVENTION GUARD: levy density f(x;0,c) = sqrt(c/(2 pi)) * exp(-c/(2x)) / x^1.5, so
    // logf = 0.5*ln(c/(2 pi)) - c/(2x) - 1.5*ln(x). Written BY HAND (NOT via @stdlib) at an
    // ASYMMETRIC point (x=2.5 != 1, c=3) so all three terms — including the -1.5*ln(x) log-x
    // coefficient — are exercised; a symmetric point like x=1 zeros that term and is slot-blind.
    {
      const c = 3
      const x = 2.5
      const expected = 0.5 * Math.log(c / (2 * Math.PI)) - c / (2 * x) - 1.5 * Math.log(x)
      expectClose(levy.logpdf(x, { c }), expected, 1e-9)
    }
    // The spec's literal example, kept as a secondary check: c=2, x=1 -> -1.5*ln(1) term is 0.
    {
      const expected = 0.5 * Math.log(2 / (2 * Math.PI)) - 1 - 0
      expectClose(levy.logpdf(1, { c: 2 }), expected, 1e-9)
    }
  })
  it('cdf matches erfc(sqrt(c/(2x))): at c=2,x=1 the argument is 1, so F = erfc(1)', () => {
    // sqrt(c/(2x)) = sqrt(2/2) = 1, an independently-verifiable anchor (F = erfc(1)).
    expectClose(levy.cdf(1, { c: 2 }), 0.15729920705028513)
  })
  it('quantile inverts cdf (round-trip), passing SCALE in the c slot', () => {
    const p = { c: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(levy.cdf(levy.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('MLE: c = n / sum(1/x_i) (harmonic-mean form), unique closed form', () => {
    // c = 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75.
    const p = levy.fit([1, 2, 4]) as LevyParams
    expectClose(p.c, 3 / 1.75, 1e-9)
  })
  it('rejects x <= 0 (0 is outside the open support; 1/x diverges)', () =>
    expect(() => levy.fit([0, 1, 2])).toThrow())
  it('k = 1', () => expect(levy.k).toBe(1))
  it("kind = 'continuous'", () => expect(levy.kind).toBe('continuous'))
})
