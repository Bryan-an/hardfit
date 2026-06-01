import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { pareto } from './pareto'

/** Mirrors the (unexported) ParetoParams slot type used inside pareto.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `pareto.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type ParetoParams = { shape: number; scale: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('pareto', () => {
  it('logpdf passes SCALE (xm) to @stdlib: matches the elementary closed form', () => {
    // CONVENTION GUARD: Pareto-I density f(x) = alpha*xm^alpha / x^(alpha+1) for x >= xm, so
    // logf = ln(alpha) + alpha*ln(xm) - (alpha+1)*ln(x). For alpha=3, xm=2, x=4 (written BY HAND,
    // not via @stdlib): catches a 1/xm rate-slot swap, which would yield a wrong-but-finite value.
    const expected = Math.log(3) + 3 * Math.log(2) - 4 * Math.log(4)
    expectClose(pareto.logpdf(4, { shape: 3, scale: 2 }), expected, 1e-9)
  })
  it('cdf passes SCALE (xm): cdf(4; alpha=3, xm=2) = 1 - (xm/x)^alpha = 0.875', () => {
    // F(x) = 1 - (xm/x)^alpha; 1 - (2/4)^3 = 1 - 1/8 = 0.875. Doubles as a scale-slot guard.
    expectClose(pareto.cdf(4, { shape: 3, scale: 2 }), 0.875, 1e-9)
  })
  it('quantile inverts cdf (round-trip)', () => {
    const p = { shape: 3, scale: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(pareto.cdf(pareto.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('MLE: xm = min(x), alpha = n / sum ln(x/xm)', () => {
    const data = [2, 4, 8]
    const p = pareto.fit(data) as ParetoParams
    // xm = 2; s = ln(2/2) + ln(4/2) + ln(8/2) = 0 + ln2 + ln4; alpha = 3/s.
    const s = Math.log(1) + Math.log(2) + Math.log(4)
    expectClose(p.scale, 2)
    expectClose(p.shape, 3 / s)
  })
  it('rejects degenerate data (all values equal → s = 0)', () =>
    expect(() => pareto.fit([5, 5, 5])).toThrow())
  it('rejects non-positive data', () => expect(() => pareto.fit([2, 0])).toThrow())
  it('k = 2', () => expect(pareto.k).toBe(2))
  it("kind = 'continuous'", () => expect(pareto.kind).toBe('continuous'))
})
