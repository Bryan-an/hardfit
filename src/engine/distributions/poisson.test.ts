import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { mean } from '../math'
import { poisson } from './poisson'

/** Mirrors the (unexported) PoissonParams slot type used inside poisson.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `poisson.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type PoissonParams = { lambda: number }

/** Integer counts; sum = 16, n = 10 -> lambda = 1.6. */
const sample = [0, 1, 2, 1, 3, 0, 2, 1, 4, 2]

/** Probabilities the discrete quantile<->cdf inequality is checked at (away from the 0/1 tails). */
const QUANTILE_PROBS = [0.25, 0.5, 0.75, 0.9]

describe('poisson', () => {
  it('MLE: lambda = mean(data)', () => {
    const p = poisson.fit(sample) as PoissonParams
    expectClose(p.lambda, mean(sample))
  })
  it('logpdf passes lambda to @stdlib: matches the elementary closed-form log-PMF', () => {
    // CONVENTION GUARD: poisson PMF p(x) = lambda^x * e^{-lambda} / x!.
    // log p(x) = x*ln(lambda) - lambda - ln(x!). For lambda=2 (ASYMMETRIC vs x), x=3:
    // ln(3!) = gammaln(4) = ln 6. Written by hand (NOT via @stdlib) to catch a wrong closed form.
    const expected = 3 * Math.log(2) - 2 - gammaln(4)
    expectClose(poisson.logpdf(3, { lambda: 2 }), expected, 1e-9)
  })
  it('cdf(0, lambda) = e^{-lambda} (only the x=0 term)', () => {
    expectClose(poisson.cdf(0, { lambda: 2 }), Math.exp(-2))
  })
  it('quantile is a discrete integer that satisfies cdf(quantile(p)) >= p', () => {
    const p = { lambda: 2 }
    for (const prob of QUANTILE_PROBS) {
      const q = poisson.quantile(prob, p)
      expect(Number.isInteger(q)).toBe(true)
      expect(poisson.cdf(q, p)).toBeGreaterThanOrEqual(prob)
    }
  })
  it('fit recovers lambda = mean on integer counts', () => {
    const p = poisson.fit(sample) as PoissonParams
    expectClose(p.lambda, 1.6)
  })
  it('rejects non-integer data', () => expect(() => poisson.fit([1, 2.5, 3])).toThrow())
  it('rejects negative integer counts', () => expect(() => poisson.fit([0, 1, -1])).toThrow())
  it('rejects all-zero data (degenerate: lambda = 0)', () =>
    expect(() => poisson.fit([0, 0, 0])).toThrow())
  it('support = { min: 0, max: +Infinity }', () => {
    const s = poisson.support?.({ lambda: 2 })
    expect(s).toEqual({ min: 0, max: Number.POSITIVE_INFINITY })
  })
  it('k = 1', () => expect(poisson.k).toBe(1))
  it("kind = 'discrete'", () => expect(poisson.kind).toBe('discrete'))
})
