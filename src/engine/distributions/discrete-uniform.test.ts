import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { discreteUniform } from './discrete-uniform'

/** Mirrors the (unexported) DiscreteUniformParams slot type used inside discrete-uniform.ts;
 *  lets the test read fitted params by name without `?? NaN` noise. `discreteUniform.fit`
 *  returns the engine-wide `FittedParams` (Record<string, number>) DTO, so a narrowing cast
 *  is expected here. */
type DiscreteUniformParams = { a: number; b: number }

/** A 1..6 fair-die sample (asymmetric endpoints a=1, b=6) whose MLE recovers exactly a=1, b=6. */
const sample = [1, 2, 3, 4, 5, 6, 2, 3, 4, 5]

describe('discreteUniform', () => {
  it('MLE: a = min(data), b = max(data) exactly', () => {
    const p = discreteUniform.fit(sample) as DiscreteUniformParams
    expect(p.a).toBe(1)
    expect(p.b).toBe(6)
  })
  it('logpdf delegates to @stdlib logpmf: matches the elementary closed-form log-PMF', () => {
    // CONVENTION GUARD: discrete-uniform PMF is uniform over the n = b-a+1 support integers,
    // so logpmf(x, a, b) = -ln(b - a + 1) for x in {a..b}. Written by hand (NOT via @stdlib)
    // with ASYMMETRIC endpoints a=1, b=6 to catch an a/b arg-slot swap (a swap yields
    // b-a+1 <= 0 -> -Infinity). For x=3 in {1..6}: logpmf = -ln(6).
    expectClose(discreteUniform.logpdf(3, { a: 1, b: 6 }), -Math.log(6), 1e-9)
  })
  it('cdf within support: F(3; a=1, b=6) = (3-1+1)/6 = 1/2', () => {
    expectClose(discreteUniform.cdf(3, { a: 1, b: 6 }), 0.5, 1e-9)
  })
  it('quantile returns an integer in support and cdf(quantile(0.5)) >= 0.5', () => {
    const p = { a: 1, b: 6 }
    const q = discreteUniform.quantile(0.5, p)
    expect(Number.isInteger(q)).toBe(true)
    expect(q).toBeGreaterThanOrEqual(1)
    expect(q).toBeLessThanOrEqual(6)
    expect(discreteUniform.cdf(q, p)).toBeGreaterThanOrEqual(0.5)
  })
  it('fit recovers a=1, b=6 from a 1..6 sample', () => {
    const p = discreteUniform.fit(sample) as DiscreteUniformParams
    expect(p.a).toBe(1)
    expect(p.b).toBe(6)
  })
  it('rejects non-integer data', () => expect(() => discreteUniform.fit([1, 2, 3.5, 4])).toThrow())
  it('rejects degenerate data (max <= min)', () =>
    expect(() => discreteUniform.fit([4, 4, 4, 4])).toThrow())
  it('k = 2', () => expect(discreteUniform.k).toBe(2))
  it("kind = 'discrete'", () => expect(discreteUniform.kind).toBe('discrete'))
  it('support(params) returns the inclusive integer bounds', () => {
    expect(discreteUniform.support?.({ a: 1, b: 6 })).toEqual({ min: 1, max: 6 })
  })
})
