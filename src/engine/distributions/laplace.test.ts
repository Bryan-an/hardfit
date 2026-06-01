import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { laplace } from './laplace'

/** Mirrors the (unexported) LaplaceParams slot type used inside laplace.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `laplace.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type LaplaceParams = { mu: number; b: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('laplace', () => {
  it('logpdf passes b as SCALE to @stdlib: matches the elementary closed form', () => {
    // CONVENTION GUARD: laplace f(x) = 1/(2b) exp(-|x-mu|/b); logf = -ln(2b) - |x-mu|/b.
    // mu=0, b=1, x=2: logf = -ln(2) - 2.
    expectClose(laplace.logpdf(2, { mu: 0, b: 1 }), -Math.log(2) - 2, 1e-9)
  })
  it('logpdf at asymmetric params (stronger mu/b slot guard)', () => {
    // CONVENTION GUARD: mu=1, b=2, x=4: logf = -ln(2*2) - |4-1|/2 = -ln(4) - 3/2.
    expectClose(laplace.logpdf(4, { mu: 1, b: 2 }), -Math.log(4) - 3 / 2, 1e-9)
  })
  it('cdf(mu) = 0.5', () => expectClose(laplace.cdf(5, { mu: 5, b: 2 }), 0.5))
  it('quantile inverts cdf (round-trip)', () => {
    const p = { mu: 5, b: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(laplace.cdf(laplace.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('MLE: mu=median (odd n), b=mean(|x-mu|)', () => {
    // data=[1,2,3,4,5] -> mu=3, b=(2+1+0+1+2)/5 = 1.2 exactly.
    const p = laplace.fit([1, 2, 3, 4, 5]) as LaplaceParams
    expectClose(p.mu, 3)
    expectClose(p.b, 1.2)
  })
  it('MLE: mu=midpoint average of the two middle order stats (even n)', () => {
    // data=[1,2,3,4] -> mu=(2+3)/2=2.5, b=(1.5+0.5+0.5+1.5)/4 = 1.
    const p = laplace.fit([1, 2, 3, 4]) as LaplaceParams
    expectClose(p.mu, 2.5)
    expectClose(p.b, 1)
  })
  it('fit sorts a numeric COPY (unsorted, multi-digit data; no lexicographic bug)', () => {
    // data=[10,2,33,4] -> sorted [2,4,10,33], median=(4+10)/2=7; default lexicographic sort would
    // order as [10,2,33,4] and pick a wrong median, so this catches a missing numeric comparator.
    const data = [10, 2, 33, 4]
    const p = laplace.fit(data) as LaplaceParams
    expectClose(p.mu, 7)
    expect(data).toEqual([10, 2, 33, 4]) // input left unmutated
  })
  it('rejects degenerate (zero scale) data', () =>
    expect(() => laplace.fit([4, 4, 4, 4])).toThrow())
  it('k = 2', () => expect(laplace.k).toBe(2))
  it("kind = 'continuous'", () => expect(laplace.kind).toBe('continuous'))
})
