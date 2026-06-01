import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { negativeBinomial } from './negative-binomial'

/** Mirrors the (unexported) NegativeBinomialParams slot type used inside negative-binomial.ts;
 *  lets the test read fitted params by name without `?? NaN` noise. `negativeBinomial.fit`
 *  returns the engine-wide `FittedParams` (Record<string, number>) DTO, so a cast is expected. */
type NegativeBinomialParams = { r: number; p: number }

/** An OVERDISPERSED integer sample (var > mean): the only regime with a finite MLE. */
const OVERDISPERSED = [0, 1, 2, 5, 8, 3, 12, 0, 4, 6, 1, 9, 2, 15, 0, 7, 3, 1, 10, 4]
/** An UNDERDISPERSED integer sample (var < mean): no finite MLE, fit must throw. */
const UNDERDISPERSED = [3, 3, 3, 4, 3, 3]

/** Reference params for the convention/cdf/quantile guards (r non-integer-free here = 3). */
const REF_R = 3
const REF_P = 0.4

const sum = (a: readonly number[]) => a.reduce((s, v) => s + v, 0)
const mean = (a: readonly number[]) => sum(a) / a.length
const popVar = (a: readonly number[]) => {
  const m = mean(a)
  return sum(a.map((v) => (v - m) * (v - m))) / a.length
}
const logLik = (p: NegativeBinomialParams) =>
  sum(OVERDISPERSED.map((x) => negativeBinomial.logpdf(x, p)))

describe('negative-binomial', () => {
  it('logpdf matches the elementary log-PMF written by hand (r/p slots, not swapped)', () => {
    // CONVENTION GUARD: logpmf(x; r, p) = lnΓ(x+r) - lnΓ(r) - lnΓ(x+1) + r*ln(p) + x*ln(1-p).
    // ASYMMETRIC params r=3, p=0.4, x=2, written by hand (NOT via @stdlib) to catch an r/p swap.
    const x = 2
    const expected =
      gammaln(x + REF_R) -
      gammaln(REF_R) -
      gammaln(x + 1) +
      REF_R * Math.log(REF_P) +
      x * Math.log(1 - REF_P)
    expectClose(negativeBinomial.logpdf(x, { r: REF_R, p: REF_P }), expected, 1e-9)
  })
  it('cdf at x=2 for r=3,p=0.4 ~ 0.31744', () => {
    expectClose(negativeBinomial.cdf(2, { r: REF_R, p: REF_P }), 0.31744, 1e-6)
  })
  it('quantile is a discrete integer with cdf(quantile(0.5)) >= 0.5', () => {
    const p = { r: REF_R, p: REF_P }
    const q = negativeBinomial.quantile(0.5, p)
    expect(Number.isInteger(q)).toBe(true)
    expect(negativeBinomial.cdf(q, p)).toBeGreaterThanOrEqual(0.5)
  })
  it('fit on overdispersed data converges to finite r>0, p in (0,1], LL >= LL at the MoM seed', () => {
    const p = negativeBinomial.fit(OVERDISPERSED) as NegativeBinomialParams
    expect(Number.isFinite(p.r)).toBe(true)
    expect(p.r).toBeGreaterThan(0)
    expect(p.p).toBeGreaterThan(0)
    expect(p.p).toBeLessThanOrEqual(1)
    // MoM seed: r0 = xbar^2 / (s2 - xbar); p0 = r0 / (r0 + xbar). MLE LL must beat the seed.
    const xbar = mean(OVERDISPERSED)
    const r0 = (xbar * xbar) / (popVar(OVERDISPERSED) - xbar)
    const seed = { r: r0, p: r0 / (r0 + xbar) }
    expect(logLik(p)).toBeGreaterThanOrEqual(logLik(seed) - 1e-9)
  })
  it('fit throws on underdispersed data (no finite MLE)', () => {
    expect(() => negativeBinomial.fit(UNDERDISPERSED)).toThrow(/overdispersed/)
  })
  it('fit never returns a degenerate Poisson-limit point on extreme data (finite-LL guard)', () => {
    // Nine zeros + one huge outlier: technically overdispersed, but a runaway Newton false-converges
    // to an enormous r (p→1, a point mass at 0) whose LL is -Infinity on the 1000. fit() must either
    // return a USABLE finite-LL fit or throw — never a silent degenerate point.
    const extreme = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000]
    let threw = false
    let params: NegativeBinomialParams | undefined
    try {
      params = negativeBinomial.fit(extreme) as NegativeBinomialParams
    } catch {
      threw = true
    }
    if (!threw && params) {
      const ll = extreme.reduce((acc, x) => acc + negativeBinomial.logpdf(x, params), 0)
      expect(Number.isFinite(ll)).toBe(true)
      expect(params.p).toBeLessThan(1)
    }
  })
  it('fit throws on non-integer data', () => {
    expect(() => negativeBinomial.fit([0, 1, 2.5, 5, 8])).toThrow(/integer/)
  })
  it('fit throws on negative data', () => {
    expect(() => negativeBinomial.fit([0, 1, -2, 5, 8])).toThrow()
  })
  it('k = 2', () => expect(negativeBinomial.k).toBe(2))
  it("kind = 'discrete'", () => expect(negativeBinomial.kind).toBe('discrete'))
  it('support = {0, +Infinity}', () => {
    const s = negativeBinomial.support?.({ r: REF_R, p: REF_P })
    expect(s).toEqual({ min: 0, max: Number.POSITIVE_INFINITY })
  })
})
