import digamma from '@stdlib/math-base-special-digamma'
import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { meanLog } from '../math'
import { chi } from './chi'

/** Mirrors the (unexported) ChiParams slot type used inside chi.ts; lets the test read the
 *  fitted param by name without `?? NaN` noise. `chi.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type ChiParams = { k: number }

/** ASYMMETRIC positive sample (no ties, varied spread) so the fit exercises a real Newton solve. */
const sample = [1.4, 2.1, 0.8, 3.2, 1.9, 2.7, 1.1, 2.4, 0.6, 3.0, 1.7, 2.2, 0.9, 2.8, 1.3]

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('chi', () => {
  it('logpdf matches the elementary chi closed form (catches chi-vs-chisquare package swap)', () => {
    // CONVENTION GUARD: f(x;k) = x^(k-1) e^{-x^2/2} / (2^(k/2-1) Γ(k/2));
    // logf = (k-1)·ln x − x²/2 − (k/2−1)·ln2 − lnΓ(k/2). At k=3, x=1 the (k-1)·ln x term zeroes.
    // chi gives ≈ −0.726; the WRONG `chisquare` package would give ≈ −1.419, so this discriminates.
    const expected = 0 - 0.5 - 0.5 * Math.log(2) - gammaln(1.5)
    expectClose(chi.logpdf(1, { k: 3 }), expected, 1e-9)
  })
  it('cdf(2, {k:2}) = 1 - e^{-2} (chi k=2 is Rayleigh σ=1; discriminates chi vs chi-squared)', () => {
    // Evaluate at x=2, NOT x=1: at x=1, x²=x so chi.cdf(1,2) == chisquare.cdf(1,2) and the test
    // could not catch the chi/chi-squared package swap. Chi k=2 CDF = 1 - e^{-x²/2}; at x=2 that
    // is 1 - e^{-2} ≈ 0.8647, whereas a chi-squared(df=2) misread would give 1 - e^{-1} ≈ 0.6321.
    expectClose(chi.cdf(2, { k: 2 }), 1 - Math.exp(-2), 1e-9)
  })
  it('quantile inverts cdf (round-trip)', () => {
    const p = { k: 3 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(chi.cdf(chi.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit solves the score equation digamma(k/2) = 2·mean(ln x) − ln2', () => {
    const p = chi.fit(sample) as ChiParams
    expect(Number.isFinite(p.k)).toBe(true)
    expect(p.k).toBeGreaterThan(0)
    expectClose(digamma(p.k / 2), 2 * meanLog(sample) - Math.log(2), 1e-7)
  })
  it('rejects non-positive data', () => expect(() => chi.fit([1, 0])).toThrow())
  it('k = 1', () => expect(chi.k).toBe(1))
  it("kind = 'continuous'", () => expect(chi.kind).toBe('continuous'))
})
