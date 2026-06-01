import digamma from '@stdlib/math-base-special-digamma'
import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { meanLog } from '../math'
import { chisquare } from './chisquare'

/** Mirrors the (unexported) ChiSquaredParams slot type used inside chisquare.ts; lets the test
 *  read fitted params by name without `?? NaN` noise. `chisquare.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type ChiSquaredParams = { df: number }

/** Strictly-positive sample (chi-squared support is x > 0). */
const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5]

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]
/** scipy chi2(df=1).ppf(0.8) — the 0.8-quantile of chi-squared with 1 d.o.f. (verified ~1.6424). */
const CHI2_1_Q80 = 1.6423744151498165

describe('chisquare', () => {
  it('logpdf matches the elementary chi-squared density (ASYMMETRIC df): catches a slot-blind point', () => {
    // CONVENTION GUARD: f(x; k) = x^(k/2-1) e^(-x/2) / (2^(k/2) Γ(k/2)); ASYMMETRIC df=4.7 so a
    // standard df (e.g. 2) cannot mask a wrong-slot pass. logf = (k/2-1)ln x - x/2 - (k/2)ln 2 - lnΓ(k/2).
    const df = 4.7
    const x = 2.3
    const expected = (df / 2 - 1) * Math.log(x) - x / 2 - (df / 2) * Math.log(2) - gammaln(df / 2)
    expectClose(chisquare.logpdf(x, { df }), expected, 1e-9)
  })
  it('cdf(1.0, df=1) ~ 0.6826894921 (the central ±1σ normal mass, chi-squared = Z²)', () => {
    expectClose(chisquare.cdf(1.0, { df: 1 }), 0.682689492137086, 1e-9)
  })
  it('quantile arg order is (prob, df): chi2(df=1) 0.8-quantile ~ 1.6424', () => {
    // CONVENTION GUARD: probability is the FIRST arg. Swapping to (df, prob) would feed prob=1 into
    // quantile(1.6424, …) territory — finite-but-wrong; the fixed reference catches it.
    expectClose(chisquare.quantile(0.8, { df: 1 }), CHI2_1_Q80, 1e-7)
  })
  it('quantile inverts cdf (round-trip)', () => {
    const p = { df: 4.7 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(chisquare.cdf(chisquare.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit converges to a finite df > 0 with score g(df) = ln2 + psi(df/2) - mean(ln x) ~ 0', () => {
    const p = chisquare.fit(sample) as ChiSquaredParams
    expect(Number.isFinite(p.df)).toBe(true)
    expect(p.df).toBeGreaterThan(0)
    const score = Math.log(2) + digamma(p.df / 2) - meanLog(sample)
    expectClose(score, 0, 1e-7, 1e-9)
  })
  it('LL at MLE >= LL at perturbed df', () => {
    const p = chisquare.fit(sample) as ChiSquaredParams
    const ll = (q: ChiSquaredParams) => sample.reduce((acc, x) => acc + chisquare.logpdf(x, q), 0)
    expect(ll(p)).toBeGreaterThanOrEqual(ll({ df: p.df * 1.2 }) - 1e-6)
  })
  it('rejects non-positive data', () => expect(() => chisquare.fit([1, 0])).toThrow())
  it('k = 1', () => expect(chisquare.k).toBe(1))
  it("kind = 'continuous'", () => expect(chisquare.kind).toBe('continuous'))
})
