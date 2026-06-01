import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { gamma } from './gamma'
import { invgamma } from './invgamma'

/** Mirrors the (unexported) InvGammaParams slot type used inside invgamma.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `invgamma.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type InvGammaParams = { shape: number; scale: number }

const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5]

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('invgamma', () => {
  it('logpdf passes SCALE directly into the beta slot: matches the elementary closed form', () => {
    // CONVENTION GUARD: invgamma pdf = beta^alpha / Gamma(alpha) * x^(-alpha-1) * exp(-beta/x), so
    // logf = alpha*ln(beta) - lnΓ(alpha) + (-alpha-1)*ln(x) - beta/x. Use ASYMMETRIC params
    // (alpha=0.5, beta=1, x=2) so the (shape,scale) slots are distinguishable and beta sits in the
    // scale slot, not a rate. Written out BY HAND (NOT via @stdlib). Expected ≈ -2.112.
    const alpha = 0.5
    const beta = 1
    const x = 2
    const expected = alpha * Math.log(beta) - gammaln(alpha) + (-alpha - 1) * Math.log(x) - beta / x
    expectClose(invgamma.logpdf(x, { shape: alpha, scale: beta }), expected, 1e-9)
  })
  it('cdf at a fixed point matches the @stdlib reference (asymmetric params)', () => {
    // alpha=3, beta=2 (scale slot): scipy invgamma(a=3, scale=2).cdf(2) ≈ 0.9197.
    expectClose(invgamma.cdf(2, { shape: 3, scale: 2 }), 0.919698602928606, 1e-9)
  })
  it('quantile inverts cdf (round-trip)', () => {
    const p = { shape: 3, scale: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(invgamma.cdf(invgamma.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit converges to finite shape>0 and scale>0 on a positive sample', () => {
    const p = invgamma.fit(sample) as InvGammaParams
    expect(Number.isFinite(p.shape)).toBe(true)
    expect(Number.isFinite(p.scale)).toBe(true)
    expect(p.shape).toBeGreaterThan(0)
    expect(p.scale).toBeGreaterThan(0)
  })
  it('fit uses the reciprocal-gamma identity: shape = gamma.fit(1/x).shape, scale = gamma.rate', () => {
    const p = invgamma.fit(sample) as InvGammaParams
    const w = gamma.fit(sample.map((x) => 1 / x)) as { shape: number; scale: number; rate: number }
    expectClose(p.shape, w.shape, 1e-12)
    expectClose(p.scale, w.rate, 1e-12)
  })
  it('rejects non-positive data', () => expect(() => invgamma.fit([1, 0, 2])).toThrow())
  it('k = 2', () => expect(invgamma.k).toBe(2))
  it("kind = 'continuous'", () => expect(invgamma.kind).toBe('continuous'))
})
