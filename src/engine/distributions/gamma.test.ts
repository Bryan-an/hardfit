import digamma from '@stdlib/math-base-special-digamma'
import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { mean, meanLog } from '../math'
import { gamma } from './gamma'

/** Mirrors the (unexported) GammaParams slot type used inside gamma.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `gamma.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type GammaParams = { shape: number; scale: number; rate: number }

const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5]

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]
/** scipy gamma(a=3, scale=1/1.5).ppf(0.5) — verified via gamma.cdf(MEDIAN, p) ≈ 0.5. */
const GAMMA_MEDIAN = 1.7827

describe('gamma', () => {
  it('shape solves ln(k) - digamma(k) = ln(mean) - mean(ln x)', () => {
    const p = gamma.fit(sample) as GammaParams
    const s = Math.log(mean(sample)) - meanLog(sample)
    expectClose(Math.log(p.shape) - digamma(p.shape), s, 1e-7)
  })
  it('scale = mean / shape; rate = 1/scale', () => {
    const p = gamma.fit(sample) as GammaParams
    expectClose(p.scale, mean(sample) / p.shape, 1e-7)
    expectClose(p.rate, 1 / p.scale, 1e-9)
  })
  it('logpdf passes RATE (not scale) to @stdlib: matches the rate-parameterized closed form', () => {
    // CONVENTION GUARD: catches a rate/scale slot-swap that self-consistency tests cannot.
    // gamma(shape a=3, rate b=1.5): logf(x) = a*ln(b) + (a-1)*ln(x) - b*x - lnΓ(a)
    const p = { shape: 3, rate: 1.5, scale: 1 / 1.5 }
    const expected = 3 * Math.log(1.5) + (3 - 1) * Math.log(2) - 1.5 * 2 - gammaln(3)
    expectClose(gamma.logpdf(2, p), expected, 1e-9)
  })
  it('cdf passes RATE: cdf(mean) for shape=3,rate=1.5 (mean=2) ~ 0.5768', () => {
    expectClose(gamma.cdf(2, { shape: 3, rate: 1.5, scale: 1 / 1.5 }), 0.5768099188731566, 1e-6)
  })
  it('LL at MLE >= LL at perturbed shape', () => {
    const p = gamma.fit(sample) as GammaParams
    const ll = (q: GammaParams) => sample.reduce((acc, x) => acc + gamma.logpdf(x, q), 0)
    const worseShape = p.shape * 1.2
    const worse = {
      shape: worseShape,
      rate: worseShape / mean(sample),
      scale: mean(sample) / worseShape,
    }
    expect(ll(p)).toBeGreaterThanOrEqual(ll(worse) - 1e-6)
  })
  it('quantile passes RATE (not scale): median of shape=3,rate=1.5 ~ 1.7827 (scale-swap gives ~4.01)', () => {
    // CONVENTION GUARD: the rate/scale slot-swap is finite-but-wrong; the fixed reference catches it.
    const p = { shape: 3, rate: 1.5, scale: 1 / 1.5 }
    expectClose(gamma.quantile(0.5, p), GAMMA_MEDIAN, 1e-3)
  })
  it('quantile inverts cdf (round-trip)', () => {
    const p = { shape: 3, rate: 1.5, scale: 1 / 1.5 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(gamma.cdf(gamma.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('rejects non-positive data', () => expect(() => gamma.fit([1, 0])).toThrow())
  it('k = 2', () => expect(gamma.k).toBe(2))
  it("kind = 'continuous'", () => expect(gamma.kind).toBe('continuous'))
})
