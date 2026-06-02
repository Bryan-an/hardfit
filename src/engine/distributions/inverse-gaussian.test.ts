import { describe, expect, it } from 'vitest'
import { mean } from '../math'
import { logLik } from '../selection'
import { inverseGaussian } from './inverse-gaussian'

/**
 * Inverse Gaussian (Wald) module tests. The closed-form MLE, the hand-composed logpdf, and the
 * overflow-safe CDF fold were all VERIFIED against scipy 1.17 (the venv-fixtures oracle):
 *   - logpdf(2; mu=1.5, lambda=2) = -1.6676412693201734          (== scipy.invgauss(1.5/2,0,2).logpdf(2))
 *   - cdf(2; mu=1.5, lambda=2)   = 0.7718200458880228
 *   - closed-form MLE on [1,2,0.5,1.5,3,0.8]: mu=mean=1.4666666666666668,
 *     lambda=n/(Σ(1/x)−n/mu)=3.6164383561643834, LL=-6.578562981427358 (== scipy.invgauss.fit(x,floc=0) LL)
 * scipy maps HardFit (mu, lambda) -> invgauss(mu_s=mu/lambda, loc=0, scale=lambda) — the TRAP.
 */

/** scipy reference logpdf at (x=2; mu=1.5, lambda=2) — pinned to machine precision. */
const REF_LOGPDF = -1.6676412693201734
/** scipy reference cdf at (x=2; mu=1.5, lambda=2). */
const REF_CDF = 0.7718200458880228
/** Verified closed-form MLE on [1,2,0.5,1.5,3,0.8] (== scipy.invgauss.fit(x, floc=0) remapped). */
const REF_FIT_MU = 1.4666666666666668
const REF_FIT_LAMBDA = 3.6164383561643834
/** Verified maximized LL at the closed-form MLE (== scipy fitted LL to ~1e-9). */
const REF_FIT_LL = -6.578562981427358

const FIT_DATA = [1, 2, 0.5, 1.5, 3, 0.8]
const FIXED_PARAMS = { mu: 1.5, lambda: 2 }

describe('inverse-gaussian: hand-composed logpdf / cdf / quantile', () => {
  it('logpdf matches scipy.invgauss.logpdf at (x=2; mu=1.5, lambda=2)', () => {
    expect(inverseGaussian.logpdf(2, FIXED_PARAMS)).toBeCloseTo(REF_LOGPDF, 12)
  })

  it('logpdf is -Infinity for x <= 0 (out of support)', () => {
    expect(inverseGaussian.logpdf(0, FIXED_PARAMS)).toBe(Number.NEGATIVE_INFINITY)
    expect(inverseGaussian.logpdf(-1, FIXED_PARAMS)).toBe(Number.NEGATIVE_INFINITY)
  })

  it('cdf matches scipy.invgauss.cdf at (x=2; mu=1.5, lambda=2)', () => {
    expect(inverseGaussian.cdf(2, FIXED_PARAMS)).toBeCloseTo(REF_CDF, 12)
  })

  it('cdf is 0 for x <= 0', () => {
    expect(inverseGaussian.cdf(0, FIXED_PARAMS)).toBe(0)
    expect(inverseGaussian.cdf(-3, FIXED_PARAMS)).toBe(0)
  })

  it('CDF overflow guard: at mu=1, lambda=400 the log-fold keeps the CDF finite (naive exp() is NaN)', () => {
    // The naive exp(2*lambda/mu)*Phi(-...) is exp(800)*~0 = Inf*0 = NaN. The log-space fold
    // exp(2*lambda/mu + normalLogcdf(-...)) stays finite. cdf(2) is ~1 here; assert finite + valid.
    const p = { mu: 1, lambda: 400 }
    const c2 = inverseGaussian.cdf(2, p)
    expect(Number.isFinite(c2)).toBe(true)
    expect(c2).toBeGreaterThan(0)
    expect(c2).toBeLessThanOrEqual(1)
    expect(c2).toBeCloseTo(1, 9)
    // A mid-body point must also be finite + monotone-consistent.
    const c1 = inverseGaussian.cdf(1, p)
    expect(Number.isFinite(c1)).toBe(true)
    expect(c1).toBeCloseTo(0.5099673351883012, 9) // scipy.invgauss(1/400,0,400).cdf(1)
  })

  it('quantile round-trips: cdf(quantile(p)) ≈ p across p in [0.01, 0.99]', () => {
    const p = { mu: 1.5, lambda: 2 }
    for (let prob = 0.01; prob <= 0.99; prob += 0.01) {
      const x = inverseGaussian.quantile(prob, p)
      expect(inverseGaussian.cdf(x, p)).toBeCloseTo(prob, 9)
    }
  })

  it('quantile clamps the degenerate probabilities (<=0 -> 0, >=1 -> +Infinity)', () => {
    const p = { mu: 1.5, lambda: 2 }
    expect(inverseGaussian.quantile(0, p)).toBe(0)
    expect(inverseGaussian.quantile(1, p)).toBe(Number.POSITIVE_INFINITY)
  })
})

describe('inverse-gaussian: closed-form MLE', () => {
  it('fit reproduces the closed form mu=mean, lambda=n/(Σ(1/x)−n/mu) (== scipy fit)', () => {
    const fitted = inverseGaussian.fit(FIT_DATA)
    expect(fitted.mu).toBeCloseTo(REF_FIT_MU, 12)
    expect(fitted.lambda).toBeCloseTo(REF_FIT_LAMBDA, 9)
    // mu IS the sample mean exactly.
    expect(fitted.mu).toBeCloseTo(mean(FIT_DATA), 12)
  })

  it("fit's LL equals scipy's fitted LL (HardFit IS the MLE)", () => {
    const fitted = inverseGaussian.fit(FIT_DATA)
    const ll = logLik(FIT_DATA, (x) => inverseGaussian.logpdf(x, fitted))
    expect(ll).toBeCloseTo(REF_FIT_LL, 9)
  })

  it('fit throws on x <= 0 in the data', () => {
    expect(() => inverseGaussian.fit([1, 2, -0.5, 3])).toThrow()
    expect(() => inverseGaussian.fit([1, 2, 0, 3])).toThrow()
  })

  it('fit throws on degenerate all-equal data (lambda not finite/positive)', () => {
    expect(() => inverseGaussian.fit([2, 2, 2, 2])).toThrow()
  })

  it('fit throws on too-small samples (n < 2)', () => {
    expect(() => inverseGaussian.fit([1])).toThrow()
  })
})

describe('inverse-gaussian: registry metadata', () => {
  it('is a 2-parameter continuous family named "inverse-gaussian"', () => {
    expect(inverseGaussian.name).toBe('inverse-gaussian')
    expect(inverseGaussian.k).toBe(2)
    expect(inverseGaussian.kind).toBe('continuous')
  })
})
