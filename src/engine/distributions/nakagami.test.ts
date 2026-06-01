import gammaCdf from '@stdlib/stats-base-dists-gamma-cdf'
import { describe, expect, it } from 'vitest'
import { mean } from '../math'
import { logLik } from '../selection'
import { nakagami } from './nakagami'

/**
 * Nakagami-m module tests. The hand-composed logpdf and the Gamma-on-x² cdf/quantile (with the
 * load-bearing RATE = m/Ω convention) were all VERIFIED against scipy 1.17 (the venv-fixtures oracle):
 *   - logpdf(2; m=2, Omega=3) = -0.7050081606432148   (== scipy.nakagami(2, 0, sqrt(3)).logpdf(2))
 *   - cdf(2; m=2, Omega=3)    =  0.7452273455163945   (== gamma.cdf(4, a=2, scale=Omega/m) == nakagami.cdf)
 *   - quantile(0.5; m=2, Omega=3) = 1.5866696206283752 (== sqrt(gamma.ppf(0.5, 2, scale=Omega/m)))
 *   - fit on [1.2,0.8,1.5,2.1,0.9,1.8,1.1,2.4,0.7,1.6]: m=1.873662875224561, Omega=mean(x²)=2.281,
 *     LL=-7.665834847566068 (== scipy.nakagami.fit(x, floc=0) LL to ~1e-9; HardFit reaches >= scipy).
 * scipy maps HardFit (m, Omega) -> nakagami(nu=m, loc=0, scale=sqrt(Omega)) — Omega = scale² (squared).
 *
 * @stdlib gamma takes (x, alpha=shape, beta=RATE). Nakagami's x² ~ Gamma(shape=m, scale=Ω/m), so the
 * RATE is m/Ω (the INVERSE of the natural scale). Passing Ω/m as the rate is wrong-but-finite — the
 * convention-guard test below pins the correct RATE and asserts it differs from the inverted one.
 */

/** scipy reference logpdf at (x=2; m=2, Omega=3) — pinned to machine precision. */
const REF_LOGPDF = -0.7050081606432148
/** scipy reference cdf at (x=2; m=2, Omega=3). */
const REF_CDF = 0.7452273455163945
/** scipy reference quantile (ppf) at (p=0.5; m=2, Omega=3). */
const REF_QUANTILE_MEDIAN = 1.5866696206283752
/** Verified MLE on FIT_DATA (== scipy.nakagami.fit(x, floc=0) remapped to ~1e-9). */
const REF_FIT_M = 1.873662875224561
const REF_FIT_OMEGA = 2.2809999999999997
/** Verified maximized LL at the MLE (== scipy fitted LL to ~1e-9; HardFit reaches >= scipy). */
const REF_FIT_LL = -7.665834847566068

/** A small positive Nakagami-ish sample for the fit checks. */
const FIT_DATA = [1.2, 0.8, 1.5, 2.1, 0.9, 1.8, 1.1, 2.4, 0.7, 1.6]
const FIXED_PARAMS = { m: 2, Omega: 3 }

describe('nakagami: hand-composed logpdf / Gamma-on-x² cdf / quantile', () => {
  it('logpdf matches scipy.nakagami.logpdf at (x=2; m=2, Omega=3)', () => {
    expect(nakagami.logpdf(2, FIXED_PARAMS)).toBeCloseTo(REF_LOGPDF, 12)
  })

  it('logpdf is -Infinity for x <= 0 (out of support)', () => {
    expect(nakagami.logpdf(0, FIXED_PARAMS)).toBe(Number.NEGATIVE_INFINITY)
    expect(nakagami.logpdf(-1, FIXED_PARAMS)).toBe(Number.NEGATIVE_INFINITY)
  })

  it('cdf matches scipy.nakagami.cdf at (x=2; m=2, Omega=3)', () => {
    expect(nakagami.cdf(2, FIXED_PARAMS)).toBeCloseTo(REF_CDF, 12)
  })

  it('cdf is 0 for x <= 0', () => {
    expect(nakagami.cdf(0, FIXED_PARAMS)).toBe(0)
    expect(nakagami.cdf(-3, FIXED_PARAMS)).toBe(0)
  })

  it('cdf passes RATE = m/Omega to @stdlib gamma (NOT the inverted Omega/m)', () => {
    // CONVENTION GUARD: x² ~ Gamma(shape=m, scale=Ω/m) ⇒ @stdlib RATE = m/Ω. The inverted rate
    // (Ω/m) is finite-but-wrong; pin both and assert the cdf equals the correct one and differs
    // from the inverted one (catches exactly the rate/scale slot-swap a round-trip cannot).
    const { m, Omega } = FIXED_PARAMS
    const x = 2
    const correctRate = gammaCdf(x * x, m, m / Omega) // RATE = m/Ω
    const invertedRate = gammaCdf(x * x, m, Omega / m) // WRONG: RATE = Ω/m (inverse)
    expect(nakagami.cdf(x, FIXED_PARAMS)).toBeCloseTo(correctRate, 12)
    expect(Math.abs(correctRate - invertedRate)).toBeGreaterThan(0.1) // the two are far apart
  })

  it('quantile matches scipy.nakagami.ppf at (p=0.5; m=2, Omega=3)', () => {
    expect(nakagami.quantile(0.5, FIXED_PARAMS)).toBeCloseTo(REF_QUANTILE_MEDIAN, 9)
  })

  it('quantile round-trips: cdf(quantile(p)) ≈ p across p in [0.05, 0.95]', () => {
    for (let prob = 0.05; prob <= 0.95; prob += 0.05) {
      const x = nakagami.quantile(prob, FIXED_PARAMS)
      expect(nakagami.cdf(x, FIXED_PARAMS)).toBeCloseTo(prob, 9)
    }
  })
})

describe('nakagami: MLE via Gamma-on-x² 1-D Newton', () => {
  it('fit reproduces the MLE: m via Newton on g(m)=ln m−ψ(m)−s, Omega=mean(x²) (== scipy fit)', () => {
    const fitted = nakagami.fit(FIT_DATA) as { m: number; Omega: number }
    expect(fitted.m).toBeCloseTo(REF_FIT_M, 6)
    expect(fitted.Omega).toBeCloseTo(REF_FIT_OMEGA, 9)
    // Omega IS the sample mean of x² exactly.
    const meanSq = mean(FIT_DATA.map((x) => x * x))
    expect(fitted.Omega).toBeCloseTo(meanSq, 12)
  })

  it("fit's LL reaches scipy's fitted LL (HardFit >= scipy)", () => {
    const fitted = nakagami.fit(FIT_DATA)
    const ll = logLik(FIT_DATA, (x) => nakagami.logpdf(x, fitted))
    expect(ll).toBeCloseTo(REF_FIT_LL, 6)
    expect(ll).toBeGreaterThanOrEqual(REF_FIT_LL - 1e-6)
  })

  it('fit throws on x <= 0 in the data', () => {
    expect(() => nakagami.fit([1, 2, -0.5, 3])).toThrow()
    expect(() => nakagami.fit([1, 2, 0, 3])).toThrow()
  })

  it('fit throws on degenerate all-equal data (s = ln Ω − 2·meanLog(x) collapses to 0)', () => {
    expect(() => nakagami.fit([2, 2, 2, 2])).toThrow()
  })

  it('fit throws on too-small samples (n < 2)', () => {
    expect(() => nakagami.fit([1])).toThrow()
  })
})

describe('nakagami: registry metadata', () => {
  it('is a 2-parameter continuous family named "nakagami"', () => {
    expect(nakagami.name).toBe('nakagami')
    expect(nakagami.k).toBe(2)
    expect(nakagami.kind).toBe('continuous')
  })
})
