import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { frechet } from './frechet'
import { weibull } from './weibull'

/** Mirrors the (unexported) FrechetParams slot type used inside frechet.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `frechet.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type FrechetParams = { shape: number; scale: number }

const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5]

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('frechet', () => {
  it('logpdf passes SCALE (not its inverse) + m=0 to @stdlib: matches the elementary closed form', () => {
    // CONVENTION GUARD: Frechet (m=0) density is f(x) = (alpha/s)*(x/s)^(-1-alpha)*exp(-(x/s)^(-alpha)).
    // Use a NON-UNIT scale (s=3) so a scale-inversion slot bug (s passed as 1/s) is detectable — at
    // s=1 the inverse equals the original and the trap is invisible. Anchor at x = scale so (x/s)=1:
    // f = (alpha/s)*1*exp(-1), hence logf = ln(2) - ln(3) - 1. Written out BY HAND (NOT via @stdlib).
    expectClose(frechet.logpdf(3, { shape: 2, scale: 3 }), Math.log(2) - Math.log(3) - 1, 1e-9)
  })
  it('cdf(scale) = exp(-1): cdf(x) = exp(-(x/s)^(-alpha)), at x=s the exponent is -1', () => {
    // CONVENTION GUARD: anchored at a NON-UNIT scale (s=3) so passing s inverted (1/3) would turn
    // the effective (x/s) from 1 into 9 and move this point far from exp(-1) — i.e. it is detectable.
    expectClose(frechet.cdf(3, { shape: 2, scale: 3 }), Math.exp(-1))
  })
  it('quantile inverts cdf (round-trip)', () => {
    const p = { shape: 2, scale: 3 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(frechet.cdf(frechet.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit converges to finite shape>0 and scale>0 on a positive sample', () => {
    const p = frechet.fit(sample) as FrechetParams
    expect(Number.isFinite(p.shape)).toBe(true)
    expect(Number.isFinite(p.scale)).toBe(true)
    expect(p.shape).toBeGreaterThan(0)
    expect(p.scale).toBeGreaterThan(0)
  })
  it('fit uses the reciprocal-Weibull identity: shape = weibull.fit(1/x).shape, scale = 1/weibull.scale', () => {
    const p = frechet.fit(sample) as FrechetParams
    const w = weibull.fit(sample.map((x) => 1 / x)) as { shape: number; scale: number }
    expectClose(p.shape, w.shape, 1e-12)
    expectClose(p.scale, 1 / w.scale, 1e-12)
  })
  it('rejects degenerate (all values equal) data', () =>
    expect(() => frechet.fit([3, 3, 3])).toThrow())
  it('rejects non-positive data', () => expect(() => frechet.fit([1, 0, 2])).toThrow())
  it('k = 2', () => expect(frechet.k).toBe(2))
  it("kind = 'continuous'", () => expect(frechet.kind).toBe('continuous'))
})
