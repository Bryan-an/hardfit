import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { mean, populationVariance } from '../math'
import { beta } from './beta'

/** Mirrors the (unexported) BetaParams slot type used inside beta.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `beta.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type BetaParams = { alpha: number; beta: number }

/** A (0,1) sample with non-trivial variance; seeds a valid, finite MoM start for Newton. */
const sample = [0.2, 0.4, 0.5, 0.6, 0.8]

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

/** Total beta log-likelihood of `sample` under params `p` (uses the public logpdf). */
function logLik(p: BetaParams): number {
  return sample.reduce((acc, x) => acc + beta.logpdf(x, p), 0)
}

/** Replicates beta.ts's method-of-moments seed so the test can score LL at the seed. */
function momSeed(): BetaParams {
  const m = mean(sample)
  const v = populationVariance(sample, m)
  const alpha = m * ((m * (1 - m)) / v - 1)
  return { alpha, beta: (alpha * (1 - m)) / m }
}

describe('beta', () => {
  it('logpdf uses (x, alpha, beta) order: matches the elementary closed form (ASYMMETRIC a=2, b=5)', () => {
    // CONVENTION GUARD: asymmetric shapes at an asymmetric x catch an alpha/beta slot-swap that a
    // symmetric point (a=b, x=0.5) is blind to. beta pdf = x^(a-1)(1-x)^(b-1)/B(a,b),
    // B(a,b) = exp(gammaln(a) + gammaln(b) - gammaln(a+b)); logf = (a-1)ln x + (b-1)ln(1-x) - ln B.
    const p = { alpha: 2, beta: 5 }
    const expected =
      (2 - 1) * Math.log(0.3) + (5 - 1) * Math.log(0.7) - (gammaln(2) + gammaln(5) - gammaln(7))
    expectClose(beta.logpdf(0.3, p), expected, 1e-9)
  })
  it('cdf at the symmetric median: cdf(0.5) for alpha=beta=2 is exactly 0.5', () => {
    expectClose(beta.cdf(0.5, { alpha: 2, beta: 2 }), 0.5, 1e-9)
  })
  it('quantile inverts cdf (round-trip) for fixed asymmetric params', () => {
    const p = { alpha: 2, beta: 5 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(beta.cdf(beta.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit returns finite, positive shapes', () => {
    const p = beta.fit(sample) as BetaParams
    expect(Number.isFinite(p.alpha) && p.alpha > 0).toBe(true)
    expect(Number.isFinite(p.beta) && p.beta > 0).toBe(true)
  })
  it('LL at the fit >= LL at the MoM seed (strictly concave MLE)', () => {
    const fit = beta.fit(sample) as BetaParams
    expect(logLik(fit)).toBeGreaterThanOrEqual(logLik(momSeed()) - 1e-6)
  })
  it('rejects data outside (0,1) — value >= 1', () => {
    expect(() => beta.fit([0.5, 1.5, 0.2])).toThrow()
  })
  it('rejects zero-variance (degenerate) data', () => {
    expect(() => beta.fit([0.4, 0.4, 0.4])).toThrow()
  })
  it('k = 2', () => expect(beta.k).toBe(2))
  it("kind = 'continuous'", () => expect(beta.kind).toBe('continuous'))
})
