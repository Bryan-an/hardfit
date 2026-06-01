import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { mean } from '../math'
import { gumbel } from './gumbel'

/** Mirrors the (unexported) GumbelParams slot type used inside gumbel.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `gumbel.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type GumbelParams = { mu: number; beta: number }

/** A right-skewed sample (annual-maxima-flavored) for the iterative MLE checks. */
const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5]

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('gumbel', () => {
  it('logpdf passes (mu, beta=SCALE) to @stdlib: matches the elementary closed form', () => {
    // CONVENTION GUARD: standard Gumbel(max) density f(x) = (1/beta) exp(-(z + e^{-z})),
    // z = (x - mu)/beta. At mu=0, beta=1, x=0: z=0 -> f = e^{-1}, so logpdf = -1 exactly.
    // Written out by hand (NOT via @stdlib) to catch a wrong arg slot (e.g. a rate-for-scale swap).
    // The spec's named example: at mu=0, beta=1, x=0 the density is e^{-1}, so logpdf = -1 exactly.
    expectClose(gumbel.logpdf(0, { mu: 0, beta: 1 }), -1, 1e-9)
    // SLOT-DISCRIMINATING case: mu != 0, beta != 1, x != mu. At mu=5,beta=1 this point would alias,
    // so beta=2 (=> 1/beta=0.5) and x-mu=1 (=> not at the location) make a rate-for-scale swap or a
    // mu/beta transposition all produce a DIFFERENT number than this hand-written closed form.
    const mu = 5
    const beta = 2
    const x = 6
    const z = (x - mu) / beta // = 0.5
    const expected = -Math.log(beta) - (z + Math.exp(-z))
    expectClose(gumbel.logpdf(x, { mu, beta }), expected, 1e-9)
  })
  it('cdf is the MAX convention with the SCALE slot live: cdf(mu) and cdf(x != mu)', () => {
    // F(x) = exp(-exp(-(x-mu)/beta)); at x = mu the inner exponent is 1, so F(mu) = e^{-1}.
    expectClose(gumbel.cdf(5, { mu: 5, beta: 2 }), Math.exp(-1), 1e-9)
    // x != mu so beta no longer cancels: a rate-for-scale swap would change this value.
    expectClose(gumbel.cdf(7, { mu: 5, beta: 2 }), Math.exp(-Math.exp(-1)), 1e-9)
  })
  it('quantile inverts cdf (round-trip)', () => {
    const p = { mu: 5, beta: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(gumbel.cdf(gumbel.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit converges to finite params and drives the profile score g(beta) to 0', () => {
    const p = gumbel.fit(sample) as GumbelParams
    expect(Number.isFinite(p.mu)).toBe(true)
    expect(Number.isFinite(p.beta)).toBe(true)
    expect(p.beta).toBeGreaterThan(0)
    // Profile score: g(beta) = xbar - beta - (sum x e^{-x/beta})/(sum e^{-x/beta}) = 0 at the MLE.
    const xbar = mean(sample)
    let S0 = 0
    let S1 = 0
    for (const x of sample) {
      const e = Math.exp(-x / p.beta)
      S0 += e
      S1 += x * e
    }
    expectClose(xbar - p.beta - S1 / S0, 0, 1e-7, 1e-7)
  })
  it('fit recovers the location MLE: sum exp(-(x-mu)/beta) = n', () => {
    // mu satisfies the location score sum e^{-(x-mu)/beta} = n at the converged beta.
    const p = gumbel.fit(sample) as GumbelParams
    let s = 0
    for (const x of sample) s += Math.exp(-(x - p.mu) / p.beta)
    expectClose(s, sample.length, 1e-7, 1e-7)
  })
  it('rejects degenerate (zero spread) data', () =>
    expect(() => gumbel.fit([2, 2, 2, 2])).toThrow())
  it('k = 2', () => expect(gumbel.k).toBe(2))
  it("kind = 'continuous'", () => expect(gumbel.kind).toBe('continuous'))
})
