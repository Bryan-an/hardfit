import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { betaprime } from './betaprime'

/** Mirrors the (unexported) BetaPrimeParams slot type used inside betaprime.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `betaprime.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type BetaPrimeParams = { alpha: number; beta: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

/** A betaprime(alpha=3, beta=4) sample (n=60, fixed seed) for the fit test. */
const sample = [
  0.4744, 0.3658, 1.3487, 0.8365, 3.8806, 0.7905, 0.267, 1.5934, 0.2607, 1.1177, 0.9331, 1.8347,
  1.4564, 0.7637, 0.4612, 0.634, 0.2388, 2.8017, 1.5899, 0.4749, 0.7842, 1.2254, 0.3936, 0.4704,
  0.878, 0.3126, 1.1852, 0.689, 1.5538, 0.1509, 0.2796, 1.9905, 0.774, 0.6367, 0.4028, 0.7928,
  0.454, 0.3093, 0.9345, 1.1884, 1.4036, 0.2249, 1.0614, 1.0267, 1.0833, 0.2949, 2.1815, 0.5951,
  0.6666, 0.8776, 0.6569, 1.3615, 0.8727, 0.9057, 1.9342, 1.7258, 0.8055, 0.7526, 5.5756, 0.4955,
]

/** Total betaprime log-likelihood via the elementary closed form (independent of betaprime.logpdf):
 *  per observation (alpha-1)ln x + (-alpha-beta)ln(1+x) - lnB(alpha,beta). */
function llClosedForm(data: readonly number[], alpha: number, beta: number): number {
  const lnB = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
  let ll = 0
  for (const x of data) {
    ll += (alpha - 1) * Math.log(x) + (-alpha - beta) * Math.log(1 + x) - lnB
  }
  return ll
}

describe('betaprime', () => {
  it('logpdf passes (alpha, beta) IN ORDER to @stdlib: matches the closed form at x != 1', () => {
    // CONVENTION GUARD with ASYMMETRIC alpha=2, beta=3 at x=2 (NOT x=1: at x=1 the density is slot-
    // blind because x^(alpha-1)=1, (1+x)^(-alpha-beta) depends only on alpha+beta, and B(a,b)=B(b,a),
    // so a swapped alpha/beta would pass). f(x) = x^(a-1)(1+x)^(-a-b)/B(a,b); B = exp(lnGamma a +
    // lnGamma b - lnGamma(a+b)). Computed BY HAND here to catch an alpha-before-beta order swap.
    const alpha = 2
    const beta = 3
    const x = 2
    const lnB = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    const expected = (alpha - 1) * Math.log(x) + (-alpha - beta) * Math.log(1 + x) - lnB
    expectClose(betaprime.logpdf(x, { alpha, beta }), expected, 1e-9)
  })
  it('logpdf is NOT symmetric in alpha/beta at x != 1 (order actually matters)', () => {
    // Sanity that the chosen guard point discriminates the slot order: swapping alpha<->beta changes
    // the value. (At x=1 these would be exactly equal, which is why the guard above lives at x=2.)
    expect(betaprime.logpdf(2, { alpha: 2, beta: 3 })).not.toBe(
      betaprime.logpdf(2, { alpha: 3, beta: 2 }),
    )
  })
  it('cdf matches the @stdlib reference value at an asymmetric point', () => {
    // betaprime cdf is the regularized incomplete beta I_{x/(1+x)}(alpha,beta); for alpha=2,beta=3,
    // x=1 -> I_{0.5}(2,3) = 0.6875 (verified against @stdlib). A direct value pins cdf orientation.
    expectClose(betaprime.cdf(1, { alpha: 2, beta: 3 }), 0.6875, 1e-9)
  })
  it('quantile inverts cdf (round-trip) at asymmetric alpha, beta', () => {
    const p = { alpha: 2, beta: 3 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(betaprime.cdf(betaprime.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit returns finite alpha, beta > 0 with LL at the fit >= LL at the MoM seed', () => {
    const p = betaprime.fit(sample) as BetaPrimeParams
    expect(Number.isFinite(p.alpha)).toBe(true)
    expect(Number.isFinite(p.beta)).toBe(true)
    expect(p.alpha).toBeGreaterThan(0)
    expect(p.beta).toBeGreaterThan(0)
    // Reconstruct the SAME method-of-moments seed fit() starts from (mean m, population var v;
    // beta0 = 2 + m(m+1)/v, alpha0 = m(beta0-1)), then assert the converged LL did not regress.
    const n = sample.length
    let sum = 0
    for (const x of sample) sum += x
    const m = sum / n
    let sumSq = 0
    for (const x of sample) sumSq += (x - m) * (x - m)
    const v = sumSq / n
    const beta0 = 2 + (m * (m + 1)) / v
    const alpha0 = m * (beta0 - 1)
    expect(llClosedForm(sample, p.alpha, p.beta)).toBeGreaterThanOrEqual(
      llClosedForm(sample, alpha0, beta0) - 1e-9,
    )
  })
  it('rejects non-positive data (support is x > 0)', () =>
    expect(() => betaprime.fit([0.5, 1.2, 0, 2.3])).toThrow(/x > 0/))
  it('throws (never returns out-of-support shapes) on extreme-spread data', () =>
    // ~24 orders of magnitude of spread drives the Newton step into a region the positivity
    // backtrack can't keep feasible; fit() must throw (→ reported failure), not return α/β <= 0.
    expect(() => betaprime.fit([1e-12, 1e12, 1, 1])).toThrow(/degenerate/))
  it('k = 2', () => expect(betaprime.k).toBe(2))
  it("kind = 'continuous'", () => expect(betaprime.kind).toBe('continuous'))
})
