import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { mean, populationVariance } from '../math'
import { normal } from './normal'

/** Mirrors the (unexported) NormalParams slot type used inside normal.ts; lets the test
 *  read fitted params by name without `?? NaN` noise. `normal.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type NormalParams = { mu: number; sigma: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

describe('normal', () => {
  it('MLE: mu=mean, sigma=sqrt(population variance) (÷n)', () => {
    const data = [2, 4, 4, 4, 5, 5, 7, 9]
    const p = normal.fit(data) as NormalParams
    expectClose(p.mu, mean(data))
    expectClose(p.sigma, Math.sqrt(populationVariance(data)))
  })
  it('logpdf matches the standard normal density at 0', () => {
    expectClose(normal.logpdf(0, { mu: 0, sigma: 1 }), -0.5 * Math.log(2 * Math.PI))
  })
  it('cdf(mu) = 0.5', () => expectClose(normal.cdf(5, { mu: 5, sigma: 2 }), 0.5))
  it('property: logpdf is maximized at the MLE vs perturbed params (sum over data)', () => {
    fc.assert(
      fc.property(
        fc
          .array(fc.double({ min: -50, max: 50, noNaN: true }), { minLength: 5, maxLength: 50 })
          .filter((d) => populationVariance(d) > 0),
        (data) => {
          const p = normal.fit(data) as NormalParams
          const ll = (q: typeof p) => data.reduce((s, x) => s + normal.logpdf(x, q), 0)
          if (p.sigma === 0) return // degenerate (constant data)
          return ll(p) >= ll({ mu: p.mu + 0.5, sigma: p.sigma }) - 1e-9
        },
      ),
    )
  })
  it('quantile(0.5) = mu (median)', () => expectClose(normal.quantile(0.5, { mu: 5, sigma: 2 }), 5))
  it('quantile inverts cdf (round-trip)', () => {
    const p = { mu: 5, sigma: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(normal.cdf(normal.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('k = 2', () => expect(normal.k).toBe(2))
  it("kind = 'continuous'", () => expect(normal.kind).toBe('continuous'))
})
