import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { rayleigh } from './rayleigh'

/** Mirrors the (unexported) RayleighParams slot type used inside rayleigh.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `rayleigh.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type RayleighParams = { sigma: number }

const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5]

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]
/** MLE denominator factor: sigma^2 = sum x^2 / (2n). */
const VARIANCE_FACTOR = 2

describe('rayleigh', () => {
  it('MLE: sigma = sqrt(sum x^2 / (2n)) exactly', () => {
    const p = rayleigh.fit(sample) as RayleighParams
    let sumSq = 0
    for (const x of sample) sumSq += x * x
    expect(p.sigma).toBe(Math.sqrt(sumSq / (VARIANCE_FACTOR * sample.length)))
  })
  it('logpdf passes SCALE to @stdlib: matches the elementary closed form', () => {
    // CONVENTION GUARD: rayleigh density f(x) = x/sigma^2 * exp(-x^2/(2 sigma^2)).
    // For sigma=2, x=1: logf = ln(1/4) - 1/8. Written by hand (NOT via @stdlib) to catch a
    // wrong arg slot (e.g. passing a rate 1/sigma instead of the scale sigma).
    const expected = Math.log(1 / 4) - 1 / 8
    expectClose(rayleigh.logpdf(1, { sigma: 2 }), expected, 1e-9)
  })
  it('cdf matches 1 - exp(-x^2/(2 sigma^2))', () => {
    // sigma=2, x=2: F = 1 - exp(-4/8) = 1 - exp(-1/2).
    expectClose(rayleigh.cdf(2, { sigma: 2 }), 1 - Math.exp(-0.5))
  })
  it('quantile passes SCALE (not rate): Q(p) = sigma*sqrt(-2 ln(1-p))', () => {
    // CONVENTION GUARD: inverse CDF in closed form, written by hand.
    const p = { sigma: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      const expected = p.sigma * Math.sqrt(-2 * Math.log(1 - prob))
      expectClose(rayleigh.quantile(prob, p), expected, 1e-9)
    }
  })
  it('quantile inverts cdf (round-trip)', () => {
    const p = { sigma: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(rayleigh.cdf(rayleigh.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('rejects negative data', () => expect(() => rayleigh.fit([1, -1, 2])).toThrow())
  it('rejects all-zero data (degenerate: sum of squares is zero)', () =>
    expect(() => rayleigh.fit([0, 0, 0])).toThrow())
  it('k = 1', () => expect(rayleigh.k).toBe(1))
  it("kind = 'continuous'", () => expect(rayleigh.kind).toBe('continuous'))
})
