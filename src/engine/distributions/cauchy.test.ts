import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { cauchy } from './cauchy'

/** Mirrors the (unexported) CauchyParams slot type used inside cauchy.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `cauchy.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type CauchyParams = { x0: number; gamma: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

/** Moderate sample (n = 15) with an obvious central location near 5 and finite spread. */
const sample = [5.1, 4.8, 5.3, 4.2, 6.1, 5.0, 4.5, 5.7, 3.9, 6.4, 4.9, 5.2, 4.6, 5.8, 5.5]

/** Sample median of `sample` (n=15, odd → middle order statistic) for the loose location check. */
function median(data: readonly number[]): number {
  const sorted = [...data].sort((a, b) => a - b)
  const mid = (sorted.length - 1) / 2
  return sorted[mid] ?? Number.NaN // `?? NaN` guards noUncheckedIndexedAccess; mid is in-bounds.
}

/** Total Cauchy log-likelihood via the elementary closed form (independent of cauchy.logpdf). */
function llClosedForm(data: readonly number[], x0: number, gamma: number): number {
  let ll = 0
  for (const x of data) {
    const r = (x - x0) / gamma
    ll += -Math.log(Math.PI) - Math.log(gamma) - Math.log(1 + r * r)
  }
  return ll
}

describe('cauchy', () => {
  it('logpdf passes (x0, gamma) to @stdlib: matches the elementary closed form at standard', () => {
    // CONVENTION GUARD: f(x) = 1 / (pi*gamma*(1 + ((x-x0)/gamma)^2)); at x0=0, gamma=1, x=0:
    // f(0) = 1/pi, so logf(0) = -ln(pi). Computed BY HAND here (not via @stdlib) to catch a slot-swap.
    expectClose(cauchy.logpdf(0, { x0: 0, gamma: 1 }), -Math.log(Math.PI), 1e-9)
  })
  it('logpdf matches the closed form at a non-standard point (x0, gamma both exercised)', () => {
    // f(x) = 1/(pi*gamma*(1+((x-x0)/gamma)^2)); x0=3, gamma=2, x=4: r=0.5, denom=pi*2*1.25.
    const expected = -Math.log(Math.PI) - Math.log(2) - Math.log(1 + 0.5 * 0.5)
    expectClose(cauchy.logpdf(4, { x0: 3, gamma: 2 }), expected, 1e-9)
  })
  it('cdf(x0) = 0.5 (x0 is the median)', () => expectClose(cauchy.cdf(3, { x0: 3, gamma: 2 }), 0.5))
  it('quantile(0.5) = x0 (median)', () => expectClose(cauchy.quantile(0.5, { x0: 3, gamma: 2 }), 3))
  it('quantile(0.75; x0=0,gamma=1) = x0 + gamma = 1 (upper-quartile half-width)', () => {
    // CONVENTION GUARD: gamma is the SCALE, so the 0.75 quantile sits exactly one gamma above x0.
    expectClose(cauchy.quantile(0.75, { x0: 0, gamma: 1 }), 1, 1e-9)
  })
  it('quantile inverts cdf (round-trip)', () => {
    const p = { x0: 3, gamma: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(cauchy.cdf(cauchy.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit converges to finite params with x0 near the median and gamma > 0', () => {
    const p = cauchy.fit(sample) as CauchyParams
    expect(Number.isFinite(p.x0)).toBe(true)
    expect(Number.isFinite(p.gamma)).toBe(true)
    expect(p.gamma).toBeGreaterThan(0)
    // The Cauchy MLE location is NOT the sample median in general — only a loose, scale-relative
    // check is valid: |x0 - median| < gamma.
    expect(Math.abs(p.x0 - median(sample))).toBeLessThan(p.gamma)
  })
  it('multi-start: LL at the fit >= LL at any single seed (defeats multimodality)', () => {
    const p = cauchy.fit(sample) as CauchyParams
    const med = median(sample)
    const sorted = [...sample].sort((a, b) => a - b)
    // `?? NaN` guards noUncheckedIndexedAccess; both indices are in-bounds for this sample.
    const q1 = sorted[Math.floor((sorted.length - 1) * 0.25)] ?? Number.NaN
    const q3 = sorted[Math.ceil((sorted.length - 1) * 0.75)] ?? Number.NaN
    const halfIQR = (q3 - q1) / 2
    const fitLL = llClosedForm(sample, p.x0, p.gamma)
    const seeds = [med, med - halfIQR, med + halfIQR]
    for (const seedX0 of seeds) {
      expect(fitLL).toBeGreaterThanOrEqual(llClosedForm(sample, seedX0, halfIQR) - 1e-9)
    }
  })
  it('rejects degenerate (zero-spread) data', () =>
    expect(() => cauchy.fit([7, 7, 7, 7])).toThrow())
  it('k = 2', () => expect(cauchy.k).toBe(2))
  it("kind = 'continuous'", () => expect(cauchy.kind).toBe('continuous'))
})
