import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { cosine } from './cosine'

/** Mirrors cosine.ts: MoM scale = pi * sd / sqrt(pi^2/3 - 2); the seed is widened past max|x-mu0|. */
const MOM_SCALE_DENOM = Math.sqrt((Math.PI * Math.PI) / 3 - 2)
const SEED_WIDEN = 1.05

/** Mirrors the (unexported) CosineParams slot type used inside cosine.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `cosine.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type CosineParams = { mu: number; s: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

/** Positive sample with a clear center near 5 and finite spread (n = 12). */
const sample = [4.8, 5.2, 4.5, 5.5, 4.1, 5.9, 4.7, 5.3, 4.4, 5.6, 5.0, 4.9]

/** Largest absolute deviation of `data` from `mu`; the fitted support half-width must exceed it. */
function maxAbsDev(data: readonly number[], mu: number): number {
  let m = 0
  for (const x of data) {
    const d = Math.abs(x - mu)
    if (d > m) m = d
  }
  return m
}

/** Total raised-cosine log-likelihood via the elementary closed form (independent of cosine.logpdf):
 *  f(x) = (1/(2s))(1 + cos(pi*(x-mu)/s)) on [mu-s, mu+s], else 0. */
function llClosedForm(data: readonly number[], mu: number, s: number): number {
  let ll = 0
  for (const x of data) {
    ll += Math.log((1 / (2 * s)) * (1 + Math.cos((Math.PI * (x - mu)) / s)))
  }
  return ll
}

describe('cosine', () => {
  it('logpdf passes (mu, s) to @stdlib: matches the elementary closed form at an ASYMMETRIC point', () => {
    // CONVENTION GUARD (slot-blind unless asymmetric): mu=2, s=3, x=2.5 ⇒ (x-mu)/s = 1/6 ≠ 0, so a
    // mu/s swap or a dropped pi factor changes the result. f(x) = (1/(2s))(1 + cos(pi*(x-mu)/s)).
    // Computed BY HAND here (not via @stdlib) to catch exactly that slot/scale error.
    const expected = Math.log((1 / (2 * 3)) * (1 + Math.cos((Math.PI * 0.5) / 3)))
    expectClose(cosine.logpdf(2.5, { mu: 2, s: 3 }), expected, 1e-9)
  })
  it('cdf(mu) = 0.5 (mu is the symmetric center/median)', () =>
    expectClose(cosine.cdf(2, { mu: 2, s: 3 }), 0.5))
  it('quantile(0.5) = mu (median)', () => expectClose(cosine.quantile(0.5, { mu: 2, s: 3 }), 2))
  it('quantile inverts cdf (round-trip)', () => {
    const p = { mu: 2, s: 3 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(cosine.cdf(cosine.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit converges: s strictly contains all data, finite LL, round-trip holds', () => {
    const p = cosine.fit(sample) as CosineParams
    expect(Number.isFinite(p.mu)).toBe(true)
    expect(p.s).toBeGreaterThan(0)
    // HARD support barrier: the fitted [mu-s, mu+s] must contain every observation.
    expect(p.s).toBeGreaterThan(maxAbsDev(sample, p.mu))
    // Log-likelihood at the fit is finite (no point falls on/outside the support).
    const ll = llClosedForm(sample, p.mu, p.s)
    expect(Number.isFinite(ll)).toBe(true)
    // Self-consistency: cdf∘quantile round-trips at the fitted params.
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(cosine.cdf(cosine.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('fit climbs: LL at the fit >= LL at the widened seed (optimizer never worsens it)', () => {
    const p = cosine.fit(sample) as CosineParams
    // Reconstruct the load-bearing seed (mu0 = mean, s0 = max(MoM, 1.05*max|x-mu0|)) and assert the
    // coordinate-ascent fit does not decrease the log-likelihood relative to it.
    const mu0 = sample.reduce((a, b) => a + b, 0) / sample.length
    let varSum = 0
    for (const x of sample) varSum += (x - mu0) * (x - mu0)
    const sd = Math.sqrt(varSum / sample.length)
    const s0 = Math.max((Math.PI * sd) / MOM_SCALE_DENOM, SEED_WIDEN * maxAbsDev(sample, mu0))
    expect(llClosedForm(sample, p.mu, p.s)).toBeGreaterThanOrEqual(
      llClosedForm(sample, mu0, s0) - 1e-9,
    )
  })
  it('rejects degenerate (zero-spread) data', () =>
    expect(() => cosine.fit([7, 7, 7, 7])).toThrow())
  it('k = 2', () => expect(cosine.k).toBe(2))
  it("kind = 'continuous'", () => expect(cosine.kind).toBe('continuous'))
})
