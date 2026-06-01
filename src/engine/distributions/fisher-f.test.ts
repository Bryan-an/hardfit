import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { fisherF } from './fisher-f'

/** Mirrors the (unexported) FisherFParams slot type inside fisher-f.ts; lets the test read
 *  fitted params by name. `fisherF.fit` returns the engine-wide `FittedParams` DTO, so a
 *  narrowing cast is expected here. */
type FisherFParams = { d1: number; d2: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

/**
 * A real F(d1=5, d2=12) sample (n=40), drawn ONCE via scipy's PCG64
 * (np.random.default_rng(20260601)) and frozen as a literal — deterministic, no live RNG in the
 * test. scipy's own `f.fit(arr, floc=0, fscale=1)` LL on this EXACT (4-decimal rounded) sample is
 * -43.99876190436338 (the reference MUST be computed on the rounded literal, not the unrounded
 * draws); HardFit must reach at least as good an LL.
 */
const F5_12_SAMPLE = [
  0.5543, 2.6359, 1.5275, 1.6237, 0.3258, 3.0493, 0.5354, 1.137, 0.3982, 1.049, 5.9368, 0.457,
  1.1215, 0.53, 0.4667, 2.029, 0.3826, 1.0985, 0.9875, 1.9698, 0.8879, 0.1112, 0.6789, 0.9741,
  0.5385, 0.4006, 3.0187, 0.9447, 0.6617, 1.0118, 0.6737, 1.1754, 0.6945, 1.976, 2.0883, 0.813,
  1.0106, 0.4779, 2.5255, 2.3879,
]

/** Total Fisher–Snedecor F log-likelihood at (d1, d2) via the public logpdf. */
function logLik(data: readonly number[], p: FisherFParams): number {
  let ll = 0
  for (const x of data) ll += fisherF.logpdf(x, p)
  return ll
}

describe('fisher-f', () => {
  // CONVENTION GUARD: d1=5, d2=12 at x=2 — the F is ASYMMETRIC (logpdf(2;5,12) ≠ logpdf(2;12,5),
  // VERIFIED), so a (d1, d2) swap would change the value. Reference computed independently against
  // scipy.stats.f.logpdf(2, 5, 12) (and the betaprime route matches it to ~1e-15).
  const guard = { d1: 5, d2: 12 }
  it('logpdf via the betaprime route matches the scipy reference (d1≠d2 at x=2 catches a swap)', () => {
    expectClose(fisherF.logpdf(2, guard), -1.8240127588464823, 1e-9)
  })
  it('cdf matches the scipy reference', () => {
    expectClose(fisherF.cdf(2, guard), 0.8490739209070631, 1e-9)
  })
  it('quantile matches the scipy reference (p=0.8)', () => {
    expectClose(fisherF.quantile(0.8, guard), 1.7403409916569361, 1e-9)
  })
  it('quantile inverts cdf (round-trip)', () => {
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(fisherF.cdf(fisherF.quantile(prob, guard), guard), prob, 1e-7)
    }
  })
  it('fit returns finite {d1, d2} with d1 > 0 and d2 > 0', () => {
    const p = fisherF.fit(F5_12_SAMPLE) as FisherFParams
    expect(p.d1).toBeGreaterThan(0)
    expect(p.d2).toBeGreaterThan(0)
  })
  it('fit LL is at least as good as scipy.f.fit on the same sample', () => {
    const p = fisherF.fit(F5_12_SAMPLE) as FisherFParams
    // scipy.stats.f.fit(arr, floc=0, fscale=1) maximized LL on the EXACT rounded literal — HardFit
    // must reach it (minus a tiny slack). scipy's f.fit under-converges, so HardFit often reaches a
    // marginally BETTER optimum here.
    const SCIPY_LL = -43.99876190436338
    expect(logLik(F5_12_SAMPLE, p)).toBeGreaterThanOrEqual(SCIPY_LL - 1e-6)
  })
  it('rejects non-positive data (x ≤ 0)', () =>
    expect(() => fisherF.fit([1, 2, -0.5, 3, 4])).toThrow())
  it('rejects too-small samples (n < 2)', () => expect(() => fisherF.fit([1])).toThrow())
  it('k = 2', () => expect(fisherF.k).toBe(2))
  it("kind = 'continuous'", () => expect(fisherF.kind).toBe('continuous'))
})
