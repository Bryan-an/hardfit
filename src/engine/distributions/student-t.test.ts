import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { studentT } from './student-t'

/** Mirrors the (unexported) StudentTParams slot type inside student-t.ts; lets the test read
 *  fitted params by name. `studentT.fit` returns the engine-wide `FittedParams` DTO, so a
 *  narrowing cast is expected here. */
type StudentTParams = { loc: number; scale: number; df: number }

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]

/**
 * A real t(df=4) sample (loc=2, scale=1.5, n=40), drawn ONCE via scipy's PCG64
 * (np.random.default_rng(20260601)) and frozen as a literal — deterministic, no live RNG in
 * the test. scipy's own `t.fit` LL on this EXACT (4-decimal rounded) sample is
 * -86.52392365752318 (the reference MUST be computed on the rounded literal, not the unrounded
 * draws); HardFit must reach at least as good an LL.
 */
const T4_SAMPLE = [
  -0.1019, 1.2461, 1.2439, 4.0658, 2.612, 4.5575, 3.3829, 0.6069, 1.2957, 0.8232, 4.5387, 0.3783,
  2.0251, 1.7231, -1.7413, 2.96, 2.6653, -0.83, 3.4253, 1.5742, 3.4106, -4.6805, 4.3307, 0.7281,
  2.7685, -0.6852, -1.1186, 1.5438, -0.7507, 7.2746, 1.0788, 2.1441, 3.3805, 1.3999, 0.7818, 2.3696,
  3.7059, 0.7341, 5.0953, 3.1666,
]

/** Total Student-t log-likelihood at (loc, scale, df) via the public logpdf. */
function logLik(data: readonly number[], p: StudentTParams): number {
  let ll = 0
  for (const x of data) ll += studentT.logpdf(x, p)
  return ll
}

describe('student-t', () => {
  // CONVENTION GUARD: loc=1.5, scale=2, df=4 with an ASYMMETRIC point x=3.7 (so a (value, df)
  // transpose or a missing location-scale wrap is caught). References computed independently
  // against scipy.stats.t(4, loc=1.5, scale=2): VERIFIED to ~1e-15.
  const guard = { loc: 1.5, scale: 2, df: 4 }
  it('logpdf applies the location-scale wrap (− ln scale): matches the scipy reference', () => {
    expectClose(studentT.logpdf(3.7, guard), -2.334690170185134, 1e-9)
  })
  it('cdf applies the location-scale wrap: matches the scipy reference', () => {
    expectClose(studentT.cdf(3.7, guard), 0.833458175226223, 1e-9)
  })
  it('quantile applies loc + scale·t-quantile: matches the scipy reference (p=0.8 and p=0.83)', () => {
    // p=0.8 → 3.381929154470363 ; p=0.83 → 3.664621409078558 (both scipy.stats.t(4,1.5,2).ppf).
    expectClose(studentT.quantile(0.8, guard), 3.381929154470363, 1e-9)
    expectClose(studentT.quantile(0.83, guard), 3.664621409078558, 1e-9)
  })
  it('quantile inverts cdf (round-trip)', () => {
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(studentT.cdf(studentT.quantile(prob, guard), guard), prob, 1e-7)
    }
  })
  it('fit returns finite {loc, scale, df} with scale > 0 and df > 0', () => {
    const p = studentT.fit(T4_SAMPLE) as StudentTParams
    expect(Number.isFinite(p.loc)).toBe(true)
    expect(p.scale).toBeGreaterThan(0)
    expect(p.df).toBeGreaterThan(0)
  })
  it('fit LL is at least as good as scipy.t.fit on the same sample', () => {
    const p = studentT.fit(T4_SAMPLE) as StudentTParams
    // scipy.stats.t.fit(T4_SAMPLE) maximized LL on the EXACT rounded literal — HardFit must reach
    // it (minus a tiny slack). HardFit in fact reaches a marginally BETTER optimum than scipy here.
    const SCIPY_LL = -86.52392365752318
    expect(logLik(T4_SAMPLE, p)).toBeGreaterThanOrEqual(SCIPY_LL - 1e-6)
  })
  it('fit LL is at least as good as the LL at the (median, IQR, df=10) seed', () => {
    const p = studentT.fit(T4_SAMPLE) as StudentTParams
    const sorted = [...T4_SAMPLE].sort((a, b) => a - b)
    const idx = (q: number): number => sorted[Math.round((sorted.length - 1) * q)] ?? Number.NaN
    const median = idx(0.5)
    const iqr = idx(0.75) - idx(0.25)
    const seed = { loc: median, scale: iqr / 1.349, df: 10 }
    expect(logLik(T4_SAMPLE, p)).toBeGreaterThanOrEqual(logLik(T4_SAMPLE, seed) - 1e-9)
  })
  it('warm-started quick refit returns finite {loc, scale, df} close to the cold fit', () => {
    // The parametric bootstrap seeds a replicate refit from the original point estimate, which lets
    // the Nelder–Mead converge in a handful of iterations from near its optimum (skipping the cold
    // median/IQR/DF_SEED seed). A warm-started quick refit on the ORIGINAL data must land at
    // essentially the same optimum as the cold fit (within a few %).
    const cold = studentT.fit(T4_SAMPLE) as StudentTParams
    const warm = studentT.fit(T4_SAMPLE, { quick: true, warmStart: cold }) as StudentTParams
    const WARM_REL_TOL = 0.03
    expect(Number.isFinite(warm.loc)).toBe(true)
    expect(warm.scale).toBeGreaterThan(0)
    expect(warm.df).toBeGreaterThan(0)
    expect(Math.abs(warm.loc - cold.loc)).toBeLessThanOrEqual(
      WARM_REL_TOL * Math.max(Math.abs(cold.loc), 1),
    )
    expect(Math.abs(warm.scale - cold.scale)).toBeLessThanOrEqual(
      WARM_REL_TOL * Math.abs(cold.scale),
    )
    expect(Math.abs(warm.df - cold.df)).toBeLessThanOrEqual(WARM_REL_TOL * Math.abs(cold.df))
  })
  it('a non-finite/non-positive warmStart falls back to the cold robust seed', () => {
    // A degenerate warmStart (NaN loc, 0 scale, etc.) must NOT poison the fit: the predicate rejects
    // it and the existing cold median/IQR seed runs, so the fit still converges to the real optimum.
    const cold = studentT.fit(T4_SAMPLE) as StudentTParams
    const bad = studentT.fit(T4_SAMPLE, {
      quick: true,
      warmStart: { loc: Number.NaN, scale: 0, df: -1 },
    }) as StudentTParams
    expect(Number.isFinite(bad.loc)).toBe(true)
    expect(bad.scale).toBeGreaterThan(0)
    expect(bad.df).toBeGreaterThan(0)
    expect(logLik(T4_SAMPLE, bad)).toBeGreaterThanOrEqual(logLik(T4_SAMPLE, cold) - 1e-6)
  })
  it('rejects degenerate (zero-spread) data', () =>
    expect(() => studentT.fit([7, 7, 7, 7, 7])).toThrow())
  it('rejects a strict majority of identical values (unbounded MLE — mirrors cauchy)', () => {
    // 4 of 5 points tie ⇒ scale → 0, df → 0 drives the LL → +∞; without the guard student-t fits
    // {scale ≈ 4e-154, df ≈ 0.008} with LL ≈ +1390 and WINS the AICc ranking with a degenerate fit.
    // (Regression for a Batch D adversarial-verification finding; cf. cauchy's identical guard.)
    expect(() => studentT.fit([3, 3, 3, 3, 9])).toThrow(/majority/)
  })
  it('a warmStart does NOT bypass the majority-tie guard (guard precedence)', () => {
    // The unbounded-MLE guard MUST run before the warm-start seed is applied — otherwise a bootstrap
    // replicate that resampled into a degenerate majority-tie sample would silently fit a spurious
    // tiny-scale optimum. Passing a (valid) warmStart must NOT let that sample through.
    expect(() =>
      studentT.fit([3, 3, 3, 3, 9], { quick: true, warmStart: { loc: 3, scale: 1, df: 5 } }),
    ).toThrow(/majority/)
  })
  it('rejects too-small samples (n < 4)', () => expect(() => studentT.fit([1, 2, 3])).toThrow())
  it('k = 3', () => expect(studentT.k).toBe(3))
  it("kind = 'continuous'", () => expect(studentT.kind).toBe('continuous'))
})
