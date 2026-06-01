import tCdf from '@stdlib/stats-base-dists-t-cdf'
import tLogpdf from '@stdlib/stats-base-dists-t-logpdf'
import tQuantile from '@stdlib/stats-base-dists-t-quantile'
import { populationVariance, sortedQuantile } from '../math'
import { minimize } from '../optimize'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 3-parameter Student-t MLE can be estimated from. (AICc needs n ≥ 5 for a
 *  finite small-sample correction; at n=4 the t's AICc is +Infinity → it sorts last with weight 0,
 *  handled gracefully by `aicc()`. The fit itself is still well-posed at n=4.) */
const MIN_SAMPLE_SIZE = 4
/** Quartile probabilities for the robust median/IQR location-scale seeds. */
const MEDIAN_PROB = 0.5
const Q1_PROB = 0.25
const Q3_PROB = 0.75
/** Converts an IQR into a Gaussian-equivalent sigma (IQR of N(0,1) = 2·Φ⁻¹(0.75) ≈ 1.349).
 *  A robust, moment-free scale seed — Student-t with small df has heavy tails, so a raw sample
 *  std seed is unstable; the IQR/1.349 anchor is. */
const IQR_TO_SIGMA = 1.349
/** Initial degrees-of-freedom guess: 10 sits between near-normal (df→∞) and heavy-tailed (df→1),
 *  a neutral start for the 1-D search in ln df. */
const DF_SEED = 10
/** Index into the unconstrained parameter vector θ = [loc, ln scale, ln df]. */
const THETA_LOC = 0
const THETA_LOG_SCALE = 1
const THETA_LOG_DF = 2

/**
 * Student-t's fitted parameters: `loc` = location (ℝ), `scale` = SCALE (> 0, the spread of the
 * standard-t after location-scale), and `df` = degrees of freedom (> 0; df→∞ recovers the normal,
 * small df gives heavy tails). The standard t is loc=0/scale=1; this module does its own
 * location-scale wrap (mirroring `cauchy.ts`): `@stdlib`'s t functions take the STANDARD value/prob
 * and df only, so `logpdf`/`cdf`/`quantile` standardize `(x − loc)/scale` first and fold the
 * Jacobian `− ln scale` into the density by hand. Passing the raw `x` (skipping the wrap) yields a
 * wrong-but-finite value (no crash) — the convention-guard tests pin the location-scaled closed
 * form to catch exactly that.
 *
 * `@stdlib` t-CONVENTION (do not transpose): `tLogpdf(value, df)`, `tCdf(value, df)` take
 * (value, df); `tQuantile(prob, df)` takes (prob, df) — prob FIRST.
 *
 * `FittedParams` is the engine-wide DTO (`Record<string, number>`) that crosses the
 * Comlink worker boundary, so it stays untyped. Inside this module we narrow `p` back
 * to the slots the wrap expects with ONE assertion per density function — the standard
 * pattern every distribution copies. The cast is for READABILITY only; it erases at
 * compile time, so a mismatched params object yields NaN — which `fitAll`'s
 * `Number.isFinite(ll)` guard turns into a reported failure rather than a crash.
 * Must be a `type` alias, not an `interface` (an interface lacks the implicit index
 * signature, so `p as StudentTParams` would not compile without an `as unknown` step).
 */
type StudentTParams = { loc: number; scale: number; df: number }

/**
 * RAW standard-t log-density at the location-scaled point — the single building block of both the
 * public `logpdf` and the optimizer's negLL. Returns `tLogpdf((x − loc)/scale, df)` WITHOUT the
 * `− ln scale` Jacobian; callers add the Jacobian exactly once (the density adds it per-point, the
 * negLL adds `n·ln scale` once for the whole sample) so the scale term is never double-counted.
 */
function stdLogpdf(x: number, loc: number, scale: number, df: number): number {
  return tLogpdf((x - loc) / scale, df)
}

// params: { loc = location, scale = SCALE (> 0), df = degrees of freedom (> 0) }.
// @stdlib t is STANDARD (df only); logpdf/cdf/quantile do the location-scale wrap here.
export const studentT: Distribution = {
  name: DistributionName.StudentT,
  label: "Student's t",
  k: 3,
  kind: 'continuous',
  fit(data): StudentTParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`student-t: need n >= ${MIN_SAMPLE_SIZE}`)
    const sorted = [...data].sort((a, b) => a - b)
    const median = sortedQuantile(sorted, MEDIAN_PROB)
    const iqr = sortedQuantile(sorted, Q3_PROB) - sortedQuantile(sorted, Q1_PROB)
    // Robust scale seed: IQR/1.349; fall back to the sample std, then to 1, if the IQR collapses
    // (e.g. a heavy central tie). A non-positive final seed means zero spread → degenerate.
    let scale0 = iqr / IQR_TO_SIGMA
    if (!(scale0 > 0)) scale0 = Math.sqrt(populationVariance(data))
    if (!(scale0 > 0)) {
      throw new Error('student-t: degenerate (zero spread)')
    }
    const loc0 = median
    // Minimize the negLL over UNCONSTRAINED θ = [loc, ln scale, ln df] (so scale, df stay > 0 with
    // no boundary). The Jacobian for the scale reparameterization adds `n·θ[LOG_SCALE]` (= n·ln scale)
    // exactly ONCE — NOT folded into stdLogpdf, which is the raw standard-t density (no − ln scale).
    const n = data.length
    const nll = (theta: readonly number[]): number => {
      const loc = theta[THETA_LOC] ?? Number.NaN
      const logScale = theta[THETA_LOG_SCALE] ?? Number.NaN
      const scale = Math.exp(logScale)
      const df = Math.exp(theta[THETA_LOG_DF] ?? Number.NaN)
      let s = 0
      for (const x of data) s += stdLogpdf(x, loc, scale, df)
      return -s + n * logScale
    }
    const result = minimize(nll, [loc0, Math.log(scale0), Math.log(DF_SEED)])
    const loc = result.x[THETA_LOC] ?? Number.NaN
    const scale = Math.exp(result.x[THETA_LOG_SCALE] ?? Number.NaN)
    // Do NOT cap df: capping at a finite ceiling drops the LL below scipy on near-normal data
    // (df→∞ is the normal limit), failing the parity gate. The unconstrained ln df is left free.
    const df = Math.exp(result.x[THETA_LOG_DF] ?? Number.NaN)
    return { loc, scale, df }
  },
  logpdf(x: number, p: FittedParams): number {
    const { loc, scale, df } = p as StudentTParams
    return stdLogpdf(x, loc, scale, df) - Math.log(scale) // − ln scale = the location-scale Jacobian
  },
  cdf(x: number, p: FittedParams): number {
    const { loc, scale, df } = p as StudentTParams
    return tCdf((x - loc) / scale, df) // standardize first; @stdlib t is (value, df)
  },
  quantile(prob: number, p: FittedParams): number {
    const { loc, scale, df } = p as StudentTParams
    return loc + scale * tQuantile(prob, df) // @stdlib t-quantile is (prob, df) — prob FIRST
  },
}
