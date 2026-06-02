import betaprimeLogpdf from '@stdlib/stats-base-dists-betaprime-logpdf'
import fCdf from '@stdlib/stats-base-dists-f-cdf'
import fQuantile from '@stdlib/stats-base-dists-f-quantile'
import {
  NM_BOOTSTRAP_F_TOL,
  NM_BOOTSTRAP_MAX_FUNCTION_EVALS,
  NM_BOOTSTRAP_MAX_ITERATIONS,
  NM_BOOTSTRAP_MAX_RESTARTS,
} from '../constants'
import { minimize } from '../optimize'
import { type Distribution, DistributionName, type FitOptions, type FittedParams } from '../types'

/** Relaxed Nelder–Mead options for a quick (bootstrap-replicate) refit; the full-precision primary
 *  fit passes `undefined` so `minimize` applies the parity-grade NM_* defaults instead. */
const QUICK_NM_OPTIONS = {
  maxIterations: NM_BOOTSTRAP_MAX_ITERATIONS,
  maxRestarts: NM_BOOTSTRAP_MAX_RESTARTS,
  fTol: NM_BOOTSTRAP_F_TOL,
  maxFunctionEvals: NM_BOOTSTRAP_MAX_FUNCTION_EVALS,
} as const

/** Fewest observations a 2-parameter Fisher–Snedecor F MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Divisor turning a degrees-of-freedom into its betaprime shape (d/2): the F density is a scaled
 *  betaprime with shapes (d1/2, d2/2). Named to avoid a bare 2 in the density. */
const DF_TO_BETAPRIME_SHAPE_DIVISOR = 2
/** Coarse log-spaced grid bounds for the (d1, d2) seed search. The MLE for moderate-df data sits
 *  well inside [0.5, 200]²; the grid finds the basin, then Nelder–Mead polishes it. */
const GRID_LO = 0.5
const GRID_HI = 200
/** Nodes per axis of the log-grid seed search (so GRID_STEPS² total LL evaluations). 8 log-spaced
 *  nodes over [0.5, 200] resolve the basin coarsely enough for the optimizer to take over. */
const GRID_STEPS = 8
/** Hard ceiling on a fitted df. Very large df pushes the F toward a point mass at 1 and lets the
 *  unconstrained optimizer run away in ln-space; this clamps the back-transform. On real moderate-df
 *  data the MLE is single/double-digit, so the cap never binds (and so never threatens the LL gate).
 *  This is the DELIBERATE OPPOSITE of Student-t's "do NOT cap df" — there df→∞ IS the normal limit
 *  and capping would drop the LL below scipy; here large df is pathological runaway, not a real fit. */
const DF_CAP = 1e6
/** Index into the unconstrained parameter vector θ = [ln d1, ln d2]. */
const THETA_LOG_D1 = 0
const THETA_LOG_D2 = 1
/** Neutral df=1 fallback for a seed coordinate. A finite-LL grid node overwrites it for any valid
 *  n>=2 strictly-positive sample (at least one of the GRID_STEPS² nodes is finite there), so it only
 *  reaches the optimizer in the degenerate all-non-finite-LL case — where it gives `minimize` a
 *  well-defined positive-df origin seed instead of a bare `1`. df=1 is chosen because `ln 1 = 0`, so
 *  the unconstrained θ = [ln d1, ln d2] seed becomes the clean origin [0, 0]. (Mirrors student-t's
 *  documented 'fall back to ... then to 1' seed.) */
const SEED_DF_FALLBACK = 1

/**
 * Fisher–Snedecor F's fitted parameters: `d1` = numerator degrees of freedom (> 0) and `d2` =
 * denominator degrees of freedom (> 0). HardFit pins scipy's `loc=0, scale=1`, so the F is the
 * standard two-parameter F on x > 0.
 *
 * DENSITY ROUTE (do NOT use `f-pdf` + ln): the F density composes from the already-vendored
 * `betaprime-logpdf` (which `f-logpdf` does not exist on npm, and `f-pdf` + `Math.log` underflows to
 * a spurious −Inf in the tail). With shapes (d1/2, d2/2):
 *   logpdf_F(x; d1, d2) = betaprimeLogpdf(d1·x/d2, d1/2, d2/2) + ln(d1/d2)
 * VERIFIED to match `scipy.f.logpdf` to ~1e-14. The F is ASYMMETRIC (`logpdf(2;5,12) ≠
 * logpdf(2;12,5)`, VERIFIED), so a (d1, d2) transpose changes the value — the convention-guard test
 * pins the d1≠d2 reference at x=2 to catch exactly that.
 *
 * `@stdlib` F-CONVENTION (do not transpose): `fCdf(x, d1, d2)`, `fQuantile(prob, d1, d2)` — d1 =
 * numerator df, d2 = denominator df; for the quantile, PROB FIRST.
 *
 * `FittedParams` is the engine-wide DTO (`Record<string, number>`) that crosses the Comlink worker
 * boundary, so it stays untyped. Inside this module we narrow `p` back to the slots the density
 * expects with ONE assertion per function — the standard pattern every distribution copies. The cast
 * is for READABILITY only; it erases at compile time, so a mismatched params object yields NaN —
 * which `fitAll`'s `Number.isFinite(ll)` guard turns into a reported failure rather than a crash.
 * Must be a `type` alias, not an `interface` (an interface lacks the implicit index signature, so
 * `p as FisherFParams` would not compile without an `as unknown` step).
 */
type FisherFParams = { d1: number; d2: number }

/** RAW F log-density at (x; d1, d2) via the betaprime route — the single building block of both the
 *  public `logpdf` and the optimizer's negLL, so the `ln(d1/d2)` constant is added in exactly one
 *  place. */
function fLogpdf(x: number, d1: number, d2: number): number {
  return (
    betaprimeLogpdf(
      (d1 * x) / d2,
      d1 / DF_TO_BETAPRIME_SHAPE_DIVISOR,
      d2 / DF_TO_BETAPRIME_SHAPE_DIVISOR,
    ) + Math.log(d1 / d2)
  )
}

/** Total F log-likelihood at (d1, d2). Used to pick the best log-grid seed node. */
function logLik(data: readonly number[], d1: number, d2: number): number {
  let ll = 0
  for (const x of data) ll += fLogpdf(x, d1, d2)
  return ll
}

// params: { d1 = numerator df (> 0), d2 = denominator df (> 0) }.
// @stdlib F is standard (loc=0, scale=1); density composes from betaprime-logpdf, cdf/quantile use
// the f-cdf / f-quantile slots (x/prob, d1, d2). F is asymmetric → never transpose d1 and d2.
export const fisherF: Distribution = {
  name: DistributionName.FisherF,
  label: 'Fisher–Snedecor F',
  k: 2,
  kind: 'continuous',
  expensiveFit: true, // 2-D Nelder–Mead MLE → bootstrap skips the O(n) BCa jackknife
  fit(data, opts?: FitOptions): FisherFParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`fisher-f: need n >= ${MIN_SAMPLE_SIZE}`)
    for (const x of data) {
      if (!(x > 0)) throw new Error('fisher-f: data must be strictly positive (x > 0)')
    }
    // WARM START (bootstrap replicate refits only): when the caller passes the original point estimate
    // with finite, strictly-positive (d1, d2), seed the optimizer there and SKIP the GRID_STEPS²
    // cold-seed scan entirely — a resample's MLE sits beside the original, so the grid (the cold-start
    // basin finder) is unneeded and is exactly the per-refit cost this avoids. A degenerate warmStart
    // (NaN/0/negative df) is rejected by the predicate so the cold log-grid below runs as before.
    let seedD1 = SEED_DF_FALLBACK
    let seedD2 = SEED_DF_FALLBACK
    const ws = opts?.warmStart as FisherFParams | undefined
    const warmStarted =
      ws !== undefined && Number.isFinite(ws.d1) && ws.d1 > 0 && Number.isFinite(ws.d2) && ws.d2 > 0
    if (warmStarted) {
      seedD1 = ws.d1
      seedD2 = ws.d2
    } else {
      // COARSE LOG-GRID SEED: the F likelihood in (d1, d2) is not convex and median/IQR give no useful
      // df anchor, so evaluate the full LL at GRID_STEPS² log-spaced nodes over [GRID_LO, GRID_HI]² and
      // keep the max-LL node. Log spacing matches the geometry of df (a multiplicative quantity).
      const logLo = Math.log(GRID_LO)
      const logHi = Math.log(GRID_HI)
      const step = (logHi - logLo) / (GRID_STEPS - 1)
      // The log-grid below overwrites the neutral df=1 fallback at the first finite-LL node (so the
      // fallback only survives the degenerate all-non-finite-LL case).
      let bestLL = Number.NEGATIVE_INFINITY
      for (let i = 0; i < GRID_STEPS; i++) {
        const d1 = Math.exp(logLo + i * step)
        for (let j = 0; j < GRID_STEPS; j++) {
          const d2 = Math.exp(logLo + j * step)
          const ll = logLik(data, d1, d2)
          if (Number.isFinite(ll) && ll > bestLL) {
            bestLL = ll
            seedD1 = d1
            seedD2 = d2
          }
        }
      }
    }
    // Minimize the negLL over UNCONSTRAINED θ = [ln d1, ln d2] (so d1, d2 stay > 0 with no boundary).
    const nll = (theta: readonly number[]): number => {
      const d1 = Math.exp(theta[THETA_LOG_D1] ?? Number.NaN)
      const d2 = Math.exp(theta[THETA_LOG_D2] ?? Number.NaN)
      return -logLik(data, d1, d2)
    }
    // Quick (bootstrap-replicate) refits use the relaxed caps; the primary/parity fit passes
    // `undefined` so `minimize` applies the full parity-grade NM_* defaults.
    const result = minimize(
      nll,
      [Math.log(seedD1), Math.log(seedD2)],
      opts?.quick ? QUICK_NM_OPTIONS : undefined,
    )
    // Back-transform with the runaway cap (see DF_CAP): df→∞ pushes the F toward a point mass at 1
    // and lets the unconstrained ln-search drift; clamp to a finite ceiling that never binds on real
    // moderate-df data.
    const d1 = Math.min(Math.exp(result.x[THETA_LOG_D1] ?? Number.NaN), DF_CAP)
    const d2 = Math.min(Math.exp(result.x[THETA_LOG_D2] ?? Number.NaN), DF_CAP)
    return { d1, d2 }
  },
  logpdf(x: number, p: FittedParams): number {
    const { d1, d2 } = p as FisherFParams
    return fLogpdf(x, d1, d2) // betaprime route + ln(d1/d2); never transpose d1, d2 (F is asymmetric)
  },
  cdf(x: number, p: FittedParams): number {
    const { d1, d2 } = p as FisherFParams
    return fCdf(x, d1, d2) // @stdlib f-cdf is (x, d1, d2)
  },
  quantile(prob: number, p: FittedParams): number {
    const { d1, d2 } = p as FisherFParams
    return fQuantile(prob, d1, d2) // @stdlib f-quantile is (prob, d1, d2) — prob FIRST
  },
}
