import normalCdf from '@stdlib/stats-base-dists-normal-cdf'
import normalLogcdf from '@stdlib/stats-base-dists-normal-logcdf'
import { mean } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter Inverse Gaussian closed-form MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Standard normal mean/sd for the @stdlib normal cdf/logcdf calls (the IG CDF folds two Φ terms
 *  of the STANDARD normal). Named to avoid bare 0/1 literals in the density. */
const STD_NORMAL_MEAN = 0
const STD_NORMAL_SD = 1
/** Density/CDF algebra constants (no-magic-literals rule). HALF = the 1/2 in the log-density prefactor
 *  and in the (x−mu)²/(2·mu²·x) exponent; THREE = the −3·ln x term of the IG log-density. TWO has
 *  THREE distinct roles, all genuinely the integer 2: (1) the 2π density normalizer in the
 *  −ln(2π)/2 term of `igLogpdf` (`Math.log(TWO·Math.PI)`); (2) the 2·mu²·x denominator of that same
 *  log-density's exponent; (3) the 2λ/μ fold exponent in the overflow-safe CDF. */
const HALF = 0.5
const THREE = 3
const TWO = 2
/** Lower bracket for the quantile bisection: a tiny positive x (the IG support is x > 0; 0 itself has
 *  CDF 0 so it can never bracket a positive probability). */
const QUANTILE_LO = 1e-12
/** Initial upper bracket for the bisection, expanded geometrically until it covers the target prob —
 *  NEVER a hardcoded final hi (a fixed ceiling would fail for heavy-tailed large-mu/small-lambda IGs). */
const QUANTILE_HI_SEED = 1
/** Geometric growth factor while expanding the upper bracket (hi *= this until cdf(hi) >= prob). */
const QUANTILE_HI_GROWTH = 2
/** Hard cap on the bracket-expansion steps. A COUNTED for-loop, not a while(true): the same
 *  no-unbounded-loop rule the optimizer documents. 2^this overflows to +Inf long before binding,
 *  so on any valid prob < 1 the loop breaks far earlier; it only fires as a freeze tripwire. */
const QUANTILE_MAX_EXPANSIONS = 200
/** Bisection iteration cap. Bisection on [lo, hi] halves the interval each step, so this many steps
 *  resolve the root well below the 1e-9 equiprobable-edge parity tol (a counted for-loop). */
const QUANTILE_MAX_ITERS = 200
/** Relative-width stopping tol for the bisection: stop when (hi − lo) <= this · hi. Tight enough that
 *  the equiprobable-edge parity check (quantile(j/k) == scipy.ppf to 1e-9) clears with headroom. */
const QUANTILE_TOL = 1e-13
/** Probability clamps: p <= 0 maps to the lower support edge 0, p >= 1 to +Infinity (no finite root). */
const PROB_LO = 0
const PROB_HI = 1

/**
 * Inverse Gaussian (Wald) fitted parameters: `mu` = mean (> 0) and `lambda` = shape/concentration
 * (> 0; larger λ → tighter, more normal-looking). HardFit's `(mu, lambda)` is NOT scipy's
 * parameterization — scipy.stats.invgauss uses `(mu_s = mu/lambda, loc=0, scale=lambda)`, so the
 * Mode-B fixture freeze is `invgauss(mu/lambda, 0, lambda)` (THE TRAP). This module hand-composes the
 * density/CDF/quantile in HardFit's own `(mu, lambda)` convention, so no scipy remap leaks into the
 * engine; the parity gate pins the closed-form params to 1e-9 against the numpy-analytic reference.
 *
 * CDF — OVERFLOW-SAFE LOG FOLD (load-bearing): the closed form is
 *   F(x) = Φ(a·(x/μ−1)) + exp(2λ/μ)·Φ(−a·(x/μ+1)),   a = sqrt(λ/x).
 * The naive `Math.exp(2λ/μ) * Φ(−·)` is `Inf · 0 = NaN` at large λ (e.g. λ=400, μ=1 ⇒ exp(800)
 * overflows). The second term is computed in LOG space and exp'd ONCE:
 *   exp(2λ/μ + normalLogcdf(−a·(x/μ+1))),
 * which stays finite (the huge positive 2λ/μ is cancelled by the huge negative log-Φ before exp).
 *
 * `FittedParams` is the engine-wide DTO (`Record<string, number>`) that crosses the Comlink worker
 * boundary, so it stays untyped. Inside this module we narrow `p` back to the slots the density
 * expects with ONE assertion per function — the standard pattern every distribution copies. The cast
 * is for READABILITY only; it erases at compile time, so a mismatched params object yields NaN —
 * which `fitAll`'s `Number.isFinite(ll)` guard turns into a reported failure rather than a crash.
 * Must be a `type` alias, not an `interface` (an interface lacks the implicit index signature, so
 * `p as InverseGaussianParams` would not compile without an `as unknown` step).
 */
type InverseGaussianParams = { mu: number; lambda: number }

/** Hand-composed IG log-density at (x; mu, lambda); −Infinity outside the open support x > 0. The
 *  single building block of the public `logpdf` and the closed-form LL — VERIFIED to match
 *  `scipy.invgauss.logpdf` to machine precision. */
function igLogpdf(x: number, mu: number, lambda: number): number {
  if (!(x > 0)) return Number.NEGATIVE_INFINITY
  return (
    HALF * (Math.log(lambda) - Math.log(TWO * Math.PI) - THREE * Math.log(x)) -
    (lambda * (x - mu) * (x - mu)) / (TWO * mu * mu * x)
  )
}

/** Hand-composed IG CDF at (x; mu, lambda) via the overflow-safe log fold (see the type docblock).
 *  0 for x <= 0 (below the open support). Shared by the public `cdf` and the quantile bisection. */
function igCdf(x: number, mu: number, lambda: number): number {
  if (!(x > 0)) return PROB_LO
  const a = Math.sqrt(lambda / x)
  // First term: ordinary Φ — never overflows.
  const lower = normalCdf(a * (x / mu - 1), STD_NORMAL_MEAN, STD_NORMAL_SD)
  // Second term in LOG space: exp(2λ/μ + lnΦ(−·)). The 2λ/μ alone overflows; the log-cdf is large
  // and negative, so their sum stays finite and exp() is well-defined (the whole reason for the fold).
  const upper = Math.exp(
    (TWO * lambda) / mu + normalLogcdf(-a * (x / mu + 1), STD_NORMAL_MEAN, STD_NORMAL_SD),
  )
  return lower + upper
}

// params: { mu = mean (> 0), lambda = shape/concentration (> 0) }.
// Hand-composed in HardFit's (mu, lambda) convention; scipy uses (mu/lambda, loc=0, scale=lambda).
// CDF uses the overflow-safe log fold; quantile is a bracketed bisection on the monotone CDF.
export const inverseGaussian: Distribution = {
  name: DistributionName.InverseGaussian,
  label: 'Inverse Gaussian',
  k: 2,
  kind: 'continuous',
  fit(data): InverseGaussianParams {
    if (data.length < MIN_SAMPLE_SIZE) {
      throw new Error(`inverse-gaussian: need n >= ${MIN_SAMPLE_SIZE}`)
    }
    for (const x of data) {
      if (!(x > 0)) throw new Error('inverse-gaussian: data must be strictly positive (x > 0)')
    }
    // Closed-form MLE (no optimizer): mu = sample mean; lambda = n / (Σ(1/x) − n/mu). VERIFIED to
    // equal scipy.invgauss.fit(x, floc=0) remapped (lambda = scale, mu = mu_s·scale) to ~1e-9.
    const mu = mean(data)
    let sumReciprocal = 0
    for (const x of data) sumReciprocal += 1 / x
    const lambda = data.length / (sumReciprocal - data.length / mu)
    // Degenerate (all x equal): Σ(1/x) = n/mean exactly ⇒ denominator 0 ⇒ lambda = +Inf (or, with
    // round-off, a huge/negative value). The IG MLE is then unbounded — reject like cauchy's tie guard.
    if (!Number.isFinite(lambda) || lambda <= 0) {
      throw new Error('inverse-gaussian: degenerate (all values identical; MLE unbounded)')
    }
    return { mu, lambda }
  },
  logpdf(x: number, p: FittedParams): number {
    const { mu, lambda } = p as InverseGaussianParams
    return igLogpdf(x, mu, lambda)
  },
  cdf(x: number, p: FittedParams): number {
    const { mu, lambda } = p as InverseGaussianParams
    return igCdf(x, mu, lambda)
  },
  quantile(prob: number, p: FittedParams): number {
    const { mu, lambda } = p as InverseGaussianParams
    // The IG CDF has no elementary inverse → bracketed bisection on the monotone CDF. The body tests
    // (cdf(quantile(p)) ≈ p) and the parity equiprobable-edge check (quantile(j/k) == scipy.ppf to
    // 1e-9) are the cross-oracles for this branch.
    if (prob <= PROB_LO) return PROB_LO // lower support edge
    if (prob >= PROB_HI) return Number.POSITIVE_INFINITY // no finite root at p = 1
    let lo = QUANTILE_LO
    let hi = QUANTILE_HI_SEED
    // Expand the upper bracket geometrically until cdf(hi) >= prob — NEVER a hardcoded ceiling (a
    // fixed hi would fail for heavy-tailed large-mu/small-lambda IGs). Counted for-loop (no while(true)).
    for (let i = 0; i < QUANTILE_MAX_EXPANSIONS; i++) {
      if (igCdf(hi, mu, lambda) >= prob) break
      hi *= QUANTILE_HI_GROWTH
    }
    // Bisect [lo, hi] to a tight relative width. Counted for-loop; halving converges far inside the cap.
    for (let i = 0; i < QUANTILE_MAX_ITERS; i++) {
      const mid = HALF * (lo + hi)
      if (igCdf(mid, mu, lambda) < prob) lo = mid
      else hi = mid
      if (hi - lo <= QUANTILE_TOL * hi) break
    }
    return HALF * (lo + hi)
  },
}
