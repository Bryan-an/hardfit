import digamma from '@stdlib/math-base-special-digamma'
import trigamma from '@stdlib/math-base-special-trigamma'
import cdf from '@stdlib/stats-base-dists-negative-binomial-cdf'
import logpmf from '@stdlib/stats-base-dists-negative-binomial-logpmf'
import quantile from '@stdlib/stats-base-dists-negative-binomial-quantile'
import { MAX_NEWTON_ITERATIONS, NEWTON_REL_TOL } from '../constants'
import { mean, populationVariance } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter negative-binomial MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2
/** Integer support floor: failures before the r-th success start at 0. */
const SUPPORT_MIN = 0
/** Newton damping factor applied to r on a non-positive/non-finite step. */
const DAMP_FACTOR = 2

/**
 * Negative-binomial's fitted parameters: `r` = SUCCESS COUNT (a positive REAL — `@stdlib`
 * accepts non-integer r) and `p` = SUCCESS PROBABILITY in (0,1]. Support is the failure
 * count {0,1,2,...} before the r-th success, so `@stdlib`'s `(x, r, p)` functions take both
 * slots directly (no reparameterization). THE TRAP: r and p must not be swapped — both live
 * in (0,1] for p but r is unbounded, so a swap is finite-but-wrong; the convention-guard test
 * compares `logpdf` against the elementary log-PMF written by hand to catch exactly that.
 *
 * `logpdf` delegates to `@stdlib`'s log-PMF (`-logpmf`): the `Distribution` interface method is
 * named `logpdf` for both continuous and discrete fits, and for a discrete law it carries log-
 * PMF mass — `selection.logLik` sums it correctly. `@stdlib`'s `logpmf` returns `-Infinity` at
 * non-integer or out-of-support x; `fit` guards integers explicitly for a clean error message.
 *
 * `FittedParams` is the engine-wide DTO (`Record<string, number>`) that crosses the
 * Comlink worker boundary, so it stays untyped. Inside this module we narrow `p` back
 * to the slots `@stdlib` expects with ONE assertion per density function — this is the
 * standard pattern every distribution copies. The cast is for READABILITY only (no
 * `?? Number.NaN` per arg) and to keep the `Distribution` interface and the registry
 * array generic-free. It does NOT validate slots at runtime: the assertion erases at
 * compile time, so a mismatched params object yields NaN — which `fitAll`'s
 * `Number.isFinite(ll)` guard turns into a reported failure rather than a crash.
 * Must be a `type` alias, not an `interface`: an interface lacks the implicit index
 * signature, so `p as NegativeBinomialParams` would not compile without an `as unknown` step.
 */
type NegativeBinomialParams = { r: number; p: number }

// params: { r = success count (positive real), p = success prob in (0,1] }.
// @stdlib negative-binomial uses (x, r, p) directly — no reparameterization.
// MLE profiles p out (p = r/(r + xbar)) and solves r by 1-D Newton on the profile score.
// The MLE is finite ONLY for overdispersed data (s2 > xbar); otherwise r -> ∞ (Poisson limit).
export const negativeBinomial: Distribution = {
  name: DistributionName.NegativeBinomial,
  label: 'Negative Binomial',
  k: 2,
  kind: 'discrete',
  fit(data): NegativeBinomialParams {
    if (data.length < MIN_SAMPLE_SIZE)
      throw new Error(`negative-binomial: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => !Number.isInteger(v)))
      throw new Error('negative-binomial requires integer counts')
    if (data.some((v) => v < SUPPORT_MIN)) throw new Error('negative-binomial requires all x >= 0')
    const xbar = mean(data)
    const s2 = populationVariance(data, xbar)
    // Finite MLE exists only when overdispersed; else r -> ∞ collapses to the Poisson limit.
    if (!(s2 > xbar)) throw new Error('negative-binomial: no finite MLE (data not overdispersed)')
    const n = data.length
    // Method-of-moments seed: r0 = xbar^2 / (s2 - xbar) (positive since s2 > xbar > 0).
    let r = (xbar * xbar) / (s2 - xbar)
    // Profile score g(r) and its derivative g'(r); g is decreasing -> a unique root.
    let converged = false
    for (let i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
      let g = n * Math.log(r / (r + xbar))
      let gp = n * (1 / r - 1 / (r + xbar))
      const digammaR = digamma(r)
      const trigammaR = trigamma(r)
      for (const x of data) {
        g += digamma(x + r) - digammaR
        gp += trigamma(x + r) - trigammaR
      }
      const step = g / gp
      const next = r - step
      if (!Number.isFinite(next) || next <= 0) {
        r = r / DAMP_FACTOR // damp toward positivity; loop continues
        continue
      }
      r = next
      if (Math.abs(step) < NEWTON_REL_TOL * r) {
        converged = true
        break
      }
    }
    if (!converged) throw new Error('negative-binomial: failed to converge')
    const p = r / (r + xbar)
    // Reject the Poisson-limit collapse: on extreme data a runaway Newton can satisfy the
    // scale-relative step test at an enormous r where p → 1 and the fit degenerates to a point mass
    // at 0 (log-likelihood −Infinity on any positive count). Require a finite total log-likelihood
    // so fit() never returns an unusable point — fitAll/bootstrap then treat it as a clean failure.
    let totalLogLik = 0
    for (const x of data) totalLogLik += logpmf(x, r, p)
    if (!Number.isFinite(totalLogLik)) {
      throw new Error('negative-binomial: no finite MLE (data too extreme)')
    }
    return { r, p }
  },
  logpdf(x: number, p: FittedParams): number {
    const { r, p: prob } = p as NegativeBinomialParams
    return logpmf(x, r, prob) // log-PMF mass (discrete); summed as the log-likelihood
  },
  cdf(x: number, p: FittedParams): number {
    const { r, p: prob } = p as NegativeBinomialParams
    return cdf(x, r, prob)
  },
  quantile(prob: number, p: FittedParams): number {
    const { r, p: success } = p as NegativeBinomialParams
    return quantile(prob, r, success)
  },
  support(): { min: number; max: number } {
    return { min: SUPPORT_MIN, max: Number.POSITIVE_INFINITY }
  },
}
