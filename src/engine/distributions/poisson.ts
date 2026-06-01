import cdf from '@stdlib/stats-base-dists-poisson-cdf'
import logpmf from '@stdlib/stats-base-dists-poisson-logpmf'
import quantile from '@stdlib/stats-base-dists-poisson-quantile'
import { mean } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 1-parameter poisson MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 1
/** Smallest count in the poisson support (the non-negative integers). */
const SUPPORT_MIN = 0

/**
 * Poisson's fitted parameter: `lambda` = MEAN (the rate). `@stdlib`'s poisson functions take
 * `lambda` directly (NO inversion), so `logpdf`/`cdf`/`quantile` pass it as-is. The MLE is
 * closed-form and unique: lambda = mean(data).
 *
 * DISCRETE distribution: `kind` is `'discrete'` and the `Distribution.logpdf` method delegates
 * to `@stdlib`'s log-PMF (`logpmf`). The interface slot is still named `logpdf`, but it carries
 * log-PMF mass, which `selection.logLik` sums correctly. `support(p)` exposes the integer
 * support bounds `{ min: 0, max: +Infinity }` driving the PMF-binned chi-square GoF.
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
 * signature, so `p as PoissonParams` would not compile without an `as unknown` step.
 */
type PoissonParams = { lambda: number }

// params: { lambda = mean }  (RATE/MEAN convention; @stdlib takes lambda directly, no inversion).
export const poisson: Distribution = {
  name: DistributionName.Poisson,
  label: 'Poisson',
  k: 1,
  kind: 'discrete',
  fit(data): PoissonParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`poisson: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => !Number.isInteger(v))) throw new Error('poisson requires integer counts')
    if (data.some((v) => v < SUPPORT_MIN))
      throw new Error('poisson requires non-negative integer counts')
    const lambda = mean(data)
    if (!(lambda > 0)) throw new Error('poisson: degenerate (all zeros)')
    return { lambda }
  },
  logpdf(x: number, p: FittedParams): number {
    const { lambda } = p as PoissonParams
    return logpmf(x, lambda) // log-PMF: discrete mass, summed by selection.logLik
  },
  cdf(x: number, p: FittedParams): number {
    const { lambda } = p as PoissonParams
    return cdf(x, lambda)
  },
  quantile(prob: number, p: FittedParams): number {
    const { lambda } = p as PoissonParams
    return quantile(prob, lambda) // discrete: returns an integer count
  },
  support(_p: FittedParams): { min: number; max: number } {
    return { min: SUPPORT_MIN, max: Number.POSITIVE_INFINITY }
  },
}
