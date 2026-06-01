import cdf from '@stdlib/stats-base-dists-geometric-cdf'
import logpmf from '@stdlib/stats-base-dists-geometric-logpmf'
import quantile from '@stdlib/stats-base-dists-geometric-quantile'
import { mean } from '../math'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 1-parameter geometric MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 1

/**
 * Geometric's fitted parameter: `p` = SUCCESS PROBABILITY in (0, 1]. HardFit uses the
 * {0, 1, 2, ...} convention — X counts the FAILURES before the first success — which is the
 * convention `@stdlib`'s geometric functions use (so `logpdf`/`cdf`/`quantile` pass `p` as-is,
 * no transform). THE MLE IS CONVENTION-SENSITIVE: for {0,1,...} it is p = 1/(1 + mean), NOT
 * 1/mean (that closed form belongs to the {1,2,...} "number of trials" convention and would
 * yield a wrong-but-finite p). Closed-form, unique, on the boundary all-zeros data gives
 * mean 0 -> p = 1 (a valid geometric, not degenerate).
 *
 * `FittedParams` is the engine-wide DTO (`Record<string, number>`) that crosses the
 * Comlink worker boundary, so it stays untyped. Inside this module we narrow `params` back
 * to the slots `@stdlib` expects with ONE assertion per density function — this is the
 * standard pattern every distribution copies. The cast is for READABILITY only (no
 * `?? Number.NaN` per arg) and to keep the `Distribution` interface and the registry
 * array generic-free. It does NOT validate slots at runtime: the assertion erases at
 * compile time, so a mismatched params object yields NaN — which `fitAll`'s
 * `Number.isFinite(ll)` guard turns into a reported failure rather than a crash.
 * Must be a `type` alias, not an `interface`: an interface lacks the implicit index
 * signature, so `params as GeometricParams` would not compile without an `as unknown` step.
 *
 * The argument is named `params` (not `p`) to avoid colliding with the math symbol `p`
 * — the success probability — that the destructure binds. `logpmf` carries log-PMF mass;
 * the `logpdf` interface name is shared with continuous distributions and `selection.logLik`
 * sums it correctly.
 */
type GeometricParams = { p: number }

// params: { p = success probability in (0,1] }  ({0,1,...} failures convention; @stdlib matches).
export const geometric: Distribution = {
  name: DistributionName.Geometric,
  label: 'Geometric',
  k: 1,
  kind: 'discrete',
  fit(data): GeometricParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`geometric: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => !Number.isInteger(v))) throw new Error('geometric requires integer counts')
    if (data.some((v) => v < 0)) throw new Error('geometric requires non-negative integer counts')
    const m = mean(data)
    if (!Number.isFinite(m)) throw new Error('geometric: degenerate (non-finite mean)')
    return { p: 1 / (1 + m) }
  },
  logpdf(x: number, params: FittedParams): number {
    const { p } = params as GeometricParams
    return logpmf(x, p)
  },
  cdf(x: number, params: FittedParams): number {
    const { p } = params as GeometricParams
    return cdf(x, p)
  },
  quantile(prob: number, params: FittedParams): number {
    const { p } = params as GeometricParams
    return quantile(prob, p)
  },
  support() {
    return { min: 0, max: Number.POSITIVE_INFINITY }
  },
}
