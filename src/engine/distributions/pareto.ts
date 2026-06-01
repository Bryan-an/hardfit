import cdf from '@stdlib/stats-base-dists-pareto-type1-cdf'
import logpdf from '@stdlib/stats-base-dists-pareto-type1-logpdf'
import quantile from '@stdlib/stats-base-dists-pareto-type1-quantile'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter Pareto (Type I) MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2

/**
 * Pareto (Type I) fitted parameters: `shape` (alpha = tail index) and `scale` (xm = the
 * minimum/lower bound). THE TRAP: `@stdlib`'s pareto-type1 functions take `(x, alpha=shape,
 * beta=SCALE=xm)` — `@stdlib` names the 2nd arg `beta`, but it is the SCALE (the lower bound xm),
 * passed DIRECTLY, NOT a rate (never 1/xm). So `logpdf`/`cdf`/`quantile` pass `scale` straight into
 * the beta slot. Passing a rate (1/xm) would silently yield a wrong-but-finite density (no crash,
 * self-consistency still holds) — the convention-guard test in pareto.test.ts compares against the
 * elementary scale-parameterized closed form to catch exactly that slot-swap.
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
 * signature, so `p as ParetoParams` would not compile without an `as unknown` step.
 */
type ParetoParams = { shape: number; scale: number }

// params: { shape = alpha (tail index), scale = xm (minimum/lower bound) }.
// @stdlib pareto-type1 uses (x, alpha=shape, beta=SCALE=xm) — logpdf/cdf/quantile pass p.scale
// directly into the beta slot, NEVER 1/xm (it is the scale = lower bound, not a rate).
export const pareto: Distribution = {
  name: DistributionName.Pareto,
  label: 'Pareto (Type I)',
  k: 2,
  kind: 'continuous',
  fit(data): ParetoParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`pareto: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('pareto requires all x > 0')
    // Closed-form MLE: xm = min(x); s = sum ln(x/xm); alpha = n / s.
    const xm = Math.min(...data)
    let s = 0
    for (const x of data) s += Math.log(x / xm)
    if (!(s > 0)) throw new Error('pareto: degenerate (all values equal)')
    const alpha = data.length / s
    return { shape: alpha, scale: xm }
  },
  logpdf(x: number, p: FittedParams): number {
    const { shape, scale } = p as ParetoParams
    return logpdf(x, shape, scale) // beta = SCALE = xm, not a rate
  },
  cdf(x: number, p: FittedParams): number {
    const { shape, scale } = p as ParetoParams
    return cdf(x, shape, scale) // beta = SCALE = xm, not a rate
  },
  quantile(prob: number, p: FittedParams): number {
    const { shape, scale } = p as ParetoParams
    return quantile(prob, shape, scale) // beta = SCALE = xm, same slot-trap as cdf
  },
}
