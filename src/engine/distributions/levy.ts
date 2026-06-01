import cdf from '@stdlib/stats-base-dists-levy-cdf'
import logpdf from '@stdlib/stats-base-dists-levy-logpdf'
import quantile from '@stdlib/stats-base-dists-levy-quantile'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 1-parameter levy MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 1
/** Fixed location parameter (mu). HardFit's Lévy is the 1-parameter family with origin at 0;
 *  `mu` is NOT estimated or stored, but @stdlib's levy functions take it as the 2nd arg, so
 *  every call passes this literal in the location slot (like frechet's `LOCATION`). */
const LEVY_LOCATION = 0

/**
 * Lévy's fitted parameter: `c` = SCALE (>0); the location mu is fixed at 0 and is NOT stored.
 * THE TRAP: `@stdlib`'s levy functions take `(x, mu=location, c=SCALE)` — `c` is the SCALE, NOT
 * a rate, so `logpdf`/`cdf`/`quantile` pass `c` directly and the literal `LEVY_LOCATION` (0) in
 * the mu slot. The MLE is closed-form and unique: with location pinned at 0, the score equation
 * gives the harmonic-mean form c = n / sum(1/x_i). Support is x > 0 STRICT (at x = 0, 1/x diverges
 * and 0 lies outside the OPEN support), hence the `<= 0` guard (NOT rayleigh's `< 0`).
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
 * signature, so `p as LevyParams` would not compile without an `as unknown` step.
 */
type LevyParams = { c: number }

// params: { c = SCALE }  (location mu fixed at 0, not stored; @stdlib takes c directly, no inversion).
// @stdlib levy uses (x, mu=location, c=SCALE) — logpdf/cdf/quantile pass p.c and LEVY_LOCATION.
export const levy: Distribution = {
  name: DistributionName.Levy,
  label: 'Lévy',
  k: 1,
  kind: 'continuous',
  fit(data): LevyParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`levy: need n >= ${MIN_SAMPLE_SIZE}`)
    if (data.some((v) => v <= 0)) throw new Error('levy requires all x > 0')
    let sumReciprocal = 0
    for (const v of data) sumReciprocal += 1 / v
    const c = data.length / sumReciprocal
    if (!(Number.isFinite(c) && c > 0)) throw new Error('levy: degenerate or non-finite scale')
    return { c }
  },
  logpdf(x: number, p: FittedParams): number {
    const { c } = p as LevyParams
    return logpdf(x, LEVY_LOCATION, c) // mu = 0, c = SCALE
  },
  cdf(x: number, p: FittedParams): number {
    const { c } = p as LevyParams
    return cdf(x, LEVY_LOCATION, c) // mu = 0, c = SCALE
  },
  quantile(prob: number, p: FittedParams): number {
    const { c } = p as LevyParams
    return quantile(prob, LEVY_LOCATION, c) // mu = 0, c = SCALE
  },
}
