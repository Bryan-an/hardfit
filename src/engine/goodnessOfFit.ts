import { adStatistic, chiSquaredGof, cramerVonMises, ksStatistic } from './gof'
import { adResult } from './gof-pvalues'
import type { ChiSquaredResult, Distribution, Fit, FittedParams, PValueMethod } from './types'

/** Method labels reused from the `PValueMethod` union (no inline string literals). */
const DIAGNOSTIC: PValueMethod = 'diagnostic'
const TABLE: PValueMethod = 'table'

/** The goodness-of-fit battery slice of a `Fit`. Tying the return type to `Fit` keeps the
 *  `fitAll` spread exact: if `Fit`'s GoF fields drift, this function fails to compile here. */
export type GoodnessOfFit = Pick<Fit, 'ks' | 'ad' | 'cvm' | 'chiSquared'>

/**
 * Map the raw `chiSquaredGof` result into a `ChiSquaredResult`. The raw `pValue` is `NaN`
 * when `df < 1` (no valid reference distribution); `ChiSquaredResult.pValue` is `number | null`,
 * so a NaN becomes `null`. Method is `'table'` (Pearson χ² against the χ²(df) table) — the
 * Chernoff–Lehmann caveat (anti-conservative for raw-data MLE; rigorous p-value is the M2.2
 * bootstrap) is documented on `chiSquaredGof` itself.
 */
function toChiSquaredResult(raw: ReturnType<typeof chiSquaredGof>): ChiSquaredResult {
  return {
    statistic: raw.statistic,
    pValue: Number.isNaN(raw.pValue) ? null : raw.pValue,
    method: TABLE,
    df: raw.df,
    bins: raw.bins,
  }
}

/**
 * Compute the goodness-of-fit battery for one fitted distribution, routing by `dist.kind`.
 *
 * CONTINUOUS — the EDF battery against the fitted CDF plus quantile-binned χ²:
 *   - `ks`  = raw Kolmogorov–Smirnov D (diagnostic; Lilliefors invalidates the standard p-value);
 *   - `ad`  = raw Anderson–Darling A² with the per-family adjusted-statistic p-value (gof-pvalues);
 *   - `cvm` = raw Cramér–von Mises W² (diagnostic-only; bootstrap p-value arrives in M2.2);
 *   - `chiSquared` = Pearson χ² over equiprobable quantile bins (df-guarded p-value → table/null).
 *
 * DISCRETE — EDF tests (KS/AD/CvM) are invalid under the ties a discrete law produces, so the
 * battery is χ²-only over PMF-based bins. That binning helper lands in M2.3 Batch C; until then
 * this branch throws. The `kind` fork itself is the routing structure M2.3 just fills in.
 */
export function goodnessOfFit(
  dist: Distribution,
  data: readonly number[],
  params: FittedParams,
): GoodnessOfFit {
  if (dist.kind === 'discrete') {
    throw new Error(
      `discrete GoF not yet implemented for '${dist.name}' (PMF-binned χ² lands in M2.3 Batch C)`,
    )
  }
  const cdf = (x: number): number => dist.cdf(x, params)
  const quantile = (prob: number): number => dist.quantile(prob, params)
  return {
    ks: ksStatistic(data, cdf),
    ad: adResult(dist.name, adStatistic(data, cdf), data.length),
    cvm: { statistic: cramerVonMises(data, cdf), pValue: null, method: DIAGNOSTIC },
    chiSquared: toChiSquaredResult(chiSquaredGof(data, quantile, dist.k)),
  }
}
