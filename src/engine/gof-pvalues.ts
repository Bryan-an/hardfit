import type { GofResult, PValueMethod } from './types'
import { DistributionName } from './types'

/**
 * Per-distribution Anderson–Darling p-values via the CANONICAL correction route:
 * modify the RAW A² statistic UP to an adjusted A*², then compare A*² to FIXED published
 * critical values (closed-form polynomial for normal/lognormal, bracketed table lookup for
 * exponential/Weibull). This is deliberately NOT scipy's route (which instead divides the
 * critical values by the same factor) — mixing the two would double-correct.
 */

/**
 * D'Agostino & Stephens (1986), "Goodness-of-Fit Techniques", Table 4.7 — piecewise
 * approximation of the upper-tail p-value for the MODIFIED normal statistic A*²
 * (location & scale estimated). Valid over the full A*² range via four branches.
 */
export function adNormalPValue(aStar: number): number {
  const a = aStar
  if (a >= 0.6) return Math.exp(1.2937 - 5.709 * a + 0.0186 * a * a)
  if (a > 0.34) return Math.exp(0.9177 - 4.279 * a - 1.38 * a * a)
  if (a > 0.2) return 1 - Math.exp(-8.318 + 42.796 * a - 59.938 * a * a)
  return 1 - Math.exp(-13.436 + 101.14 * a - 223.73 * a * a)
}

/**
 * Stephens critical-value tables for the modified statistic A*², paired column-for-column:
 * `*_SIG[i]` is the significance level whose upper-tail critical value is `*_CRIT[i]`.
 * Criticals ascend as significance descends (more extreme A*² ⇒ smaller p).
 * Source: M.A. Stephens (1974/1986), tables for the EDF goodness-of-fit statistics.
 */
const EXP_SIG = [0.15, 0.1, 0.05, 0.025, 0.01]
const EXP_CRIT = [0.916, 1.062, 1.321, 1.591, 1.959] // exponential, scale estimated
// 2-parameter Weibull via its Gumbel/extreme-value equivalent — a FIXED table (NOT scipy's
// shape-dependent one, which targets the 3-parameter Weibull).
const GUMBEL_SIG = [0.25, 0.1, 0.05, 0.025, 0.01]
const GUMBEL_CRIT = [0.474, 0.637, 0.757, 0.877, 1.038]

/**
 * Bracketed p-value by linear interpolation of A*² within a Stephens critical-value table.
 * The table is ascending in `crit` and descending in `sig`. Clamping convention at the ends:
 *  - A*² ≤ smallest critical → p = the LARGEST significance boundary (e.g. ">0.15"/">0.25",
 *    reported as that numeric sig);
 *  - A*² ≥ largest critical → p = the SMALLEST significance boundary (e.g. "<0.01", reported
 *    as that numeric sig).
 * Always returns a finite number.
 */
function bracketedPValue(aStar: number, sig: readonly number[], crit: readonly number[]): number {
  const lastIdx = crit.length - 1
  const firstCrit = crit.at(0)
  const lastCrit = crit.at(lastIdx)
  const firstSig = sig.at(0)
  const lastSig = sig.at(lastIdx)
  // Guard the table shape; a malformed (empty/unequal) table yields a finite fallback.
  if (
    firstCrit === undefined ||
    lastCrit === undefined ||
    firstSig === undefined ||
    lastSig === undefined
  ) {
    return Number.NaN
  }
  if (aStar <= firstCrit) return firstSig // below the table → largest significance boundary
  if (aStar >= lastCrit) return lastSig // above the table → smallest significance boundary
  for (let i = 0; i < lastIdx; i++) {
    const lo = crit.at(i)
    const hi = crit.at(i + 1)
    const sigLo = sig.at(i)
    const sigHi = sig.at(i + 1)
    if (lo === undefined || hi === undefined || sigLo === undefined || sigHi === undefined) continue
    if (aStar >= lo && aStar <= hi) {
      const t = (aStar - lo) / (hi - lo)
      return sigLo + t * (sigHi - sigLo)
    }
  }
  return lastSig // unreachable given the bounds checks; keeps the return finite + total
}

const CLOSED_FORM: PValueMethod = 'closed-form'
const TABLE: PValueMethod = 'table'
const DIAGNOSTIC: PValueMethod = 'diagnostic'

/**
 * Anderson–Darling GoF result for a distribution from its RAW A² and sample size n.
 * Each branch applies the family's published adjustment to get A*², then maps A*² to a
 * p-value (closed-form for normal/lognormal, bracketed table for exponential/Weibull).
 * Gamma (and any unrecognised name) has no closed-form AD criticals → diagnostic/null
 * (the rigorous bootstrap p-value arrives in M2.2). `statistic` is always the raw A².
 */
export function adResult(name: string, a2: number, n: number): GofResult {
  switch (name) {
    case DistributionName.Normal:
    case DistributionName.Lognormal: {
      // Lognormal = the normal test on ln x; A² is log-invariant, so the same adjustment applies.
      const aStar = a2 * (1 + 0.75 / n + 2.25 / (n * n))
      return { statistic: a2, pValue: adNormalPValue(aStar), method: CLOSED_FORM }
    }
    case DistributionName.Exponential: {
      const aStar = a2 * (1 + 0.6 / n)
      return { statistic: a2, pValue: bracketedPValue(aStar, EXP_SIG, EXP_CRIT), method: TABLE }
    }
    case DistributionName.Weibull: {
      const aStar = a2 * (1 + 0.2 / Math.sqrt(n))
      return {
        statistic: a2,
        pValue: bracketedPValue(aStar, GUMBEL_SIG, GUMBEL_CRIT),
        method: TABLE,
      }
    }
    default: // gamma + anything else: no closed-form AD critical values yet
      return { statistic: a2, pValue: null, method: DIAGNOSTIC }
  }
}
