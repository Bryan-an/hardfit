/**
 * One-sample Kolmogorov–Smirnov statistic D = max(D+, D-) against a fitted CDF.
 * NOTE: with parameters estimated from the same data, the standard KS p-value is INVALID
 * (Lilliefors). Use D as a diagnostic / comparison metric only.
 */
export function ksStatistic(data: readonly number[], cdf: (x: number) => number): number {
  const x = [...data].sort((a, b) => a - b) // copy; never mutate caller's array
  const n = x.length
  if (n === 0) return Number.NaN
  let dPlus = -Infinity
  let dMinus = -Infinity
  for (const [j, value] of x.entries()) {
    const f = Math.min(1, Math.max(0, cdf(value))) // clamp F into [0,1]
    dPlus = Math.max(dPlus, (j + 1) / n - f) // i/n - F(x_i),  i = j+1
    dMinus = Math.max(dMinus, f - j / n) // F(x_i) - (i-1)/n
  }
  return Math.max(0, dPlus, dMinus) // clamp FP noise
}

/** Open-interval clamp for the Anderson–Darling u_i: keeps ln(u) and ln(1−u) finite at the support edges. */
const AD_CLAMP_EPS = 1e-12

/**
 * Anderson–Darling A² statistic (raw) against a fitted CDF, weighting the tails more heavily than KS.
 * Sorted u_i = clamp(F(x_(i)), [EPS, 1−EPS]);
 *   A² = max(0, −n − (1/n)·Σ_{i=0..n−1}(2i+1)·[ln u_i + ln(1 − u_{n−1−i})]).
 * The reversed-index term is reindexed (j = n−1−i) to the algebraically identical forward form
 *   A² = max(0, −n − (1/n)·Σ_i [ (2i+1)·ln u_i + (2n−2i−1)·ln(1 − u_i) ])
 * so each u_i is touched once — no reversed lookups, matching ksStatistic's iteration style.
 * NOTE: like KS, with parameters estimated from the same data the standard A² p-value is invalid;
 * the per-distribution adjusted-statistic route (gof-pvalues) handles that. Use raw A² as a diagnostic here.
 */
export function adStatistic(data: readonly number[], cdf: (x: number) => number): number {
  const x = [...data].sort((a, b) => a - b) // copy; never mutate caller's array
  const n = x.length
  if (n === 0) return Number.NaN
  let s = 0
  for (const [i, value] of x.entries()) {
    const u = Math.min(1 - AD_CLAMP_EPS, Math.max(AD_CLAMP_EPS, cdf(value))) // open-interval clamp
    s += (2 * i + 1) * Math.log(u) + (2 * n - 2 * i - 1) * Math.log(1 - u)
  }
  return Math.max(0, -n - s / n) // clamp tiny negative FP noise
}
