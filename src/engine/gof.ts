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
