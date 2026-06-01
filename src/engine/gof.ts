import chi2cdf from '@stdlib/stats-base-dists-chisquare-cdf'

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

/**
 * Cramér–von Mises W² statistic against a fitted CDF (scipy's `n·ω²` normalization).
 * Sorted u_i = clamp(F(x_(i)), [0,1]); W² = 1/(12n) + Σ_{i=0..n−1}(u_i − (2(i+1)−1)/(2n))².
 * Like KS/AD, the raw statistic is a diagnostic when parameters are estimated from the same data.
 */
export function cramerVonMises(data: readonly number[], cdf: (x: number) => number): number {
  const x = [...data].sort((a, b) => a - b) // copy; never mutate caller's array
  const n = x.length
  if (n === 0) return Number.NaN
  let sum = 0
  for (const [i, value] of x.entries()) {
    const u = Math.min(1, Math.max(0, cdf(value))) // clamp F into [0,1]
    const d = u - (2 * (i + 1) - 1) / (2 * n)
    sum += d * d
  }
  return 1 / (12 * n) + sum
}

/** Min expected count per equiprobable chi-square bin; k is chosen so E_j = n/k ≥ this. */
const MIN_EXPECTED_PER_BIN = 5
/** Defensive cap on the discrete support enumeration. The upper tail `n·(1−cdf(v))` strictly
 *  decreases to 0, so the scan always terminates by dropping below MIN_EXPECTED_PER_BIN; this only
 *  guards a pathological cdf that never reaches 1. Far above any realistic count support. */
const MAX_DISCRETE_SCAN = 100_000

/**
 * Pearson chi-squared GoF for a CONTINUOUS fit using equiprobable bins from the fitted quantile.
 * Bin edges are the fitted quantiles Q(j/k) for j=1..k−1, with k = max(2, ⌊n/MIN_EXPECTED_PER_BIN⌋)
 * so the expected count per bin E_j = n/k ≥ MIN_EXPECTED_PER_BIN. Equiprobable quantile bins make
 * the test bounded-support-safe automatically: every edge is a real quantile, never out of range.
 * Binning is strict (x > edge), so a point landing exactly on an edge falls into the lower bin.
 *   X² = Σ_j (O_j − E_j)² / E_j;  df = k − 1 − nParams;  p = 1 − χ²cdf(X², df).
 * NOTE (Chernoff–Lehmann): like KS (Lilliefors) and AD, with parameters estimated from the same
 * data the χ²(k−1−p) reference distribution is anti-conservative for raw-data MLE (binned vs raw
 * estimation), so this p-value is APPROXIMATE. The rigorous p-value is the bootstrap (M2.2).
 */
export function chiSquaredGof(
  data: readonly number[],
  quantile: (prob: number) => number, // Q(p; θ̂) in data scale
  nParams: number,
): { statistic: number; df: number; bins: number; pValue: number } {
  const n = data.length
  const k = Math.max(2, Math.floor(n / MIN_EXPECTED_PER_BIN)) // equiprobable: E_j = n/k ≥ min
  const edges: number[] = []
  for (let j = 1; j < k; j++) edges.push(quantile(j / k))
  const observed = new Array<number>(k).fill(0)
  for (const x of data) {
    let b = 0
    for (const edge of edges) {
      if (x > edge) b++
      else break // edges are sorted ascending; first non-exceeded edge fixes the bin
    }
    const count = observed[b]
    if (count !== undefined) observed[b] = count + 1
  }
  const expected = n / k
  let statistic = 0
  for (const o of observed) {
    const d = o - expected
    statistic += (d * d) / expected
  }
  const df = k - 1 - nParams
  const pValue = df >= 1 ? 1 - chi2cdf(statistic, df) : Number.NaN
  return { statistic, df, bins: k, pValue }
}

/** One chi-square cell of a discrete fit: the inclusive integer range [lo, hi] (hi may be the
 *  support max, possibly +Infinity for the merged upper tail), with its observed + expected counts. */
export interface DiscreteChiSquaredCell {
  lo: number
  hi: number
  observed: number
  expected: number
}

/**
 * Pearson chi-squared GoF for a DISCRETE fit using PMF/count-based bins (the EDF tests KS/AD/CvM are
 * invalid under the ties a discrete law produces). Cells are built by walking the integer support
 * upward from `supportMin`, accumulating an open cell until BOTH its expected count `E = n·Σpmf(v)`
 * reaches MIN_EXPECTED_PER_BIN AND the remaining upper tail `n·(1−cdf(v))` is still ≥ the minimum
 * (so no sub-minimum tail is stranded). When the tail would drop below the minimum (or the finite
 * support ends), the open cell and all remaining mass collapse into ONE final cell `[openLo, max]`
 * whose expected is the exact remaining mass `n·(1−cdf(openLo−1))`. This greedy rule is fully
 * deterministic and is mirrored byte-for-byte by `chi_squared_binning_discrete` in gen_fixtures.py;
 * the parity gate pins the resulting cell edges + observed + expected, not just the statistic.
 *   X² = Σ_j (O_j − E_j)² / E_j;  df = k − 1 − nParams;  p = df ≥ 1 ? 1 − χ²cdf(X², df) : NaN.
 * NOTE (Chernoff–Lehmann): like the continuous χ², with parameters estimated from the same data the
 * χ²(k−1−p) reference is anti-conservative for raw-data MLE, so this p-value is APPROXIMATE.
 */
export function chiSquaredGofDiscrete(
  data: readonly number[],
  pmf: (x: number) => number, // P(X = v) = exp(logpmf)
  cdf: (x: number) => number, // F(v) = P(X ≤ v)
  supportMin: number,
  supportMax: number, // may be Number.POSITIVE_INFINITY (unbounded counts)
  nParams: number,
): { statistic: number; df: number; bins: number; pValue: number; cells: DiscreteChiSquaredCell[] } {
  const n = data.length
  // Observed count in the inclusive integer cell [lo, hi] (data are integers for a discrete fit).
  const countIn = (lo: number, hi: number): number => {
    let c = 0
    for (const x of data) if (x >= lo && x <= hi) c++
    return c
  }
  const cells: DiscreteChiSquaredCell[] = []
  let openLo = supportMin
  let accExpected = 0
  for (let v = supportMin; v <= supportMax && v < supportMin + MAX_DISCRETE_SCAN; v++) {
    accExpected += n * pmf(v)
    const remainingTail = n * (1 - cdf(v)) // expected mass strictly above v
    // Fold the open cell + all remaining mass into the final cell when the tail can no longer
    // sustain its own minimum-expected cell, or the (finite) support ends at v.
    if (remainingTail < MIN_EXPECTED_PER_BIN || v >= supportMax) break
    if (accExpected >= MIN_EXPECTED_PER_BIN) {
      cells.push({ lo: openLo, hi: v, observed: countIn(openLo, v), expected: accExpected })
      openLo = v + 1
      accExpected = 0
    }
  }
  // Final cell [openLo, supportMax]: exact remaining mass via the CDF (not the partial accumulator),
  // so it is correct whether the support is bounded (cdf(max)=1) or unbounded.
  cells.push({
    lo: openLo,
    hi: supportMax,
    observed: countIn(openLo, supportMax),
    expected: n * (1 - cdf(openLo - 1)),
  })
  let statistic = 0
  for (const cell of cells) {
    const d = cell.observed - cell.expected
    statistic += (d * d) / cell.expected
  }
  const k = cells.length
  const df = k - 1 - nParams
  const pValue = df >= 1 ? 1 - chi2cdf(statistic, df) : Number.NaN
  return { statistic, df, bins: k, pValue, cells }
}
