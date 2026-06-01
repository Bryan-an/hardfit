export function mean(x: readonly number[]): number {
  if (x.length === 0) throw new Error('mean: empty array')
  let s = 0
  for (const v of x) s += v
  return s / x.length
}

export function meanLog(x: readonly number[]): number {
  if (x.length === 0) throw new Error('meanLog: empty array')
  let s = 0
  for (const v of x) s += Math.log(v)
  return s / x.length
}

/** Population (MLE) variance: divides by n. */
export function populationVariance(x: readonly number[], mu = mean(x)): number {
  let s = 0
  for (const v of x) s += (v - mu) * (v - mu)
  return s / x.length
}

/** Linearly-interpolated sample quantile of an ASCENDING-sorted array (numpy 'linear' / R type-7):
 *  h = (n−1)p; q = sorted[⌊h⌋] + (h−⌊h⌋)·(sorted[⌈h⌉] − sorted[⌊h⌋]). Index reads are guarded for
 *  `noUncheckedIndexedAccess`; a malformed (empty) array yields NaN rather than throwing. */
export function sortedQuantile(sorted: readonly number[], prob: number): number {
  const h = (sorted.length - 1) * prob
  const lo = Math.floor(h)
  const hi = Math.ceil(h)
  const vLo = sorted[lo]
  const vHi = sorted[hi]
  if (vLo === undefined || vHi === undefined) return Number.NaN
  return vLo + (h - lo) * (vHi - vLo)
}
