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
