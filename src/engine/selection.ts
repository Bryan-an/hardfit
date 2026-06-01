export function logLik(data: readonly number[], logpdf: (x: number) => number): number {
  let ll = 0
  for (const x of data) ll += logpdf(x) // sum in log space; never log(product)
  return ll
}

export function aic(ll: number, k: number): number {
  return 2 * k - 2 * ll
}

export function aicc(ll: number, k: number, n: number): number {
  const denom = n - k - 1
  if (denom <= 0) return Number.POSITIVE_INFINITY // correction undefined -> sorts last, weight 0
  return aic(ll, k) + (2 * k * (k + 1)) / denom
}

export function bic(ll: number, k: number, n: number): number {
  return k * Math.log(n) - 2 * ll // Math.log is ln
}

export interface Rankable {
  name: string
  logLik: number
  k: number
}
export interface Ranked {
  rank: number
  aicc: number
  deltaAICc: number
  weight: number
}

export function rankByAICc<T extends Rankable>(items: readonly T[], n: number): (T & Ranked)[] {
  const scored = items.map((it) => ({ it, aiccVal: aicc(it.logLik, it.k, n) }))
  const minAICc = Math.min(...scored.map((s) => s.aiccVal))
  // delta form IS the numerically stable form: max(-delta/2)=0 -> exp=1, no overflow.
  const withRel = scored.map((s) => {
    const delta = s.aiccVal - minAICc
    return { ...s, delta, rel: Math.exp(-delta / 2) } // exp(-Inf)=0, safe
  })
  const sumRel = withRel.reduce((acc, s) => acc + s.rel, 0)
  return withRel
    .map((s) => ({
      ...s.it,
      aicc: s.aiccVal,
      deltaAICc: s.delta,
      weight: sumRel > 0 ? s.rel / sumRel : 0,
      rank: 0,
    }))
    .sort((a, b) => a.aicc - b.aicc)
    .map((r, i) => ({ ...r, rank: i + 1 }))
}
