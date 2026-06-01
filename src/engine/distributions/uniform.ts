import cdf from '@stdlib/stats-base-dists-uniform-cdf'
import logpdf from '@stdlib/stats-base-dists-uniform-logpdf'
import quantile from '@stdlib/stats-base-dists-uniform-quantile'
import { type Distribution, DistributionName, type FittedParams } from '../types'

/** Fewest observations a 2-parameter uniform MLE can be estimated from. */
const MIN_SAMPLE_SIZE = 2

/**
 * Uniform's fitted parameters: `a` = lower bound, `b` = upper bound. `b` is the absolute
 * MAXIMUM of the support, NOT a width — `@stdlib`'s uniform functions take `(x, a, b)` with
 * `a`/`b` as the endpoints, so `logpdf`/`cdf`/`quantile` pass `a`,`b` directly. No rate/scale
 * trap here, but the convention-guard test in uniform.test.ts still compares against the
 * elementary density 1/(b-a) to catch a wrong @stdlib arg slot.
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
 * signature, so `p as UniformParams` would not compile without an `as unknown` step.
 */
type UniformParams = { a: number; b: number }

// params: { a = lower bound, b = upper bound }  (b is the absolute MAX, not a width).
export const uniform: Distribution = {
  name: DistributionName.Uniform,
  label: 'Uniform (continuous)',
  k: 2,
  kind: 'continuous',
  fit(data): UniformParams {
    if (data.length < MIN_SAMPLE_SIZE) throw new Error(`uniform: need n >= ${MIN_SAMPLE_SIZE}`)
    // Single pass for the support bounds. NOT Math.min/max(...data): argument spread overflows the
    // call stack on large arrays (RangeError at ~100k+); the sibling distributions use this loop.
    let a = data[0] ?? Number.NaN
    let b = a
    for (const x of data) {
      if (x < a) a = x
      if (x > b) b = x
    }
    // Require finite bounds with positive width: catches all-equal data, NaN inputs, AND a ±Infinity
    // bound (Infinity > a is true, so the bare `b > a` check would let a degenerate fit through).
    if (!(Number.isFinite(a) && Number.isFinite(b) && b > a)) {
      throw new Error('uniform: degenerate or non-finite support (need finite max > min)')
    }
    return { a, b }
  },
  logpdf(x: number, p: FittedParams): number {
    const { a, b } = p as UniformParams
    return logpdf(x, a, b)
  },
  cdf(x: number, p: FittedParams): number {
    const { a, b } = p as UniformParams
    return cdf(x, a, b)
  },
  quantile(prob: number, p: FittedParams): number {
    const { a, b } = p as UniformParams
    return quantile(prob, a, b)
  },
}
