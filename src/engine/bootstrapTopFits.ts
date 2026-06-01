import type { BootstrapFitResult } from './bootstrap'
import { bootstrapFit } from './bootstrap'
import {
  BOOTSTRAP_SEED_SALT,
  BOOTSTRAP_TOP_K,
  DEFAULT_BOOTSTRAP_B,
  DEFAULT_CI_ALPHA,
} from './constants'
import type { Distribution, RankedFit } from './types'

/**
 * Top-k parametric-bootstrap orchestration (M2.2 U4): run the FUSED {@link bootstrapFit}
 * for the best few ranked fits and assemble the per-distribution results. This module is
 * PURE (no Comlink/worker glue) so it is unit-testable; the worker calls it and supplies
 * the cooperative-cancel poll + the progress proxy.
 */

/** Knobs for {@link bootstrapTopFits}: which top fits to bootstrap and the shared `seed`. */
export interface BootstrapOptions {
  /** Number of top-ranked (lowest-AICc) fits to bootstrap. Default {@link BOOTSTRAP_TOP_K}. */
  topK?: number
  /** Replicates per fit. Default {@link DEFAULT_BOOTSTRAP_B}. */
  B?: number
  /** Two-sided CI miscoverage. Default {@link DEFAULT_CI_ALPHA}. */
  alpha?: number
  /** Master seed; each fit gets a distinct derived seed (see {@link deriveFitSeed}). */
  seed: number
}

/** Bootstrap results keyed by distribution name (the {@link bootstrapFit} return per fit). */
export type BootstrapResult = Record<string, BootstrapFitResult>

/**
 * Derive a distinct, reproducible sampler seed for the fit at `index` from the master
 * `seed`: mix in `BOOTSTRAP_SEED_SALT·(index+1)` so adjacent fits get far-apart (and thus
 * uncorrelated) streams, then coerce to a positive uint32 with `>>> 0`. mt19937 rejects
 * seed 0, so the rare XOR-to-zero case is bumped to 1.
 */
export function deriveFitSeed(seed: number, index: number): number {
  const derived = (seed ^ (BOOTSTRAP_SEED_SALT * (index + 1))) >>> 0
  return derived === 0 ? 1 : derived
}

/**
 * Bootstrap the top `opts.topK ?? BOOTSTRAP_TOP_K` of `ranked` (already AICc-sorted, best
 * first). Each ranked fit is matched to its {@link Distribution} by `name` (unmatched fits
 * are skipped), given a distinct derived seed, and run through {@link bootstrapFit}.
 *
 * Progress is the OVERALL fraction across the resolved fits: each fit's per-chunk fraction
 * `frac` maps to `(doneFits + frac)/k`, and a completion tick `doneFits/k` after each fit
 * guarantees the final value reaches exactly 1 ({@link bootstrapFit}'s own `onChunk` only
 * fires at chunk boundaries and never emits 1). Fits run SEQUENTIALLY so the running count
 * and the cooperative `isCancelled` poll both see a single fit in flight.
 *
 * `isCancelled` is polled by {@link bootstrapFit} at each chunk boundary; when it returns
 * true that call throws `BootstrapCancelledError`, which propagates out of here unchanged.
 */
export async function bootstrapTopFits(
  distributions: readonly Distribution[],
  data: readonly number[],
  ranked: readonly RankedFit[],
  opts: BootstrapOptions,
  onProgress?: (fraction: number) => void,
  isCancelled?: () => boolean,
): Promise<BootstrapResult> {
  const topK = opts.topK ?? BOOTSTRAP_TOP_K
  const B = opts.B ?? DEFAULT_BOOTSTRAP_B
  const alpha = opts.alpha ?? DEFAULT_CI_ALPHA

  // Resolve {dist, fit} pairs first (skip unregistered names) so `k` is the real count and
  // the overall-progress denominator can reach 1.
  const pairs: { dist: Distribution; fit: RankedFit }[] = []
  for (const fit of ranked.slice(0, topK)) {
    const dist = distributions.find((d) => d.name === fit.name)
    if (dist !== undefined) pairs.push({ dist, fit })
  }

  const result: BootstrapResult = {}
  const k = pairs.length
  if (k === 0) return result

  let doneFits = 0
  for (let i = 0; i < k; i++) {
    const { dist, fit } = pairs[i] as { dist: Distribution; fit: RankedFit }
    const fitProgress =
      onProgress !== undefined
        ? (frac: number): void => onProgress((doneFits + frac) / k)
        : undefined
    result[dist.name] = await bootstrapFit(dist, data, fit.params, {
      B,
      alpha,
      seed: deriveFitSeed(opts.seed, i),
      ...(fitProgress ? { onChunk: fitProgress } : {}),
      ...(isCancelled ? { isCancelled } : {}),
    })
    doneFits++
    onProgress?.(doneFits / k) // completion tick; the last iteration reports exactly 1.
  }

  return result
}
