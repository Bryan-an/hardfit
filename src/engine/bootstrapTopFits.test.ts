import { describe, expect, it } from 'vitest'
import { bootstrapTopFits, deriveFitSeed } from './bootstrapTopFits'
import { BOOTSTRAP_SEED_SALT } from './constants'
import { DISTRIBUTIONS } from './distributions/index'
import { makeSampler } from './sampling'
import type { Distribution, FittedParams, RankedFit } from './types'

/** Two-sided CI level (95% CI). */
const ALPHA = 0.05
/** Master seed for the orchestrator (deterministic). */
const SEED = 4242
/** Modest B for test speed; production default is DEFAULT_BOOTSTRAP_B (999). */
const B = 80

/** Look up a registered distribution by name, throwing (rather than `!`) when absent. */
function requireDist(name: string): Distribution {
  const dist = DISTRIBUTIONS.find((d) => d.name === name)
  if (dist === undefined) throw new Error(`test setup: no distribution '${name}'`)
  return dist
}

/** Deterministic dataset drawn FROM `dist` at `params` via the seeded sampler. */
function simulate(dist: Distribution, params: FittedParams, n: number, seed: number): number[] {
  const draw = makeSampler(dist.name, params, seed)
  return Array.from({ length: n }, () => draw())
}

/** A minimal RankedFit stub: only `name`, `params`, and `rank` matter to the orchestrator. */
function rankedStub(name: string, params: FittedParams, rank: number): RankedFit {
  return {
    name,
    label: name,
    k: Object.keys(params).length,
    params,
    logLik: 0,
    aic: 0,
    aicc: 0,
    bic: 0,
    ks: 0,
    ad: { statistic: 0, pValue: null, method: 'diagnostic' },
    cvm: { statistic: 0, pValue: null, method: 'diagnostic' },
    chiSquared: { statistic: 0, pValue: null, method: 'diagnostic', df: 1, bins: 2 },
    rank,
    deltaAICc: 0,
    weight: 1,
  }
}

describe('deriveFitSeed', () => {
  it('maps (seed, index) to a positive uint32 via the named SALT', () => {
    const i = 0
    const raw = (SEED ^ (BOOTSTRAP_SEED_SALT * (i + 1))) >>> 0
    expect(deriveFitSeed(SEED, i)).toBe(raw === 0 ? 1 : raw)
  })

  it('never returns 0 (mt19937 rejects seed 0) — the XOR-to-zero master maps to 1', () => {
    const zeroMaster = (BOOTSTRAP_SEED_SALT * 1) >>> 0 // seed ^ SALT === 0 at index 0
    expect(deriveFitSeed(zeroMaster, 0)).toBe(1)
  })

  it('yields distinct, decorrelated seeds across adjacent indices', () => {
    const seeds = [0, 1, 2].map((i) => deriveFitSeed(SEED, i))
    expect(new Set(seeds).size).toBe(3)
    // adjacent indices must differ by far more than 1 (not seed^0, seed^1, …)
    expect(Math.abs((seeds[0] ?? 0) - (seeds[1] ?? 0))).toBeGreaterThan(1000)
  })
})

describe('bootstrapTopFits', () => {
  it('returns results keyed by the top-k distribution names; progress reaches 1', async () => {
    const data = simulate(requireDist('normal'), { mu: 5, sigma: 2 }, 60, 777)
    // ranked order (best first): normal, exponential, gamma — orchestrator takes top 2.
    const ranked: RankedFit[] = [
      rankedStub('normal', { mu: 5, sigma: 2 }, 1),
      rankedStub('exponential', { rate: 0.2 }, 2),
      rankedStub('gamma', { shape: 3, rate: 0.6 }, 3),
    ]
    const fractions: number[] = []
    const result = await bootstrapTopFits(
      DISTRIBUTIONS,
      data,
      ranked,
      { topK: 2, B, alpha: ALPHA, seed: SEED },
      (f) => fractions.push(f),
    )

    expect(Object.keys(result).sort()).toEqual(['exponential', 'normal'])
    expect(result.normal?.gofPValues.ks).toBeGreaterThan(0)
    // overall progress monotone non-decreasing, ends at exactly 1.
    for (let i = 1; i < fractions.length; i++) {
      expect(fractions[i] ?? 0).toBeGreaterThanOrEqual(fractions[i - 1] ?? 0)
    }
    expect(fractions.at(-1)).toBe(1)
  })

  it('uses a distinct derived seed per fit (decorrelated streams)', async () => {
    const data = simulate(requireDist('normal'), { mu: 5, sigma: 2 }, 40, 777)
    const ranked: RankedFit[] = [
      rankedStub('normal', { mu: 5, sigma: 2 }, 1),
      rankedStub('exponential', { rate: 0.2 }, 2),
    ]
    const result = await bootstrapTopFits(DISTRIBUTIONS, data, ranked, {
      topK: 2,
      B,
      alpha: ALPHA,
      seed: SEED,
    })
    expect(result.normal?.seed).toBe(deriveFitSeed(SEED, 0))
    expect(result.exponential?.seed).toBe(deriveFitSeed(SEED, 1))
    expect(result.normal?.seed).not.toBe(result.exponential?.seed)
  })

  it('skips a ranked fit whose distribution is not registered', async () => {
    const data = simulate(requireDist('normal'), { mu: 5, sigma: 2 }, 40, 777)
    const ranked: RankedFit[] = [
      rankedStub('normal', { mu: 5, sigma: 2 }, 1),
      rankedStub('does-not-exist', { foo: 1 }, 2),
    ]
    const result = await bootstrapTopFits(DISTRIBUTIONS, data, ranked, {
      topK: 5, // ask for more than exist
      B,
      alpha: ALPHA,
      seed: SEED,
    })
    expect(Object.keys(result)).toEqual(['normal'])
  })

  it('defaults topK to BOOTSTRAP_TOP_K and is cancellable at a chunk boundary', async () => {
    const data = simulate(requireDist('normal'), { mu: 5, sigma: 2 }, 40, 777)
    const ranked: RankedFit[] = [rankedStub('normal', { mu: 5, sigma: 2 }, 1)]
    await expect(
      bootstrapTopFits(
        DISTRIBUTIONS,
        data,
        ranked,
        { B, alpha: ALPHA, seed: SEED },
        undefined,
        () => true,
      ),
    ).rejects.toThrow(/cancel/i)
  })

  it('returns an empty result and never divides by zero when no fit resolves', async () => {
    const data = simulate(requireDist('normal'), { mu: 5, sigma: 2 }, 40, 777)
    const fractions: number[] = []
    const result = await bootstrapTopFits(
      DISTRIBUTIONS,
      data,
      [rankedStub('does-not-exist', { foo: 1 }, 1)],
      { topK: 2, B, alpha: ALPHA, seed: SEED },
      (f) => fractions.push(f),
    )
    expect(result).toEqual({})
    for (const f of fractions) expect(Number.isNaN(f)).toBe(false)
  })
})
