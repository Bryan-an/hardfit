import exponential from '@stdlib/random-base-exponential'
import gamma from '@stdlib/random-base-gamma'
import lognormal from '@stdlib/random-base-lognormal'
import normal from '@stdlib/random-base-normal'
import weibull from '@stdlib/random-base-weibull'
import { describe, expect, it } from 'vitest'
import { expectClose } from '../test/relClose'

/** Seed shared by every reproducibility check; any fixed value works since we only compare
 *  two generators built from the SAME seed (identical streams) — never against a golden draw. */
const SEED = 1234
/** First-k draws compared when asserting two same-seed generators agree. */
const REPRO_DRAWS = 5
/** Large n so the exponential sample mean concentrates near its closed form 1/rate. */
const SAMPLE_SIZE = 20000
/** Rate of the exponential smoke sample; closed-form mean = 1/EXP_RATE = 0.5. */
const EXP_RATE = 2
/** Relative tolerance for the stochastic-but-seeded sample mean (~5% of the value). */
const MEAN_RTOL = 0.05

/** Draws `n` values from a generator and returns their arithmetic mean. */
function sampleMean(draw: () => number, n: number): number {
  let sum = 0
  for (let i = 0; i < n; i++) sum += draw()
  return sum / n
}

/** Collects the first `k` draws of a generator (order-preserving). */
function firstDraws(draw: () => number, k: number): number[] {
  return Array.from({ length: k }, () => draw())
}

describe('@stdlib random-base samplers: factory(...params, { seed }) interop + reproducibility', () => {
  it('exponential.factory takes the RATE: large-n mean ≈ 1/rate', () => {
    const draw = exponential.factory(EXP_RATE, { seed: SEED })
    expectClose(sampleMean(draw, SAMPLE_SIZE), 1 / EXP_RATE, MEAN_RTOL)
  })

  it('same seed → identical first draws (normal)', () => {
    const a = normal.factory(0, 1, { seed: SEED })
    const b = normal.factory(0, 1, { seed: SEED })
    expect(firstDraws(a, REPRO_DRAWS)).toEqual(firstDraws(b, REPRO_DRAWS))
  })

  it('same seed → identical first draws (lognormal)', () => {
    const a = lognormal.factory(0, 1, { seed: SEED })
    const b = lognormal.factory(0, 1, { seed: SEED })
    expect(firstDraws(a, REPRO_DRAWS)).toEqual(firstDraws(b, REPRO_DRAWS))
  })

  it('same seed → identical first draws (exponential)', () => {
    const a = exponential.factory(EXP_RATE, { seed: SEED })
    const b = exponential.factory(EXP_RATE, { seed: SEED })
    expect(firstDraws(a, REPRO_DRAWS)).toEqual(firstDraws(b, REPRO_DRAWS))
  })

  it('same seed → identical first draws (gamma)', () => {
    const a = gamma.factory(3, 1.5, { seed: SEED })
    const b = gamma.factory(3, 1.5, { seed: SEED })
    expect(firstDraws(a, REPRO_DRAWS)).toEqual(firstDraws(b, REPRO_DRAWS))
  })

  it('same seed → identical first draws (weibull)', () => {
    const a = weibull.factory(2, 3, { seed: SEED })
    const b = weibull.factory(2, 3, { seed: SEED })
    expect(firstDraws(a, REPRO_DRAWS)).toEqual(firstDraws(b, REPRO_DRAWS))
  })
})
