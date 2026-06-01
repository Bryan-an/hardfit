import { describe, expect, it } from 'vitest'
import { expectClose } from '../test/relClose'
import { aic, aicc, bic, logLik, rankByAICc } from './selection'

describe('selection', () => {
  it('logLik sums per-point log densities', () => {
    expectClose(
      logLik([0, 1, 2], (x) => -x),
      -(0 + 1 + 2),
    )
  })
  it('aic = 2k - 2LL', () => expectClose(aic(-10, 2), 2 * 2 - 2 * -10))
  it('aicc adds the small-sample correction', () => {
    // k=2, n=10: AIC + 2*2*3/(10-2-1) = AIC + 12/7
    expectClose(aicc(-10, 2, 10), aic(-10, 2) + 12 / 7)
  })
  it('aicc = +Infinity when n - k - 1 <= 0', () => {
    expect(aicc(-10, 2, 3)).toBe(Number.POSITIVE_INFINITY)
  })
  it('bic = k ln n - 2LL', () => expectClose(bic(-10, 2, 10), 2 * Math.log(10) - 2 * -10))
  it('rankByAICc sorts ascending, computes delta + Akaike weights summing to 1', () => {
    const fits = [
      { name: 'a', logLik: -20, k: 2 },
      { name: 'b', logLik: -10, k: 2 }, // best
      { name: 'c', logLik: -15, k: 1 },
    ]
    const ranked = rankByAICc(fits, 30)
    const best = ranked[0]
    if (!best) throw new Error('expected at least one ranked fit')
    expect(best.name).toBe('b')
    expect(best.rank).toBe(1)
    expectClose(best.deltaAICc, 0)
    const wsum = ranked.reduce((s, r) => s + r.weight, 0)
    expectClose(wsum, 1, 1e-12)
  })
})
