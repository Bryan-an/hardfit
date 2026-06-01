import { describe, expect, it } from 'vitest'
import { expectClose } from '../test/relClose'
import { mean, meanLog, populationVariance } from './math'

describe('math helpers', () => {
  it('mean', () => expectClose(mean([2, 4, 6]), 4))
  it('populationVariance divides by n (MLE), not n-1', () => {
    // values 2,4,6: mean 4, sum sq dev = 4+0+4 = 8, /3 = 2.6667
    expectClose(populationVariance([2, 4, 6]), 8 / 3)
  })
  it('meanLog', () => expectClose(meanLog([1, Math.E]), 0.5))
  it('mean throws on empty', () => expect(() => mean([])).toThrow())
})
