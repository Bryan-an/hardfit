import { describe, expect, it } from 'vitest'
import { expectClose } from '../test/relClose'
import { mean, meanLog, populationVariance, sortedQuantile } from './math'

describe('math helpers', () => {
  it('mean', () => expectClose(mean([2, 4, 6]), 4))
  it('populationVariance divides by n (MLE), not n-1', () => {
    // values 2,4,6: mean 4, sum sq dev = 4+0+4 = 8, /3 = 2.6667
    expectClose(populationVariance([2, 4, 6]), 8 / 3)
  })
  it('meanLog', () => expectClose(meanLog([1, Math.E]), 0.5))
  it('mean throws on empty', () => expect(() => mean([])).toThrow())
})

describe('sortedQuantile (numpy linear / R type-7 interpolation)', () => {
  // n=4, h = 3*prob. median: h=1.5 -> sorted[1] + 0.5*(sorted[2]-sorted[1]) = 2 + 0.5 = 2.5
  it('median of [1,2,3,4] is 2.5', () => expect(sortedQuantile([1, 2, 3, 4], 0.5)).toBe(2.5))
  // h=0.75 -> sorted[0] + 0.75*(sorted[1]-sorted[0]) = 1 + 0.75 = 1.75
  it('Q1 of [1,2,3,4] is 1.75', () => expect(sortedQuantile([1, 2, 3, 4], 0.25)).toBe(1.75))
  // h=2.25 -> sorted[2] + 0.25*(sorted[3]-sorted[2]) = 3 + 0.25 = 3.25
  it('Q3 of [1,2,3,4] is 3.25', () => expect(sortedQuantile([1, 2, 3, 4], 0.75)).toBe(3.25))
  it('empty array yields NaN (malformed input, not a throw)', () =>
    expect(Number.isNaN(sortedQuantile([], 0.5))).toBe(true))
  it('single element returns that element', () => expect(sortedQuantile([7], 0.5)).toBe(7))
})
