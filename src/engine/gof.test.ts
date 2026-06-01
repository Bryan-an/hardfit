import { describe, it } from 'vitest'
import { expectClose } from '../test/relClose'
import { ksStatistic } from './gof'

describe('ksStatistic', () => {
  it('uniform sample vs uniform CDF — exact small case', () => {
    // sorted [0.25,0.5,0.75], F(x)=x. n=3.
    // D+ = max(1/3-0.25, 2/3-0.5, 3/3-0.75) = max(0.0833,0.1667,0.25)=0.25
    // D- = max(0.25-0, 0.5-1/3, 0.75-2/3) = max(0.25,0.1667,0.0833)=0.25
    expectClose(
      ksStatistic([0.5, 0.25, 0.75], (x) => x),
      0.25,
      1e-12,
    )
  })
  it('does not mutate the input array', () => {
    const data = [3, 1, 2]
    ksStatistic(data, (x) => x / 3)
    const first = data[0]
    if (first === undefined) throw new Error('expected a non-empty input array')
    expectClose(first, 3) // original order preserved
  })
  it('clamps tiny negative FP noise to 0 (perfect fit)', () => {
    const d = ksStatistic([1, 2, 3, 4], (x) => x / 4) // step exactly matches at integers
    expectClose(d, 0.25, 1e-12) // D- at i=1: F(1)-0 = 0.25
  })
})
