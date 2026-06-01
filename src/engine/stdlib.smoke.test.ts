import digamma from '@stdlib/math-base-special-digamma'
import trigamma from '@stdlib/math-base-special-trigamma'
import expCdf from '@stdlib/stats-base-dists-exponential-cdf'
import gammaCdf from '@stdlib/stats-base-dists-gamma-cdf'
import lognormalCdf from '@stdlib/stats-base-dists-lognormal-cdf'
import normalCdf from '@stdlib/stats-base-dists-normal-cdf'
import normalLogpdf from '@stdlib/stats-base-dists-normal-logpdf'
import weibullCdf from '@stdlib/stats-base-dists-weibull-cdf'
import { describe, it } from 'vitest'
import { expectClose } from '../test/relClose'

describe('@stdlib import + interop + parameter conventions', () => {
  it('normal cdf(0,0,1) = 0.5; logpdf finite', () => {
    expectClose(normalCdf(0, 0, 1), 0.5)
    expectClose(normalLogpdf(0, 0, 1), -0.5 * Math.log(2 * Math.PI))
  })
  it('exponential cdf uses RATE: cdf(x,lambda)=1-e^{-lambda x}', () => {
    expectClose(expCdf(1, 2), 1 - Math.exp(-2)) // lambda=2 (rate)
  })
  it('gamma cdf uses shape+RATE (alpha,beta): mean=alpha/beta', () => {
    // shape=2, rate=0.5 -> mean=4; cdf(4,2,0.5) ~ 0.59399415
    expectClose(gammaCdf(4, 2, 0.5), 0.5939941502901619, 1e-6)
  })
  it('weibull cdf uses shape+SCALE (k,lambda): F(lambda)=1-e^{-1}', () => {
    expectClose(weibullCdf(10, 2, 10), 1 - Math.exp(-1)) // x=scale -> 1-1/e
  })
  it('lognormal cdf params are on the LOG scale: F(1;0,1)=0.5', () => {
    expectClose(lognormalCdf(1, 0, 1), 0.5) // median = e^mu = 1
  })
  it('digamma/trigamma known values', () => {
    expectClose(digamma(1), -0.5772156649015329, 1e-9)
    expectClose(trigamma(1), (Math.PI * Math.PI) / 6, 1e-9)
  })
})
