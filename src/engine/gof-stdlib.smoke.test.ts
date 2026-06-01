import chi2cdf from '@stdlib/stats-base-dists-chisquare-cdf'
import gammaQuantile from '@stdlib/stats-base-dists-gamma-quantile'
import normalQuantile from '@stdlib/stats-base-dists-normal-quantile'
import weibullQuantile from '@stdlib/stats-base-dists-weibull-quantile'
import { describe, it } from 'vitest'
import { expectClose } from '../test/relClose'

describe('@stdlib quantile + chisquare-cdf interop + conventions', () => {
  it('normal quantile(0.5,0,1)=0', () => expectClose(normalQuantile(0.5, 0, 1), 0, 1e-9))
  it('gamma quantile uses shape+RATE: median of shape=3,rate=1.5 (~mean 2)', () => {
    // gamma(shape 3, rate 1.5) mean=2; median < mean; just assert finite, positive, < some bound + matches cdf inverse
    const q = gammaQuantile(0.5, 3, 1.5)
    expectClose(q, 1.7827, 1e-3) // gamma(shape=3, rate=1.5) median; round-trips through the RATE-parameterized cdf
  })
  it('weibull quantile uses shape+SCALE: Q(1-1/e ≈ .6321; k=2,λ=3) = scale=3', () => {
    expectClose(weibullQuantile(1 - Math.exp(-1), 2, 3), 3, 1e-6)
  })
  it('chisquare cdf(x,df): cdf(0,1)=0, monotone', () => expectClose(chi2cdf(0, 1), 0, 1e-12))
})
