import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { mean, populationVariance } from '../math'
import { logistic } from './logistic'

/** Mirrors the (unexported) LogisticParams slot type used inside logistic.ts; lets the test read
 *  fitted params by name without `?? NaN` noise. `logistic.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type LogisticParams = { mu: number; s: number }

const sample = [1.2, 3.4, 2.1, 5.6, 4.0, 2.8, 3.3, 1.9, 4.7, 3.0, 2.5, 4.2, 1.7, 3.8, 2.9]

/** Probabilities the quantile↔cdf round-trip is checked at (away from the 0/1 tails). */
const ROUND_TRIP_PROBS = [0.1, 0.5, 0.9]
/** Method-of-moments seed scale factor: s0 = sd(x)·√3/π. */
const MOM_SCALE_FACTOR = Math.sqrt(3) / Math.PI

/** Total log-likelihood of the sample under (mu, s). */
function logLik(p: LogisticParams): number {
  let ll = 0
  for (const x of sample) ll += logistic.logpdf(x, p)
  return ll
}

describe('logistic', () => {
  it('logpdf passes SCALE to @stdlib: matches the elementary closed form', () => {
    // CONVENTION GUARD: f(x) = e^{-z} / (s·(1+e^{-z})²), z=(x-mu)/s. For mu=0,s=1,x=0: f(0)=1/4.
    expectClose(logistic.logpdf(0, { mu: 0, s: 1 }), -Math.log(4), 1e-9)
    // A non-trivial point computed by hand from the same closed form (mu=2, s=1.5, x=3.5).
    const mu = 2
    const s = 1.5
    const x = 3.5
    const z = (x - mu) / s
    const expected = -z - Math.log(s) - 2 * Math.log(1 + Math.exp(-z))
    expectClose(logistic.logpdf(x, { mu, s }), expected, 1e-9)
  })
  it('cdf(mu) = 0.5', () => expectClose(logistic.cdf(5, { mu: 5, s: 2 }), 0.5))
  it('quantile(0.5) = mu (median)', () => expectClose(logistic.quantile(0.5, { mu: 5, s: 2 }), 5))
  it('quantile inverts cdf (round-trip)', () => {
    const p = { mu: 5, s: 2 }
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(logistic.cdf(logistic.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('MLE converges to finite params that maximize the likelihood vs the MoM seed', () => {
    const p = logistic.fit(sample) as LogisticParams
    expect(Number.isFinite(p.mu)).toBe(true)
    expect(Number.isFinite(p.s)).toBe(true)
    expect(p.s).toBeGreaterThan(0)
    // The fitted LL must be at least the LL at the method-of-moments seed (it maximizes).
    const seed: LogisticParams = {
      mu: mean(sample),
      s: Math.sqrt(populationVariance(sample)) * MOM_SCALE_FACTOR,
    }
    expect(logLik(p)).toBeGreaterThanOrEqual(logLik(seed) - 1e-9)
  })
  it('MLE satisfies the score equations Σ(2F-1)=0 and Σz(2F-1)=n', () => {
    const p = logistic.fit(sample) as LogisticParams
    let a = 0
    let b = 0
    for (const x of sample) {
      const z = (x - p.mu) / p.s
      const f = 1 / (1 + Math.exp(-z))
      a += 2 * f - 1
      b += z * (2 * f - 1)
    }
    expectClose(a, 0, 1e-7, 1e-7)
    expectClose(b, sample.length, 1e-7, 1e-7)
  })
  it('fitted params round-trip through quantile/cdf', () => {
    const p = logistic.fit(sample) as LogisticParams
    for (const prob of ROUND_TRIP_PROBS) {
      expectClose(logistic.cdf(logistic.quantile(prob, p), p), prob, 1e-7)
    }
  })
  it('rejects degenerate (zero variance) data', () =>
    expect(() => logistic.fit([3, 3, 3, 3])).toThrow('logistic: degenerate (zero variance)'))
  it('k = 2', () => expect(logistic.k).toBe(2))
  it("kind = 'continuous'", () => expect(logistic.kind).toBe('continuous'))
})
