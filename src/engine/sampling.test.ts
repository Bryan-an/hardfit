import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { mean, populationVariance } from './math'
import { makeSampler } from './sampling'
import { DistributionName } from './types'

/** Master seed for the moment checks; the values are deterministic given the seed. */
const SEED = 20260601
/** Distinct seed for the reproducibility check (any fixed value works). */
const REPRO_SEED = 99
/** Large n so the empirical mean/variance concentrate near their closed forms. */
const SAMPLE_SIZE = 20000
/** Relative tolerance for the stochastic-but-seeded moment checks (~5% of the value). */
const MOMENT_RTOL = 0.05
/** First-k draws compared when asserting two same-seed samplers agree. */
const REPRO_DRAWS = 10

/** Draws `n` values from a sampler into an array (advances the one stream). */
function drawSample(sampler: () => number, n: number): number[] {
  return Array.from({ length: n }, () => sampler())
}

/** numpy.allclose-style relative check tailored to the noisy empirical moments. */
function expectNear(actual: number, expected: number): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(MOMENT_RTOL * Math.abs(expected))
}

describe('makeSampler: convention-guarded empirical moments', () => {
  it('exponential(rate): mean ≈ 1/rate, var ≈ 1/rate²', () => {
    const rate = 2
    const sample = drawSample(
      makeSampler(DistributionName.Exponential, { rate }, SEED),
      SAMPLE_SIZE,
    )
    expectNear(mean(sample), 1 / rate)
    expectNear(populationVariance(sample), 1 / (rate * rate))
  })

  it('gamma(shape, RATE): mean ≈ shape/rate, var ≈ shape/rate² (catches a rate/scale swap)', () => {
    const shape = 3
    const rate = 1.5
    const params = { shape, rate, scale: 1 / rate }
    const sample = drawSample(makeSampler(DistributionName.Gamma, params, SEED), SAMPLE_SIZE)
    expectNear(mean(sample), shape / rate)
    expectNear(populationVariance(sample), shape / (rate * rate))
  })

  it('weibull(shape, SCALE): mean ≈ scale·Γ(1+1/shape) (catches a scale/rate swap)', () => {
    const shape = 2
    const scale = 3
    const sample = drawSample(
      makeSampler(DistributionName.Weibull, { shape, scale }, SEED),
      SAMPLE_SIZE,
    )
    const expectedMean = scale * Math.exp(gammaln(1 + 1 / shape))
    expectNear(mean(sample), expectedMean)
  })

  it('normal(mu, sigma): mean ≈ mu, sd ≈ sigma', () => {
    const mu = 5
    const sigma = 2
    const sample = drawSample(
      makeSampler(DistributionName.Normal, { mu, sigma }, SEED),
      SAMPLE_SIZE,
    )
    expectNear(mean(sample), mu)
    expectNear(Math.sqrt(populationVariance(sample)), sigma)
  })

  it('lognormal(mu, sigma): mean ≈ exp(mu + sigma²/2)', () => {
    const mu = 0
    const sigma = 0.5
    const sample = drawSample(
      makeSampler(DistributionName.Lognormal, { mu, sigma }, SEED),
      SAMPLE_SIZE,
    )
    expectNear(mean(sample), Math.exp(mu + (sigma * sigma) / 2))
  })
})

describe('makeSampler: reproducibility + guards', () => {
  it('same seed → identical first draws', () => {
    const params = { shape: 3, rate: 1.5, scale: 1 / 1.5 }
    const a = makeSampler(DistributionName.Gamma, params, REPRO_SEED)
    const b = makeSampler(DistributionName.Gamma, params, REPRO_SEED)
    const drawsA = Array.from({ length: REPRO_DRAWS }, () => a())
    const drawsB = Array.from({ length: REPRO_DRAWS }, () => b())
    expect(drawsA).toEqual(drawsB)
  })

  it('throws for an unknown distribution name', () => {
    expect(() => makeSampler('pareto', { rate: 1 }, SEED)).toThrow(/no sampler/)
  })
})
