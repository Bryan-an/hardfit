import gammaln from '@stdlib/math-base-special-gammaln'
import { describe, expect, it } from 'vitest'
import { DISTRIBUTIONS } from './distributions/index'
import { mean, populationVariance } from './math'
import { makeSampler } from './sampling'
import { DistributionName, type FittedParams } from './types'

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
/** Inverse-Gaussian MSH sampler statistical test (the ONLY check on this stream — every other IG
 *  path is pinned to scipy at 1e-9). The IG variance estimator is tail-sensitive (excess kurtosis
 *  ≈ 6), so use a larger n and a TIGHTER tol than the global 5% to actually distinguish an unbiased
 *  sampler from a biased-but-plausible one (a correlated-stream bug). Per the plan: N≈50k, ~2–3%. */
const IG_SAMPLE_SIZE = 50000
const IG_MOMENT_RTOL = 0.03

/** Draws `n` values from a sampler into an array (advances the one stream). */
function drawSample(sampler: () => number, n: number): number[] {
  return Array.from({ length: n }, () => sampler())
}

/** numpy.allclose-style relative check tailored to the noisy empirical moments. */
function expectNear(actual: number, expected: number): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(MOMENT_RTOL * Math.abs(expected))
}

/** Sample quantile (type-7 / linear interpolation, numpy's default) over a sorted copy. */
function sampleQuantile(sample: readonly number[], p: number): number {
  const sorted = [...sample].sort((a, b) => a - b)
  const idx = (sorted.length - 1) * p
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  const vLo = sorted[lo] ?? Number.NaN
  const vHi = sorted[hi] ?? Number.NaN
  return lo === hi ? vLo : vLo + (idx - lo) * (vHi - vLo)
}

/**
 * Batch A sampler convention guard. A sampler with a swapped/inverted parameter slot draws from
 * the WRONG distribution, so its empirical median + IQR diverge from the distribution's OWN
 * quantile function (which the scipy parity gate independently pins). Cross-checking sampler ↔
 * quantile catches the slot trap WITHOUT any hand-computed theoretical constant, and works for the
 * heavy-tailed families (pareto/cauchy/frechet) whose mean/variance are undefined or infinite.
 */
function expectSamplerMatchesQuantile(name: string, params: FittedParams): void {
  const dist = DISTRIBUTIONS.find((d) => d.name === name)
  if (dist === undefined) throw new Error(`no distribution "${name}"`)
  const sample = drawSample(makeSampler(name, params, SEED), SAMPLE_SIZE)
  const iqr = sampleQuantile(sample, 0.75) - sampleQuantile(sample, 0.25)
  expectNear(sampleQuantile(sample, 0.5), dist.quantile(0.5, params))
  expectNear(iqr, dist.quantile(0.75, params) - dist.quantile(0.25, params))
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

describe('makeSampler: Batch A sampler↔quantile convention guards', () => {
  it('uniform(a, b)', () => expectSamplerMatchesQuantile(DistributionName.Uniform, { a: 2, b: 8 }))
  it('rayleigh(sigma)', () => expectSamplerMatchesQuantile(DistributionName.Rayleigh, { sigma: 3 }))
  it('pareto(shape, SCALE=xm) — heavy tail, median/IQR not moments', () =>
    expectSamplerMatchesQuantile(DistributionName.Pareto, { shape: 3, scale: 2 }))
  it('laplace(mu, b)', () =>
    expectSamplerMatchesQuantile(DistributionName.Laplace, { mu: 4, b: 1.5 }))
  it('logistic(mu, s)', () =>
    expectSamplerMatchesQuantile(DistributionName.Logistic, { mu: 3, s: 1.2 }))
  it('gumbel(mu, beta) — MAX convention', () =>
    expectSamplerMatchesQuantile(DistributionName.Gumbel, { mu: 2, beta: 1.5 }))
  it('cauchy(x0, gamma) — moment-less, median/IQR', () =>
    expectSamplerMatchesQuantile(DistributionName.Cauchy, { x0: 5, gamma: 2 }))
  it('frechet(shape, SCALE=s) — heavy tail, median/IQR', () =>
    expectSamplerMatchesQuantile(DistributionName.Frechet, { shape: 3, scale: 2 }))
})

describe('makeSampler: Batch B sampler↔quantile convention guards', () => {
  it('levy(c) — moment-less (mu fixed 0), median/IQR', () =>
    expectSamplerMatchesQuantile(DistributionName.Levy, { c: 2 }))
  it('chisquare(df)', () => expectSamplerMatchesQuantile(DistributionName.ChiSquared, { df: 5 }))
  it('chi(k)', () => expectSamplerMatchesQuantile(DistributionName.Chi, { k: 4 }))
  it('invgamma(shape, SCALE=beta) — heavy tail, median/IQR', () =>
    expectSamplerMatchesQuantile(DistributionName.InvGamma, { shape: 4, scale: 3 }))
  it('betaprime(alpha, beta) — heavy tail, median/IQR', () =>
    expectSamplerMatchesQuantile(DistributionName.BetaPrime, { alpha: 5, beta: 4 }))
  it('cosine(mu, s)', () => expectSamplerMatchesQuantile(DistributionName.Cosine, { mu: 3, s: 2 }))
  it('beta(alpha, beta) — support (0,1)', () =>
    expectSamplerMatchesQuantile(DistributionName.Beta, { alpha: 2, beta: 5 }))
})

describe('makeSampler: Batch C discrete sampler convention guards (empirical moments)', () => {
  // Discrete quantiles are step functions, so the median/IQR cross-check is coarse; assert the
  // closed-form mean/variance of the integer draws instead, which catch a wrong slot/convention.
  it('poisson(lambda): mean ≈ var ≈ lambda', () => {
    const lambda = 3
    const s = drawSample(makeSampler(DistributionName.Poisson, { lambda }, SEED), SAMPLE_SIZE)
    expectNear(mean(s), lambda)
    expectNear(populationVariance(s), lambda)
  })
  it('geometric(p): mean ≈ (1-p)/p for the {0,1,…} failures convention', () => {
    const p = 0.4
    const s = drawSample(makeSampler(DistributionName.Geometric, { p }, SEED), SAMPLE_SIZE)
    expectNear(mean(s), (1 - p) / p) // 1.5 — NOT 1/p (that would be the {1,2,…} convention)
  })
  it('negative-binomial(r, p): mean ≈ r(1-p)/p (0-based failure counts)', () => {
    const r = 4
    const p = 0.4
    const s = drawSample(
      makeSampler(DistributionName.NegativeBinomial, { r, p }, SEED),
      SAMPLE_SIZE,
    )
    expectNear(mean(s), (r * (1 - p)) / p) // 6
  })
  it('discrete-uniform(a, b): mean ≈ (a+b)/2, var ≈ ((b-a+1)²-1)/12', () => {
    const a = 1
    const b = 6
    const s = drawSample(makeSampler(DistributionName.DiscreteUniform, { a, b }, SEED), SAMPLE_SIZE)
    expectNear(mean(s), (a + b) / 2) // 3.5
    expectNear(populationVariance(s), ((b - a + 1) ** 2 - 1) / 12) // 35/12 ≈ 2.917
  })
})

describe('makeSampler: Batch D sampler↔quantile convention guards', () => {
  it('student-t(loc, scale, df) — location-scaled standard t, median/IQR (catches loc/scale swap)', () =>
    expectSamplerMatchesQuantile(DistributionName.StudentT, { loc: 3, scale: 2, df: 5 }))
  it('fisher-f(d1, d2) — asymmetric positive support, median/IQR (catches a d1/d2 swap)', () =>
    expectSamplerMatchesQuantile(DistributionName.FisherF, { d1: 8, d2: 12 }))
  // Inverse-Gaussian gets a STATISTICAL test (not the median/IQR cross-check): the MSH sampler with
  // two CORRELATED streams (shared seed) would be biased-but-plausible — median/IQR could still look
  // right while the accept step is wrong. Mean ≈ mu and variance ≈ mu³/lambda catch a stream
  // correlation / fold bug. (mu=2, lambda=5 ⇒ mean 2, var 1.6.)
  it('inverse-gaussian(mu, lambda) — MSH sampler: mean ≈ mu, var ≈ mu³/lambda (catches stream bias)', () => {
    const mu = 2
    const lambda = 5
    const s = drawSample(
      makeSampler(DistributionName.InverseGaussian, { mu, lambda }, SEED),
      IG_SAMPLE_SIZE,
    )
    // Measured at this seed: mean 2.0032 (0.16% err), var 1.5997 (0.02% err) — both far inside the
    // 3% tol, so a correlated-stream bias (which would shift these well past 3%) is ruled out.
    expect(Math.abs(mean(s) - mu)).toBeLessThanOrEqual(IG_MOMENT_RTOL * mu)
    const expVar = (mu * mu * mu) / lambda
    expect(Math.abs(populationVariance(s) - expVar)).toBeLessThanOrEqual(IG_MOMENT_RTOL * expVar)
  })
  // Nakagami draws √(Gamma) on the squared scale, so its DEFINING moment is E[x²] = Ω. A rate/scale
  // swap in the gamma factory (passing Ω/m instead of RATE=m/Ω) would shift mean(x²) far off Ω; the
  // sampler↔quantile cross-check below ALSO catches a swap (median/IQR diverge), belt-and-suspenders.
  it('nakagami(m, Omega) — √Gamma sampler: mean(x²) ≈ Omega (catches a rate/scale swap)', () => {
    const m = 1.5
    const Omega = 4
    const s = drawSample(makeSampler(DistributionName.Nakagami, { m, Omega }, SEED), SAMPLE_SIZE)
    expectNear(mean(s.map((x) => x * x)), Omega)
  })
  it('nakagami(m, Omega) — sampler↔quantile median/IQR convention guard', () =>
    expectSamplerMatchesQuantile(DistributionName.Nakagami, { m: 1.5, Omega: 4 }))
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
    expect(() => makeSampler('nonexistent-distribution', { rate: 1 }, SEED)).toThrow(/no sampler/)
  })
})
