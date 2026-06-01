import betaSampler from '@stdlib/random-base-beta'
import betaprimeSampler from '@stdlib/random-base-betaprime'
import cauchySampler from '@stdlib/random-base-cauchy'
import chiSampler from '@stdlib/random-base-chi'
import chisquareSampler from '@stdlib/random-base-chisquare'
import cosineSampler from '@stdlib/random-base-cosine'
import exponentialSampler from '@stdlib/random-base-exponential'
import frechetSampler from '@stdlib/random-base-frechet'
import gammaSampler from '@stdlib/random-base-gamma'
import gumbelSampler from '@stdlib/random-base-gumbel'
import invgammaSampler from '@stdlib/random-base-invgamma'
import laplaceSampler from '@stdlib/random-base-laplace'
import levySampler from '@stdlib/random-base-levy'
import logisticSampler from '@stdlib/random-base-logistic'
import lognormalSampler from '@stdlib/random-base-lognormal'
import normalSampler from '@stdlib/random-base-normal'
import paretoSampler from '@stdlib/random-base-pareto-type1'
import rayleighSampler from '@stdlib/random-base-rayleigh'
import uniformSampler from '@stdlib/random-base-uniform'
import weibullSampler from '@stdlib/random-base-weibull'
import { DistributionName, type FittedParams } from './types'

/**
 * Per-distribution slot types mirroring the (unexported) param shapes inside each
 * `distributions/*.ts`. As with the densities, `FittedParams` is the engine-wide DTO
 * (`Record<string, number>`), so we narrow `p` back to the slots each `@stdlib` factory
 * expects with ONE `p as XParams` assertion per case — the standard zero-warning pattern
 * the densities use (no `as number` per argument). The assertion is for readability only;
 * it erases at compile time and does not validate slots at runtime.
 */
type NormalParams = { mu: number; sigma: number }
type LognormalParams = { mu: number; sigma: number }
type ExponentialParams = { rate: number }
type GammaParams = { shape: number; rate: number }
type WeibullParams = { shape: number; scale: number }
// M2.3 Batch A — same convention as each distribution's own module + density slots.
type UniformParams = { a: number; b: number }
type RayleighParams = { sigma: number }
type ParetoParams = { shape: number; scale: number } // alpha=shape, beta=scale=xm
type LaplaceParams = { mu: number; b: number }
type LogisticParams = { mu: number; s: number }
type GumbelParams = { mu: number; beta: number }
type CauchyParams = { x0: number; gamma: number }
type FrechetParams = { shape: number; scale: number } // alpha=shape, s=scale, m=0
// M2.3 Batch B — same convention as each distribution's own module + density slots.
type LevyParams = { c: number } // location fixed at 0; c = scale
type ChiSquaredParams = { df: number }
type ChiParams = { k: number }
type InvGammaParams = { shape: number; scale: number } // alpha=shape, beta=scale (direct, not a rate)
type BetaPrimeParams = { alpha: number; beta: number }
type CosineParams = { mu: number; s: number }
type BetaParams = { alpha: number; beta: number }

/** Fréchet's location is fixed at 0 in HardFit (a 2-parameter Fréchet); the sampler's 3rd
 *  positional arg is that location `m`. Named to avoid a bare 0 literal in the factory call. */
const FRECHET_LOCATION = 0
/** Lévy's location is fixed at 0 in HardFit (a 1-parameter Lévy); the sampler's 1st positional
 *  arg is that location `mu`. Named to avoid a bare 0 literal in the factory call. */
const LEVY_LOCATION = 0

/**
 * Builds a seeded iid sampler for one distribution in HardFit's param convention. Each
 * `@stdlib` factory returns a STATEFUL generator: seed once, then call the returned
 * function repeatedly to advance a single reproducible stream (never rebuild the factory
 * mid-stream — that would replay identical draws). Same seed → identical draws.
 *
 * THE SLOT-TRAP (matches the M1 densities 1:1): gamma takes shape + RATE (`p.rate`, NOT
 * `p.scale`); weibull takes shape + SCALE (`p.scale`, NOT a rate); exponential takes the
 * RATE; normal/lognormal take (mu, sigma) — lognormal's on the log scale. The
 * convention-guard tests in sampling.test.ts compare large-n empirical moments against the
 * closed forms to catch any rate/scale swap.
 */
export function makeSampler(name: string, p: FittedParams, seed: number): () => number {
  switch (name) {
    case DistributionName.Normal: {
      const { mu, sigma } = p as NormalParams
      return normalSampler.factory(mu, sigma, { seed })
    }
    case DistributionName.Lognormal: {
      const { mu, sigma } = p as LognormalParams
      return lognormalSampler.factory(mu, sigma, { seed }) // (mu, sigma) on the LOG scale
    }
    case DistributionName.Exponential: {
      const { rate } = p as ExponentialParams
      return exponentialSampler.factory(rate, { seed }) // RATE
    }
    case DistributionName.Gamma: {
      const { shape, rate } = p as GammaParams
      return gammaSampler.factory(shape, rate, { seed }) // shape, RATE (not scale)
    }
    case DistributionName.Weibull: {
      const { shape, scale } = p as WeibullParams
      return weibullSampler.factory(shape, scale, { seed }) // shape, SCALE (not rate)
    }
    case DistributionName.Uniform: {
      const { a, b } = p as UniformParams
      return uniformSampler.factory(a, b, { seed }) // (a, b) support bounds
    }
    case DistributionName.Rayleigh: {
      const { sigma } = p as RayleighParams
      return rayleighSampler.factory(sigma, { seed }) // SCALE sigma
    }
    case DistributionName.Pareto: {
      const { shape, scale } = p as ParetoParams
      return paretoSampler.factory(shape, scale, { seed }) // alpha=shape, beta=SCALE=xm
    }
    case DistributionName.Laplace: {
      const { mu, b } = p as LaplaceParams
      return laplaceSampler.factory(mu, b, { seed }) // (mu, b=SCALE)
    }
    case DistributionName.Logistic: {
      const { mu, s } = p as LogisticParams
      return logisticSampler.factory(mu, s, { seed }) // (mu, s=SCALE)
    }
    case DistributionName.Gumbel: {
      const { mu, beta } = p as GumbelParams
      return gumbelSampler.factory(mu, beta, { seed }) // (mu, beta=SCALE); MAX convention
    }
    case DistributionName.Cauchy: {
      const { x0, gamma } = p as CauchyParams
      return cauchySampler.factory(x0, gamma, { seed }) // (x0=location, gamma=SCALE)
    }
    case DistributionName.Frechet: {
      const { shape, scale } = p as FrechetParams
      return frechetSampler.factory(shape, scale, FRECHET_LOCATION, { seed }) // alpha, s=SCALE, m=0
    }
    case DistributionName.Levy: {
      const { c } = p as LevyParams
      return levySampler.factory(LEVY_LOCATION, c, { seed }) // (mu=0, c=SCALE)
    }
    case DistributionName.ChiSquared: {
      const { df } = p as ChiSquaredParams
      return chisquareSampler.factory(df, { seed }) // k = degrees of freedom
    }
    case DistributionName.Chi: {
      const { k } = p as ChiParams
      return chiSampler.factory(k, { seed }) // k = degrees of freedom
    }
    case DistributionName.InvGamma: {
      const { shape, scale } = p as InvGammaParams
      return invgammaSampler.factory(shape, scale, { seed }) // alpha=shape, beta=SCALE (direct)
    }
    case DistributionName.BetaPrime: {
      const { alpha, beta } = p as BetaPrimeParams
      return betaprimeSampler.factory(alpha, beta, { seed }) // two shapes
    }
    case DistributionName.Cosine: {
      const { mu, s } = p as CosineParams
      return cosineSampler.factory(mu, s, { seed }) // (mu=location, s=SCALE)
    }
    case DistributionName.Beta: {
      const { alpha, beta } = p as BetaParams
      return betaSampler.factory(alpha, beta, { seed }) // two shapes, support (0,1)
    }
    default:
      throw new Error(`makeSampler: no sampler for '${name}'`)
  }
}
