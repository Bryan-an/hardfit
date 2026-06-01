import exponentialSampler from '@stdlib/random-base-exponential'
import gammaSampler from '@stdlib/random-base-gamma'
import lognormalSampler from '@stdlib/random-base-lognormal'
import normalSampler from '@stdlib/random-base-normal'
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
    default:
      throw new Error(`makeSampler: no sampler for '${name}'`)
  }
}
