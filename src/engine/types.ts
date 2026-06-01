/** Machine ids for the distributions HardFit fits — single source of truth.
 *  Later units reference these constants instead of repeating raw id strings. */
export const DistributionName = {
  Normal: 'normal',
  Lognormal: 'lognormal',
  Exponential: 'exponential',
  Gamma: 'gamma',
  Weibull: 'weibull',
} as const
export type DistributionName = (typeof DistributionName)[keyof typeof DistributionName]

/** Fitted parameters in HardFit's own convention (documented per distribution). */
export type FittedParams = Record<string, number>

export interface Distribution {
  readonly name: string // machine id, e.g. 'normal'
  readonly label: string // display label, e.g. 'Normal'
  readonly k: number // number of estimated parameters (for AIC)
  /** MLE fit. Throws an Error if `data` violates the distribution's support. */
  fit(data: readonly number[]): FittedParams
  /** Natural-log density at x for fitted params (data scale). */
  logpdf(x: number, p: FittedParams): number
  /** Cumulative distribution function at x for fitted params. */
  cdf(x: number, p: FittedParams): number
}

/** A successful fit of one distribution with its diagnostics. */
export interface Fit {
  name: string
  label: string
  k: number
  params: FittedParams
  logLik: number
  aic: number
  aicc: number
  bic: number
  ks: number // Kolmogorov-Smirnov D (diagnostic only; p-value invalid for estimated params)
}

/** A Fit plus its ranking position. */
export interface RankedFit extends Fit {
  rank: number // 1 = best (lowest AICc)
  deltaAICc: number // AICc - min(AICc)
  weight: number // Akaike weight in (0,1]
}

/** A distribution that could not be fit to the given data (e.g. support violation). */
export interface FitFailure {
  name: string
  label: string
  error: string
}

export interface FitAllResult {
  ranked: RankedFit[]
  failures: FitFailure[]
  n: number
}
