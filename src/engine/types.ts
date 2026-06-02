/** Machine ids for the distributions HardFit fits — single source of truth.
 *  Later units reference these constants instead of repeating raw id strings. */
export const DistributionName = {
  Normal: 'normal',
  Lognormal: 'lognormal',
  Exponential: 'exponential',
  Gamma: 'gamma',
  Weibull: 'weibull',
  // M2.3 Batch A — drop-in continuous distributions.
  Uniform: 'uniform',
  Rayleigh: 'rayleigh',
  Pareto: 'pareto',
  Laplace: 'laplace',
  Logistic: 'logistic',
  Gumbel: 'gumbel',
  Cauchy: 'cauchy',
  Frechet: 'frechet',
  // M2.3 Batch B — drop-in continuous distributions.
  Levy: 'levy',
  ChiSquared: 'chisquare',
  Chi: 'chi',
  InvGamma: 'invgamma',
  BetaPrime: 'betaprime',
  Cosine: 'cosine',
  Beta: 'beta',
  // M2.3 Batch C — discrete distributions.
  Poisson: 'poisson',
  Geometric: 'geometric',
  NegativeBinomial: 'negative-binomial',
  DiscreteUniform: 'discrete-uniform',
  // M2.3 Batch D — multi-parameter MLE via the vendored Nelder–Mead optimizer.
  StudentT: 'student-t',
  FisherF: 'fisher-f',
  InverseGaussian: 'inverse-gaussian',
  Nakagami: 'nakagami',
} as const
export type DistributionName = (typeof DistributionName)[keyof typeof DistributionName]

/** Fitted parameters in HardFit's own convention (documented per distribution). */
export type FittedParams = Record<string, number>

/** Sample space of a distribution. Routes goodness-of-fit: EDF tests (KS/AD/CvM) for
 *  continuous fits, chi-square for discrete. M2.3's discrete batch just flips this flag. */
export type DistributionKind = 'continuous' | 'discrete'

/** Per-fit knobs. The optional shape keeps families implementing `fit(data)` assignable to the
 *  `fit(data, opts?)` interface method with NO change — only the optimizer-fit families read it. */
export interface FitOptions {
  /** When true, the fit uses RELAXED Nelder–Mead caps (the `NM_BOOTSTRAP_*` constants) instead of the
   *  full parity-grade `NM_*` defaults. Set ONLY by the parametric bootstrap's B-replicate refits,
   *  where Monte-Carlo error across replicates dwarfs any single replicate's sub-1e-6 point-estimate
   *  imprecision. The primary/displayed fit and the scipy-parity path NEVER pass this. Closed-form and
   *  Newton families ignore it. */
  quick?: boolean
}

export interface Distribution {
  readonly name: string // machine id, e.g. 'normal'
  readonly label: string // display label, e.g. 'Normal'
  readonly k: number // number of estimated parameters (for AIC)
  readonly kind: DistributionKind // GoF routing: EDF tests for continuous, chi-square for discrete
  /** true for families whose MLE uses the iterative Nelder–Mead optimizer (Student-t, Fisher-F). The
   *  parametric bootstrap skips the O(n) BCa jackknife for them — n leave-one-out NM refits would be an
   *  O(n·NM) blowup — and uses percentile CIs instead. */
  readonly expensiveFit?: boolean
  /** MLE fit. Throws an Error if `data` violates the distribution's support. */
  fit(data: readonly number[], opts?: FitOptions): FittedParams
  /** Natural-log density at x for fitted params (data scale). */
  logpdf(x: number, p: FittedParams): number
  /** Cumulative distribution function at x for fitted params. */
  cdf(x: number, p: FittedParams): number
  /** Inverse CDF (quantile). Used for chi-square binning + (M2.2) bootstrap sampling. */
  quantile(prob: number, p: FittedParams): number
  /** Integer support bounds for a DISCRETE distribution at the fitted params, driving the
   *  PMF-binned chi-square (M2.3 Batch C). `max` may be `+Infinity` (unbounded counts).
   *  Continuous distributions omit this (their GoF uses the EDF/quantile path instead). */
  support?(p: FittedParams): { min: number; max: number }
}

/** How a reported p-value was obtained. */
export type PValueMethod = 'closed-form' | 'table' | 'diagnostic' | 'bootstrap'

/** A goodness-of-fit test result with its statistic and (when available) p-value. */
export interface GofResult {
  statistic: number
  pValue: number | null // null when not available (diagnostic-only until bootstrap)
  method: PValueMethod
}

/** Chi-squared GoF result, carrying its binning so callers can report df + bin count. */
export interface ChiSquaredResult extends GofResult {
  df: number
  bins: number
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
  ad: GofResult // Anderson-Darling
  cvm: GofResult // Cramér-von Mises
  chiSquared: ChiSquaredResult // Chi-squared (equiprobable quantile bins)
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
