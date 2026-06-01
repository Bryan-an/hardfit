/** Minimum sample size fitAll will accept.
 *  AICc's small-sample correction needs n - k - 1 > 0; the richest model here has k = 2,
 *  so n >= k + 2 = 4 is the smallest n yielding a finite AICc for every distribution. */
export const MIN_FIT_SAMPLE_SIZE = 4
/** Max iterations for the gamma/weibull shape-MLE Newton loops. */
export const MAX_NEWTON_ITERATIONS = 100
/** Relative convergence tolerance for the Newton step (|step| < tol * k). */
export const NEWTON_REL_TOL = 1e-12

/** Default number of parametric-bootstrap replicates. B+1 = 1000 gives good p-value
 *  discreteness (the cited `(1+ge)/(B+1)` form) and an adequate BCa interval. */
export const DEFAULT_BOOTSTRAP_B = 999
/** Default two-sided CI miscoverage: a 95% CI has tails at alpha/2 and 1 − alpha/2. */
export const DEFAULT_CI_ALPHA = 0.05
/** Above this n the jackknife (O(n) refits, O(n²) for weibull) is a freeze hazard, so
 *  BCa skips the acceleration pass and every parameter CI falls back to percentile. */
export const BCA_JACKKNIFE_MAX_N = 2000
/** Bootstrap replicates per chunk: at each chunk boundary the loop reports progress,
 *  checks for cancellation, and yields (0 ms timer) so a worker can service messages. */
export const CHUNK = 50
/** Default number of top-ranked (lowest-AICc) fits the bootstrap orchestrator runs.
 *  Bootstrapping every distribution is wasteful; only the best few are worth the cost. */
export const BOOTSTRAP_TOP_K = 3
/** Odd 32-bit multiplier (Knuth's golden-ratio constant) for deriving a distinct sampler
 *  seed per bootstrapped fit. Mixing the master seed with `SALT·(index+1)` spreads the
 *  per-fit streams far apart so they are uncorrelated (adjacent indices alone would yield
 *  near-identical seeds — a silent statistical bug). */
export const BOOTSTRAP_SEED_SALT = 0x9e3779b1
