/** Minimum sample size fitAll will accept.
 *  AICc's small-sample correction needs n - k - 1 > 0; the richest model here has k = 2,
 *  so n >= k + 2 = 4 is the smallest n yielding a finite AICc for every distribution. */
export const MIN_FIT_SAMPLE_SIZE = 4
/** Max iterations for the gamma/weibull shape-MLE Newton loops. */
export const MAX_NEWTON_ITERATIONS = 100
/** Relative convergence tolerance for the Newton step (|step| < tol * k). */
export const NEWTON_REL_TOL = 1e-12
