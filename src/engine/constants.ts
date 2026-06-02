/** Minimum sample size fitAll will accept.
 *  AICc's small-sample correction needs n - k - 1 > 0. Since Batch D the richest model is
 *  k = 3 (Student-t), so a FINITE Student-t AICc needs n >= k + 2 = 5; at n = 4 the t's
 *  AICc is +Infinity (aicc() returns POSITIVE_INFINITY when n - k - 1 <= 0) → it ranks last
 *  with weight 0, gracefully, while the k <= 2 families still fit and score finitely. We KEEP
 *  this floor at 4 (not 5): n = 4 yields a finite AICc for every k <= 2 distribution, and the
 *  lone k = 3 t simply sorts to the bottom rather than being excluded. */
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

// --- Nelder–Mead general minimizer (Batch D, src/engine/optimize.ts) ---
/** Hard cap on simplex iterations PER restart. Bounds the only inner loop. NO unbounded while-loop —
 *  this project lost a session to one in a beta test, so every loop here is a counted `for`. */
export const NM_MAX_ITERATIONS = 200
/** Hard cap on objective evaluations across the WHOLE minimize() call (all restarts). Independent of
 *  the iteration cap: a pathological objective can burn evals without advancing iterations, so the
 *  eval budget — checked INSIDE the safe-f wrapper — is the true freeze guard. */
export const NM_MAX_FUNCTION_EVALS = 2000
/** Restart-from-best count. A restart re-expands a FULL-SIZE fresh simplex around the incumbent best
 *  and re-searches; this escapes a collapsed simplex AND performs the final polish that clears the
 *  1e-6 LL parity gate (verified: restart-from-best reaches the closed-form MLE LL to machine zero). */
export const NM_MAX_RESTARTS = 2
/** ABSOLUTE simplex-size tolerance (max |coord − bestCoord| over all vertices). Absolute (not
 *  relative) because the parity gate is absolute in LL units. */
export const NM_X_TOL = 1e-10
/** ABSOLUTE objective-spread tolerance f(worst)−f(best). MUST be absolute and well below the 1e-6 LL
 *  gate; 1e-9 gives ~3 orders of headroom (verified to reach machine-zero LL gap). */
export const NM_F_TOL = 1e-9
/** Relative per-coordinate perturbation building the initial simplex from a NONZERO seed coord
 *  (vertex_i coord_i *= 1+this). scipy's `nonzdelt`; deterministic, no RNG. */
export const NM_INITIAL_STEP = 0.05
/** Absolute perturbation for a ZERO seed coord (else a relative bump of 0 gives a degenerate simplex).
 *  scipy's `zdelt`. */
export const NM_ZERO_STEP = 0.00025
/** Standard Nelder–Mead reflection / expansion / contraction / shrink coefficients. */
export const NM_REFLECTION = 1
export const NM_EXPANSION = 2
export const NM_CONTRACTION = 0.5
export const NM_SHRINK = 0.5

// --- Relaxed Nelder–Mead caps for bootstrap replicate refits (fit(data, { quick: true })) ---
// The B parametric-bootstrap replicate refits do NOT need parity-gate precision: Monte-Carlo error
// across the B replicates dominates any single replicate's sub-1e-6 point-estimate imprecision. So
// the quick path trades the full NM_* caps for these relaxed ones, which bounds the per-refit cost
// for the optimizer-fit families (Student-t, Fisher-F). The primary/displayed fit and the scipy-
// parity path keep the FULL NM_* caps — they never pass `{ quick: true }`.
/** Relaxed iteration cap per restart for quick refits (vs NM_MAX_ITERATIONS=200). */
export const NM_BOOTSTRAP_MAX_ITERATIONS = 60
/** No restarts for quick refits (vs NM_MAX_RESTARTS=2): the parity-clearing fresh-simplex polish is
 *  the most expensive phase and is unnecessary when MC error across replicates dominates. */
export const NM_BOOTSTRAP_MAX_RESTARTS = 0
/** Relaxed objective-spread tolerance for quick refits (vs NM_F_TOL=1e-9): 1e-6 is well below the
 *  scale of the MC error a single replicate contributes. */
export const NM_BOOTSTRAP_F_TOL = 1e-6
/** Relaxed eval budget for quick refits (vs NM_MAX_FUNCTION_EVALS=2000): the true freeze guard,
 *  sized so a quick refit terminates quickly even on a pathological synthetic replicate. */
export const NM_BOOTSTRAP_MAX_FUNCTION_EVALS = 400
