import {
  NM_CONTRACTION,
  NM_EXPANSION,
  NM_F_TOL,
  NM_INITIAL_STEP,
  NM_MAX_FUNCTION_EVALS,
  NM_MAX_ITERATIONS,
  NM_MAX_RESTARTS,
  NM_REFLECTION,
  NM_SHRINK,
  NM_X_TOL,
  NM_ZERO_STEP,
} from './constants'

/**
 * Domain-agnostic deterministic Nelder–Mead simplex minimizer for `f: R^n -> R`.
 *
 * SCOPE BOUNDARY: this file ONLY minimizes a scalar objective. No location/scale, no parity, no
 * distribution logic lives here — families reparameterize their MLE into unconstrained coordinates
 * (e.g. `[loc, ln sigma, ln df]`) and call `minimize`.
 *
 * DETERMINISM: there is NO runtime randomness. The initial simplex is built by a fixed perturbation
 * rule (scipy's `nonzdelt`/`zdelt`), and the vertex sort uses an explicit comparator with an
 * original-index tie-break — so identical `(f, x0, options)` always yields bit-identical output.
 *
 * TERMINATION INVARIANT: ZERO while-loops. Three bounded `for` loops (restart, iteration, shrink)
 * plus a global evaluation tripwire inside `safeF`. No path can spin without consuming either an
 * iteration or a function evaluation, so the optimizer cannot hang on a pathological objective.
 */

/** A scalar objective to minimize. Receives a read-only point and returns a single number. */
export type Objective = (x: readonly number[]) => number

/** The outcome of a `minimize` call: the best point found, its objective value, the total
 *  iterations consumed across all restarts, and whether the inner tolerance test was met
 *  (NOT merely that the loops ran out — exhausting iterations/restarts does NOT set `converged`). */
export interface MinimizeResult {
  x: number[]
  fx: number
  iterations: number
  converged: boolean
}

/** Per-call overrides; any omitted field falls back to the corresponding `NM_*` constant. */
export interface NelderMeadOptions {
  maxIterations?: number
  maxFunctionEvals?: number
  maxRestarts?: number
  xTol?: number
  fTol?: number
  initialStep?: number
  zeroStep?: number
  reflection?: number
  expansion?: number
  contraction?: number
  shrink?: number
}

/** Resolved options after applying defaults — every field present so the algorithm reads no `?`. */
interface ResolvedOptions {
  maxIterations: number
  maxFunctionEvals: number
  maxRestarts: number
  xTol: number
  fTol: number
  initialStep: number
  zeroStep: number
  reflection: number
  expansion: number
  contraction: number
  shrink: number
}

function resolveOptions(o: NelderMeadOptions | undefined): ResolvedOptions {
  return {
    maxIterations: o?.maxIterations ?? NM_MAX_ITERATIONS,
    maxFunctionEvals: o?.maxFunctionEvals ?? NM_MAX_FUNCTION_EVALS,
    maxRestarts: o?.maxRestarts ?? NM_MAX_RESTARTS,
    xTol: o?.xTol ?? NM_X_TOL,
    fTol: o?.fTol ?? NM_F_TOL,
    initialStep: o?.initialStep ?? NM_INITIAL_STEP,
    zeroStep: o?.zeroStep ?? NM_ZERO_STEP,
    reflection: o?.reflection ?? NM_REFLECTION,
    expansion: o?.expansion ?? NM_EXPANSION,
    contraction: o?.contraction ?? NM_CONTRACTION,
    shrink: o?.shrink ?? NM_SHRINK,
  }
}

/** A guarded element read for `noUncheckedIndexedAccess`; an out-of-bounds index yields NaN, which
 *  `safeF` then maps to +Inf — never a silent `undefined` arithmetic. */
function at(v: readonly number[], i: number): number {
  return v[i] ?? Number.NaN
}

/**
 * The ONLY place `f` is invoked. Wraps the objective with a hard evaluation budget and a
 * non-finite rejection rule. `evaluate()` is the single counted gate (named to avoid the JS global `eval`; it never executes arbitrary code — it only invokes the stored objective `f`):
 *   - If the budget is already exhausted, it returns +Inf WITHOUT calling `f` (so the caller's
 *     incumbent best is never displaced by a forbidden extra call) and sets `exhausted`.
 *   - Otherwise it calls `f`, increments the counter, and maps any non-finite result
 *     (NaN, +Inf, AND −Inf) to +Infinity.
 *
 * The −Inf→+Inf map is INTENTIONAL: a genuinely −Inf negLL (e.g. sigma→0 overfit) is rejected
 * exactly like NaN so the simplex can never collapse onto a degenerate maximum.
 */
class SafeObjective {
  private calls = 0
  exhausted = false
  // Explicit field declarations + a plain assignment constructor: TS parameter properties are not
  // erasable syntax, which the project's `erasableSyntaxOnly` compiler option forbids.
  private readonly f: Objective
  private readonly budget: number
  constructor(f: Objective, budget: number) {
    this.f = f
    this.budget = budget
  }
  evaluate(x: readonly number[]): number {
    if (this.calls >= this.budget) {
      this.exhausted = true
      return Number.POSITIVE_INFINITY
    }
    const r = this.f(x)
    this.calls += 1
    return Number.isFinite(r) ? r : Number.POSITIVE_INFINITY
  }
}

/** Build the deterministic initial simplex (n+1 vertices) around `x0 = v[0]`. For each coordinate
 *  `i`, vertex `i+1` copies `x0` and perturbs coord `i`: relatively (`*= 1+initialStep`) when nonzero,
 *  absolutely (`= zeroStep`) when zero — else a relative bump of 0 would give a degenerate simplex.
 *  Works for n=1 (a 2-vertex simplex). NO RNG. */
function buildSimplex(x0: readonly number[], opts: ResolvedOptions): number[][] {
  const n = x0.length
  const simplex: number[][] = [[...x0]]
  for (let i = 0; i < n; i++) {
    const vertex = [...x0]
    const seed = at(x0, i)
    vertex[i] = seed !== 0 ? seed * (1 + opts.initialStep) : opts.zeroStep
    simplex.push(vertex)
  }
  return simplex
}

/** Comparator over vertex indices by their cached objective value, with a deterministic tie-break
 *  on the ORIGINAL index. Never uses `fa - fb`: when two values are both +Inf (all-NaN objective),
 *  `Inf - Inf = NaN` would corrupt the sort order and destroy bit-determinism. */
function compareByValue(fv: readonly number[], ia: number, ib: number): number {
  const fa = at(fv, ia)
  const fb = at(fv, ib)
  if (fa < fb) return -1
  if (fa > fb) return 1
  return ia - ib
}

/** Centroid of all vertices EXCEPT the worst (the last in `order`): the reflection anchor. */
function centroidExcludingWorst(simplex: readonly number[][], order: readonly number[]): number[] {
  const n = simplex[0]?.length ?? 0
  const c = new Array<number>(n).fill(0)
  const keep = order.length - 1 // all but the worst vertex
  for (let r = 0; r < keep; r++) {
    const vertex = simplex[at(order, r)] ?? []
    for (let j = 0; j < n; j++) c[j] = (c[j] ?? 0) + at(vertex, j)
  }
  for (let j = 0; j < n; j++) c[j] = (c[j] ?? 0) / keep
  return c
}

/** `centroid + coeff·(centroid − point)` — the affine move shared by reflection/expansion/contraction. */
function step(centroid: readonly number[], point: readonly number[], coeff: number): number[] {
  const out = new Array<number>(centroid.length)
  for (let j = 0; j < centroid.length; j++)
    out[j] = at(centroid, j) + coeff * (at(centroid, j) - at(point, j))
  return out
}

/** State threaded through and out of a single restart's inner loop. */
interface RestartResult {
  bestX: number[]
  bestF: number
  iterations: number
  converged: boolean
}

/**
 * One restart: run the bounded Nelder–Mead iteration loop on a fresh full-size simplex built around
 * `seedX`. Returns the best vertex, its value, the iterations consumed, and whether the inner
 * tolerance test (BOTH `fTol` on the objective spread AND `xTol` on the simplex size) was met.
 * Stops immediately — `converged=false` — if `safe.exhausted` trips at any evaluation.
 */
function runRestart(
  safe: SafeObjective,
  seedX: readonly number[],
  opts: ResolvedOptions,
): RestartResult {
  const n = seedX.length
  const simplex = buildSimplex(seedX, opts)
  const fv = simplex.map((v) => safe.evaluate(v))
  let iterations = 0

  for (let it = 0; it < opts.maxIterations; it++) {
    if (safe.exhausted) break
    iterations = it + 1

    // Deterministic ascending sort by value; order[0]=best, order[last]=worst.
    const order = simplex.map((_, idx) => idx).sort((a, b) => compareByValue(fv, a, b))
    const best = at(order, 0)
    const worst = at(order, order.length - 1)
    const secondWorst = at(order, order.length - 2)
    const fBest = at(fv, best)
    const fWorst = at(fv, worst)
    const bestVertex = simplex[best] ?? []

    // Convergence requires BOTH the objective spread AND the simplex size to be within tolerance.
    let maxCoordSpread = 0
    for (let v = 0; v < simplex.length; v++) {
      const vertex = simplex[v] ?? []
      for (let j = 0; j < n; j++) {
        const d = Math.abs(at(vertex, j) - at(bestVertex, j))
        if (d > maxCoordSpread) maxCoordSpread = d
      }
    }
    if (fWorst - fBest <= opts.fTol && maxCoordSpread <= opts.xTol) {
      return { bestX: [...bestVertex], bestF: fBest, iterations, converged: true }
    }

    const centroid = centroidExcludingWorst(simplex, order)
    const worstVertex = simplex[worst] ?? []

    // Reflection.
    const reflected = step(centroid, worstVertex, opts.reflection)
    const fReflected = safe.evaluate(reflected)
    if (safe.exhausted) break

    if (fReflected < fBest) {
      // Expansion: the reflection beat the best, so try going further.
      const expanded = step(centroid, worstVertex, opts.reflection * opts.expansion)
      const fExpanded = safe.evaluate(expanded)
      if (safe.exhausted) break
      if (fExpanded < fReflected) {
        simplex[worst] = expanded
        fv[worst] = fExpanded
      } else {
        simplex[worst] = reflected
        fv[worst] = fReflected
      }
    } else if (fReflected < at(fv, secondWorst)) {
      // Reflection is better than the second-worst: accept it.
      simplex[worst] = reflected
      fv[worst] = fReflected
    } else {
      // Contraction.
      if (fReflected < fWorst) {
        // Outside contraction (reflection improved on the worst but is still high).
        const contracted = step(centroid, worstVertex, opts.reflection * opts.contraction)
        const fContracted = safe.evaluate(contracted)
        if (safe.exhausted) break
        if (fContracted <= fReflected) {
          simplex[worst] = contracted
          fv[worst] = fContracted
        } else {
          shrink(simplex, fv, best, opts, safe)
          if (safe.exhausted) break
        }
      } else {
        // Inside contraction (reflection no better than the worst).
        const contracted = step(centroid, worstVertex, -opts.contraction)
        const fContracted = safe.evaluate(contracted)
        if (safe.exhausted) break
        if (fContracted < fWorst) {
          simplex[worst] = contracted
          fv[worst] = fContracted
        } else {
          shrink(simplex, fv, best, opts, safe)
          if (safe.exhausted) break
        }
      }
    }
  }

  // Loop exhausted (iterations or eval budget) without meeting tolerance: return current best,
  // NOT converged. The post-loop best is recomputed deterministically.
  let bestIdx = 0
  for (let v = 1; v < fv.length; v++) {
    if (compareByValue(fv, v, bestIdx) < 0) bestIdx = v
  }
  const bestVertex = simplex[bestIdx] ?? [...seedX]
  return { bestX: [...bestVertex], bestF: at(fv, bestIdx), iterations, converged: false }
}

/** Shrink every vertex except `best` toward `best` by factor `shrink` (a bounded `for k=1..n`).
 *  Re-evaluates the shrunk vertices through `safe`, so it can trip the eval budget. */
function shrink(
  simplex: number[][],
  fv: number[],
  best: number,
  opts: ResolvedOptions,
  safe: SafeObjective,
): void {
  const bestVertex = simplex[best] ?? []
  const n = bestVertex.length
  for (let k = 0; k < simplex.length; k++) {
    if (k === best) continue
    const vertex = simplex[k] ?? []
    const shrunk = new Array<number>(n)
    for (let j = 0; j < n; j++)
      shrunk[j] = at(bestVertex, j) + opts.shrink * (at(vertex, j) - at(bestVertex, j))
    simplex[k] = shrunk
    fv[k] = safe.evaluate(shrunk)
    if (safe.exhausted) return
  }
}

/**
 * Minimize `f` from seed `x0`. The outer driver runs up to `maxRestarts + 1` restarts: each builds
 * a fresh FULL-SIZE simplex around the incumbent best (this is what escapes a collapsed simplex and
 * polishes the optimum to the parity tolerance) and runs the bounded inner loop. The eval-budget
 * tripwire unwinds BOTH loops. `converged` reflects the inner tolerance test only.
 */
export function minimize(
  f: Objective,
  x0: readonly number[],
  options?: NelderMeadOptions,
): MinimizeResult {
  const opts = resolveOptions(options)
  const safe = new SafeObjective(f, opts.maxFunctionEvals)

  let bestX = [...x0]
  let bestF = safe.evaluate(bestX) // seed value: the incumbent every restart must strictly beat to move.
  let totalIterations = 0
  let converged = false

  for (let r = 0; r <= opts.maxRestarts; r++) {
    if (safe.exhausted) break
    // Capture the incumbent BEFORE this restart (on r=0 this is the seed value f(x0)). The polish
    // decision compares against THIS, not the post-update best — comparing against the already-
    // updated best would always read a ~0 improvement and stop after the first converged restart,
    // skipping the fresh-simplex re-descent that tightens the optimum to the parity tolerance.
    const prevBestF = bestF
    const result = runRestart(safe, bestX, opts)
    totalIterations += result.iterations
    if (result.bestF < bestF) {
      bestF = result.bestF
      bestX = result.bestX
    }
    if (safe.exhausted) {
      converged = false
      break
    }
    // A restart that met tolerance certifies convergence. Stop only when the restart-over-restart
    // improvement is itself below fTol (further polish is futile) or restarts are exhausted.
    // Exhausting restarts alone does NOT set converged.
    if (result.converged) {
      converged = true
      if (prevBestF - bestF < opts.fTol || r === opts.maxRestarts) break
    }
  }

  return { x: bestX, fx: bestF, iterations: totalIterations, converged }
}
