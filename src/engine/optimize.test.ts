import { describe, expect, it } from 'vitest'
import { expectClose } from '../test/relClose'
import { mean, populationVariance } from './math'
import { minimize } from './optimize'

/** Probabilities are not used here; this suite exercises the domain-agnostic minimizer alone:
 *  no loc/scale/parity logic lives in optimize.ts, so the objectives are hand-rolled. */

/** Separable quadratic with minimum at `c` and value 0 there. */
function separableQuadratic(c: readonly number[]): (x: readonly number[]) => number {
  return (x: readonly number[]) => {
    let s = 0
    for (let i = 0; i < c.length; i++) {
      const ci = c[i] ?? Number.NaN
      const xi = x[i] ?? Number.NaN
      s += (xi - ci) * (xi - ci)
    }
    return s
  }
}

/** 2-D Rosenbrock banana: f(x,y) = (1−x)² + 100(y−x²)²; global min 0 at (1,1). The curved valley
 *  starves a single Nelder–Mead pass UNDER A LIMITED PER-RESTART ITERATION BUDGET — used by test 2 to
 *  show restart-from-best rescues a stalled single pass at the same per-pass budget. (NOTE: with the
 *  default uncapped budget this implementation's single pass already reaches machine zero, so the
 *  feature only becomes observable when the per-restart iteration cap is the binding constraint.) */
function rosenbrock(v: readonly number[]): number {
  const x = v[0] ?? Number.NaN
  const y = v[1] ?? Number.NaN
  const a = 1 - x
  const b = y - x * x
  return a * a + 100 * b * b
}

/** Normal negative log-likelihood in UNCONSTRAINED coords θ = (mu, ln sigma) — the parity-shaped
 *  objective. negLL = (n/2)·ln(2π) + n·θ1 + Σ(x−θ0)² / (2·e^{2θ1}); its minimizer is mu = sample mean,
 *  e^{θ1} = population (MLE) std. Proves the tolerance clears the 1e-6 LL gate. */
function normalNegLL(data: readonly number[]): (theta: readonly number[]) => number {
  const n = data.length
  return (theta: readonly number[]) => {
    const mu = theta[0] ?? Number.NaN
    const lnSigma = theta[1] ?? Number.NaN
    const sigma2 = Math.exp(2 * lnSigma)
    let ss = 0
    for (const x of data) ss += (x - mu) * (x - mu)
    return (n / 2) * Math.log(2 * Math.PI) + n * lnSigma + ss / (2 * sigma2)
  }
}

const NORMAL_SAMPLE = [4.2, 5.1, 4.8, 6.3, 5.5, 4.9, 5.7, 5.0, 4.4, 6.0, 5.3, 4.6]

describe('minimize (Nelder–Mead)', () => {
  it('1: separable quadratic recovers the minimizer to 1e-9 with fx≈0 and converged', () => {
    const c = [1.5, -2.25, 0.75]
    const r = minimize(separableQuadratic(c), [0, 0, 0])
    expect(r.converged).toBe(true)
    for (let i = 0; i < c.length; i++) expectClose(r.x[i] ?? Number.NaN, c[i] ?? Number.NaN, 1e-9)
    expect(Math.abs(r.fx)).toBeLessThan(1e-9)
  })

  it('2: 2-D Rosenbrock — restart-from-best clears 1e-6 where a same-budget single pass stalls', () => {
    // DISCRIMINATING test of the restart feature. `maxIterations` is PER RESTART, and the only way a
    // run can exceed K total iterations under this API is via restart-from-best (there is no
    // "continue the same simplex" path). So holding the per-restart budget fixed at K and toggling
    // maxRestarts isolates restart-from-best exactly: more total iterations IS the restart feature.
    //
    // At K=40 the single pass stalls in the Rosenbrock valley at fx≈0.31 (~300× above the 1e-6 gate,
    // verified by a K-sweep over 10..70), while three restarts (it≈120) polish to fx≈9e-14 (~8 orders
    // below the gate) — a wide, platform-robust window on both sides. (The committed test asserted only
    // the OUTCOME at the default budget, where the single pass ALREADY reaches ~1e-22 with ZERO
    // restarts, so it pinned nothing about restart-from-best.)
    const PER_RESTART_ITERS = 40
    const single = minimize(rosenbrock, [-1.2, 1], {
      maxRestarts: 0,
      maxIterations: PER_RESTART_ITERS,
    })
    const restarted = minimize(rosenbrock, [-1.2, 1], {
      maxRestarts: 2,
      maxIterations: PER_RESTART_ITERS,
    })
    // Single pass at this per-restart budget stalls well above the gate (the premise the feature rescues)...
    expect(single.fx).toBeGreaterThan(1e-3)
    // ...and restart-from-best clears it at the SAME per-pass budget. Assert on fx, not `converged`: a
    // final restart that hits its iteration cap mid-polish leaves the flag false even with fx < 1e-6.
    expect(restarted.fx).toBeLessThan(1e-6)
    // The improvement came from extra restart passes, not a longer single loop (K is per restart).
    expect(restarted.iterations).toBeGreaterThan(single.iterations)
  })

  it('3: normal-MLE in (mu, ln sigma) reaches the MLE objective to ~machine zero (parity-shaped)', () => {
    const mu = mean(NORMAL_SAMPLE)
    const sigma = Math.sqrt(populationVariance(NORMAL_SAMPLE, mu))
    const f = normalNegLL(NORMAL_SAMPLE)
    const r = minimize(f, [0, 0])
    // The negLL is flat to machine precision within ~1e-8 of the optimum (curvature ~n/sigma^2 with a
    // large additive constant ~11 → ULP ~2.4e-15), so Nelder–Mead cannot resolve the COORDINATE below
    // that floating-point floor — exactly the flat-likelihood reason the parity contract pins the LL,
    // not the params (ITERATIVE_PARAM_DIAGNOSTIC_SKIP). The objective gap is the faithful convergence
    // proof (the real gate is in LL units, 1e-6); the coordinates get only a parity-scale sanity tol.
    expect(Math.abs(r.fx - f([mu, Math.log(sigma)]))).toBeLessThanOrEqual(1e-9)
    expectClose(r.x[0] ?? Number.NaN, mu, 1e-6)
    expectClose(Math.exp(r.x[1] ?? Number.NaN), sigma, 1e-6)
  })

  it('4: determinism — two identical calls return bit-identical x, fx, iterations', () => {
    const f = separableQuadratic([2.5, -1.5])
    const a = minimize(f, [10, 10])
    const b = minimize(f, [10, 10])
    expect(a.x).toStrictEqual(b.x)
    expect(a.fx).toBe(b.fx)
    expect(a.iterations).toBe(b.iterations)
    expect(a.converged).toBe(b.converged)
  })

  it('5: n=1 univariate f(x)=(x−3)² from x0=[0] recovers 3 to 1e-9 (2-vertex simplex)', () => {
    const r = minimize((x: readonly number[]) => ((x[0] ?? Number.NaN) - 3) ** 2, [0])
    expect(r.converged).toBe(true)
    expectClose(r.x[0] ?? Number.NaN, 3, 1e-9)
  })

  it('6: zero-coordinate seed [0,0] with min at (0.5,−0.5) exercises NM_ZERO_STEP and converges', () => {
    const r = minimize(separableQuadratic([0.5, -0.5]), [0, 0])
    expect(r.converged).toBe(true)
    expectClose(r.x[0] ?? Number.NaN, 0.5, 1e-9)
    expectClose(r.x[1] ?? Number.NaN, -0.5, 1e-9)
  })

  it('7: iteration cap (tiny maxIterations, maxRestarts=0) on Rosenbrock → not converged, no hang', () => {
    const cap = 5
    const r = minimize(rosenbrock, [-1.2, 1], { maxIterations: cap, maxRestarts: 0 })
    expect(r.converged).toBe(false)
    expect(r.iterations).toBeLessThanOrEqual(cap)
    expect(Number.isFinite(r.fx)).toBe(true)
    expect(Number.isFinite(r.x[0] ?? Number.NaN)).toBe(true)
  })

  it('8: eval-budget cap — objective called ≤ maxFunctionEvals, returns, not converged', () => {
    const budget = 30
    let calls = 0
    const f = (x: readonly number[]) => {
      calls += 1
      return rosenbrock(x)
    }
    const r = minimize(f, [-1.2, 1], { maxFunctionEvals: budget })
    expect(calls).toBeLessThanOrEqual(budget)
    expect(r.converged).toBe(false)
    expect(Number.isFinite(r.fx)).toBe(true)
  })

  it('9: all-NaN objective → safeF maps to +Inf, terminates within caps, returns seed, no throw', () => {
    const seed = [3.3, -7.1]
    let r: ReturnType<typeof minimize> | undefined
    expect(() => {
      r = minimize(() => Number.NaN, seed)
    }).not.toThrow()
    expect(r?.converged).toBe(false)
    expect(r?.x).toStrictEqual(seed)
    expect(r?.fx).toBe(Number.POSITIVE_INFINITY)
  })

  it('10: −Inf on part of the domain is rejected like NaN (intentional −Inf→+Inf map)', () => {
    // The −Inf region must be the ATTRACTOR for this to test anything: f = (x+3)² for x>0 (its
    // UNCONSTRAINED minimum is x=−3, which lies INSIDE the rejected x≤0 region), and −Inf for x≤0.
    // Descent is therefore driven leftward straight into the wall, so the simplex genuinely PROBES
    // x≤0 (a counter proves it). The −Inf→+Inf map must reject those probes: with the map present the
    // optimizer is pinned against the wall at x≈0⁺ (fx≈9, x>0); WITHOUT the map (if −Inf were accepted)
    // it would collapse to x<0 onto the degenerate basin. A naive (x−2)² objective would NOT test this:
    // its minimum is at x=2, away from the wall, so descent marches rightward and never probes x≤0
    // (probedNonPositive=0) — the assertions would pass identically with or without the guard.
    let probedNonPositive = 0
    const f = (v: readonly number[]) => {
      const x = v[0] ?? Number.NaN
      if (x <= 0) {
        probedNonPositive += 1
        return Number.NEGATIVE_INFINITY
      }
      return (x + 3) ** 2
    }
    const r = minimize(f, [1])
    // The objective forced the simplex to actually probe the rejected region...
    expect(probedNonPositive).toBeGreaterThan(0)
    // ...and the −Inf→+Inf map kept the result on the finite side of the wall (never the −Inf basin).
    expect(r.x[0] ?? Number.NaN).toBeGreaterThan(0)
  })

  it('11: options override the constant defaults (custom fTol + maxIterations honored)', () => {
    // A loose fTol stops earlier (fewer iterations) than the default tight tol on the same problem.
    const f = separableQuadratic([5, 5])
    const loose = minimize(f, [0, 0], { fTol: 1e-2, xTol: 1e-2 })
    const tight = minimize(f, [0, 0], { fTol: 1e-12, xTol: 1e-12 })
    expect(loose.converged).toBe(true)
    expect(loose.iterations).toBeLessThanOrEqual(tight.iterations)
    // A tiny maxIterations with no restarts is honored (cannot converge a 2-D quadratic in 1 iter).
    const capped = minimize(f, [0, 0], { maxIterations: 1, maxRestarts: 0 })
    expect(capped.iterations).toBeLessThanOrEqual(1)
  })
})
