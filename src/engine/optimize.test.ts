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

/** 2-D Rosenbrock banana: f(x,y) = (1−x)² + 100(y−x²)²; global min 0 at (1,1).
 *  A single-pass Nelder–Mead from (−1.2, 1) stalls near ~1e-3; only restart-from-best reaches ~1e-6. */
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

  it('2: 2-D Rosenbrock from (−1.2,1) reaches (1,1) to ~1e-6 (proves restart-from-best fired)', () => {
    const r = minimize(rosenbrock, [-1.2, 1])
    expectClose(r.x[0] ?? Number.NaN, 1, 1e-5, 1e-6)
    expectClose(r.x[1] ?? Number.NaN, 1, 1e-5, 1e-6)
    expect(r.fx).toBeLessThan(1e-6)
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
    // f = (x−2)² where x>0, but −Inf for x ≤ 0 (a degenerate "overfit" region). The optimizer must
    // stay on the finite region and recover x=2, never collapse onto the −Inf basin.
    const f = (v: readonly number[]) => {
      const x = v[0] ?? Number.NaN
      if (x <= 0) return Number.NEGATIVE_INFINITY
      return (x - 2) ** 2
    }
    const r = minimize(f, [1])
    expect(r.converged).toBe(true)
    expectClose(r.x[0] ?? Number.NaN, 2, 1e-9)
    expect(r.fx).toBeLessThan(1e-9)
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
