import { describe, expect, it } from 'vitest'
import { expectClose } from '../../test/relClose'
import { mean } from '../math'
import { geometric } from './geometric'

/** Mirrors the (unexported) GeometricParams slot type used inside geometric.ts; lets the test
 *  read fitted params by name without `?? NaN` noise. `geometric.fit` returns the engine-wide
 *  `FittedParams` (Record<string, number>) DTO, so a narrowing cast is expected here. */
type GeometricParams = { p: number }

/** Asymmetric integer counts ({0,1,...} failures convention): non-uniform, includes a zero. */
const sample = [0, 1, 0, 2, 3, 0, 1, 1, 4, 0]

describe('geometric', () => {
  it('MLE: p = 1 / (1 + mean) exactly ({0,1,...} convention, not 1/mean)', () => {
    const par = geometric.fit(sample) as GeometricParams
    expect(par.p).toBe(1 / (1 + mean(sample)))
  })
  it('logpdf delegates to @stdlib logpmf: matches the elementary closed-form log-PMF', () => {
    // CONVENTION GUARD: geometric PMF (failures convention) P(X=x) = (1-p)^x * p.
    // log-PMF = x*ln(1-p) + ln(p). For p=0.4, x=2 (ASYMMETRIC), written BY HAND (NOT via @stdlib).
    const expected = 2 * Math.log(0.6) + Math.log(0.4)
    expectClose(geometric.logpdf(2, { p: 0.4 }), expected, 1e-9)
  })
  it('cdf at 0 equals p (P(X=0) = p)', () => {
    // {0,1,...} convention: F(0) = P(X=0) = p. For p=0.4 -> 0.4 (1/mean convention would give P(X=1)).
    expectClose(geometric.cdf(0, { p: 0.4 }), 0.4)
  })
  it('quantile returns an integer whose cdf reaches the probability', () => {
    // Discrete quantile is integer-valued; check the defining inequality cdf(Q(r)) >= r.
    const par = { p: 0.4 }
    const q = geometric.quantile(0.5, par)
    expect(Number.isInteger(q)).toBe(true)
    expect(geometric.cdf(q, par)).toBeGreaterThanOrEqual(0.5)
  })
  it('all-zeros is the valid boundary p = 1 (no throw)', () => {
    const par = geometric.fit([0, 0, 0, 0]) as GeometricParams
    expect(par.p).toBe(1)
  })
  it('rejects non-integer data', () => expect(() => geometric.fit([0.5, 1, 2])).toThrow())
  it('rejects negative data (support violation)', () =>
    expect(() => geometric.fit([-1, 0, 1])).toThrow())
  it('k = 1', () => expect(geometric.k).toBe(1))
  it("kind = 'discrete'", () => expect(geometric.kind).toBe('discrete'))
  it('support is {0, 1, 2, ...}: min 0, max +Infinity', () => {
    expect(geometric.support?.({ p: 0.4 })).toEqual({ min: 0, max: Number.POSITIVE_INFINITY })
  })
})
