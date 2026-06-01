import { describe, expect, it } from 'vitest'
import { expectClose } from '../test/relClose'
import { adNormalPValue, adResult } from './gof-pvalues'
import { DistributionName } from './types'

describe("adNormalPValue (D'Agostino & Stephens 1986 piecewise)", () => {
  // CRITICAL GATE: the classic A*² normal critical values. The literature target is an
  // ABSOLUTE band (the published approximation does not reproduce the table to 4 decimals,
  // ~3e-4 error is expected) — so 5e-3 goes in the atol slot, NOT rtol.
  it('A*²=0.752 → p≈0.05 (5% normal critical value)', () => {
    expectClose(adNormalPValue(0.752), 0.05, 0, 5e-3)
  })
  it('A*²=1.035 → p≈0.01 (1% normal critical value)', () => {
    expectClose(adNormalPValue(1.035), 0.01, 0, 5e-3)
  })

  // Branch coverage: 0.752/1.035 only exercise the a≥0.6 branch. Cover the other three
  // segments so a coefficient typo cannot ship silently. Loose tolerance — these are the
  // approximation's own outputs, asserted only to pin the active branch + sane magnitude.
  it('a≤0.2 branch: A*²=0.1 → p≈0.996', () => {
    expectClose(adNormalPValue(0.1), 0.996, 0, 5e-3)
  })
  it('0.2<a≤0.34 branch: A*²=0.3 → p≈0.583', () => {
    expectClose(adNormalPValue(0.3), 0.583, 0, 5e-3)
  })
  it('0.34<a<0.6 branch: A*²=0.5 → p≈0.209', () => {
    expectClose(adNormalPValue(0.5), 0.209, 0, 5e-3)
  })
})

describe('adResult — Normal & Lognormal (closed-form)', () => {
  it("normal uses the D'Agostino-Stephens closed form", () => {
    const r = adResult(DistributionName.Normal, 0.4, 50)
    expect(r.method).toBe('closed-form')
    expect(r.statistic).toBe(0.4) // raw A² reported alongside
    expect(r.pValue).not.toBeNull()
  })
  it('lognormal uses the same closed form (normal test on ln x)', () => {
    const r = adResult(DistributionName.Lognormal, 0.4, 50)
    expect(r.method).toBe('closed-form')
    expect(r.statistic).toBe(0.4)
    expect(r.pValue).not.toBeNull()
  })
  it('applies the A*² = A²(1 + 0.75/n + 2.25/n²) adjustment before the p-value', () => {
    // n=50: A*² = 0.4·(1 + 0.75/50 + 2.25/2500) = 0.4·1.0159 = 0.40636 → adNormalPValue branch a≥0.6? no.
    const n = 50
    const a2 = 0.4
    const aStar = a2 * (1 + 0.75 / n + 2.25 / (n * n))
    const r = adResult(DistributionName.Normal, a2, n)
    expectClose(r.pValue ?? Number.NaN, adNormalPValue(aStar), 0, 1e-12)
  })
})

describe('adResult — Exponential (bracketed table, A*²=A²(1+0.6/n))', () => {
  it('A*² landing on the 5% critical (1.321) → p≈0.05', () => {
    // n=10 → A² = 1.321/(1+0.6/10) so the adjusted A*² equals the 5% critical exactly.
    const n = 10
    const a2 = 1.321 / (1 + 0.6 / n)
    const r = adResult(DistributionName.Exponential, a2, n)
    expect(r.method).toBe('table')
    expect(r.statistic).toBe(a2)
    expectClose(r.pValue ?? Number.NaN, 0.05, 0, 1e-9)
  })
  it('interpolates strictly between table nodes (discriminates the arithmetic)', () => {
    // A*²=1.19 sits between 1.062@0.10 and 1.321@0.05 → linear interp p≈0.07529.
    // Build A² so A*² = 1.19 exactly at n=10.
    const n = 10
    const a2 = 1.19 / (1 + 0.6 / n)
    const r = adResult(DistributionName.Exponential, a2, n)
    expectClose(r.pValue ?? Number.NaN, 0.0752895752895753, 0, 1e-9)
  })
  it('clamps below the smallest critical to the largest significance (0.15)', () => {
    // A*²=0.3 < 0.916 → clamp to the top sig boundary.
    const n = 10
    const a2 = 0.3 / (1 + 0.6 / n)
    const r = adResult(DistributionName.Exponential, a2, n)
    expectClose(r.pValue ?? Number.NaN, 0.15, 0, 1e-12)
  })
  it('clamps above the largest critical to the smallest significance (0.01)', () => {
    // A*²=2.5 > 1.959 → clamp to the bottom sig boundary.
    const n = 10
    const a2 = 2.5 / (1 + 0.6 / n)
    const r = adResult(DistributionName.Exponential, a2, n)
    expectClose(r.pValue ?? Number.NaN, 0.01, 0, 1e-12)
  })
})

describe('adResult — Weibull (fixed Gumbel/EV table, A*²=A²(1+0.2/√n))', () => {
  it('A*² landing on the 5% Gumbel critical (0.757) → p≈0.05', () => {
    // n=16 → A² = 0.757/(1+0.2/√16) so the adjusted A*² equals the 5% critical exactly.
    const n = 16
    const a2 = 0.757 / (1 + 0.2 / Math.sqrt(n))
    const r = adResult(DistributionName.Weibull, a2, n)
    expect(r.method).toBe('table')
    expect(r.statistic).toBe(a2)
    expectClose(r.pValue ?? Number.NaN, 0.05, 0, 1e-9)
  })
  it('clamps below the smallest Gumbel critical to the largest significance (0.25)', () => {
    // A*²=0.2 < 0.474 → clamp to the top Gumbel sig boundary.
    const n = 16
    const a2 = 0.2 / (1 + 0.2 / Math.sqrt(n))
    const r = adResult(DistributionName.Weibull, a2, n)
    expectClose(r.pValue ?? Number.NaN, 0.25, 0, 1e-12)
  })
})

describe('adResult — Gamma + unknown (diagnostic, no closed-form criticals)', () => {
  it('gamma yields a diagnostic result with no p-value', () => {
    const r = adResult(DistributionName.Gamma, 0.8, 40)
    expect(r.method).toBe('diagnostic')
    expect(r.pValue).toBeNull()
    expect(r.statistic).toBe(0.8) // raw A² still reported
  })
  it('an unknown distribution name falls through to diagnostic/null', () => {
    const r = adResult('pareto', 0.8, 40)
    expect(r.method).toBe('diagnostic')
    expect(r.pValue).toBeNull()
  })
})
