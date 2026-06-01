import { expect } from 'vitest'

/** numpy.allclose-style check: |a-b| <= rtol*|b| + atol. Better than toBeCloseTo for large/small magnitudes. */
export function expectClose(actual: number, expected: number, rtol = 1e-9, atol = 1e-12): void {
  expect(Number.isFinite(actual), `expected finite, got ${actual}`).toBe(true)
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rtol * Math.abs(expected) + atol)
}
