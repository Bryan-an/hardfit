import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DISTRIBUTIONS } from '../../../src/engine/distributions/index'
import { adStatistic, chiSquaredGof, cramerVonMises, ksStatistic } from '../../../src/engine/gof'
import { logLik } from '../../../src/engine/selection'
import type { Distribution, FittedParams } from '../../../src/engine/types'
import { DistributionName } from '../../../src/engine/types'
import { expectClose } from '../../../src/test/relClose'

/**
 * scipy reference-fixture parity gate. Reads the committed JSON fixtures
 * (generated locally by scripts/gen_fixtures.py — NEVER Python in CI) and
 * asserts HardFit's engine against scipy in two modes:
 *
 *  Mode B — GoF arithmetic at the FIXED scipy-derived params, rtol 1e-9. This
 *    is the real numerical oracle for the GoF layer: KS D, raw A², CvM n·ω².
 *    Chi-square is a SOFTER gate (binning-coupled): the bin shape (k/df/observed)
 *    must match exactly; the statistic only at a loose tol (a near-edge bin flip
 *    between scipy.ppf and @stdlib.quantile would shift it, not a real bug).
 *
 *  Mode A — fit-from-data. Closed-form families (normal/lognormal/exponential)
 *    assert params within 1e-9 of the numpy-analytic reference. Iterative
 *    families (gamma/weibull) assert HardFit's log-likelihood is AT LEAST AS
 *    GOOD as scipy's (≥ oracle − tol) — scipy under-converges, so param equality
 *    is not the contract; params are checked only as a loose diagnostic.
 */

// --- Tolerances (named; no magic literals) ---------------------------------

/** Mode B EDF statistics + closed-form Mode A params: machine-precision parity. */
const PARITY_RTOL = 1e-9
/** Chi-square statistic: softer, absorbs a scipy.ppf vs @stdlib.quantile edge flip. */
const CHI_SQUARED_RTOL = 1e-6
/** Mode A iterative LL: HardFit must reach within this of scipy's maximized LL. */
const LOG_LIK_SLACK = 1e-6
/** Loose diagnostic on iterative fit params (NOT a gate; scipy under-converges). */
const ITERATIVE_PARAM_RTOL = 1e-3

const AICC_INFINITY_SENTINEL = 'Infinity'

// --- Fixture shape ---------------------------------------------------------

interface ChiSquaredOracle {
  bins: number
  edges: number[]
  observed: number[]
  expected: number
  df: number
  statistic: number
}
interface ModeB {
  ks: number
  adRaw: number
  cvm: number
  chiSquared: ChiSquaredOracle
}
interface ModeA {
  form: 'closed-form' | 'iterative'
  params: FittedParams
  logLik: number
  aicc: number | string
}
interface Fixture {
  dataset: string
  data: number[]
  fixedParams: FittedParams
  modeB: ModeB
  modeA: ModeA
}
interface FixtureFile {
  distribution: string
  manifest: { python: string; numpy: string; scipy: string }
  fixtures: Fixture[]
}

function loadFixture(distName: string): FixtureFile {
  const path = join(import.meta.dirname, `${distName}.json`)
  return JSON.parse(readFileSync(path, 'utf8')) as FixtureFile
}

function distributionByName(name: string): Distribution {
  const dist = DISTRIBUTIONS.find((d) => d.name === name)
  if (dist === undefined) throw new Error(`no HardFit distribution named "${name}"`)
  return dist
}

/** Decode the JSON AICc, which carries Infinity as a string sentinel. */
function decodeAicc(aicc: number | string): number {
  return aicc === AICC_INFINITY_SENTINEL ? Number.POSITIVE_INFINITY : (aicc as number)
}

// --- Distributions under test (closed-form families gate params; iterative gate LL) ---

const CLOSED_FORM_NAMES: readonly string[] = [
  DistributionName.Normal,
  DistributionName.Lognormal,
  DistributionName.Exponential,
]

for (const distName of Object.values(DistributionName)) {
  const file = loadFixture(distName)
  const dist = distributionByName(distName)
  const isClosedForm = CLOSED_FORM_NAMES.includes(distName)

  describe(`scipy parity — ${distName} (numpy ${file.manifest.numpy}, scipy ${file.manifest.scipy})`, () => {
    for (const fx of file.fixtures) {
      describe(`dataset ${fx.dataset} (n=${fx.data.length})`, () => {
        // Mode B: GoF arithmetic at the FIXED scipy-derived params.
        const cdf = (x: number): number => dist.cdf(x, fx.fixedParams)
        const quantile = (prob: number): number => dist.quantile(prob, fx.fixedParams)

        it('KS D matches scipy at fixed params', () => {
          expectClose(ksStatistic(fx.data, cdf), fx.modeB.ks, PARITY_RTOL)
        })
        it('raw A² matches scipy at fixed params', () => {
          expectClose(adStatistic(fx.data, cdf), fx.modeB.adRaw, PARITY_RTOL)
        })
        it('Cramér-von Mises n·ω² matches scipy at fixed params', () => {
          expectClose(cramerVonMises(fx.data, cdf), fx.modeB.cvm, PARITY_RTOL)
        })
        it('chi-square binning shape + edges + statistic match scipy at fixed params', () => {
          const oracle = fx.modeB.chiSquared
          const got = chiSquaredGof(fx.data, quantile, dist.k)
          // Binning shape is the hard coupling — must match exactly.
          expect(got.bins).toBe(oracle.bins)
          expect(got.df).toBe(oracle.df)
          expect(oracle.edges.length).toBe(oracle.bins - 1)
          // Assert the equiprobable edges Q(j/k) at machine precision. This pins the binning
          // (equal edges + no datum within 1e-9 of an edge ⇒ identical observed counts) AND
          // directly exercises the quantile-slot mapping (gamma RATE / weibull SCALE), which
          // the cdf-based checks only cover indirectly.
          for (let j = 1; j < oracle.bins; j++) {
            const oracleEdge = oracle.edges[j - 1]
            expect(oracleEdge, `missing oracle edge ${j}`).toBeTypeOf('number')
            expectClose(quantile(j / oracle.bins), oracleEdge as number, PARITY_RTOL)
          }
          // The statistic itself is a softer, binning-coupled gate: a near-edge bin flip
          // between scipy.ppf and @stdlib.quantile could shift it without indicating a bug.
          // With the edges pinned above this is belt-and-suspenders, hence the looser tol.
          expectClose(got.statistic, oracle.statistic, CHI_SQUARED_RTOL)
        })

        // Mode A: fit-from-data.
        it(`fit-from-data ${isClosedForm ? 'reproduces the analytic MLE' : "is at least as good as scipy's"}`, () => {
          const fitted = dist.fit(fx.data)
          const ll = logLik(fx.data, (x) => dist.logpdf(x, fitted))

          if (isClosedForm) {
            for (const [key, ref] of Object.entries(fx.modeA.params)) {
              const value = fitted[key]
              expect(value, `missing fitted param "${key}"`).toBeTypeOf('number')
              expectClose(value as number, ref, PARITY_RTOL)
            }
          } else {
            // HardFit must achieve a log-likelihood at least as good as scipy's.
            expect(ll).toBeGreaterThanOrEqual(fx.modeA.logLik - LOG_LIK_SLACK)
            // Params only as a loose diagnostic (scipy under-converges; not a gate).
            for (const [key, ref] of Object.entries(fx.modeA.params)) {
              const value = fitted[key]
              if (typeof value === 'number') expectClose(value, ref, ITERATIVE_PARAM_RTOL)
            }
          }
        })

        it('AICc decodes (string "Infinity" sentinel → +Infinity, else a finite number)', () => {
          const decoded = decodeAicc(fx.modeA.aicc)
          if (fx.modeA.aicc === AICC_INFINITY_SENTINEL) {
            // n ≤ k + 1: HardFit's AICc small-sample correction is undefined → +Infinity.
            expect(decoded).toBe(Number.POSITIVE_INFINITY)
          } else {
            expect(Number.isFinite(decoded)).toBe(true)
          }
        })
      })
    }
  })
}
