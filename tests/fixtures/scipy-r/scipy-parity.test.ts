import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { DISTRIBUTIONS } from '../../../src/engine/distributions/index'
import {
  adStatistic,
  chiSquaredGof,
  chiSquaredGofDiscrete,
  cramerVonMises,
  ksStatistic,
} from '../../../src/engine/gof'
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

/** Params whose tight iterative diagnostic (ITERATIVE_PARAM_RTOL) is SKIPPED because scipy's
 *  own .fit under-converges / the likelihood is flat there, so HardFit legitimately differs while
 *  PASSING the LL cross-check (HardFit LL >= scipy LL). The LL gate is the real contract; this
 *  diagnostic is informational. Keyed by distribution name -> param keys to skip. */
const ITERATIVE_PARAM_DIAGNOSTIC_SKIP: Record<string, readonly string[]> = {
  // Near-normal/heavy-tail data: the LL is flat in log(df), so df is not 1e-3-identifiable even
  // though HardFit reaches >= scipy's LL (often a marginally BETTER optimum). Skip the df diagnostic;
  // the LL cross-check still runs and is the gate.
  [DistributionName.StudentT]: ['df'],
}

const AICC_INFINITY_SENTINEL = 'Infinity'

// --- Fixture shape ---------------------------------------------------------

/** One discrete chi-square cell from the gen_fixtures oracle; `hi` is the "Infinity" sentinel
 *  string for the unbounded upper-tail cell. */
interface DiscreteCellOracle {
  lo: number
  hi: number | string
  observed: number
  expected: number
}
interface ChiSquaredOracle {
  bins: number
  df: number
  statistic: number
  // Continuous (equiprobable) fixtures carry edges + a scalar expected; discrete fixtures carry cells.
  edges?: number[]
  observed?: number[]
  expected?: number
  cells?: DiscreteCellOracle[]
}
interface ModeB {
  // Discrete fixtures emit null for the EDF tests (invalid under ties); continuous emit numbers.
  ks: number | null
  adRaw: number | null
  cvm: number | null
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
  // M2.3 Batch A closed-form MLE families (params gated to 1e-9 + the universal LL cross-check).
  DistributionName.Uniform,
  DistributionName.Rayleigh,
  DistributionName.Pareto,
  // M2.3 Batch B closed-form MLE family (Lévy: c = n/Σ(1/x); params 1e-9 + LL floor).
  DistributionName.Levy,
  // M2.3 Batch C closed-form discrete MLE families (Negative-Binomial is iterative → LL-only).
  DistributionName.Poisson,
  DistributionName.Geometric,
  DistributionName.DiscreteUniform,
]

/** Decode a discrete-cell upper bound: the gen_fixtures oracle emits the unbounded tail's `hi`
 *  as the string "Infinity"; the engine uses `Number.POSITIVE_INFINITY`. */
function decodeCellHi(hi: number | string): number {
  return hi === AICC_INFINITY_SENTINEL ? Number.POSITIVE_INFINITY : (hi as number)
}

for (const distName of Object.values(DistributionName)) {
  const file = loadFixture(distName)
  const dist = distributionByName(distName)
  const isClosedForm = CLOSED_FORM_NAMES.includes(distName)

  const isDiscrete = dist.kind === 'discrete'

  describe(`scipy parity — ${distName} (numpy ${file.manifest.numpy}, scipy ${file.manifest.scipy})`, () => {
    for (const fx of file.fixtures) {
      describe(`dataset ${fx.dataset} (n=${fx.data.length})`, () => {
        // Mode B: GoF arithmetic at the FIXED scipy-derived params.
        const cdf = (x: number): number => dist.cdf(x, fx.fixedParams)
        const quantile = (prob: number): number => dist.quantile(prob, fx.fixedParams)

        if (!isDiscrete) {
          it('KS D matches scipy at fixed params', () => {
            expectClose(ksStatistic(fx.data, cdf), fx.modeB.ks as number, PARITY_RTOL)
          })
          it('raw A² matches scipy at fixed params', () => {
            expectClose(adStatistic(fx.data, cdf), fx.modeB.adRaw as number, PARITY_RTOL)
          })
          it('Cramér-von Mises n·ω² matches scipy at fixed params', () => {
            expectClose(cramerVonMises(fx.data, cdf), fx.modeB.cvm as number, PARITY_RTOL)
          })
          it('chi-square binning shape + edges + statistic match scipy at fixed params', () => {
            const oracle = fx.modeB.chiSquared
            const edges = oracle.edges ?? []
            const got = chiSquaredGof(fx.data, quantile, dist.k)
            // Binning shape is the hard coupling — must match exactly.
            expect(got.bins).toBe(oracle.bins)
            expect(got.df).toBe(oracle.df)
            expect(edges.length).toBe(oracle.bins - 1)
            // Assert the equiprobable edges Q(j/k) at machine precision. This pins the binning
            // (equal edges + no datum within 1e-9 of an edge ⇒ identical observed counts) AND
            // directly exercises the quantile-slot mapping (gamma RATE / weibull SCALE), which
            // the cdf-based checks only cover indirectly.
            for (let j = 1; j < oracle.bins; j++) {
              const oracleEdge = edges[j - 1]
              expect(oracleEdge, `missing oracle edge ${j}`).toBeTypeOf('number')
              expectClose(quantile(j / oracle.bins), oracleEdge as number, PARITY_RTOL)
            }
            // The statistic itself is a softer, binning-coupled gate: a near-edge bin flip
            // between scipy.ppf and @stdlib.quantile could shift it without indicating a bug.
            // With the edges pinned above this is belt-and-suspenders, hence the looser tol.
            expectClose(got.statistic, oracle.statistic, CHI_SQUARED_RTOL)
          })
        } else {
          it('discrete PMF-binned χ² cells + observed + expected match scipy at fixed params', () => {
            const oracle = fx.modeB.chiSquared
            const cells = oracle.cells ?? []
            const pmf = (x: number): number => Math.exp(dist.logpdf(x, fx.fixedParams))
            const support = dist.support?.(fx.fixedParams) ?? {
              min: 0,
              max: Number.POSITIVE_INFINITY,
            }
            const got = chiSquaredGofDiscrete(fx.data, pmf, cdf, support.min, support.max, dist.k)
            // Pin the binning itself, not just the statistic: cell count, edges, observed counts,
            // and expected counts must match the gen_fixtures.py mirror exactly (the batch's #1
            // correctness risk — a TS/Python merge divergence surfaces here).
            expect(got.bins).toBe(oracle.bins)
            expect(got.df).toBe(oracle.df)
            expect(got.cells.length).toBe(cells.length)
            for (let i = 0; i < cells.length; i++) {
              const oc = cells[i]
              const gc = got.cells[i]
              expect(oc, `missing oracle cell ${i}`).toBeTypeOf('object')
              if (oc === undefined || gc === undefined) continue
              expect(gc.lo).toBe(oc.lo)
              expect(decodeCellHi(gc.hi)).toBe(decodeCellHi(oc.hi))
              expect(gc.observed).toBe(oc.observed)
              expectClose(gc.expected, oc.expected, PARITY_RTOL)
            }
            // With cells pinned, the statistic can be asserted tightly.
            expectClose(got.statistic, oracle.statistic, PARITY_RTOL)
          })
        }

        // Mode A: fit-from-data.
        it(`fit-from-data ${isClosedForm ? 'reproduces the analytic MLE' : "is at least as good as scipy's"}`, () => {
          const fitted = dist.fit(fx.data)
          const ll = logLik(fx.data, (x) => dist.logpdf(x, fitted))

          // UNIVERSAL independent cross-check (closed-form AND iterative): HardFit's fitted
          // log-likelihood must reach scipy's independent `.fit` LL (minus slack). scipy's
          // optimizer does not share HardFit's MLE formula, so a formula bug replicated
          // identically in the Python emit and the TS engine yields suboptimal params whose LL
          // falls below scipy's here — it cannot self-cancel the way a param check against a
          // shared-formula reference could. Degrades gracefully at boundary MLEs (uniform/Pareto),
          // where a gradient≈0 check would misfire but HardFit-correct ≥ scipy ≥ HardFit-wrong.
          expect(ll).toBeGreaterThanOrEqual(fx.modeA.logLik - LOG_LIK_SLACK)

          if (isClosedForm) {
            // Closed-form families ALSO pin params to the numpy-analytic MLE at machine precision.
            for (const [key, ref] of Object.entries(fx.modeA.params)) {
              const value = fitted[key]
              expect(value, `missing fitted param "${key}"`).toBeTypeOf('number')
              expectClose(value as number, ref, PARITY_RTOL)
            }
          } else {
            // Iterative families: params are a loose diagnostic only (scipy under-converges).
            const skip = ITERATIVE_PARAM_DIAGNOSTIC_SKIP[distName] ?? []
            for (const [key, ref] of Object.entries(fx.modeA.params)) {
              if (skip.includes(key)) continue // documented: scipy under-converges / LL flat here
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
