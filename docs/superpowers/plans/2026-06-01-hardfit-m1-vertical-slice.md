# HardFit M1 — Vertical Slice (5-distribution fitting, end-to-end, deployed) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user can paste/upload a column of numbers, click *Fit*, and see the 5 distributions (Normal, Lognormal, Exponential, Gamma, Weibull) fit by MLE, **ranked by AICc** with KS shown as a diagnostic, plus a histogram with the best-fit PDF overlaid — all computed in a Web Worker and **deployed live**.

**Architecture:** A pure, framework-agnostic `src/engine/` (zero DOM/React/Comlink imports) holds the math: per-distribution MLE + `logpdf`/`cdf` (densities/CDFs reused from `@stdlib`; the fitting/GoF/ranking layer is ours), a KS statistic, AIC/AICc/BIC + Akaike-weight ranking, and a `fitAll` orchestrator. The engine is exposed over one dedicated Comlink worker; React talks only to the typed worker proxy. The chart is Plotly via a thin custom component (no `react-plotly.js`). Builds on M0 (scaffold, strict TS, Vitest, Playwright, Cloudflare deploy).

**Tech Stack:** TypeScript (strictest), `@stdlib` per-function packages (distribution `logpdf`/`cdf` + `digamma`/`trigamma`), Comlink 4.4.2, `plotly.js-cartesian-dist-min` 3.5.1, Vitest + fast-check, Playwright. Deploys to the existing Cloudflare Workers Static Assets project on push.

> ### Locked technical facts (verified 2026-06-01 — do not re-derive)
> - **`@stdlib` packages are independently versioned.** Pin EACH at the exact verified version (several have NO 0.3.x). Imports are CommonJS `export = fn` (default export is the function; **no named exports** — never `import { pdf }`); they typecheck under `esModuleInterop: true` (set by `@tsconfig/strictest`, already in `tsconfig.app.json`). Task 1 is a smoke test that proves install + interop before any engine code.
> - **Parameter conventions (LOAD-BEARING):** normal/lognormal `(x, mu, sigma)` σ = std-dev (lognormal: of `ln X`). exponential `(x, lambda)` λ = **RATE**. gamma `(x, alpha, beta)` α = shape, β = **RATE**. weibull `(x, k, lambda)` k = shape, λ = **SCALE**. Same letter `lambda` = rate in exponential but scale in weibull.
> - **MLE:** normal/lognormal/exponential closed-form (variance uses **1/n**, not 1/(n−1)); gamma via Minka-seeded 1-D Newton on `ln(k) − ψ(k) = ln(mean) − mean(ln x)` (scale θ = mean/k); weibull via Newton on the shape score (scale λ = (mean(xᵏ))^(1/k)). All require `x > 0` except normal (ℝ) and exponential (`x ≥ 0`). **Lognormal LL must subtract `Σ ln xᵢ`** (Jacobian) to be comparable.
> - **KS:** `D = max(D⁺, D⁻)`; over sorted data with 0-based index `j` (math `i = j+1`): `D⁺ = max((j+1)/n − F(x_j))`, `D⁻ = max(F(x_j) − j/n)`. The KS p-value is **invalid for fitted params (Lilliefors)** → v1 reports `D` as a diagnostic only and **ranks by AICc**.
> - **Selection:** `LL = Σ logpdf(xᵢ)` (sum logs, never log of product); `AIC = 2k − 2·LL`; `AICc = AIC + 2k(k+1)/(n−k−1)`, **`+Infinity` when `n−k−1 ≤ 0`**; `BIC = k·ln n − 2·LL`. Rank ascending by AICc; `Δᵢ = AICcᵢ − min`; weights `wᵢ = exp(−Δᵢ/2)/Σ exp(−Δⱼ/2)` (the Δ form IS the stable form — never exponentiate raw AICc). `k`: normal/lognormal/gamma/weibull = 2, exponential = 1.
> - **Deploy:** the M0 Cloudflare Workers Static Assets project auto-deploys on push to `main` (config in `wrangler.jsonc`, SPA fallback via `not_found_handling`). The Worker is created with the Vite `new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' })` form (first arg MUST be a string literal, second MUST be `import.meta.url`).

> **Conventions for the executor:** repo root `/Users/bryan-andagoya/Development/personal/hard-fit-project/hard-fit`; on branch `main` (create a feature branch `feat/m1-vertical-slice` first — see Task 0). pnpm via Corepack only. After every code change keep these green: `pnpm typecheck`, `pnpm check`, `pnpm test`. Commit after each task with a `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer. Use **opus** for any execution subagent.

---

## File Structure

```
src/
├─ engine/                      # PURE TS — no DOM/React/Comlink/Worker imports; runs in Node
│  ├─ types.ts                  # FittedParams, Distribution, Fit, RankedFit DTOs
│  ├─ math.ts                   # mean, sampleStats helpers (shared, tiny)
│  ├─ distributions/
│  │  ├─ normal.ts  lognormal.ts  exponential.ts  gamma.ts  weibull.ts
│  │  └─ index.ts               # DISTRIBUTIONS registry (the 5)
│  ├─ gof.ts                    # ksStatistic(data, cdf)
│  ├─ selection.ts              # logLik, aic, aicc, bic, rankByAICc
│  ├─ fitAll.ts                 # fitAll(data, onProgress?) -> RankedFit[]
│  ├─ stdlib.smoke.test.ts      # interop/version gate for @stdlib
│  └─ index.ts                  # public engine API (re-exports fitAll, types)
├─ engine.worker.ts             # Comlink.expose(engineApi)  (Vite worker chunk)
├─ engineClient.ts              # Comlink.wrap<EngineApi> + runFitAll + teardown
├─ lib/parseNumbers.ts          # parse a numeric column from pasted text / CSV (single column)
├─ ui/
│  ├─ PlotlyChart.tsx           # thin Plotly component (Plotly.react/purge)
│  ├─ DataInput.tsx             # textarea + file input -> number[]
│  ├─ ResultsTable.tsx          # ranked fit table
│  └─ buildHistogramPdf.ts      # build Plotly traces (histogram + fitted PDF grid)
├─ plotly-cartesian.d.ts        # ambient shim for plotly.js-cartesian-dist-min
├─ App.tsx                      # wires DataInput -> worker fitAll -> ResultsTable + chart
└─ test/relClose.ts             # relative-tolerance assertion helper for numeric tests
e2e/fit.spec.ts                 # upload/paste -> fit -> table + chart render (under CSP)
```

**Module boundaries:** `engine/` imports only `@stdlib` (+ its own files) — never React/DOM/Comlink. `engine.worker.ts` is the only file importing Comlink + the engine. `App.tsx`/`ui/` import only `engineClient` (the typed proxy), never `engine/` directly. `import type { EngineApi }` is type-only.

---

## Task 0: Branch + dependencies + @stdlib interop smoke test

**Files:** Create `src/engine/stdlib.smoke.test.ts`, `src/test/relClose.ts`. Modify `package.json`, `pnpm-lock.yaml`.

- [ ] **Step 1: Branch from main**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b feat/m1-vertical-slice
```

- [ ] **Step 2: Install the exact @stdlib packages (versions are NOT uniform — pin each)**

```bash
pnpm add -E \
  @stdlib/stats-base-dists-normal-logpdf@0.3.1 @stdlib/stats-base-dists-normal-cdf@0.3.1 \
  @stdlib/stats-base-dists-lognormal-logpdf@0.2.3 @stdlib/stats-base-dists-lognormal-cdf@0.2.3 \
  @stdlib/stats-base-dists-exponential-logpdf@0.3.1 @stdlib/stats-base-dists-exponential-cdf@0.3.1 \
  @stdlib/stats-base-dists-gamma-logpdf@0.3.1 @stdlib/stats-base-dists-gamma-cdf@0.3.1 \
  @stdlib/stats-base-dists-weibull-logpdf@0.3.1 @stdlib/stats-base-dists-weibull-cdf@0.3.1 \
  @stdlib/math-base-special-digamma@0.3.1 @stdlib/math-base-special-trigamma@0.3.1 \
  @stdlib/math-base-special-gammaln@0.3.1
```
Expected: installs cleanly (these are pure-JS, no build scripts → no `onlyBuiltDependencies` change). If any version 404s, run `npm view <pkg> version` and pin the latest that exists. Then run `pnpm run licenses` — all `@stdlib` packages are `Apache-2.0 AND BSL-1.0` (both already in the allow-list) → must still exit 0.

- [ ] **Step 3: Write a relative-tolerance test helper**

`src/test/relClose.ts`:
```typescript
import { expect } from 'vitest'

/** numpy.allclose-style check: |a-b| <= rtol*|b| + atol. Better than toBeCloseTo for large/small magnitudes. */
export function expectClose(actual: number, expected: number, rtol = 1e-9, atol = 1e-12): void {
  expect(Number.isFinite(actual), `expected finite, got ${actual}`).toBe(true)
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(rtol * Math.abs(expected) + atol)
}
```

- [ ] **Step 4: Write the @stdlib smoke test (interop + version + convention gate)**

`src/engine/stdlib.smoke.test.ts`:
```typescript
import { describe, it } from 'vitest'
import normalCdf from '@stdlib/stats-base-dists-normal-cdf'
import normalLogpdf from '@stdlib/stats-base-dists-normal-logpdf'
import expCdf from '@stdlib/stats-base-dists-exponential-cdf'
import gammaCdf from '@stdlib/stats-base-dists-gamma-cdf'
import weibullCdf from '@stdlib/stats-base-dists-weibull-cdf'
import lognormalCdf from '@stdlib/stats-base-dists-lognormal-cdf'
import digamma from '@stdlib/math-base-special-digamma'
import trigamma from '@stdlib/math-base-special-trigamma'
import { expectClose } from '../test/relClose'

describe('@stdlib import + interop + parameter conventions', () => {
  it('normal cdf(0,0,1) = 0.5; logpdf finite', () => {
    expectClose(normalCdf(0, 0, 1), 0.5)
    expectClose(normalLogpdf(0, 0, 1), -0.5 * Math.log(2 * Math.PI))
  })
  it('exponential cdf uses RATE: cdf(x,lambda)=1-e^{-lambda x}', () => {
    expectClose(expCdf(1, 2), 1 - Math.exp(-2)) // lambda=2 (rate)
  })
  it('gamma cdf uses shape+RATE (alpha,beta): mean=alpha/beta', () => {
    // shape=2, rate=0.5 -> mean=4; cdf(4,2,0.5) ~ 0.59399415
    expectClose(gammaCdf(4, 2, 0.5), 0.5939941502901619, 1e-6)
  })
  it('weibull cdf uses shape+SCALE (k,lambda): F(lambda)=1-e^{-1}', () => {
    expectClose(weibullCdf(10, 2, 10), 1 - Math.exp(-1)) // x=scale -> 1-1/e
  })
  it('lognormal cdf params are on the LOG scale: F(1;0,1)=0.5', () => {
    expectClose(lognormalCdf(1, 0, 1), 0.5) // median = e^mu = 1
  })
  it('digamma/trigamma known values', () => {
    expectClose(digamma(1), -0.5772156649015329, 1e-9)
    expectClose(trigamma(1), Math.PI * Math.PI / 6, 1e-9)
  })
})
```

- [ ] **Step 5: Run the smoke test + typecheck**

```bash
pnpm test -- stdlib.smoke && pnpm typecheck
```
Expected: PASS. **If `pnpm typecheck` errors** with "Module ... can only be default-imported using the 'esModuleInterop' flag" or a `verbatimModuleSyntax` complaint on these CJS `export =` modules: confirm `tsconfig.app.json` has `"esModuleInterop": true` and `"allowSyntheticDefaultImports": true` (both come from `@tsconfig/strictest`; add them explicitly if missing). Do NOT switch to `import * as x` (the default export is the callable function). Re-run until green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test(engine): add @stdlib deps + interop/version/convention smoke test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 1: Engine types + shared math helpers

**Files:** Create `src/engine/types.ts`, `src/engine/math.ts`, `src/engine/math.test.ts`.

- [ ] **Step 1: Define the engine DTOs**

`src/engine/types.ts`:
```typescript
/** Fitted parameters in HardFit's own convention (documented per distribution). */
export type FittedParams = Record<string, number>

export interface Distribution {
  readonly name: string // machine id, e.g. 'normal'
  readonly label: string // display label, e.g. 'Normal'
  readonly k: number // number of estimated parameters (for AIC)
  /** MLE fit. Throws an Error if `data` violates the distribution's support. */
  fit(data: readonly number[]): FittedParams
  /** Natural-log density at x for fitted params (data scale). */
  logpdf(x: number, p: FittedParams): number
  /** Cumulative distribution function at x for fitted params. */
  cdf(x: number, p: FittedParams): number
}

/** A successful fit of one distribution with its diagnostics. */
export interface Fit {
  name: string
  label: string
  k: number
  params: FittedParams
  logLik: number
  aic: number
  aicc: number
  bic: number
  ks: number // Kolmogorov-Smirnov D (diagnostic only; p-value invalid for estimated params)
}

/** A Fit plus its ranking position. */
export interface RankedFit extends Fit {
  rank: number // 1 = best (lowest AICc)
  deltaAICc: number // AICc - min(AICc)
  weight: number // Akaike weight in (0,1]
}

/** A distribution that could not be fit to the given data (e.g. support violation). */
export interface FitFailure {
  name: string
  label: string
  error: string
}

export interface FitAllResult {
  ranked: RankedFit[]
  failures: FitFailure[]
  n: number
}
```

- [ ] **Step 2: Write the failing test for math helpers**

`src/engine/math.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { mean, meanLog, populationVariance } from './math'
import { expectClose } from '../test/relClose'

describe('math helpers', () => {
  it('mean', () => expectClose(mean([2, 4, 6]), 4))
  it('populationVariance divides by n (MLE), not n-1', () => {
    // values 2,4,6: mean 4, sum sq dev = 4+0+4 = 8, /3 = 2.6667
    expectClose(populationVariance([2, 4, 6]), 8 / 3)
  })
  it('meanLog', () => expectClose(meanLog([1, Math.E]), 0.5))
  it('mean throws on empty', () => expect(() => mean([])).toThrow())
})
```

- [ ] **Step 3: Run → fail**

```bash
pnpm test -- engine/math
```
Expected: FAIL (cannot resolve `./math`).

- [ ] **Step 4: Implement the helpers**

`src/engine/math.ts`:
```typescript
export function mean(x: readonly number[]): number {
  if (x.length === 0) throw new Error('mean: empty array')
  let s = 0
  for (const v of x) s += v
  return s / x.length
}

export function meanLog(x: readonly number[]): number {
  if (x.length === 0) throw new Error('meanLog: empty array')
  let s = 0
  for (const v of x) s += Math.log(v)
  return s / x.length
}

/** Population (MLE) variance: divides by n. */
export function populationVariance(x: readonly number[], mu = mean(x)): number {
  let s = 0
  for (const v of x) s += (v - mu) * (v - mu)
  return s / x.length
}
```

- [ ] **Step 5: Run → pass; commit**

```bash
pnpm test -- engine/math && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(engine): add engine DTOs and shared math helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Normal distribution (closed-form MLE) — the template

**Files:** Create `src/engine/distributions/normal.ts`, `src/engine/distributions/normal.test.ts`.

- [ ] **Step 1: Write the failing test**

`src/engine/distributions/normal.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import fc from 'fast-check'
import { normal } from './normal'
import { mean, populationVariance } from '../math'
import { expectClose } from '../../test/relClose'

describe('normal', () => {
  it('MLE: mu=mean, sigma=sqrt(population variance) (÷n)', () => {
    const data = [2, 4, 4, 4, 5, 5, 7, 9]
    const p = normal.fit(data)
    expectClose(p.mu, mean(data))
    expectClose(p.sigma, Math.sqrt(populationVariance(data)))
  })
  it('logpdf matches the standard normal density at 0', () => {
    expectClose(normal.logpdf(0, { mu: 0, sigma: 1 }), -0.5 * Math.log(2 * Math.PI))
  })
  it('cdf(mu) = 0.5', () => expectClose(normal.cdf(5, { mu: 5, sigma: 2 }), 0.5))
  it('property: logpdf is maximized at the MLE vs perturbed params (sum over data)', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: -50, max: 50, noNaN: true }), { minLength: 5, maxLength: 50 }), (data) => {
        const p = normal.fit(data)
        const ll = (q: typeof p) => data.reduce((s, x) => s + normal.logpdf(x, q), 0)
        if (p.sigma === 0) return // degenerate (constant data)
        return ll(p) >= ll({ mu: p.mu + 0.5, sigma: p.sigma }) - 1e-9
      }),
    )
  })
  it('k = 2', () => expect(normal.k).toBe(2))
})
```

- [ ] **Step 2: Run → fail**

```bash
pnpm test -- distributions/normal
```
Expected: FAIL (cannot resolve `./normal`).

- [ ] **Step 3: Implement**

`src/engine/distributions/normal.ts`:
```typescript
import logpdf from '@stdlib/stats-base-dists-normal-logpdf'
import cdf from '@stdlib/stats-base-dists-normal-cdf'
import type { Distribution, FittedParams } from '../types'
import { mean, populationVariance } from '../math'

// params: { mu = mean, sigma = standard deviation }  (data scale)
export const normal: Distribution = {
  name: 'normal',
  label: 'Normal',
  k: 2,
  fit(data) {
    if (data.length < 2) throw new Error('normal: need n >= 2')
    const mu = mean(data)
    const sigma = Math.sqrt(populationVariance(data, mu))
    if (!(sigma > 0)) throw new Error('normal: degenerate (zero variance)')
    return { mu, sigma }
  },
  logpdf(x: number, p: FittedParams): number {
    return logpdf(x, p.mu, p.sigma)
  },
  cdf(x: number, p: FittedParams): number {
    return cdf(x, p.mu, p.sigma)
  },
}
```

- [ ] **Step 4: Run → pass; commit**

```bash
pnpm test -- distributions/normal && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(engine): normal distribution (closed-form MLE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Lognormal & Exponential (closed-form)

**Files:** Create `lognormal.ts`/`lognormal.test.ts`, `exponential.ts`/`exponential.test.ts` under `src/engine/distributions/`.

- [ ] **Step 1: Lognormal failing test**

`src/engine/distributions/lognormal.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { lognormal } from './lognormal'
import { expectClose } from '../../test/relClose'

describe('lognormal', () => {
  it('fits Normal on ln(x): for data=[1, e, e^2] -> mu=1, sigma^2=2/3', () => {
    const p = lognormal.fit([1, Math.E, Math.E * Math.E]) // ln -> [0,1,2], mean 1, var(÷3)=2/3
    expectClose(p.mu, 1)
    expectClose(p.sigma, Math.sqrt(2 / 3))
  })
  it('rejects non-positive data', () => expect(() => lognormal.fit([1, 0, 2])).toThrow())
  it('cdf(median=e^mu)=0.5', () => expectClose(lognormal.cdf(Math.E, { mu: 1, sigma: 0.5 }), 0.5))
  it('k = 2', () => expect(lognormal.k).toBe(2))
})
```

- [ ] **Step 2: Run → fail, then implement**

`src/engine/distributions/lognormal.ts`:
```typescript
import logpdf from '@stdlib/stats-base-dists-lognormal-logpdf'
import cdf from '@stdlib/stats-base-dists-lognormal-cdf'
import type { Distribution, FittedParams } from '../types'

// params: { mu, sigma } = mean & std-dev of ln(X) (LOG scale, NOT data scale)
export const lognormal: Distribution = {
  name: 'lognormal',
  label: 'Lognormal',
  k: 2,
  fit(data) {
    if (data.length < 2) throw new Error('lognormal: need n >= 2')
    if (data.some((v) => v <= 0)) throw new Error('lognormal requires all x > 0')
    const n = data.length
    let s = 0
    for (const v of data) s += Math.log(v)
    const mu = s / n
    let ss = 0
    for (const v of data) {
      const d = Math.log(v) - mu
      ss += d * d
    }
    const sigma = Math.sqrt(ss / n)
    if (!(sigma > 0)) throw new Error('lognormal: degenerate (constant logs)')
    return { mu, sigma }
  },
  logpdf(x: number, p: FittedParams): number {
    return logpdf(x, p.mu, p.sigma)
  },
  cdf(x: number, p: FittedParams): number {
    return cdf(x, p.mu, p.sigma)
  },
}
```
> Note: `@stdlib`'s `lognormal-logpdf` already includes the `−ln x` Jacobian, so `Σ logpdf` is on the data scale and is directly comparable to the other distributions' LLs — no manual Jacobian needed when LL is computed via `logpdf`.

- [ ] **Step 3: Exponential failing test**

`src/engine/distributions/exponential.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { exponential } from './exponential'
import { expectClose } from '../../test/relClose'

describe('exponential', () => {
  it('MLE rate = 1/mean', () => {
    const p = exponential.fit([1, 2, 3, 4]) // mean 2.5 -> rate 0.4
    expectClose(p.rate, 0.4)
  })
  it('rejects negative data', () => expect(() => exponential.fit([1, -1])).toThrow())
  it('cdf(x,rate) = 1 - e^{-rate x}', () => expectClose(exponential.cdf(1, { rate: 2 }), 1 - Math.exp(-2)))
  it('logpdf passes RATE: = ln(rate) - rate*x', () => expectClose(exponential.logpdf(2, { rate: 0.5 }), Math.log(0.5) - 0.5 * 2))
  it('k = 1', () => expect(exponential.k).toBe(1))
})
```

- [ ] **Step 4: Implement exponential**

`src/engine/distributions/exponential.ts`:
```typescript
import logpdf from '@stdlib/stats-base-dists-exponential-logpdf'
import cdf from '@stdlib/stats-base-dists-exponential-cdf'
import type { Distribution, FittedParams } from '../types'
import { mean } from '../math'

// params: { rate = lambda } (RATE; mean = 1/rate)
export const exponential: Distribution = {
  name: 'exponential',
  label: 'Exponential',
  k: 1,
  fit(data) {
    if (data.length < 1) throw new Error('exponential: empty')
    if (data.some((v) => v < 0)) throw new Error('exponential requires x >= 0')
    const m = mean(data)
    if (!(m > 0)) throw new Error('exponential: degenerate (mean <= 0)')
    return { rate: 1 / m }
  },
  logpdf(x: number, p: FittedParams): number {
    return logpdf(x, p.rate)
  },
  cdf(x: number, p: FittedParams): number {
    return cdf(x, p.rate)
  },
}
```

- [ ] **Step 5: Run both → pass; commit**

```bash
pnpm test -- distributions/lognormal distributions/exponential && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(engine): lognormal + exponential distributions (closed-form MLE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Gamma distribution (Minka-seeded 1-D Newton)

**Files:** Create `src/engine/distributions/gamma.ts`, `gamma.test.ts`.

- [ ] **Step 1: Failing test (self-consistency + maximality, no external golden needed)**

`src/engine/distributions/gamma.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import digamma from '@stdlib/math-base-special-digamma'
import gammaln from '@stdlib/math-base-special-gammaln'
import { gamma } from './gamma'
import { mean, meanLog } from '../math'
import { expectClose } from '../../test/relClose'

const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5]

describe('gamma', () => {
  it('shape solves ln(k) - digamma(k) = ln(mean) - mean(ln x)', () => {
    const p = gamma.fit(sample)
    const s = Math.log(mean(sample)) - meanLog(sample)
    expectClose(Math.log(p.shape) - digamma(p.shape), s, 1e-7)
  })
  it('scale = mean / shape; rate = 1/scale', () => {
    const p = gamma.fit(sample)
    expectClose(p.scale, mean(sample) / p.shape, 1e-7)
    expectClose(p.rate, 1 / p.scale, 1e-9)
  })
  it('logpdf passes RATE (not scale) to @stdlib: matches the rate-parameterized closed form', () => {
    // CONVENTION GUARD: catches a rate/scale slot-swap that self-consistency tests cannot.
    // gamma(shape a=3, rate b=1.5): logf(x) = a*ln(b) + (a-1)*ln(x) - b*x - lnΓ(a)
    const p = { shape: 3, rate: 1.5, scale: 1 / 1.5 }
    const expected = 3 * Math.log(1.5) + (3 - 1) * Math.log(2) - 1.5 * 2 - gammaln(3)
    expectClose(gamma.logpdf(2, p), expected, 1e-9)
  })
  it('cdf passes RATE: cdf(mean) for shape=3,rate=1.5 (mean=2) ~ 0.5768', () => {
    expectClose(gamma.cdf(2, { shape: 3, rate: 1.5, scale: 1 / 1.5 }), 0.5768099188731566, 1e-6)
  })
  it('LL at MLE >= LL at perturbed shape', () => {
    const p = gamma.fit(sample)
    const ll = (q: typeof p) => sample.reduce((acc, x) => acc + gamma.logpdf(x, q), 0)
    const worse = { ...p, shape: p.shape * 1.2, rate: (p.shape * 1.2) / mean(sample), scale: mean(sample) / (p.shape * 1.2) }
    expect(ll(p)).toBeGreaterThanOrEqual(ll(worse) - 1e-6)
  })
  it('rejects non-positive data', () => expect(() => gamma.fit([1, 0])).toThrow())
  it('k = 2', () => expect(gamma.k).toBe(2))
})
```

- [ ] **Step 2: Run → fail, then implement** (uses `@stdlib` `digamma`/`trigamma`; gamma `@stdlib` density wants shape+RATE)

`src/engine/distributions/gamma.ts`:
```typescript
import logpdf from '@stdlib/stats-base-dists-gamma-logpdf'
import cdf from '@stdlib/stats-base-dists-gamma-cdf'
import digamma from '@stdlib/math-base-special-digamma'
import trigamma from '@stdlib/math-base-special-trigamma'
import type { Distribution, FittedParams } from '../types'
import { mean, meanLog } from '../math'

// params: { shape, scale, rate }  where scale = 1/rate, mean = shape*scale.
// @stdlib gamma uses (x, alpha=shape, beta=RATE) — pass p.rate, not p.scale.
export const gamma: Distribution = {
  name: 'gamma',
  label: 'Gamma',
  k: 2,
  fit(data) {
    if (data.length < 2) throw new Error('gamma: need n >= 2')
    if (data.some((v) => v <= 0)) throw new Error('gamma requires all x > 0')
    const m = mean(data)
    const s = Math.log(m) - meanLog(data) // >= 0
    if (!(s > 0)) throw new Error('gamma: degenerate (zero log-variance)')
    // Minka (2002) closed-form seed, then 1-D Newton on g(k)=ln k - psi(k) - s.
    let k = (3 - s + Math.sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s)
    for (let i = 0; i < 100; i++) {
      const g = Math.log(k) - digamma(k) - s
      const gp = 1 / k - trigamma(k) // < 0
      const step = g / gp
      const next = k - step
      if (!Number.isFinite(next) || next <= 0) {
        k = k / 2 // damp toward positivity; loop continues
        continue
      }
      k = next
      if (Math.abs(step) < 1e-12 * k) break
    }
    const scale = m / k
    return { shape: k, scale, rate: 1 / scale }
  },
  logpdf(x: number, p: FittedParams): number {
    return logpdf(x, p.shape, p.rate) // beta = RATE
  },
  cdf(x: number, p: FittedParams): number {
    return cdf(x, p.shape, p.rate) // beta = RATE
  },
}
```

- [ ] **Step 3: Run → pass; commit**

```bash
pnpm test -- distributions/gamma && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(engine): gamma distribution (Minka-seeded Newton MLE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Weibull distribution (Newton on shape score)

**Files:** Create `src/engine/distributions/weibull.ts`, `weibull.test.ts`.

- [ ] **Step 1: Failing test**

`src/engine/distributions/weibull.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { weibull } from './weibull'
import { meanLog } from '../math'
import { expectClose } from '../../test/relClose'

const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5]

describe('weibull', () => {
  it('shape solves the score: 1/k + meanLn - (sum x^k ln x)/(sum x^k) = 0', () => {
    const p = weibull.fit(sample)
    const k = p.shape
    let S0 = 0
    let S1 = 0
    for (const x of sample) {
      const xk = Math.pow(x, k)
      S0 += xk
      S1 += xk * Math.log(x)
    }
    expectClose(1 / k + meanLog(sample) - S1 / S0, 0, 1e-7, 1e-7)
  })
  it('scale = (mean(x^k))^(1/k)', () => {
    const p = weibull.fit(sample)
    let S0 = 0
    for (const x of sample) S0 += Math.pow(x, p.shape)
    expectClose(p.scale, Math.pow(S0 / sample.length, 1 / p.shape), 1e-7)
  })
  it('cdf(scale) = 1 - 1/e', () => expectClose(weibull.cdf(3, { shape: 2, scale: 3 }), 1 - Math.exp(-1)))
  it('logpdf passes SCALE to @stdlib: matches the elementary closed form', () => {
    // CONVENTION GUARD: weibull(shape k=2, scale lam=3): logf = ln(k) - ln(lam) + (k-1)ln(x/lam) - (x/lam)^k
    const expected = Math.log(2) - Math.log(3) + (2 - 1) * Math.log(2 / 3) - (2 / 3) ** 2
    expectClose(weibull.logpdf(2, { shape: 2, scale: 3 }), expected, 1e-9)
  })
  it('rejects non-positive data', () => expect(() => weibull.fit([1, 0])).toThrow())
  it('k = 2', () => expect(weibull.k).toBe(2))
})
```

- [ ] **Step 2: Run → fail, then implement** (@stdlib weibull wants shape+SCALE)

`src/engine/distributions/weibull.ts`:
```typescript
import logpdf from '@stdlib/stats-base-dists-weibull-logpdf'
import cdf from '@stdlib/stats-base-dists-weibull-cdf'
import type { Distribution, FittedParams } from '../types'
import { meanLog } from '../math'

// params: { shape = k, scale = lambda }  (lambda is SCALE, not rate). @stdlib: (x, k, lambda).
export const weibull: Distribution = {
  name: 'weibull',
  label: 'Weibull',
  k: 2,
  fit(data) {
    if (data.length < 2) throw new Error('weibull: need n >= 2')
    if (data.some((v) => v <= 0)) throw new Error('weibull requires all x > 0')
    const n = data.length
    const lnx = data.map((v) => Math.log(v))
    const meanLn = meanLog(data)
    // Menon (1963) seed: k0 = (pi/sqrt6) / sd(ln x)
    let sdLnSq = 0
    for (const v of lnx) sdLnSq += (v - meanLn) * (v - meanLn)
    const sdLn = Math.sqrt(sdLnSq / n)
    let k = sdLn > 0 ? (Math.PI / Math.sqrt(6)) / sdLn : 1
    for (let i = 0; i < 100; i++) {
      let S0 = 0
      let S1 = 0
      let S2 = 0
      for (let j = 0; j < n; j++) {
        const xk = Math.pow(data[j]!, k)
        S0 += xk
        S1 += xk * lnx[j]!
        S2 += xk * lnx[j]! * lnx[j]!
      }
      const g = 1 / k + meanLn - S1 / S0
      const gp = -1 / (k * k) - (S2 * S0 - S1 * S1) / (S0 * S0) // < 0 (g monotone decreasing)
      const step = g / gp
      const next = k - step
      if (!Number.isFinite(next) || next <= 0) {
        k = k / 2
        continue
      }
      k = next
      if (Math.abs(step) < 1e-12 * k) break
    }
    let S0 = 0
    for (const v of data) S0 += Math.pow(v, k)
    const scale = Math.pow(S0 / n, 1 / k)
    return { shape: k, scale }
  },
  logpdf(x: number, p: FittedParams): number {
    return logpdf(x, p.shape, p.scale)
  },
  cdf(x: number, p: FittedParams): number {
    return cdf(x, p.shape, p.scale)
  },
}
```
> `noUncheckedIndexedAccess` is on (strictest) → the `data[j]!`/`lnx[j]!` non-null assertions are needed inside the bounded loop; that's correct here since `j < n`.

- [ ] **Step 3: Run → pass; commit**

```bash
pnpm test -- distributions/weibull && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(engine): weibull distribution (Newton MLE on shape score)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Distribution registry

**Files:** Create `src/engine/distributions/index.ts`, `index.test.ts`.

- [ ] **Step 1: Failing test**

`src/engine/distributions/index.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { DISTRIBUTIONS } from './index'

describe('DISTRIBUTIONS registry', () => {
  it('contains the 5 M1 distributions with unique names', () => {
    const names = DISTRIBUTIONS.map((d) => d.name)
    expect(names).toEqual(['normal', 'lognormal', 'exponential', 'gamma', 'weibull'])
    expect(new Set(names).size).toBe(5)
  })
  it('every entry exposes fit/logpdf/cdf and a k of 1 or 2', () => {
    for (const d of DISTRIBUTIONS) {
      expect(typeof d.fit).toBe('function')
      expect(typeof d.logpdf).toBe('function')
      expect(typeof d.cdf).toBe('function')
      expect([1, 2]).toContain(d.k)
    }
  })
})
```

- [ ] **Step 2: Implement**

`src/engine/distributions/index.ts`:
```typescript
import type { Distribution } from '../types'
import { normal } from './normal'
import { lognormal } from './lognormal'
import { exponential } from './exponential'
import { gamma } from './gamma'
import { weibull } from './weibull'

export const DISTRIBUTIONS: readonly Distribution[] = [normal, lognormal, exponential, gamma, weibull]
```

- [ ] **Step 3: Run → pass; commit**

```bash
pnpm test -- distributions/index && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(engine): distribution registry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Goodness-of-fit — KS statistic

**Files:** Create `src/engine/gof.ts`, `gof.test.ts`.

- [ ] **Step 1: Failing test (locks the 0-based indexing)**

`src/engine/gof.test.ts`:
```typescript
import { describe, it } from 'vitest'
import { ksStatistic } from './gof'
import { expectClose } from '../test/relClose'

describe('ksStatistic', () => {
  it('uniform sample vs uniform CDF — exact small case', () => {
    // sorted [0.25,0.5,0.75], F(x)=x. n=3.
    // D+ = max(1/3-0.25, 2/3-0.5, 3/3-0.75) = max(0.0833,0.1667,0.25)=0.25
    // D- = max(0.25-0, 0.5-1/3, 0.75-2/3) = max(0.25,0.1667,0.0833)=0.25
    expectClose(ksStatistic([0.5, 0.25, 0.75], (x) => x), 0.25, 1e-12)
  })
  it('does not mutate the input array', () => {
    const data = [3, 1, 2]
    ksStatistic(data, (x) => x / 3)
    expectClose(data[0]!, 3) // original order preserved
  })
  it('clamps tiny negative FP noise to 0 (perfect fit)', () => {
    const d = ksStatistic([1, 2, 3, 4], (x) => x / 4) // step exactly matches at integers
    expectClose(d, 0.25, 1e-12) // D- at i=1: F(1)-0 = 0.25
  })
})
```

- [ ] **Step 2: Implement**

`src/engine/gof.ts`:
```typescript
/**
 * One-sample Kolmogorov–Smirnov statistic D = max(D+, D-) against a fitted CDF.
 * NOTE: with parameters estimated from the same data, the standard KS p-value is INVALID
 * (Lilliefors). Use D as a diagnostic / comparison metric only.
 */
export function ksStatistic(data: readonly number[], cdf: (x: number) => number): number {
  const x = [...data].sort((a, b) => a - b) // copy; never mutate caller's array
  const n = x.length
  if (n === 0) return Number.NaN
  let dPlus = -Infinity
  let dMinus = -Infinity
  for (let j = 0; j < n; j++) {
    const f = Math.min(1, Math.max(0, cdf(x[j]!))) // clamp F into [0,1]
    dPlus = Math.max(dPlus, (j + 1) / n - f) // i/n - F(x_i),  i = j+1
    dMinus = Math.max(dMinus, f - j / n) // F(x_i) - (i-1)/n
  }
  return Math.max(0, dPlus, dMinus) // clamp FP noise
}
```

- [ ] **Step 3: Run → pass; commit**

```bash
pnpm test -- engine/gof && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(engine): KS goodness-of-fit statistic

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Model selection — LL, AIC/AICc/BIC, ranking

**Files:** Create `src/engine/selection.ts`, `selection.test.ts`.

- [ ] **Step 1: Failing test**

`src/engine/selection.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { logLik, aic, aicc, bic, rankByAICc } from './selection'
import { expectClose } from '../test/relClose'

describe('selection', () => {
  it('logLik sums per-point log densities', () => {
    expectClose(logLik([0, 1, 2], (x) => -x), -(0 + 1 + 2))
  })
  it('aic = 2k - 2LL', () => expectClose(aic(-10, 2), 2 * 2 - 2 * -10))
  it('aicc adds the small-sample correction', () => {
    // k=2, n=10: AIC + 2*2*3/(10-2-1) = AIC + 12/7
    expectClose(aicc(-10, 2, 10), aic(-10, 2) + 12 / 7)
  })
  it('aicc = +Infinity when n - k - 1 <= 0', () => {
    expect(aicc(-10, 2, 3)).toBe(Number.POSITIVE_INFINITY)
  })
  it('bic = k ln n - 2LL', () => expectClose(bic(-10, 2, 10), 2 * Math.log(10) - 2 * -10))
  it('rankByAICc sorts ascending, computes delta + Akaike weights summing to 1', () => {
    const fits = [
      { name: 'a', logLik: -20, k: 2 },
      { name: 'b', logLik: -10, k: 2 }, // best
      { name: 'c', logLik: -15, k: 1 },
    ]
    const ranked = rankByAICc(fits, 30)
    expect(ranked[0]!.name).toBe('b')
    expect(ranked[0]!.rank).toBe(1)
    expectClose(ranked[0]!.deltaAICc, 0)
    const wsum = ranked.reduce((s, r) => s + r.weight, 0)
    expectClose(wsum, 1, 1e-12)
  })
})
```

- [ ] **Step 2: Implement**

`src/engine/selection.ts`:
```typescript
export function logLik(data: readonly number[], logpdf: (x: number) => number): number {
  let ll = 0
  for (const x of data) ll += logpdf(x) // sum in log space; never log(product)
  return ll
}

export function aic(ll: number, k: number): number {
  return 2 * k - 2 * ll
}

export function aicc(ll: number, k: number, n: number): number {
  const denom = n - k - 1
  if (denom <= 0) return Number.POSITIVE_INFINITY // correction undefined -> sorts last, weight 0
  return aic(ll, k) + (2 * k * (k + 1)) / denom
}

export function bic(ll: number, k: number, n: number): number {
  return k * Math.log(n) - 2 * ll // Math.log is ln
}

export interface Rankable {
  name: string
  logLik: number
  k: number
}
export interface Ranked<T> {
  rank: number
  aicc: number
  deltaAICc: number
  weight: number
}

export function rankByAICc<T extends Rankable>(items: readonly T[], n: number): (T & Ranked<T>)[] {
  const scored = items.map((it) => ({ it, aiccVal: aicc(it.logLik, it.k, n) }))
  const minAICc = Math.min(...scored.map((s) => s.aiccVal))
  // delta form IS the numerically stable form: max(-delta/2)=0 -> exp=1, no overflow.
  const withRel = scored.map((s) => {
    const delta = s.aiccVal - minAICc
    return { ...s, delta, rel: Math.exp(-delta / 2) } // exp(-Inf)=0, safe
  })
  const sumRel = withRel.reduce((acc, s) => acc + s.rel, 0)
  return withRel
    .map((s) => ({
      ...s.it,
      aicc: s.aiccVal,
      deltaAICc: s.delta,
      weight: sumRel > 0 ? s.rel / sumRel : 0,
      rank: 0,
    }))
    .sort((a, b) => a.aicc - b.aicc)
    .map((r, i) => ({ ...r, rank: i + 1 }))
}
```

- [ ] **Step 3: Run → pass; commit**

```bash
pnpm test -- engine/selection && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(engine): log-likelihood + AIC/AICc/BIC + AICc ranking

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: fitAll orchestrator + public engine API

**Files:** Create `src/engine/fitAll.ts`, `fitAll.test.ts`, `src/engine/index.ts`.

- [ ] **Step 1: Failing test (end-to-end on synthetic data)**

`src/engine/fitAll.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { fitAll } from './fitAll'

// Deterministic positive sample (looks roughly gamma/lognormal-ish).
const sample = [2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0, 3.7, 1.2, 4.8, 2.2, 3.9, 5.5, 2.7, 3.1, 4.4]

describe('fitAll', () => {
  it('fits all 5 distributions, ranks by AICc, weights sum to 1', () => {
    const res = fitAll(sample)
    expect(res.n).toBe(sample.length)
    expect(res.ranked.length).toBe(5)
    expect(res.failures.length).toBe(0)
    expect(res.ranked[0]!.rank).toBe(1)
    // ascending AICc
    for (let i = 1; i < res.ranked.length; i++) {
      expect(res.ranked[i]!.aicc).toBeGreaterThanOrEqual(res.ranked[i - 1]!.aicc)
    }
    const wsum = res.ranked.reduce((s, r) => s + r.weight, 0)
    expect(Math.abs(wsum - 1)).toBeLessThan(1e-9)
    // each ranked fit carries finite diagnostics
    for (const r of res.ranked) {
      expect(Number.isFinite(r.logLik)).toBe(true)
      expect(Number.isFinite(r.aic)).toBe(true)
      expect(r.ks).toBeGreaterThanOrEqual(0)
    }
  })
  it('reports failures (not crashes) when a distribution rejects the data', () => {
    const withNeg = [-1, 2, 3, 4, 5, 6] // exponential needs x>=0; lognormal/gamma/weibull need x>0
    const res = fitAll(withNeg)
    const failed = res.failures.map((f) => f.name).sort()
    expect(failed).toEqual(['exponential', 'gamma', 'lognormal', 'weibull'])
    expect(res.ranked.some((r) => r.name === 'normal')).toBe(true)
  })
  it('throws on too-small samples', () => expect(() => fitAll([1, 2])).toThrow())
})
```

- [ ] **Step 2: Implement fitAll**

`src/engine/fitAll.ts`:
```typescript
import type { Fit, FitAllResult, FitFailure, RankedFit } from './types'
import { DISTRIBUTIONS } from './distributions/index'
import { ksStatistic } from './gof'
import { logLik, aic, aicc, bic, rankByAICc } from './selection'

export interface FitAllOptions {
  onProgress?: (completed: number, total: number) => void
}

export function fitAll(data: readonly number[], opts: FitAllOptions = {}): FitAllResult {
  const n = data.length
  if (n < 4) throw new Error('fitAll: need at least 4 data points')
  const fits: Fit[] = []
  const failures: FitFailure[] = []
  const total = DISTRIBUTIONS.length
  DISTRIBUTIONS.forEach((dist, i) => {
    try {
      const params = dist.fit(data)
      const ll = logLik(data, (x) => dist.logpdf(x, params))
      if (!Number.isFinite(ll)) throw new Error('non-finite log-likelihood')
      fits.push({
        name: dist.name,
        label: dist.label,
        k: dist.k,
        params,
        logLik: ll,
        aic: aic(ll, dist.k),
        aicc: aicc(ll, dist.k, n),
        bic: bic(ll, dist.k, n),
        ks: ksStatistic(data, (x) => dist.cdf(x, params)),
      })
    } catch (e) {
      failures.push({ name: dist.name, label: dist.label, error: e instanceof Error ? e.message : String(e) })
    }
    opts.onProgress?.(i + 1, total)
  })
  const ranked = rankByAICc(
    fits.map((f) => ({ ...f, logLik: f.logLik, k: f.k })),
    n,
  ) as unknown as RankedFit[]
  return { ranked, failures, n }
}
```
> The `rankByAICc` generic returns each input object augmented with `rank`/`aicc`/`deltaAICc`/`weight`; since each `Fit` already has `aicc`, the ranker overwrites it with the same value. The `as RankedFit[]` cast is sound because every `Fit` field plus the ranking fields are present.

- [ ] **Step 3: Public engine API**

`src/engine/index.ts`:
```typescript
export { fitAll } from './fitAll'
export type { FitAllOptions } from './fitAll'
export type { Fit, RankedFit, FitFailure, FitAllResult, FittedParams } from './types'
export { DISTRIBUTIONS } from './distributions/index'
```

- [ ] **Step 4: Run → pass; commit**

```bash
pnpm test -- engine/fitAll && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(engine): fitAll orchestrator + public engine API

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: CSV / paste number parser

**Files:** Create `src/lib/parseNumbers.ts`, `parseNumbers.test.ts`.

- [ ] **Step 1: Failing test**

`src/lib/parseNumbers.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { parseNumbers } from './parseNumbers'

describe('parseNumbers', () => {
  it('parses newline/comma/space separated numbers', () => {
    expect(parseNumbers('1\n2.5, 3\t4 5')).toEqual([1, 2.5, 3, 4, 5])
  })
  it('ignores blank lines, a header row, and non-numeric tokens', () => {
    expect(parseNumbers('value\n1\n\n2\nNA\n3')).toEqual([1, 2, 3])
  })
  it('handles a single CSV column with a header', () => {
    expect(parseNumbers('x\r\n10\r\n20\r\n30')).toEqual([10, 20, 30])
  })
  it('returns [] for empty input', () => expect(parseNumbers('   ')).toEqual([]))
})
```

- [ ] **Step 2: Implement** (simple single-column parser; PapaParse/Excel/multi-column is M3)

`src/lib/parseNumbers.ts`:
```typescript
/** Parse a flat list of numbers from pasted text or a single CSV column.
 *  Splits on any whitespace/comma/semicolon; drops tokens that aren't finite numbers
 *  (so a header row like "value" or an "NA" cell is ignored). */
export function parseNumbers(text: string): number[] {
  const out: number[] = []
  for (const tok of text.split(/[\s,;]+/)) {
    if (tok === '') continue
    const v = Number(tok)
    if (Number.isFinite(v)) out.push(v)
  }
  return out
}
```

- [ ] **Step 3: Run → pass; commit**

```bash
pnpm test -- lib/parseNumbers && pnpm typecheck && pnpm check
git add -A && git commit -m "feat(data): single-column number parser for paste/CSV input

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Comlink worker + typed client

**Files:** Create `src/engine.worker.ts`, `src/engineClient.ts`. Modify `package.json` (add `comlink`).

- [ ] **Step 1: Install Comlink (exact pin; ships its own types)**

```bash
pnpm add -E comlink@4.4.2
```

- [ ] **Step 2: Worker exposing the engine**

`src/engine.worker.ts`:
```typescript
import * as Comlink from 'comlink'
import { fitAll, type FitAllResult } from './engine/index'

const api = {
  /** Fit all distributions. Streams progress via a Comlink.proxy'd callback. */
  fitAll(samples: Float64Array, onProgress?: (completed: number, total: number) => void): FitAllResult {
    // samples.buffer was transferred in -> use directly. Engine wants number[].
    const data = Array.from(samples)
    return fitAll(data, { onProgress })
  },
}

export type EngineApi = typeof api // client imports this TYPE only
Comlink.expose(api)
```

- [ ] **Step 3: Typed client**

`src/engineClient.ts`:
```typescript
import * as Comlink from 'comlink'
import type { EngineApi } from './engine.worker' // TYPE-ONLY: keeps worker code out of the main bundle
import type { FitAllResult } from './engine/index'

// Vite statically detects this exact form (string literal + import.meta.url). Do NOT parameterize.
const worker = new Worker(new URL('./engine.worker.ts', import.meta.url), { type: 'module' })
const engine = Comlink.wrap<EngineApi>(worker)

export async function runFitAll(
  data: readonly number[],
  onProgress?: (completed: number, total: number) => void,
): Promise<FitAllResult> {
  const samples = Float64Array.from(data)
  // transfer the buffer (zero-copy); `samples` is neutered on this side afterward.
  return engine.fitAll(
    Comlink.transfer(samples, [samples.buffer]),
    onProgress ? Comlink.proxy(onProgress) : undefined,
  )
}
```

- [ ] **Step 4: Verify it typechecks + builds (worker chunk emitted)**

```bash
pnpm typecheck && pnpm check && pnpm build
```
Expected: build succeeds and `dist/assets/` includes a separate worker chunk. (No unit test here — the worker is exercised by the e2e in Task 14. Engine logic is already unit-tested in Node.)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(worker): Comlink worker exposing the fitting engine + typed client

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Plotly chart component + histogram/PDF trace builder

**Files:** Create `src/plotly-cartesian.d.ts`, `src/ui/PlotlyChart.tsx`, `src/ui/buildHistogramPdf.ts`, `buildHistogramPdf.test.ts`. Modify `package.json`.

- [ ] **Step 1: Install Plotly cartesian bundle + types**

```bash
pnpm add -E plotly.js-cartesian-dist-min@3.5.1
pnpm add -D -E @types/plotly.js@3.0.10
```
> Use `-cartesian-` (NOT `-basic-`, which has no histogram trace and throws at runtime; NOT the full 4.85 MB bundle).
> **Interop note:** `import Plotly from 'plotly.js-cartesian-dist-min'` is a default import of an `export =` module — the SAME CommonJS shape as the `@stdlib` packages, so it relies on the same `esModuleInterop: true` that Task 0 Step 5 confirmed. A green Task 0 does not guarantee this line typechecks until the shim (Step 2) is in place; if `pnpm typecheck` complains here, the remedy is identical (ensure `esModuleInterop`/`allowSyntheticDefaultImports`), not switching import styles.

- [ ] **Step 2: Ambient type shim (no `@types/plotly.js-cartesian-dist-min` exists)**

`src/plotly-cartesian.d.ts`:
```typescript
declare module 'plotly.js-cartesian-dist-min' {
  import * as Plotly from 'plotly.js'
  export = Plotly
}
```

- [ ] **Step 3: Failing test for the trace builder**

`src/ui/buildHistogramPdf.test.ts`:
```typescript
import { describe, it, expect } from 'vitest'
import { buildHistogramPdf } from './buildHistogramPdf'
import { normal } from '../engine/distributions/normal'

describe('buildHistogramPdf', () => {
  it('returns a histogram trace (probability density) + a fitted-PDF line trace', () => {
    const data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    const traces = buildHistogramPdf(data, normal, normal.fit(data), 64)
    expect(traces).toHaveLength(2)
    expect(traces[0]!.type).toBe('histogram')
    expect((traces[0] as { histnorm?: string }).histnorm).toBe('probability density')
    expect(traces[1]!.type).toBe('scatter')
    const line = traces[1] as { x: number[]; y: number[] }
    expect(line.x).toHaveLength(64)
    expect(line.y).toHaveLength(64)
    expect(line.y.every((v) => v >= 0 && Number.isFinite(v))).toBe(true)
  })
})
```

- [ ] **Step 4: Implement the trace builder** (pure; no Plotly import needed for the data)

`src/ui/buildHistogramPdf.ts`:
```typescript
import type { Data } from 'plotly.js'
import type { Distribution, FittedParams } from '../engine/types'

/** Build [histogram(density), fitted-PDF line] traces for the data + a fitted distribution. */
export function buildHistogramPdf(
  data: readonly number[],
  dist: Distribution,
  params: FittedParams,
  gridPoints = 256,
): Data[] {
  let lo = Infinity
  let hi = -Infinity
  for (const v of data) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  } // plain loop, not Math.min(...data) — spreading a large array overflows the stack
  const span = hi - lo || 1
  const x0 = lo - span * 0.05
  const x1 = hi + span * 0.05
  const xGrid: number[] = []
  const yGrid: number[] = []
  for (let i = 0; i < gridPoints; i++) {
    const x = x0 + ((x1 - x0) * i) / (gridPoints - 1)
    xGrid.push(x)
    yGrid.push(Math.exp(dist.logpdf(x, params))) // pdf = exp(logpdf)
  }
  return [
    {
      type: 'histogram',
      x: data as number[],
      histnorm: 'probability density',
      name: 'data',
      opacity: 0.6,
      marker: { color: '#7aa6ff' },
    },
    {
      type: 'scatter',
      mode: 'lines',
      x: xGrid,
      y: yGrid,
      name: `${dist.label} fit`,
      line: { color: '#ff5d5d', width: 2 },
    },
  ]
}
```

- [ ] **Step 5: Thin Plotly component**

`src/ui/PlotlyChart.tsx`:
```tsx
import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-cartesian-dist-min'
import type { Data, Layout } from 'plotly.js'

export function PlotlyChart({ data, layout }: { data: Data[]; layout?: Partial<Layout> }) {
  const divRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = divRef.current
    if (!el) return
    void Plotly.react(el, data, layout ?? {}, { responsive: true })
    return () => {
      Plotly.purge(el) // capture el; divRef.current may be null at cleanup (React 19 StrictMode double-mount)
    }
  }, [data, layout])
  return <div ref={divRef} style={{ width: '100%', height: 400 }} />
}
```

- [ ] **Step 6: Run test + typecheck + build; commit**

```bash
pnpm test -- ui/buildHistogramPdf && pnpm typecheck && pnpm check && pnpm build
git add -A && git commit -m "feat(ui): Plotly chart component + histogram/PDF trace builder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: UI wiring — data input, results table, App

**Files:** Create `src/ui/DataInput.tsx`, `src/ui/ResultsTable.tsx`, `ResultsTable.test.tsx`. Modify `src/App.tsx`.

- [ ] **Step 1: Failing component test for the results table**

`src/ui/ResultsTable.test.tsx`:
```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ResultsTable } from './ResultsTable'
import type { RankedFit } from '../engine/types'

const ranked: RankedFit[] = [
  { name: 'gamma', label: 'Gamma', k: 2, params: { shape: 2, scale: 1.5, rate: 1 / 1.5 }, logLik: -30, aic: 64, aicc: 64.5, bic: 67, ks: 0.08, rank: 1, deltaAICc: 0, weight: 0.7 },
  { name: 'normal', label: 'Normal', k: 2, params: { mu: 3, sigma: 1.2 }, logLik: -32, aic: 68, aicc: 68.5, bic: 71, ks: 0.12, rank: 2, deltaAICc: 4, weight: 0.3 },
]

describe('ResultsTable', () => {
  it('renders a row per fit, best first, with label + AICc', () => {
    render(<ResultsTable ranked={ranked} />)
    const rows = screen.getAllByRole('row')
    expect(rows.length).toBe(1 + ranked.length) // header + 2
    expect(screen.getByText('Gamma')).toBeInTheDocument()
    expect(screen.getByText('Normal')).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Implement ResultsTable**

`src/ui/ResultsTable.tsx`:
```tsx
import type { RankedFit } from '../engine/types'

const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '—')

export function ResultsTable({ ranked }: { ranked: RankedFit[] }) {
  return (
    <table className="w-full text-sm border-collapse">
      <thead>
        <tr className="text-left border-b border-slate-300">
          <th className="py-1 pr-3">#</th>
          <th className="py-1 pr-3">Distribution</th>
          <th className="py-1 pr-3">AICc</th>
          <th className="py-1 pr-3">ΔAICc</th>
          <th className="py-1 pr-3">Weight</th>
          <th className="py-1 pr-3">KS (D)</th>
          <th className="py-1 pr-3">log-lik</th>
        </tr>
      </thead>
      <tbody>
        {ranked.map((r) => (
          <tr key={r.name} className="border-b border-slate-100">
            <td className="py-1 pr-3">{r.rank}</td>
            <td className="py-1 pr-3 font-medium">{r.label}</td>
            <td className="py-1 pr-3">{fmt(r.aicc)}</td>
            <td className="py-1 pr-3">{fmt(r.deltaAICc)}</td>
            <td className="py-1 pr-3">{fmt(r.weight)}</td>
            <td className="py-1 pr-3">{fmt(r.ks)}</td>
            <td className="py-1 pr-3">{fmt(r.logLik)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
```

- [ ] **Step 3: Implement DataInput** (textarea + file upload → number[])

`src/ui/DataInput.tsx`:
```tsx
import { useState } from 'react'
import { parseNumbers } from '../lib/parseNumbers'

const SAMPLE = '2.1 3.4 1.8 5.2 2.9 4.1 3.0 2.5 6.0 3.7 1.2 4.8 2.2 3.9 5.5 2.7 3.1 4.4'

export function DataInput({ onData }: { onData: (data: number[]) => void }) {
  const [text, setText] = useState('')
  const submit = (raw: string) => onData(parseNumbers(raw))
  return (
    <div className="flex flex-col gap-2">
      <textarea
        aria-label="data"
        className="border border-slate-300 rounded p-2 font-mono text-sm h-32"
        placeholder="Paste numbers (one per line or comma/space separated)"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex gap-2 items-center">
        <button
          type="button"
          className="px-3 py-1 rounded bg-slate-900 text-white"
          onClick={() => submit(text)}
        >
          Fit distributions
        </button>
        <button
          type="button"
          className="px-3 py-1 rounded border border-slate-300"
          onClick={() => {
            setText(SAMPLE)
            submit(SAMPLE)
          }}
        >
          Load sample
        </button>
        <input
          type="file"
          accept=".csv,.txt"
          aria-label="upload data file"
          onChange={async (e) => {
            const file = e.target.files?.[0]
            if (file) {
              const content = await file.text()
              setText(content)
              submit(content)
            }
          }}
        />
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Wire App.tsx**

`src/App.tsx`:
```tsx
import { useMemo, useState } from 'react'
import { runFitAll } from './engineClient'
import type { FitAllResult } from './engine/index'
import { DISTRIBUTIONS } from './engine/index'
import { DataInput } from './ui/DataInput'
import { ResultsTable } from './ui/ResultsTable'
import { PlotlyChart } from './ui/PlotlyChart'
import { buildHistogramPdf } from './ui/buildHistogramPdf'

export default function App() {
  const [data, setData] = useState<number[]>([])
  const [result, setResult] = useState<FitAllResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function onData(d: number[]) {
    setError(null)
    setResult(null)
    if (d.length < 4) {
      setError('Please provide at least 4 numeric values.')
      setData([])
      return
    }
    setData(d)
    setBusy(true)
    try {
      setResult(await runFitAll(d))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const traces = useMemo(() => {
    if (!result || result.ranked.length === 0 || data.length === 0) return null
    const best = result.ranked[0]!
    const dist = DISTRIBUTIONS.find((d) => d.name === best.name)
    return dist ? buildHistogramPdf(data, dist, best.params) : null
  }, [result, data])

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 p-6 flex flex-col gap-6 max-w-3xl mx-auto">
      <header>
        <h1 className="text-3xl font-bold">HardFit</h1>
        <p className="text-slate-600">Fit your data to probability distributions, in your browser.</p>
      </header>

      <DataInput onData={onData} />

      {error && <p role="alert" className="text-red-600">{error}</p>}
      {busy && <p>Fitting…</p>}

      {result && result.ranked.length > 0 && (
        <section className="flex flex-col gap-4">
          <h2 className="text-xl font-semibold">Ranked fits (by AICc)</h2>
          <ResultsTable ranked={result.ranked} />
          {traces && <PlotlyChart data={traces} layout={{ bargap: 0.02, showlegend: true, xaxis: { title: { text: 'value' } }, yaxis: { title: { text: 'density' } } }} />}
          {result.failures.length > 0 && (
            <p className="text-sm text-slate-500">
              Not applicable: {result.failures.map((f) => f.label).join(', ')} (data outside support).
            </p>
          )}
        </section>
      )}
    </main>
  )
}
```

- [ ] **Step 5: Run tests + typecheck + check + build; fix any a11y/lint; commit**

```bash
pnpm test && pnpm typecheck && pnpm check && pnpm build
git add -A && git commit -m "feat(ui): wire data input -> worker fit -> ranked table + chart

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: End-to-end test + deploy

**Files:** Modify `e2e/smoke.spec.ts` (or add `e2e/fit.spec.ts`).

- [ ] **Step 1: Add the e2e fit flow (runs under the real CSP via wrangler)**

Append to `e2e/smoke.spec.ts` (keep the existing 3 tests):
```typescript
test('fits the sample dataset end-to-end and renders ranked table + chart', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
  page.on('pageerror', (e) => errors.push(e.message))

  await page.goto('/')
  await page.getByRole('button', { name: 'Load sample' }).click()

  // worker computes; ranked table appears with all 5 distributions
  await expect(page.getByRole('heading', { name: 'Ranked fits (by AICc)' })).toBeVisible({ timeout: 15_000 })
  for (const label of ['Normal', 'Lognormal', 'Exponential', 'Gamma', 'Weibull']) {
    await expect(page.getByText(label, { exact: true })).toBeVisible()
  }
  // Plotly chart rendered
  await expect(page.locator('.plotly').first()).toBeVisible()

  expect(errors).toEqual([]) // no CSP violations / runtime errors (worker + Plotly under script-src 'self')
})
```
> If the worker or Plotly trips the CSP (a `Refused to ...` console error fails this test), check `public/_headers`: `worker-src 'self' blob:` and `script-src 'self'` are already set in M0; Vite emits the worker as a same-origin module chunk (allowed). Do not loosen the CSP without diagnosing the exact directive.

- [ ] **Step 2: Run the full local gate**

```bash
pnpm typecheck && pnpm check && pnpm test && pnpm run licenses && pnpm build && CI=1 pnpm e2e
```
Expected: all green (Vitest unit suite + 4 Playwright tests). Fix any failures before proceeding.

- [ ] **Step 3: Commit, push the branch, open a PR (CI runs on it)**

```bash
git add -A && git commit -m "test(e2e): fit-and-render flow under the real CSP

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push -u origin feat/m1-vertical-slice
gh pr create --fill --base main
```

- [ ] **Step 4: Merge after CI is green, then verify the live deploy**

After the PR's GitHub CI is green, merge to `main` (squash). The push to `main` auto-triggers the Cloudflare Workers build. Then:
```bash
gh pr merge --squash --delete-branch
# wait for the Cloudflare build, then verify the live app actually fits:
curl -sS -o /dev/null -w "%{http_code}\n" https://hardfit.bryanandagoya.workers.dev/
```
Expected: `200`. Open the URL, click **Load sample**, and confirm the ranked table (5 distributions) + histogram-with-PDF render with a clean console — the M1 slice is live.

---

## Self-Review

**1. Spec coverage (M1 milestone):**
- 5 distributions (Normal, Lognormal, Exponential, Gamma, Weibull) with MLE → Tasks 2–5. ✓
- Pure, TDD'd `engine/` validated against known values + invariants → Tasks 1–9. ✓
- KS statistic (diagnostic) + AICc ranking + Akaike weights → Tasks 7–8. ✓
- `fitAll` orchestration with failure handling → Task 9. ✓
- Comlink worker + typed client (UI never blocks) → Task 11. ✓
- CSV/paste data import → Task 10. ✓
- Ranked table + histogram-with-fitted-PDF chart → Tasks 12–13. ✓
- Deployed end-to-end + e2e under CSP → Task 14. ✓
- *Correctly deferred to later milestones:* Anderson-Darling / Cramér-von Mises / Chi-Squared, bootstrap CIs, the other ~48 distributions, i18n, the full scipy/R fixture pipeline, multi-column/Excel import, the remaining chart types.

**2. Placeholder scan:** Every code step contains complete, runnable code (verified `@stdlib` packages/versions/conventions, validated MLE algorithms, exact KS/AICc formulas). The only judgment lookups are: re-pinning an `@stdlib` version if it 404s (with `npm view`), and the `esModuleInterop` remedy in Task 0 Step 5 — both are explicit verification steps, not omissions.

**3. Type/name consistency:** `Distribution` (`fit`/`logpdf`/`cdf`/`k`) is used identically across all 5 distribution files, the registry, and `fitAll`. `RankedFit`/`Fit`/`FitAllResult` are defined in `types.ts` and consumed consistently by `fitAll`, `ResultsTable`, and `App`. Parameter-name conventions are fixed per distribution (`{mu,sigma}`, `{rate}`, `{shape,scale,rate}`, `{shape,scale}`) and the `@stdlib` calls pass the documented convention (gamma → `rate`, weibull → `scale`, exponential → `rate`). `runFitAll`/`EngineApi`/`fitAll` signatures align across worker, client, and engine.

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh Opus subagent per task, spec + code-quality review between tasks (same as M0). The engine tasks (2–9) especially benefit from the two-stage review since they're the correctness-critical core.
2. **Inline** — execute here in batches with checkpoints.

Next milestone after M1: **M2** — widen the engine to all ~53 distributions + add Anderson-Darling/Cramér-von Mises/Chi-Squared, bootstrap CIs, and the scipy/R reference-fixture pipeline (the "beat EasyFit" numerical-parity gate).
