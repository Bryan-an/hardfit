# HardFit — Technical Strategy & Architecture

> **Status:** Decision-made strategy (research-backed, adversarially verified). Pending: confirm deadline → finalize the v1 scope cut → write the implementation plan.
> **Locked decisions:** beat EasyFit · experienced solo maintainer (Bryan) → modern best-in-class stack · **stateless** (defined: *no server, no accounts, no backend database*) + client-side only · English + Spanish.
> **Date:** 2026-05-31. All versions below are current as of this date — **re-pin at scaffold time** (`pnpm create vite@latest`); the stack moves fast.
> **How this was produced:** an 11-dimension research workflow → 6 adversarial verification passes → synthesis → completeness critic. Corrections from verification are folded in throughout.

---

## 1. Vision & how HardFit beats EasyFit

EasyFit (MathWave, v5.6, Windows-only, paid, effectively dormant since ~2015) defined the category: import data → auto-fit 50+ distributions → rank by three classic goodness-of-fit (GoF) tests → plot → export. It is the right thing to copy *and* the right thing to leapfrog.

**"Stateless"** here means: HardFit has **no server, no user accounts, and no backend database**. The entire app is static files + client-side computation. User data is parsed and fit **entirely in the browser** and never transmitted anywhere. Optional local history (IndexedDB) is **opt-in and one-click-clearable**; results are shared via URL encoding, never via a server.

**HardFit matches the full EasyFit loop and then exceeds it on six axes EasyFit cannot reach:**

1. **Free, zero-install, cross-platform, privacy-first.** A static web app, enforced by a strict Content-Security-Policy (§8/§9) so data *provably* can't leave the browser.
2. **Modern model selection for free.** AIC / AICc / BIC + Akaike weights — criteria @RISK reserves for its paid Business tier. EasyFit has none.
3. **Statistically honest GoF.** Anderson-Darling (distribution-specific adjusted statistics), Cramér-von Mises, Chi-Squared, and KS — **with each p-value's provenance labeled** (exact / table / bootstrap) and the "estimated-parameter" KS caveat handled correctly (Lilliefors / parametric bootstrap). EasyFit ships none of this rigor.
4. **Bootstrap confidence intervals** on fitted parameters (top candidates) + asymptotic standard errors everywhere.
5. **Modern interactive, exportable charts** (zoom/pan/hover, toggle distributions, PNG **and** SVG), a side-by-side comparison view, and integrated data cleaning/outlier handling.
6. **Bilingual (EN/ES) accessible UI** (WCAG 2.2 AA) + **shareable result links** + **offline (PWA)** support.

The wedge is the *combination*: no single tool — free or paid — bundles auto-fit breadth + information-criterion ranking + bootstrap CIs + interactive bilingual accessible web delivery at zero cost.

---

## 2. Feature scope

### Comparison: EasyFit vs HardFit

| Feature | EasyFit | HardFit |
|---|---|---|
| Platform / cost | Windows desktop; paid (academic ~$99, single-user ~$499, site ~$4,999, subscription ~$1/day); install required | Web, free, zero-install, runs anywhere |
| Privacy | Local app | Fully client-side; data never leaves the browser (CSP-enforced) |
| Auto "fit-all + rank" | Yes (markets 55+) | Yes (~53 families), streamed with live progress |
| Manual single-fit mode | Yes | Yes |
| Distribution catalog | ~51 named families (markets "55+" counting variants) | ~53 families covering **all common EasyFit families** (a few niche ones — Log-Pearson III, Wakeby — explicitly deferred, see §13) |
| Estimation | MoM / MLE / LSE / L-moments (fixed per family) | MLE (default) + MoM seeds; selectable where it matters; shows fitted params + standard errors |
| GoF tests | KS, Anderson-Darling, Chi-Squared | KS (+Lilliefors), Anderson-Darling, **Cramér-von Mises**, Chi-Squared |
| Model selection | None | **AIC, AICc, BIC + Akaike weights** |
| p-value honesty | Tabulated critical values only | Provenance-labeled (exact/table/bootstrap), valid under estimated params |
| Confidence intervals | None | **Bootstrap CIs** (top candidates) + asymptotic SEs |
| Charts | Static-ish: hist+PDF, CDF, survival, hazard, P-P, Q-Q, prob-diff | Interactive: same set + comparison overlay; PNG **and** SVG export |
| Descriptive stats | Full | Full (n, min, max, range, mean, var, sd, CV, skew, kurtosis, median, mode, percentiles) |
| Data cleaning | Spreadsheet only | Outlier flag (IQR/z/MAD), trim/winsorize, missing handling, log/Box-Cox toggle |
| Import | Excel / CSV / text | CSV (streaming) / Excel / paste; multi-column + frequency/weighted input (SHOULD) |
| RNG from fit | Yes | Yes |
| Comparison view | Overlay on chart | Overlay chart + cross-criterion comparison table |
| Sharing | None | **Shareable URL** (config + summary results), versioned payload |
| Local history | Project files | **Opt-in** IndexedDB recent datasets + fit history (one-click clear) |
| Offline | Desktop (always offline) | **PWA / service worker** (works offline) |
| Onboarding | Manual | Bundled example dataset, inline glossary (AICc / Akaike weights / Lilliefors), first-run guide — EN & ES |
| i18n | Single language | **English + Spanish**, runtime switch, locale-aware number formatting |
| Accessibility | None | **WCAG 2.2 AA**, keyboard nav, accessible data-table alternative for every chart |

### Prioritized roadmap (tiers — the final v1 cut depends on the deadline; see §11)

**MUST — parity to credibly replace EasyFit**
- Auto fit-all (~53 families) → ranked table; manual single-fit mode.
- Distribution catalog (§4) covering all common EasyFit families + core discrete.
- MLE (default) + MoM, showing fitted parameter values.
- KS, Anderson-Darling, Chi-Squared with a sortable rank column (1 = best).
- Full plot set: histogram+PDF, CDF (empirical overlay), survival, hazard, P-P, Q-Q, probability-difference.
- Descriptive statistics block.
- Import CSV / Excel / paste; large-dataset handling.
- RNG from any fitted distribution.
- Export: PNG/SVG charts, CSV results, HTML/print report.

**SHOULD — the EXCEED layer**
- AIC / AICc / BIC + Akaike weights as ranking criteria.
- Cramér-von Mises; Lilliefors-corrected KS; provenance-labeled p-values.
- Bootstrap CIs on parameters for the **top 3–5** ranked fits + asymptotic SEs.
- Interactive charts (zoom/pan/hover, toggle dists), comparison view.
- Data cleaning / outlier handling, transform toggles.
- Bilingual UI (EN/ES) + locale-aware number formatting.
- Shareable URL links; opt-in local history (IndexedDB).
- WCAG 2.2 AA accessibility.
- PWA / offline support.
- Onboarding: example dataset + inline glossary + first-run guide (EN/ES).
- Multi-column / frequency (grouped/weighted) import.

**COULD — post-launch, explicitly deferred (§12 YAGNI)**
- Niche distributions: Log-Pearson III, Wakeby, distinct Error-Function.
- Censored/truncated data; extra estimators (QME/MGE/MSE/L-moments).
- BCa bootstrap intervals (v1 uses percentile intervals); Monte-Carlo GoF p-values; Cullen-Frey pre-screen.
- Mixture/compound distributions; StatAssist-style parameter explorer.
- "Export to Python/R code"; embeddable widget; Monte-Carlo simulation.
- Accounts / cloud save (would break the privacy pitch and the stateless lock).

---

## 3. Recommended tech stack

| Concern | Choice | Version (re-pin at scaffold) | Why | Alternative |
|---|---|---|---|---|
| Language | TypeScript | 6.0.x | Strict-by-default; types catch numeric/array bugs | — |
| Framework | React (plain SPA, no meta-framework) | 19.2.x | Compute runs in a Worker → React runtime weight is a non-issue; deepest charts/stats/i18n/test ecosystem | SvelteKit + adapter-static (smaller bundle, thinner ecosystem) |
| Build tool | Vite | 8.0.x (Rolldown) | First-class `?worker` chunking, trivial static `dist/`. **ESM-only → CI/host build image must be Node ≥20.19 or ≥22.12** | — |
| Package manager | **pnpm**, pinned via **Corepack** (`packageManager` field) | 10.x | Strict + fast + disk-efficient; **blocks dependency lifecycle scripts by default** (`onlyBuiltDependencies` allowlist) and supports a **`minimumReleaseAge` cooldown** — the two strongest supply-chain defenses (§8) | — (npm/yarn lack the script-block default) |
| Styling | Tailwind CSS v4 via `@tailwindcss/vite` | 4.3.x | CSS-first config, near-zero setup, auto content detection | — |
| UI components | shadcn/ui (prefer **Base UI** primitives; Radix fallback) | current | Copy-in components you own; Base UI (MUI team) now better-maintained than post-WorkOS Radix | Headless UI (lighter) |
| Charts | Plotly.js via a **thin custom React component** (NOT react-plotly.js) | plotly.js 3.5.x | react-plotly.js is frozen at Sep 2022 (pre-React-19); Plotly itself does free PNG **and** SVG export, native hist+PDF, scattergl for 10k+ pts. **Ship the partial `plotly.js-cartesian-dist` bundle** (full is ~1.3 MB gz) | **Vega-Lite 6.4.x + react-vega** (native Q-Q/density, React-19-current) |
| Stats / distributions / special fns | **@stdlib** (per-function packages), pinned + wrapped behind one internal adapter module | 0.3.x / 0.2.x | 35+ dists (pdf/cdf/quantile/log-variants/moments) + **all** special fns incl. `digamma`/`trigamma`; per-package `.d.ts`. **Note:** pre-1.0 (pin!); npm packages are CommonJS (Vite consumes fine); license `Apache-2.0 AND BSL-1.0` (both permissive) | jStat (MIT; **no `@types/jstat` exists**; lacks digamma/trigamma — alternative only) |
| Optimization / MLE | **fmin** (Nelder-Mead) + **numeric.js** (BFGS, matrix inverse), **vendored & frozen** | fmin 0.0.4 / numeric 1.2.6 | Derivative-free 1–3-param NLL minimization; tiny; we own the loop. Both dormant + ship **no types** → vendor, pin, hand-write `.d.ts` | minimize-golden-section-1d (1-D only); **ml-levenberg-marquardt = least-squares ONLY, never the general MLE optimizer** |
| i18n | react-i18next + i18next + browser-languagedetector + http-backend | 17.x / 26.x | Only candidate with **reactive runtime switch, no reload**; auto-detect + localStorage persist built-in | Paraglide JS (smaller, reload-on-switch) |
| CSV parsing | PapaParse | 5.5.x | Streaming in a Worker (`worker:true`, `step`), flat memory | — |
| Excel parsing | SheetJS/xlsx — **vendored** (download the 0.20.x tarball from `cdn.sheetjs.com`, commit it / add as a `file:` dep; npm is stale at 0.18.5) | 0.20.x | Bundled at build time, **no runtime CDN** → satisfies the strict `default-src 'self'` CSP **and** removes a runtime supply-chain vector; no streaming → parse in worker, size-guard, steer big data to CSV | — |
| Compression (share links) | lz-string (`compressToEncodedURIComponent`) | 1.5.0 | URL-safe, tiny output | pako gzip + base64url |
| Local persistence | Dexie.js (IndexedDB) + localStorage for tiny prefs | 4.x | Clean schema/version/query; **opt-in + clearable** | idb (smaller, more boilerplate) |
| State | Zustand | 5.0.x | Tiny client store (no server → no TanStack Query) | — |
| Unit/component test | Vitest + React Testing Library + fast-check | vitest 4.x | Shares Vite pipeline; property tests for distribution invariants | — |
| E2E test | Playwright + @axe-core/playwright | 1.6x | E2E + a11y scans + ARIA snapshots | — |
| Lint / format | Biome | 2.x | One fast binary/config; add narrow ESLint only for `jsx-a11y`/`react-hooks` if needed | ESLint flat + Prettier |
| Worker RPC | Comlink (raw, via Vite `?worker`), pinned | 4.4.2 | Typed async RPC over postMessage. ~18 mo stale but stable; **avoid the wrapper plugin** | threads.js; hand-rolled postMessage |
| PWA / offline | `vite-plugin-pwa` (Workbox) | current | Precache the static shell for offline use | hand-rolled service worker |
| CI | GitHub Actions | — | typecheck → biome → vitest → build → playwright; deploy separate | — |
| Hosting | **Cloudflare Pages** (or Workers Static Assets for a new project) | — | Uncapped static bandwidth, no card, no commercial clause, auto SPA fallback, free custom-domain SSL | GitHub Pages (backup) |

---

## 4. The statistics & fitting engine

This is HardFit's core. **Reuse vs. implement is explicit.**

### Reuse (from @stdlib)
- PDF/PMF, log-PDF/log-PMF, CDF, quantile/inverse-CDF, moments for supported families.
- Special functions: `gammaln`, regularized incomplete gamma + inverse, regularized incomplete beta + inverse, `erf`/`erfc`/`erfinv`, **`digamma`/`trigamma`** (verified present — the reason to pick @stdlib over jStat).
- Seedable RNG samplers (bootstrap + "generate from fit").
- One-sample KS statistic (`@stdlib/stats-kstest`) and Chi-Squared GoF (`@stdlib/stats-chi2gof`). **Note:** kstest treats parameters as *known* — feeding it estimated parameters makes the p-value invalid (handled below).

### Implement ourselves (bugs concentrate here — test hardest, validate vs scipy/R)
- The whole **MLE fit loop**, MoM seeding, multi-start, log/logit reparameterization.
- **Anderson-Darling** + distribution-specific adjusted statistics & p-value approximations (@stdlib has no AD test).
- **Cramér-von Mises** statistic.
- **AIC / AICc / BIC**, Δ and Akaike weights, ranking orchestration.
- **Lilliefors** correction + **parametric-bootstrap** p-values; the optimizer (vendored fmin/numeric.js, frozen, hand-written `.d.ts`).

### Distributions (~53)

**Continuous (~45):** Normal, Lognormal, Log-Gamma, Exponential, Gamma, Generalized Gamma, Erlang, Weibull (2P/3P), Rayleigh, Chi-Squared, Student's t, F, Beta, Kumaraswamy, Uniform, Triangular, PERT, Cauchy, Laplace, Logistic, Generalized Logistic, Gumbel, Fréchet, GEV, Pareto, Lomax (Pareto-II), Generalized Pareto, Burr, Dagum, Birnbaum-Saunders, Inverse Gaussian, Lévy, Nakagami, Rice, Hyperbolic Secant, Johnson SB/SU, Inverse Gamma (Pearson-5), Pearson-6, Power Function, Reciprocal, Error/Exponential-Power.

**Discrete (~8):** Bernoulli, Binomial (n known), Geometric, Hypergeometric, Logarithmic, Negative Binomial, Poisson, Discrete Uniform.

*(Families beyond @stdlib's built-ins are composed from its special functions. Pin @stdlib and re-audit its catalog at build time. Log-Pearson III & Wakeby deferred to COULD — niche hydrology distributions.)*

### Parameter estimation strategy
- **Closed-form MLE where it exists** (exact + used as seeds): Normal, Lognormal (on logs), Exponential, Uniform, Pareto-I, Poisson, Geometric, Binomial (n known).
- **Numerical MLE otherwise:** minimize the negative log-likelihood over **unconstrained transformed params** — `θ = exp(φ)` for positive scale/shape, `θ = a + (b−a)·sigmoid(φ)` for bounded — seeded by **method-of-moments**, solved with **Nelder-Mead (fmin)** for 1–3 params (BFGS via numeric.js as a cross-check). Dedicated Newton fast-paths (Gamma: Choi-Wette/Minka; Weibull shape: 1-D Newton; Beta: 2-D Newton with trigamma) where speed matters, always with MoM-seeded NLL as fallback.
- **Multi-start** (5–20 perturbed inits, keep lowest NLL) — cheap at this dimensionality.
- **Standard errors:** fresh central-difference Hessian of the NLL at the MLE; `SE = sqrt(diag(inv(Hessian)))`; Wald CIs. **Do not** reuse the BFGS-accumulated inverse Hessian. Delta-method back-transform from log/logit space.
- Everything in **log-space** (`logpdf`/`logpmf`, sum logs); clamp CDFs to `(1e-12, 1−1e-12)` before logs; prefer `erfc`/upper-gamma in tails.

### Goodness-of-fit
- **KS / Lilliefors:** D = sup|F̂ − F|. The standard Kolmogorov p-value is **invalid when params are estimated from the same data** → use **Lilliefors** (normal/exponential) or **parametric bootstrap**. Always show the statistic; label the p-value's provenance.
- **Anderson-Darling:** primary EDF statistic (tail-weighted). Adjusted A\*² + closed-form p-value approximations where they exist (normal/lognormal `A²(1+0.75/n+2.25/n²)`; exponential `A²(1+0.6/n)`; Weibull/Gumbel: Stephens); parametric bootstrap otherwise.
- **Cramér-von Mises:** computed cheaply from the same sorted CDF array.
- **Chi-Squared:** equiprobable bins, expected ≥ 5 (pool tails), df = k − 1 − m. **Recommended for discrete fits** (EDF tests break under ties).
- **Every emitted p-value is tagged `exact` / `table` / `bootstrap`** in the result object and the UI.

### Model ranking
- `AIC = 2k − 2·LL`; `AICc = AIC + 2k(k+1)/(n−k−1)` (use when n/k < ~40; undefined for n ≤ k+1 → guard); `BIC = k·ln(n) − 2·LL`.
- **Default rank by AICc** (fall back to AIC for large n), secondary-sort by Anderson-Darling. Report Δᵢ and Akaike weights. Never rank by raw LL or by GoF p-value alone. Only compare candidates fit to the **same data on the same support**.

### "Fit all + rank" workflow
Per candidate: seed (MoM) → fit (closed-form or NLL minimization, multi-start) → compute LL, AIC/AICc/BIC, KS/AD/CvM/Chi-Sq → **stream the row back as it completes**. Sort by chosen criterion. **Bootstrap CIs only for the top 3–5** — bootstrapping all ~53 with multi-start is the difference between seconds and an unusable run.

---

## 5. Visualization plan

**Library:** Plotly.js wrapped in a **thin custom React component** calling `Plotly.react`/`Plotly.newPlot` against a ref. Avoid `react-plotly.js` (frozen Sep 2022, pre-React-19). Ship the **partial `plotly.js-cartesian-dist`** bundle (drops 3D/geo/mapbox). Free client-side PNG **and** SVG via `Plotly.downloadImage`/`toImage` (the "subscription" line in Plotly docs is a Chart-Studio artifact, not the OSS lib). Use `scattergl` (WebGL) for large on-screen scatters; the SVG renderer for crisp vector export (note: GL traces rasterize on SVG export).

**Alt-primary:** Vega-Lite 6.4.x + react-vega — declarative, React-19-current, native `quantileNormal`/`quantileUniform` (Q-Q recipe) + `density`/`regression`.

The stats layer (fits, plotting positions, inverse-CDF) lives **upstream** of the chart — the chart only renders points. Plotting position `pᵢ = (i − 0.5)/n` (Hazen).

| Chart | How (Plotly) |
|---|---|
| Histogram + fitted PDF | `histnorm:'probability density'` histogram + `scatter` line of the fitted PDF on a fine x-grid, same axes (multiple PDFs = multiple traces) |
| CDF (empirical vs fitted) | Empirical step (`scatter`, `line.shape:'hv'`) + smooth fitted-CDF line |
| P-P | `scatter` of (theoretical CDF(xᵢ), empirical i/n) on [0,1]² + `y=x` reference |
| Q-Q | `scatter` of (theoretical quantile(pᵢ), sample quantile xᵢ) + `y=x` reference (`scattergl` for 10k+) |
| Probability difference | line of (theoretical CDF, empirical − theoretical) + `y=0` rule |
| Survival / hazard | line traces of 1−F and f/(1−F) |
| Comparison | multiple fitted traces overlaid; legend toggle; paired cross-criterion table |

**Export:** PNG + SVG per chart, CSV of results, HTML/print report. Every chart ships an **accessible data-table alternative** (also the screen-reader path — §8).

---

## 6. i18n strategy

**Library:** react-i18next + i18next + `i18next-browser-languagedetector` + `i18next-http-backend` — the only candidate that switches language **reactively at runtime with no reload** (preserves filters, scroll, in-memory chart/fit state).

**Catalog:** **plain JSON** (not ICU). EN/ES have simple one/other plurals, so i18next's `Intl.PluralRules`-driven pluralization suffices. Namespaced by feature: `/public/locales/{en,es}/{common,stats,charts,settings}.json`, lazy-loaded by `http-backend` (`loadPath:'/locales/{{lng}}/{{ns}}.json'`).

**Switching / persistence / auto-detect:** detector order `['localStorage','navigator','htmlTag']`, auto-persists to localStorage. **Set `load:'languageOnly'`** so `es-ES`/`en-US` regional variants don't 404 against `/locales/es` and `/locales/en`. No URL-locale routing (pointless for a client SPA). Configure `useSuspense` deliberately (Suspense boundary or `ready` gate) given lazy loading.

**Type safety:** augment the `i18next` module (`i18next.d.ts`) from EN JSON for typed `t()` keys; **`i18next-parser` CI check** so missing ES keys fail the build instead of silently falling back.

**Locale-aware numbers (the es-vs-en separators):** this is the **browser's `Intl` job**. **Keep all standalone statistical values OUT of the catalog** and route them through a typed, memoized formatter hook keyed to `i18n.language`:

```ts
// src/i18n/formatters.ts — memoized Intl instances per locale
useFormatters(locale) -> { num, pct, decimal, int, sci, date }
```
- `en-US`: `1,234,567.89` and `12.5%`
- `es-ES`: `1.234.567,89` and `12,5 %` — **note the non-breaking space (U+00A0) before `%`**; assert with ` ` in snapshot tests.

Only numbers embedded **inside sentences/plurals** use inline `{{val, number}}`. Update `<html lang>` and chart ARIA labels on switch; run a11y audits in **both** locales.

---

## 7. Architecture & project structure

```
hardfit/
├─ public/
│  ├─ locales/{en,es}/{common,stats,charts,settings}.json
│  ├─ sample-data/                 # bundled example dataset (onboarding)
│  └─ _headers                     # Cloudflare CSP + security headers
├─ src/
│  ├─ engine/                      # PURE, framework-agnostic. NO DOM/React/Comlink imports.
│  │  ├─ distributions/            # one module per family: logpdf, cdf, quantile, momFromData
│  │  ├─ estimation/               # closed-form MLE, NLL minimization, MoM seeds, multi-start
│  │  ├─ optimize/                 # vendored fmin (Nelder-Mead) + numeric.js (BFGS, inv), Hessian/SE
│  │  ├─ gof/                      # ks, lilliefors, andersonDarling(+adjusted), cramerVonMises, chiSquared
│  │  ├─ selection/                # logLik, aic, aicc, bic, akaikeWeights, rank
│  │  ├─ bootstrap/                # resample + refit (top-k only), percentile CIs
│  │  ├─ descriptive/              # n, min, max, mean, var, sd, CV, skew, kurtosis, median, mode, pcts
│  │  ├─ data/                     # cleaning: outliers (IQR/z/MAD), trim/winsorize, transforms
│  │  ├─ plotData/                 # build P-P/Q-Q/CDF/prob-diff/hist series (numbers only)
│  │  ├─ fitAll.ts                 # orchestration: fit-all + rank, emits progress callbacks
│  │  └─ index.ts                  # public API: fitAll(data, opts, onProgress?, abort?) ...
│  ├─ engine.worker.ts             # thin shim: Comlink.expose(engineApi)  (Vite `?worker` chunk)
│  ├─ workerClient.ts              # Comlink.wrap(...) + typed proxy + transfer helpers
│  ├─ ui/                          # React: views, custom Plotly component, tables, controls, onboarding
│  ├─ i18n/                        # index.ts (init), formatters.ts, i18next.d.ts
│  ├─ state/                       # Zustand stores (dataset, fit results, prefs)
│  ├─ persistence/                 # Dexie (opt-in history, schema-versioned) + lz-string share-link (versioned payload)
│  ├─ types/                       # shared DTOs across engine ↔ worker ↔ UI (include schemaVersion)
│  └─ main.tsx
├─ tests/ { engine/, fixtures/scipy-r/, e2e/ }
├─ scripts/gen-fixtures.py         # scipy/R reference generator
├─ vite.config.ts  biome.json  tsconfig.json
└─ .github/workflows/{ci,deploy}.yml
```

**Module boundaries.** `engine/` is pure TypeScript with **zero** DOM/React/Comlink/Worker imports — depends only on @stdlib + vendored optimizers, unit-tests instantly in Node, independently reusable. `engine.worker.ts` is the only file that knows Comlink. The React UI talks only to a **typed Comlink proxy**, never the engine directly. `types/` holds serializable DTOs (each carries a `schemaVersion` for forward-compat of share links + IndexedDB rows).

**Worker boundary.** All heavy work (fits + GoF + bootstrap) runs in **one long-lived dedicated worker** via Vite's `new Worker(new URL('./engine.worker.ts', import.meta.url), {type:'module'})` + raw Comlink. The dataset moves zero-copy as a `Float64Array` via `Comlink.transfer(arr,[arr.buffer])` (**buffer is neutered afterward — copy first if the UI still needs raw values**). Progress streams back via a `Comlink.proxy(onProgress)` callback per finished distribution.

**Cancellation.** `AbortSignal` is **NOT** transferable/postMessage-able (WHATWG DOM #948). Cancellation is **cooperative**: a proxied `abort()` flag the worker **polls between distributions** and throws (main-thread promise rejects cleanly); a `worker.terminate()` watchdog is the hard backstop for a runaway optimizer.

**Persistence (opt-in).** Local history is **off by default**; a clear toggle enables it and a one-click "Clear all local data" wipes IndexedDB + localStorage. Share-link payloads and Dexie schemas are **versioned** (embedded `schemaVersion`) so old links/rows degrade gracefully instead of breaking.

**Data flow.** `import (CSV stream / Excel-in-worker / paste; multi-col + frequency)` → `parse` → `clean` → `fit-all-in-worker (stream ranked rows)` → `rank (AICc default)` → `bootstrap top-k` → `visualize` → `export / share`. Only small config + summary results go into the shareable **URL hash fragment** (never sent to any server).

---

## 8. Best practices

**TypeScript.** TS 6.0 `strict` extending `@tsconfig/strictest`: `noUncheckedIndexedAccess` (critical for fit-loop indexing), `exactOptionalPropertyTypes`, `verbatimModuleSyntax`, `moduleResolution:'bundler'`, explicit `types:[...]`.

**Testing & numerical validation.**
- **Reference fixtures:** `scripts/gen-fixtures.py` generates pdf/cdf/quantile + full fit results + KS/AD/Chi-Sq stats from **scipy.stats / R** per family at known params; commit as JSON in `tests/fixtures/scipy-r/`. *(These are HardFit's correctness oracle — see §13.)*
- **Relative-tolerance matcher** (`|a−b| ≤ rtol·|b| + atol`), not bare `toBeCloseTo` (absolute/decimal-digit-based; fails on tiny PDF tails / large log-likelihoods).
- **Property tests** (fast-check): CDF monotone in [0,1], PDF integrates to ~1, quantile = CDF inverse, ranking stable under permutation.
- Engine tests in Node (no jsdom); component tests via Vitest browser mode/RTL; E2E via Playwright.

**Security (the privacy guarantee).** Strict **Content-Security-Policy** via Cloudflare `public/_headers`: `default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'` — this technically **enforces "data never leaves the browser."** Add `Strict-Transport-Security`, `Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`, `Permissions-Policy` (deny camera/mic/geo). Validate/parse user CSV/JSON defensively (XSS via labels/tooltips, prototype pollution, ReDoS).

**Telemetry stance.** **No third-party analytics, no remote error reporting** (consistent with the privacy pitch + CSP). Crash visibility comes from an in-app React error boundary that shows a copyable local error report the user can choose to paste into a GitHub issue.

**Accessibility (WCAG 2.2 AA).** Never convey meaning by color alone (patterns/markers/labels); 4.5:1 text / 3:1 non-text contrast; keyboard-operable charts (`tabindex=0`, arrow-key traversal); **always provide an accessible data-table alternative** (doubles as the screen-reader path). Update `<html lang>` + chart ARIA on language switch; audit in **both** locales. `@axe-core/playwright` + manual AT testing.

**Performance + budgets.** All compute off the main thread. Lazy-load Plotly + results view via `React.lazy`/Suspense; partial Plotly bundle; conservative `manualChunks` (chart vendor only). **Budgets (Lighthouse pass/fail):** initial JS < ~200 KB gz (excluding lazy chart chunk), TBT < 200 ms, LCP < 2.5 s, CLS < 0.1. **Browser/device matrix:** last 2 versions of evergreen Chrome/Edge/Firefox/Safari + mobile Safari/Chrome; on mobile, cap auto-fit concurrency and warn before bootstrapping large data.

**Licensing.** App bundles MIT/Apache/BSD/BSL-1.0 code (incl. vendored fmin/numeric.js, @stdlib, Plotly, SheetJS). Add a **`license-checker` CI step** that fails on non-permissive licenses and **generates a `THIRD-PARTY-LICENSES` / NOTICE** file shipped with the app.

**Supply-chain security (treat npm as hostile-by-default).** Recent worm-style attacks (e.g. the 2025 *Shai-Hulud*–style campaigns) compromise popular packages via malicious lifecycle scripts and stolen maintainer tokens. Layered, all-free defenses:
- **Block install scripts (biggest single win).** pnpm 10+ does **not** run dependencies' `preinstall`/`install`/`postinstall` scripts unless explicitly allowlisted via **`onlyBuiltDependencies`** (in `package.json` / `pnpm-workspace.yaml`). Keep that allowlist tiny and code-reviewed (typically just `esbuild`). Most malware executes in these scripts — this shuts the main door.
- **Release-age cooldown.** Set pnpm's **`minimumReleaseAge`** (e.g. `1440`–`4320` minutes = 1–3 days) so a freshly-published version can't be installed until it has survived a detection window. The most effective defense against *zero-day* package compromises; pair with `minimumReleaseAgeExclude` only for trusted internal pkgs.
- **Pin + freeze + verify integrity.** `save-exact=true` (no `^` ranges), commit **`pnpm-lock.yaml`** (stores per-package integrity hashes), and **`pnpm install --frozen-lockfile`** everywhere (CI fails if the lockfile would change). pnpm verifies the content-addressable store on install.
- **Pin the toolchain itself.** Lock pnpm via the **`packageManager`** field + **Corepack** (a compromised package manager is also a threat), and pin Node (`.nvmrc` / `engines`).
- **Minimize + vendor.** The stack deliberately keeps few dependencies and **vendors small critical libs** (fmin, numeric.js, lz-string, SheetJS) — frozen, reviewed, no transitive surprises, no runtime CDN. Fewer deps = smaller attack surface.
- **Scan in CI (free).** `pnpm audit` (fail on high/critical) + **GitHub Dependabot** alerts + the **Dependency Review** action (blocks PRs that introduce vulnerable/malicious deps) + **Socket** (free for OSS — flags install scripts, obfuscated code, network/filesystem access, and known supply-chain attacks at PR time).
- **Prefer provenance.** Favor packages publishing npm **provenance attestations** (sigstore); check before adding any new dependency.
- **Harden CI as its own supply chain.** **Pin third-party GitHub Actions by full commit SHA** (not floating tags), default `GITHUB_TOKEN` to read-only with least-privilege per-job, and never expose deploy secrets to PR-triggered workflows.
- **Runtime containment (defense in depth).** The strict **CSP `connect-src 'self'`** (above) means that even if a malicious dependency shipped, it cannot exfiltrate the user's data to an external origin from the browser.

**Error handling.** Guard edges: AICc undefined for n ≤ k+1; MoM failures (neg-binomial needs var>mean; beta needs var<m(1−m)) → grid-seeded NLL fallback; near-singular Hessian → prefer bootstrap CIs; NaN/Inf likelihoods surface as a clear "failed to converge" row, not a crash. Worker exceptions re-throw on the main thread via Comlink → recoverable UI error.

**CI/CD.** One GitHub Actions workflow (push/PR), **Corepack-enabled pnpm**, fail-fast: `pnpm install --frozen-lockfile` → `pnpm audit --audit-level=high` → `tsc --noEmit` → `biome ci` → `vitest run --coverage` → `license-checker` → `pnpm build` → Playwright E2E (cached browsers). Third-party Actions **pinned by commit SHA**; read-only `GITHUB_TOKEN`. Deploy in a **separate** workflow on `main`/tag. `security-review` once before first deploy; Dependabot + Dependency-Review + Socket on PRs.

---

## 9. Free hosting (no database needed)

**Choice: Cloudflare Pages** (or **Workers Static Assets** for a brand-new project — identical economics; Pages is in maintenance mode but fully functional, *not* deprecated). **Verified:** uncapped static-asset bandwidth (fair-use, no overage billing), 500 builds/month, free custom-domain SSL, **no credit card, no commercial restriction**, automatic SPA fallback. **No backend database is needed** — static files + client compute + opt-in IndexedDB + URL-hash sharing.

**Deploy / setup**
1. Push repo to GitHub.
2. Cloudflare dashboard → Workers & Pages → Create → Pages → connect repo.
3. Build command `pnpm build`; output dir `dist`. (Cloudflare auto-detects pnpm from `pnpm-lock.yaml` + the `packageManager` field; ensure the build image is Node ≥20.19.)
4. SPA fallback is **automatic** (no `404.html` present ⇒ unmatched paths serve `index.html`); optionally add `public/_redirects` `/* /index.html 200`.
5. Add `public/_headers` (CSP + security headers, §8) — Cloudflare applies it automatically.
6. **Vite base gotcha:** Cloudflare serves at the domain root → keep `base:'/'`.
7. Custom domain: free `*.pages.dev`, or a **subdomain** (`app.example.com` via CNAME). An **apex** domain needs moving nameservers to Cloudflare (free, extra step) — avoid unless wanted.

**Backup: GitHub Pages.** Public repo, soft 100 GB/mo (emails, no hard suspend), free HTTPS. Gotchas: set **`base:'/<repo>/'`** in `vite.config.ts` for a project site (wrong value ⇒ blank white page), and add the **SPA fallback hack** `cp dist/index.html dist/404.html` post-build (or HashRouter). Official Vite Actions workflow, Source = GitHub Actions.

**Avoid as primary:** Vercel Hobby (non-commercial clause), Netlify free (credit cap **suspends** the site), Surge (paid custom-domain HTTPS).

**Discoverability (light):** a short landing section + `<title>`/meta description/Open-Graph tags + a sitemap; submit the `*.pages.dev` (or custom) URL to search engines. Enough for an assignment; not a marketing project.

---

## 10. Claude Code skills / plugins / MCP — what to install, how, and when

Most of these are **already available in your session**. For a fresh machine, here's the install mechanism per type.

### How to install
- **Plugins (bundle the skills):** manage with the **`/plugin`** command → browse a marketplace → install. Relevant plugins: `superpowers` (brainstorming, writing-plans, executing-plans, subagent-driven-development, test-driven-development, systematic-debugging, requesting/receiving-code-review, verification-before-completion, using-git-worktrees), `frontend-design`, `hookify` (hooks), `commit-commands`, `chrome-devtools-mcp`, `claude-md-management`, `skill-creator`. Confirm with `/plugin`.
- **MCP servers (live tools, not skills):** configure per-project in a committed **`.mcp.json`** (reproducible for anyone who clones), or globally via **`claude mcp add`**. Verify with `claude mcp list`.
  - **context7** (live, version-correct library docs) — already connected here. Fresh setup: add via `claude mcp add` using its HTTP transport (see Context7's docs; a free key raises limits). **Commit it to `.mcp.json`.**
  - **chrome-devtools-mcp** (Lighthouse, perf traces, a11y, network) — via the `chrome-devtools-mcp` plugin or `claude mcp add chrome-devtools -- npx chrome-devtools-mcp@latest`. Needs Chrome installed.
  - **claude-in-chrome** (browser automation / live smoke tests / demo GIFs) — the built-in Claude-in-Chrome tooling (`claude --chrome` / Chrome extension).
- **Built-in commands (no install):** `/code-review`, `/security-review`, `/init`, `/verify`, `/run`.

### When to use each
| Phase | Tool / skill | When |
|---|---|---|
| Brainstorm | **superpowers:brainstorming** | Up front (done): lock distributions, bilingual scope, chart types, perf budget. |
| Spec + plan | **superpowers:writing-plans** + `/init` (CLAUDE.md) | Turn this strategy into 2–5-min tasks with file paths + verification steps. |
| Isolate | **superpowers:using-git-worktrees** | Separate engine / UI / i18n streams from `main`. |
| Versioned docs | **context7 MCP** | Every config/API touch: Vite `?worker`/`base`, Vitest, Playwright, i18next, Plotly. Batch resolves. |
| TDD engine | **superpowers:test-driven-development** + Vitest + scipy/R fixtures | The pure `engine/` only. RED-GREEN-REFACTOR against reference fixtures + property tests. |
| Debug numerics | **superpowers:systematic-debugging** | Any NaN/Inf fit, non-convergence, worker clone failure, flaky test. |
| Build UI | **superpowers:subagent-driven-development** / **executing-plans** | Dispatch independent UI/i18n/persistence tasks. |
| Review | **superpowers:requesting-code-review** + **`/code-review`** | After engine, after worker bridge, after UI. |
| Design polish | **frontend-design** | One pass once the UI is functional — escape generic AI styling. |
| Browser audit | **chrome-devtools-mcp** + **a11y-debugging** + **debug-optimize-lcp** | Against localhost: a11y (both locales), LCP/CLS/TBT vs the §8 budgets. Enable only this phase (~18k tokens of tool defs). |
| E2E | **Playwright (committed code)** + @axe-core/playwright | EN↔ES toggle persists, upload→fit→render, SPA deep-link, UI never freezes during fit. |
| Verify | **superpowers:verification-before-completion** | Before any "done" — show actual command output. |
| Security | **`/security-review`** | Once before first deploy + before any data-handling change. |
| Deploy smoke | **claude-in-chrome** | Smoke-test the deployed URL (real Accept-Language, real CDN) + record a demo GIF. **Never gate CI on it** (beta, non-deterministic). |
| Finish | **superpowers:finishing-a-development-branch** | Merge/PR/cleanup. |

**Hooks (via `hookify`/settings.json), few + fast:** PostToolUse typecheck+lint+affected-tests on edit; block-commit-to-`main`; an i18n hardcoded-string guard; optional Stop hook running `vitest run`.

**CLAUDE.md outline:** project one-liner + locked decisions (stateless, client-only, EN/ES) · **architecture invariants** (`engine/` has zero DOM/React/Comlink imports; Comlink message contract; cooperative cancellation — no AbortSignal transfer) · **stats rules** (rank by AICc; KS p-value invalid for fitted params → Lilliefors/bootstrap; label p-value provenance; bootstrap top-k only; SE from fresh Hessian) · **i18n rules** (strings via i18next; standalone numbers via the Intl formatter module; EN/ES files in sync via CI; `load:'languageOnly'`) · **numerics base** (@stdlib only; vendored fmin/numeric.js frozen) · **build/deploy** (pnpm via Corepack, exact commands, Vite `base` + SPA fallback per host) · **supply-chain rules** (pnpm `onlyBuiltDependencies` allowlist + `minimumReleaseAge` cooldown; exact-pin + commit lockfile + `--frozen-lockfile`; vendor critical libs / no runtime CDN; pin Actions by SHA; never add a dep without a provenance/Socket check).

---

## 11. Phased build roadmap — **thin vertical slice first**

> Rationale: building the entire ~53-distribution engine before any UI is big-bang and the #1 *delivery* risk for an app that must ship and be demoed. M1 cuts a **deployable, demoable end-to-end slice**, surfacing the worker↔chart↔UI integration risks early; only then do we go wide.

| Milestone | Deliverable |
|---|---|
| **M0 — Scaffold + CI + empty deploy** | Vite 8 + React 19 + TS (strictest); **pnpm via Corepack (pinned `packageManager`)**; Biome; Vitest + Playwright wired; **supply-chain hardening** (`onlyBuiltDependencies` allowlist, `minimumReleaseAge` cooldown, exact-pin + committed lockfile, `pnpm audit` + Dependabot + Dependency-Review + Socket, Actions pinned by SHA); GitHub Actions (audit/typecheck/lint/test/build + license-checker); **empty app deploys green to Cloudflare Pages** from `main`; CLAUDE.md + hooks in place. |
| **M1 — Vertical slice (END-TO-END, deployed)** | ~5 distributions (Normal, Lognormal, Exponential, Gamma, Weibull) with MLE; KS + AICc; the Comlink worker + cooperative cancel; CSV import; fit-all → ranked table; **one chart** (histogram + fitted PDF) via the custom Plotly component; **deployed and usable**. Engine TDD'd against scipy/R fixtures for these 5. |
| **M2 — Engine breadth (TDD)** | All ~53 families; closed-form + numerical MLE (MoM seeds, multi-start, log/logit); full GoF (KS/Lilliefors, AD+adjusted, CvM, Chi-Sq); AIC/AICc/BIC + Akaike weights; descriptive stats; bootstrap (top-k) + SEs. **All validated vs committed scipy/R fixtures + property tests.** |
| **M3 — Full visualization + import + export** | Remaining charts (CDF, survival, hazard, P-P, Q-Q, prob-diff) + comparison view; data cleaning/outliers/transforms; Excel + paste import (multi-col/frequency); export PNG/SVG/CSV + HTML/print report; RNG-from-fit. |
| **M4 — i18n (EN/ES)** | react-i18next (plain JSON, `languageOnly`); EN+ES catalogs; reactive switcher + persistence + auto-detect; typed-key augmentation + CI sync check; Intl formatter module wired through every numeric display (es-vs-en separators). |
| **M5 — Polish / a11y / perf / onboarding** | frontend-design pass; WCAG 2.2 AA (keyboard, ARIA, accessible data tables, both locales); lazy chart loading + partial Plotly bundle; chrome-devtools-mcp audits green vs §8 budgets; **onboarding** (example dataset, inline glossary, first-run guide) EN/ES. |
| **M6 — Deploy hardening + share + offline** | CSP + `_headers`; opt-in IndexedDB history + one-click clear (schema-versioned); lz-string URL-hash share links (versioned payload); **PWA/offline** (vite-plugin-pwa); `/security-review`; production deploy; claude-in-chrome live smoke test + demo GIF. |

*(If the deadline is tight, ship M0→M1→M2→M3 as v1 — that already matches EasyFit's core — and fold M4–M6 into a fast-follow. If comfortable, the full M0→M6 is the "beat EasyFit" product.)*

---

## 12. Risks & mitigations + YAGNI

| Risk | Mitigation |
|---|---|
| **Numerical correctness** (#1 risk — MLE loop, AD adjustments, AIC/BIC, bootstrap are our code) | scipy/R reference fixtures + relative-tolerance matcher + property tests; cross-check reused numerics; multi-start + MoM seeds; guard AICc (n≤k+1), MoM validity, near-singular Hessian. |
| **KS p-value misuse** (most likely defect) | Never emit a raw KS p-value for fitted params; Lilliefors/bootstrap; rank by AICc; **label every p-value's provenance**. |
| **Big-bang delivery** | **Vertical-slice-first roadmap (M1)** → deployable/demoable early; widen after. |
| **Performance / responsiveness** | All compute in a dedicated worker; stream per-distribution; **bootstrap only top 3–5**; zero-copy transfer; partial Plotly bundle; scattergl for big scatters; mobile concurrency cap. |
| **Worker cancellation** | Cooperative abort polled between distributions + `terminate()` watchdog (AbortSignal not transferable). |
| **Excel memory** | SheetJS has no `.xlsx` streaming → parse in worker, size-guard, steer big data to CSV. |
| **Dependency staleness** | Vendor + pin frozen deps (fmin, numeric.js, lz-string); pin @stdlib (pre-1.0) behind one adapter module; raw Comlink (not the plugin); custom Plotly component; re-pin via `pnpm create vite@latest`. |
| **Supply-chain attack (malicious npm dependency)** | pnpm blocks install scripts by default (`onlyBuiltDependencies` allowlist) + `minimumReleaseAge` cooldown; exact-pin + committed lockfile + `--frozen-lockfile` + integrity hashes; Corepack-pinned pnpm; vendored critical libs (no runtime CDN); `pnpm audit` + Dependabot + Dependency-Review + Socket in CI; GitHub Actions pinned by SHA; CSP `connect-src 'self'` blocks browser-side exfiltration as defense-in-depth. |
| **Privacy claim unenforced** | Strict CSP + `_headers`; no telemetry/remote error reporting; client-only error boundary. |
| **Free-tier limits** | Cloudflare Pages uncapped bandwidth; avoid Netlify (suspends)/Vercel (non-commercial) as primary; no server/DB to exhaust. |
| **Broken old share links / stored rows** | Versioned share-link payload + Dexie schema (`schemaVersion` + migrations). |
| **Scope creep** | Ship MUST + core SHOULD; gate COULD behind explicit post-launch decision. |

**Explicit YAGNI cuts (not in v1):** niche dists (Log-Pearson III, Wakeby, distinct Error-Function); censored/truncated data; QME/MGE/MSE/L-moment estimators; BCa intervals (use percentile); mixture/compound distributions; Monte-Carlo simulation; accounts/cloud save (breaks privacy + the stateless lock); Cullen-Frey pre-screen; Monte-Carlo GoF p-values; "export to Python/R code"; embeddable widget; **WASM** (premature — add only if profiling proves the worker's inner log-likelihood kernel is the bottleneck).

---

## 13. What "beat EasyFit" means — acceptance criteria

EasyFit is Windows-only, paid, and uses **mixed estimators** (MoM/LSE/L-moments per family), so diffing HardFit's MLE output directly against EasyFit would manufacture false mismatches and requires a license we don't have. Instead, "beat EasyFit" is made **checkable** via two gates:

1. **Numerical parity with the authoritative oracle (scipy/R).** For every supported family, HardFit's fitted parameters, log-likelihood, and KS/AD/Chi-Sq statistics must match `scipy.stats` / R `fitdistrplus` within relative tolerance on the committed fixtures. *(This is a hard CI gate from M1 onward.)*
2. **Demonstrable feature superiority.** A checklist HardFit satisfies that EasyFit does not: AIC/AICc/BIC + Akaike weights · Cramér-von Mises + Lilliefors + p-value provenance · bootstrap CIs · interactive PNG+SVG charts · comparison view · bilingual EN/ES + locale number formatting · WCAG 2.2 AA · shareable links · PWA/offline · free + zero-install + privacy-first.

A canonical demo dataset (bundled in `public/sample-data/`) exercises the full loop end-to-end and is the artifact shown when defending the project.

---

*End of strategy. Next step: confirm the deadline → finalize the v1 cut (§11) → `superpowers:writing-plans` to produce the task-level implementation plan.*
