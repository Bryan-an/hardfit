# scipy reference-fixture parity gate

The committed JSON under [`scipy-r/`](./scipy-r/) is HardFit's **external numerical
oracle**. It is generated locally from scipy/numpy and asserted against the engine by
[`scipy-r/scipy-parity.test.ts`](./scipy-r/scipy-parity.test.ts). This is what makes
"beat EasyFit" a *checkable* numerical-correctness criterion: the goodness-of-fit
arithmetic and the MLE fits are pinned to an independent, widely-trusted implementation.

## The rule: fixtures are committed; CI never runs Python

- The JSON in `scipy-r/*.json` is **generated locally and committed**.
- **CI runs only `pnpm test`** — it reads the committed JSON and asserts. **Never run
  Python in CI.** The generator is a developer tool, not a build step.
- Regenerate (and re-commit) only when the canonical datasets, the scipy↔HardFit mapping,
  or a deliberate scipy/numpy bump changes the expected numbers.

## Regenerate

```bash
python3 -m venv .venv-fixtures
.venv-fixtures/bin/pip install -r scripts/requirements-fixtures.txt
.venv-fixtures/bin/python scripts/gen_fixtures.py
pnpm exec biome format --write tests/fixtures/scipy-r/   # match the repo's JSON formatting
```

The final `biome format` step is required: `json.dump` writes one array element per line,
but `pnpm check` (`biome check .`) enforces Biome's canonical JSON formatting, so the
committed fixtures must be Biome-formatted or the lint gate fails on them.

## Pinned dependencies (verified on this machine)

`scripts/requirements-fixtures.txt`:

| package | pinned version |
| ------- | -------------- |
| numpy   | `2.3.4`        |
| scipy   | `1.17.1`       |

Generated with **Python 3.14.3, numpy 2.3.4, scipy 1.17.1** (cp314 wheels exist for both
pins — no deviation needed). Each JSON carries the actual `{python, numpy, scipy}` versions
in its `manifest` block, and the test surfaces them in the suite name.

## scipy ↔ HardFit parameter mapping (the gate's load-bearing detail)

Every oracle number is computed from a **scipy-native frozen distribution**; the params are
mapped into HardFit's convention **only on emit** (once, in `gen_fixtures.py`). The TS test
feeds those mapped params into `@stdlib` and must reproduce the scipy-derived oracle — so a
mapping typo makes the gate **fail**, it cannot silently self-cancel.

| distribution | scipy fit call            | scipy params → | HardFit convention                       |
| ------------ | ------------------------- | -------------- | ---------------------------------------- |
| normal       | `norm.fit(x)` (NO `floc`) | `(loc, scale)` | `mu = loc`, `sigma = scale` (÷n std)     |
| lognormal    | `lognorm.fit(x, floc=0)`  | `(s, 0, scale)`| `mu = ln(scale)`, `sigma = s`            |
| exponential  | `expon.fit(x, floc=0)`    | `(0, scale)`   | `rate = 1/scale`                         |
| gamma        | `gamma.fit(x, floc=0)`    | `(a, 0, scale)`| `shape = a`, `rate = 1/scale` (RATE trap)|
| weibull      | `weibull_min.fit(x, floc=0)` | `(c, 0, scale)` | `shape = c`, `scale = scale`          |

`@stdlib` slot reminders: gamma takes `(x, shape, RATE)` and weibull takes `(x, shape, SCALE)`
— the gamma fixtures store `rate`, the weibull fixtures store `scale`, matching the engine.

## What the gate asserts

Each fixture file (one per distribution) holds four canonical datasets. The datasets are
**fixed literal arrays** (no live RNG) so the oracle is reproducible: the M1 demo sample
(`n = 18`) plus two larger positive samples (`n = 25`, `n = 30`) that push `n` past 20 so the
equiprobable chi-square binning yields `df ≥ 1` for the 2-parameter families, plus a tiny
`n = 3` sample that trips the AICc small-sample guard (`n ≤ k + 1`) so the `"Infinity"`
sentinel and its decode path are actually exercised for the 2-parameter families.

- **Mode B — GoF arithmetic at FIXED params (`rtol 1e-9`).** At the scipy-derived params,
  HardFit's `ksStatistic` / `adStatistic` (raw A²) / `cramerVonMises` (scipy's `n·ω²`
  normalization) must match the scipy oracle to machine precision. **Chi-square is a softer,
  binning-coupled gate:** the bin shape (`k`, `df`, edge count) must match *exactly*, and the
  statistic only to a looser tol (a near-edge bin flip between `scipy.ppf` and
  `@stdlib.quantile` would shift the statistic, not indicate a real bug). The binning
  replicates HardFit's `k = max(2, ⌊n/5⌋)` equiprobable rule via `scipy.ppf(j/k)` edges.
- **Mode A — fit-from-data.** Closed-form families (normal/lognormal/exponential) have their
  MLE computed **analytically in numpy** (mean + ÷n population std; logs for lognormal;
  `rate = 1/mean` for exponential) and HardFit's fit must match within `1e-9` — the reference
  is *not* capped at scipy's optimizer wobble. Iterative families (gamma/weibull) use
  `scipy.*.fit(x, floc=0)` and the gate is `HardFit_LL ≥ oracle_LL − 1e-6` (HardFit is at
  least as good as scipy, which under-converges); params are a loose `1e-3` diagnostic only.

The gate asserts **statistics, never p-values**. JSON cannot carry `Infinity`, so AICc uses a
`"Infinity"` string sentinel (emitted when `n ≤ k + 1`) which the test decodes.
