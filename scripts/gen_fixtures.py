#!/usr/bin/env python3
"""Generate the scipy reference fixtures for HardFit's parity gate.

This is a LOCAL developer script (never run in CI). It emits one JSON file per
distribution under ``tests/fixtures/scipy-r/``; the Vitest suite
``scipy-parity.test.ts`` asserts HardFit's engine against the committed JSON.

Two oracle modes per (distribution, dataset):

* **Mode B (GoF arithmetic at FIXED params).** We fit a scipy-native frozen
  distribution, then compute every goodness-of-fit number (KS D, raw A^2,
  Cramer-von Mises n*omega^2, and the equiprobable chi-square binning) FROM THAT
  FROZEN DIST. The fixed params are emitted in HardFit convention; the TS test
  feeds those mapped params into @stdlib and must reproduce the scipy-derived
  oracle to rtol 1e-9. The scipy<->HardFit parameter mapping therefore appears
  exactly once, on the emit side: a mapping typo makes the gate FAIL (it cannot
  silently self-cancel, because the oracle never round-trips through the map).

* **Mode A (fit-from-data).** For the closed-form families (normal, lognormal,
  exponential) we compute the MLE ANALYTICALLY in numpy (authoritative, not
  capped at scipy's optimizer wobble) and store the maximized log-likelihood.
  For the iterative families (gamma, weibull) we use ``scipy.*.fit(x, floc=0)``
  and store the log-likelihood as the gate quantity: the TS test asserts HardFit
  achieves a log-likelihood at least as good (HardFit may converge tighter).

Regenerate with::

    python3 -m venv .venv-fixtures
    .venv-fixtures/bin/pip install -r scripts/requirements-fixtures.txt
    .venv-fixtures/bin/python scripts/gen_fixtures.py
    pnpm exec biome format --write tests/fixtures/scipy-r/

See tests/fixtures/README.md for the full mapping table and the CI rule.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy import stats

# --- Output location -------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "scipy-r"

# --- Fixed canonical datasets (literal arrays; NO live RNG) ----------------
#
# Named so the JSON records which sample each fixture came from. All values are
# strictly positive so every dataset is valid for all five families (lognormal,
# exponential, gamma, weibull require x > 0; normal is unconstrained but happy
# with positives). ``m1_sample`` is the canonical M1 demo sample (n = 18); the
# larger samples push n past 20 so the chi-square binning yields df >= 1
# (k = floor(n/5) >= 4) for the 2-parameter families.

M1_SAMPLE = [
    2.1, 3.4, 1.8, 5.2, 2.9, 4.1, 3.0, 2.5, 6.0,
    3.7, 1.2, 4.8, 2.2, 3.9, 5.5, 2.7, 3.1, 4.4,
]

# n = 25, moderate spread; gives k = 5 -> df = 2 (2-param), df = 3 (exponential).
SPREAD_25 = [
    1.4, 2.8, 0.9, 3.3, 5.1, 2.2, 4.0, 1.1, 6.7, 3.0,
    2.5, 0.6, 4.9, 1.9, 3.6, 7.2, 2.0, 5.8, 1.3, 4.2,
    2.9, 3.8, 0.4, 6.1, 2.4,
]

# n = 30, right-skewed (gamma/weibull/lognormal-friendly); k = 6 -> df = 3 / 4.
SKEWED_30 = [
    0.5, 1.2, 0.8, 2.1, 1.5, 3.4, 0.9, 1.8, 4.2, 2.6,
    1.1, 0.7, 5.3, 1.9, 2.4, 0.6, 3.1, 1.4, 6.8, 2.0,
    1.6, 0.4, 2.9, 1.0, 3.7, 0.3, 4.5, 1.3, 2.2, 8.1,
]

# n = 3 — deliberately tiny so the AICc small-sample guard (n <= k + 1) fires for the
# 2-parameter families (n - k - 1 = 0), exercising the "Infinity" sentinel + its TS decode.
# Three distinct positives keep every family's fit non-degenerate. (Exponential has k = 1,
# so n - k - 1 = 1 > 0 → its AICc stays finite here; the sentinel fires for the rest.)
TINY_3 = [1.5, 2.5, 4.0]

# Strictly-positive datasets — valid for every family (the x>0 families require positivity;
# the R-support and bounded families are happy with positives too).
DATASETS: dict[str, list[float]] = {
    "tiny_3": TINY_3,
    "m1_sample": M1_SAMPLE,
    "spread_25": SPREAD_25,
    "skewed_30": SKEWED_30,
}

# n = 26, spans negatives and positives. Applied ONLY to the real-support families added in
# M2.3 Batch A (uniform, laplace, logistic, gumbel, cauchy) so the gate exercises negative data,
# not just positives. The x>0 families (lognormal/exponential/gamma/weibull/rayleigh/pareto/
# frechet) and the pre-existing `normal` fixtures stay on the positive datasets only — no churn
# to the already-green fixtures. k = floor(26/5) = 5 bins -> chi-square df = 5-1-2 = 2 for k=2.
SIGNED_26 = [
    -3.2, 1.4, -0.7, 2.9, -1.1, 0.3, 4.6, -2.5, 1.8, -0.2,
    3.1, -1.9, 0.9, 2.2, -0.5, 1.1, -3.8, 2.7, -1.3, 0.6,
    3.9, -0.9, 1.6, -2.1, 0.1, 2.4,
]

# Real-support families added in Batch A that additionally get the signed dataset.
REAL_SUPPORT_FAMILIES = {"uniform", "laplace", "logistic", "gumbel", "cauchy"}


def datasets_for(dist_name: str) -> dict[str, list[float]]:
    """Datasets to emit for a family: the positive set for everyone, plus the signed set for the
    real-support Batch A families (so negative data is exercised)."""
    if dist_name in REAL_SUPPORT_FAMILIES:
        return {**DATASETS, "signed_26": SIGNED_26}
    return DATASETS

# --- HardFit engine constants (mirror src/engine) --------------------------

MIN_EXPECTED_PER_BIN = 5  # gof.ts: k = max(2, floor(n / MIN_EXPECTED_PER_BIN))
AICC_INFINITY_SENTINEL = "Infinity"  # JSON cannot carry Inf; decode in the TS gate.


# --- Goodness-of-fit oracles (computed from a scipy-native frozen dist) ----


def ad_raw(xs_sorted: np.ndarray, cdf: np.ndarray) -> float:
    """Raw Anderson-Darling A^2 from the fitted CDF at the sorted data.

    A^2 = -n - (1/n) * sum_{i=1..n} (2i-1) * [ln F_i + ln(1 - F_{n+1-i})],
    matching HardFit's ``adStatistic`` (open-interval clamp, max(0, .)).
    """
    n = len(xs_sorted)
    clamped = np.clip(cdf, 1e-12, 1 - 1e-12)  # HardFit's AD_CLAMP_EPS
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(clamped) + np.log(1 - clamped[::-1])))
    return float(max(0.0, -n - s / n))


def ks_d(xs_sorted: np.ndarray, cdf: np.ndarray) -> float:
    """One-sample KS D = max(D+, D-) against the fitted CDF (HardFit's convention)."""
    n = len(xs_sorted)
    f = np.clip(cdf, 0.0, 1.0)
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - f)
    d_minus = np.max(f - (i - 1) / n)
    return float(max(0.0, d_plus, d_minus))


def cramer_von_mises(xs_sorted: np.ndarray, cdf: np.ndarray) -> float:
    """Cramer-von Mises n*omega^2 = 1/(12n) + sum (F_i - (2i-1)/(2n))^2 (scipy normalization)."""
    n = len(xs_sorted)
    f = np.clip(cdf, 0.0, 1.0)
    i = np.arange(1, n + 1)
    d = f - (2 * i - 1) / (2 * n)
    return float(1.0 / (12.0 * n) + np.sum(d * d))


def chi_squared_binning(
    data: list[float],
    ppf: Callable[[float], float],
    n_params: int,
) -> dict:
    """Equiprobable chi-square binning replicating HardFit's ``chiSquaredGof`` exactly.

    Edges are the fitted quantiles Q(j/k) for j=1..k-1 with
    k = max(2, floor(n / MIN_EXPECTED_PER_BIN)); E_j = n/k. Binning is strict
    (x > edge -> next bin), so a point exactly on an edge falls into the lower
    bin (measure-zero for a continuous fit, but replicated for fidelity).
    """
    n = len(data)
    k = max(2, n // MIN_EXPECTED_PER_BIN)
    edges = [float(ppf(j / k)) for j in range(1, k)]
    observed = [0] * k
    for x in data:
        b = 0
        for edge in edges:
            if x > edge:
                b += 1
            else:
                break
        observed[b] += 1
    expected = n / k
    statistic = float(sum((o - expected) ** 2 / expected for o in observed))
    df = k - 1 - n_params
    return {
        "bins": k,
        "edges": edges,
        "observed": observed,
        "expected": expected,
        "df": df,
        "statistic": statistic,
    }


def gof_block(data: list[float], rv, n_params: int) -> dict:
    """All Mode-B GoF arithmetic at the FIXED params of a frozen scipy dist ``rv``."""
    xs_sorted = np.sort(np.asarray(data, dtype=float))
    cdf = rv.cdf(xs_sorted)
    return {
        "ks": ks_d(xs_sorted, cdf),
        "adRaw": ad_raw(xs_sorted, cdf),
        "cvm": cramer_von_mises(xs_sorted, cdf),
        "chiSquared": chi_squared_binning(data, rv.ppf, n_params),
    }


def log_lik(data: list[float], rv) -> float:
    """Maximized log-likelihood of a frozen scipy dist at the data (sum of logpdf)."""
    return float(np.sum(rv.logpdf(np.asarray(data, dtype=float))))


def aicc_or_sentinel(log_lik_value: float, k: int, n: int) -> float | str:
    """AICc with HardFit's small-sample guard: emit the sentinel when n <= k + 1."""
    if n - k - 1 <= 0:
        return AICC_INFINITY_SENTINEL
    aic = 2 * k - 2 * log_lik_value
    return float(aic + (2 * k * (k + 1)) / (n - k - 1))


# --- Per-distribution fixture builders -------------------------------------
#
# Each builder fits a scipy-native frozen dist for Mode B, computes the Mode-A
# fit reference, and returns the (fixedParams, modeB, modeA) tuple. Params are
# emitted in HardFit convention via the verified scipy<->HardFit mapping:
#   expon.fit(x, floc=0)       -> (0, scale);        HardFit rate = 1/scale
#   gamma.fit(x, floc=0)       -> (a, 0, scale);     HardFit shape=a, rate=1/scale
#   weibull_min.fit(x, floc=0) -> (c, 0, scale);     HardFit shape=c, scale=scale
#   lognorm.fit(x, floc=0)     -> (s, 0, scale);     HardFit mu=ln(scale), sigma=s
#   norm.fit(x)                -> (loc, scale);      HardFit mu=loc, sigma=scale
# Closed-form Mode-A references (normal/lognormal/exponential) are computed
# ANALYTICALLY in numpy with ddof=0 (population std = HardFit's MLE).


def build_normal(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # Mode B: scipy norm.fit (NO floc) -> (loc, scale); for an MLE fit this equals
    # the analytic mean / population std, so the frozen dist IS the oracle.
    loc, scale = stats.norm.fit(arr)
    rv = stats.norm(loc=loc, scale=scale)
    fixed = {"mu": float(loc), "sigma": float(scale)}
    n = len(arr)
    # Mode A params: analytic MLE (mean, population std with ddof=0).
    mu_a = float(np.mean(arr))
    sigma_a = float(np.std(arr))  # numpy default ddof=0 -> population std
    # modeA.logLik comes from the INDEPENDENT scipy.fit frozen dist `rv` (NOT our analytic
    # reference), so the universal "HardFit LL >= scipy.fit LL" cross-check cannot self-cancel a
    # formula bug shared between this emit code and the TS engine. For the closed-form MLE
    # families scipy.fit == analytic, so the emitted value is unchanged — only its provenance
    # becomes formula-independent.
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "closed-form",
            "params": {"mu": mu_a, "sigma": sigma_a},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_lognormal(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    s, _loc, scale = stats.lognorm.fit(arr, floc=0)
    rv = stats.lognorm(s, loc=0, scale=scale)
    # HardFit: mu = ln(scale), sigma = s.
    fixed = {"mu": float(np.log(scale)), "sigma": float(s)}
    n = len(arr)
    # Mode A params: analytic normal MLE on the logs (population std, ddof=0).
    logs = np.log(arr)
    mu_a = float(np.mean(logs))
    sigma_a = float(np.std(logs))
    # modeA.logLik from the INDEPENDENT scipy.fit `rv` (see build_normal) — formula-independent
    # cross-check; scipy.fit == analytic here, so the value is unchanged.
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "closed-form",
            "params": {"mu": mu_a, "sigma": sigma_a},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_exponential(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    _loc, scale = stats.expon.fit(arr, floc=0)
    rv = stats.expon(loc=0, scale=scale)
    # HardFit: rate = 1 / scale.
    fixed = {"rate": float(1.0 / scale)}
    n = len(arr)
    # Mode A params: analytic MLE rate = 1 / mean.
    rate_a = float(1.0 / np.mean(arr))
    # modeA.logLik from the INDEPENDENT scipy.fit `rv` (see build_normal) — formula-independent
    # cross-check; scipy.fit (floc=0) == analytic here, so the value is unchanged.
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=1),
        "modeA": {
            "form": "closed-form",
            "params": {"rate": rate_a},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 1, n),
        },
    }


def build_gamma(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    a, _loc, scale = stats.gamma.fit(arr, floc=0)
    rv = stats.gamma(a, loc=0, scale=scale)
    # HardFit: shape = a, rate = 1 / scale (the @stdlib gamma RATE slot-trap).
    fixed = {"shape": float(a), "rate": float(1.0 / scale)}
    n = len(arr)
    # Mode A: iterative fit -> gate on log-likelihood (HardFit must be >= this).
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"shape": float(a), "rate": float(1.0 / scale)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_weibull(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    c, _loc, scale = stats.weibull_min.fit(arr, floc=0)
    rv = stats.weibull_min(c, loc=0, scale=scale)
    # HardFit: shape = c, scale = scale (the @stdlib weibull SCALE slot).
    fixed = {"shape": float(c), "scale": float(scale)}
    n = len(arr)
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"shape": float(c), "scale": float(scale)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


BUILDERS: dict[str, Callable[[list[float]], dict]] = {
    "normal": build_normal,
    "lognormal": build_lognormal,
    "exponential": build_exponential,
    "gamma": build_gamma,
    "weibull": build_weibull,
}


def manifest() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mani = manifest()
    for dist_name, build in BUILDERS.items():
        fixtures = []
        for dataset_name, data in datasets_for(dist_name).items():
            built = build(data)
            fixtures.append({"dataset": dataset_name, "data": data, **built})
        payload = {
            "distribution": dist_name,
            "manifest": mani,
            "fixtures": fixtures,
        }
        path = OUT_DIR / f"{dist_name}.json"
        with path.open("w", encoding="utf-8") as f:
            # allow_nan=False: fail loudly rather than emit bare NaN/Infinity
            # (invalid JSON, rejected by JS JSON.parse and Biome). AICc uses the
            # "Infinity" string sentinel; p-values are intentionally NOT stored
            # (the gate asserts statistics, never p-values).
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write("\n")
        print(f"wrote {path.relative_to(OUT_DIR.parent.parent)}", file=sys.stderr)
    print(f"manifest: {mani}", file=sys.stderr)


if __name__ == "__main__":
    main()
