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
import math
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

# Strictly-(0,1) data for the Beta family (M2.3 Batch B): the positive datasets (values > 1) and
# signed_26 are OUT OF SUPPORT for Beta. Values are strictly between 0 and 1 (no exact 0/1, which
# break ln(x)/ln(1-x)); interior/unimodal with variance < m(1-m) so the MoM seed stays positive.
# n = 22 -> chi-square k = floor(22/5) = 4 bins -> df = 4-1-2 = 1 for the k=2 Beta fit.
UNIT_22 = [
    0.42, 0.55, 0.38, 0.61, 0.47, 0.33, 0.58, 0.50, 0.44, 0.66, 0.29,
    0.52, 0.39, 0.63, 0.48, 0.36, 0.57, 0.45, 0.60, 0.41, 0.53, 0.34,
]
# n = 3 in (0,1): trips the AICc n <= k+1 sentinel for the k=2 Beta fit.
UNIT_TINY_3 = [0.3, 0.5, 0.7]

# Bounded-(0,1) families (M2.3 Batch B) that use the unit-interval datasets instead of the positives.
UNIT_INTERVAL_FAMILIES = {"beta"}

# --- M2.3 Batch D: Student-t real-support samples ---------------------------
#
# Student-t is real-support (the positive/unit/signed sets above are all the wrong shape for
# gating a heavy-tailed real-support fit), so it gets its OWN datasets: real t(df) draws. They
# are generated ONCE here from a SEEDED PCG64 stream (np.random.default_rng — version-stable, NOT
# live system RNG) and rounded to 4 decimals, so the committed JSON is byte-stable across reruns.
# Per the plan's fixture discipline: use small-ish df (4, 8) so df is IDENTIFIABLE, and the
# project's moderate-n convention (n=25, n=2000) — NEVER adversarial tiny-n (where scipy's t.fit
# diverges to degenerate df→0/scale→0 maxima and the gate would mislabel HardFit's correct fit).
_T_RNG = np.random.default_rng(20260601)
# n=25, t(df=4, loc=2, scale=1.5): a small heavy-tailed sample where df is well-identified.
STUDENT_T4_25 = [round(float(v), 4) for v in stats.t(4, loc=2, scale=1.5).rvs(size=25, random_state=_T_RNG)]
# n=2000, t(df=8, loc=0, scale=1): a large near-normal-ish sample; df identifiable, exercises the
# 399-edge equiprobable chi-square at scale (verified: @stdlib t-quantile ↔ scipy.ppf to ~1e-15).
STUDENT_T8_2000 = [round(float(v), 4) for v in stats.t(8, loc=0, scale=1.0).rvs(size=2000, random_state=_T_RNG)]

# --- M2.3 Batch D: Fisher–Snedecor F positive-support samples ---------------
#
# F is positive-support (x > 0) and heavy-tailed, so it gets its OWN real F(d1, d2) draws (the
# generic positive DATASETS are the wrong shape for gating a moderate-df F fit). Generated ONCE from
# a SEEDED PCG64 stream and rounded to 4 decimals → byte-stable JSON. Per the plan's fixture
# discipline: moderate df (5,12) / (10,20) so the optimizer's basin is well-defined, and the
# project's moderate-n convention (n=25, n=2000) — NEVER adversarial tiny-n.
_F_RNG = np.random.default_rng(20260601)
# n=25, F(d1=5, d2=12): a small moderate-df sample. scipy.f.fit under-converges here (the parity
# skip-set covers the d1/d2 diagnostic), so the LL cross-check is the gate.
FISHER_F5_12_25 = [round(float(v), 4) for v in stats.f(5, 12).rvs(size=25, random_state=_F_RNG)]
# n=2000, F(d1=10, d2=20): a large moderate-df sample; exercises the 399-edge equiprobable chi-square
# at scale (verified: @stdlib f-quantile ↔ scipy.ppf to ~1e-15).
FISHER_F10_20_2000 = [round(float(v), 4) for v in stats.f(10, 20).rvs(size=2000, random_state=_F_RNG)]

# --- M2.3 Batch D: Inverse Gaussian (Wald) positive-support samples ---------
#
# Inverse Gaussian is positive-support (x > 0) with a CLOSED-FORM MLE, so it gets its OWN real
# invgauss draws (the generic positive DATASETS are the wrong shape and — crucially — too small to
# stress the bisection quantile). THE TRAP: scipy uses invgauss(mu_s=mu/lambda, loc=0, scale=lambda),
# NOT mu directly. Generated ONCE from a SEEDED PCG64 stream and rounded to 4 decimals → byte-stable.
# The n=2000 set is LOAD-BEARING: its 399 equiprobable chi-square edges (out to Q(0.9975)) are the
# ONLY cross-oracle for the engine's geometric-bracket bisection quantile (the unit tests only check
# self-consistency cdf(quantile(p))≈p); a tail/bracket bug surfaces here against scipy.ppf to ~1e-9.
_IG_RNG = np.random.default_rng(20260601)
# n=25, IG(mu=1.5, lambda=2): a small skewed sample (small lambda → heavy right tail).
INVGAUSS_1_5_2_25 = [
    round(float(v), 4) for v in stats.invgauss(1.5 / 2.0, loc=0, scale=2.0).rvs(size=25, random_state=_IG_RNG)
]
# n=2000, IG(mu=2, lambda=5): a large sample that stresses the bisection's geometric bracket out to
# the deep upper tail (Q(0.9975)); verified the closed-form MLE == scipy.invgauss.fit remap to ~1e-9.
INVGAUSS_2_5_2000 = [
    round(float(v), 4) for v in stats.invgauss(2.0 / 5.0, loc=0, scale=5.0).rvs(size=2000, random_state=_IG_RNG)
]

# --- M2.3 Batch C: integer COUNT datasets (the continuous datasets are out-of-support for discrete
# fits — non-integer values make logPMF -> -inf). Each discrete family gets count data on its own
# support; values are stored as ints so the engine and the discrete chi-square round-trip them.
POISSON_COUNTS = [0, 1, 2, 1, 3, 0, 2, 1, 4, 2, 1, 0, 3, 2, 1, 2, 0, 1, 3, 1, 2, 1, 0, 2, 1]  # n=25
GEOM_COUNTS = [0, 1, 0, 2, 3, 0, 1, 1, 4, 0, 2, 1, 0, 1, 2, 0, 3, 1, 0, 1]  # n=20, {0,1,...} failures
# OVERDISPERSED (var > mean) — mandatory for a finite negative-binomial MLE.
NBINOM_COUNTS = [0, 1, 2, 5, 8, 3, 12, 0, 4, 6, 1, 9, 2, 15, 0, 7, 3, 1, 10, 4, 2, 6, 0, 8, 1, 5, 3, 11, 2, 7]  # n=30
DISCRETE_UNIFORM_COUNTS = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 3, 4, 2, 5]  # n=20, a=1 b=6
# Tiny integer sets to trip the AICc n<=k+1 sentinel (k=1 -> fires at n=2; k=2 -> n=3).
COUNT_TINY = [1, 3]  # n=2: sentinel for the k=1 families (poisson, geometric)
# n=3: sentinel for the k=2 families. MUST be overdispersed (var > mean) so the negative-binomial
# MLE is finite: [0,1,5] has mean 2, var 4.67. Also valid discrete-uniform data (a=0, b=5).
COUNT_TINY_3 = [0, 1, 5]

# Per-family discrete datasets (each on its own integer support). Keyed like the other family maps.
DISCRETE_DATASETS: dict[str, dict[str, list[float]]] = {
    "poisson": {"count_tiny": COUNT_TINY, "poisson_counts": POISSON_COUNTS},
    "geometric": {"count_tiny": COUNT_TINY, "geom_counts": GEOM_COUNTS},
    "negative-binomial": {"count_tiny_3": COUNT_TINY_3, "nbinom_counts": NBINOM_COUNTS},
    "discrete-uniform": {"count_tiny_3": COUNT_TINY_3, "discrete_uniform_counts": DISCRETE_UNIFORM_COUNTS},
}


def datasets_for(dist_name: str) -> dict[str, list[float]]:
    """Datasets to emit for a family: integer counts for discrete families, unit-interval for the
    bounded-(0,1) Batch B family, the signed set for real-support Batch A families, else positives."""
    if dist_name in DISCRETE_DATASETS:
        return DISCRETE_DATASETS[dist_name]
    if dist_name in UNIT_INTERVAL_FAMILIES:
        return {"unit_tiny_3": UNIT_TINY_3, "unit_22": UNIT_22}
    if dist_name == "student-t":
        # Real-support t(df) draws (Batch D); NOT the positive/signed sets.
        return {"student_t4_25": STUDENT_T4_25, "student_t8_2000": STUDENT_T8_2000}
    if dist_name == "fisher-f":
        # Real F(d1, d2) draws (Batch D); positive-support, moderate df.
        return {"fisher_f5_12_25": FISHER_F5_12_25, "fisher_f10_20_2000": FISHER_F10_20_2000}
    if dist_name == "inverse-gaussian":
        # Real invgauss draws (Batch D); positive-support. The n=2000 set stresses the bisection
        # quantile's geometric bracket at the 399 equiprobable edges (its only scipy cross-oracle).
        return {"invgauss_1_5_2_25": INVGAUSS_1_5_2_25, "invgauss_2_5_2000": INVGAUSS_2_5_2000}
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


# --- M2.3 Batch C: DISCRETE goodness-of-fit oracles ------------------------
#
# Discrete fits use PMF/count-based chi-square (the EDF tests KS/AD/CvM are invalid under ties).
# scipy DISCRETE dists have NO .fit, so Mode-A MLEs are analytic (below) and the independent LL
# cross-check is a formula-free grid scan (grid_scan_loglik). The binning here MIRRORS
# src/engine/gof.ts chiSquaredGofDiscrete byte-for-byte; the parity gate pins the emitted
# cells + observed + expected so any divergence fails loudly.

MAX_DISCRETE_SCAN = 100_000  # mirrors gof.ts; defensive cap (the tail -> 0 guarantees termination)
EXPECTED_EPS = 1e-9  # mirrors gof.ts: slack so an E exactly at MIN merges the same way TS-vs-Python


def chi_squared_binning_discrete(
    data: list[float],
    pmf: Callable[[int], float],  # P(X = v)
    cdf: Callable[[int], float],  # F(v) = P(X <= v)
    support_min: int,
    support_max: float,  # may be math.inf (unbounded counts)
    n_params: int,
) -> dict:
    """Group-by-integer-value chi-square binning, byte-identical to gof.ts chiSquaredGofDiscrete:
    walk the support upward accumulating an open cell until BOTH its expected >= MIN and the
    remaining upper tail n*(1-cdf(v)) >= MIN; fold the open cell + remaining mass into one final
    cell [openLo, support_max] whose expected is the exact n*(1-cdf(openLo-1)). df = bins-1-n_params.
    """
    n = len(data)

    def count_in(lo: int, hi: float) -> int:
        return int(sum(1 for x in data if lo <= x <= hi))

    cells: list[dict] = []
    open_lo = support_min
    acc_expected = 0.0
    v = support_min
    while v <= support_max and v < support_min + MAX_DISCRETE_SCAN:
        acc_expected += n * pmf(v)
        remaining_tail = n * (1 - cdf(v))  # expected mass strictly above v
        # -EXPECTED_EPS: mirror gof.ts so an E exactly at MIN merges identically TS-vs-Python.
        if remaining_tail < MIN_EXPECTED_PER_BIN - EXPECTED_EPS or v >= support_max:
            break
        if acc_expected >= MIN_EXPECTED_PER_BIN - EXPECTED_EPS:
            cells.append(
                {
                    "lo": open_lo,
                    "hi": v,
                    "observed": count_in(open_lo, v),
                    "expected": acc_expected,
                }
            )
            open_lo = v + 1
            acc_expected = 0.0
        v += 1
    cells.append(
        {
            "lo": open_lo,
            "hi": support_max if math.isinf(support_max) else int(support_max),
            "observed": count_in(open_lo, support_max),
            "expected": n * (1 - cdf(open_lo - 1)),
        }
    )
    statistic = float(sum((c["observed"] - c["expected"]) ** 2 / c["expected"] for c in cells))
    k = len(cells)
    df = k - 1 - n_params
    # JSON cannot carry inf; the unbounded final cell's hi is emitted as the sentinel string.
    cells_json = [
        {**c, "hi": "Infinity" if (isinstance(c["hi"], float) and math.isinf(c["hi"])) else c["hi"]}
        for c in cells
    ]
    return {"bins": k, "cells": cells_json, "df": df, "statistic": statistic}


def gof_block_discrete(data: list[float], rv, n_params: int, support_min: int, support_max: float) -> dict:
    """Discrete Mode-B GoF block: PMF-binned chi-square only (KS/AD/CvM are null — invalid for
    discrete). `rv` is a frozen scipy discrete dist already in HardFit's support convention
    (e.g. geom(p, loc=-1), randint(a, b+1)), so rv.pmf/rv.cdf at integer v match @stdlib."""
    return {
        "ks": None,
        "adRaw": None,
        "cvm": None,
        "chiSquared": chi_squared_binning_discrete(
            data,
            lambda v: float(rv.pmf(v)),
            lambda v: float(rv.cdf(v)),
            support_min,
            support_max,
            n_params,
        ),
    }


def log_lik_discrete(data: list[float], rv) -> float:
    """Maximized log-likelihood of a frozen DISCRETE scipy dist (sum of logPMF, not logpdf)."""
    return float(np.sum(rv.logpmf(np.asarray(data, dtype=float))))


def grid_scan_loglik(data: list[float], logpmf_fn: Callable[..., float], grids: list[np.ndarray]) -> float:
    """Formula-INDEPENDENT log-likelihood floor: the max of sum(logpmf) over a coarse parameter
    grid. Never calls the MLE formula, so a bug shared between this emitter and the TS engine
    cannot self-cancel — the engine's fitted LL must still clear this floor. Load-bearing for the
    iterative discrete families (e.g. negative-binomial); the closed-form ones gate params directly.
    """
    arr = np.asarray(data, dtype=float)
    best = -np.inf
    meshes = np.meshgrid(*grids, indexing="ij")
    for idx in np.ndindex(meshes[0].shape):
        params = [float(m[idx]) for m in meshes]
        ll = float(np.sum(logpmf_fn(arr, *params)))
        if np.isfinite(ll) and ll > best:
            best = ll
    return best


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


# --- M2.3 Batch A builders -------------------------------------------------


def build_uniform(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # Mode B: scipy uniform.fit -> (loc=min, scale=max-min); the MLE is exact (min, max).
    loc, scale = stats.uniform.fit(arr)
    rv = stats.uniform(loc=loc, scale=scale)
    # HardFit: a = loc (lower bound = min), b = loc + scale (upper bound = MAX, NOT the raw
    # width/scale). Emitting `scale` into b is a silent gate failure — this is the load-bearing map.
    fixed = {"a": float(loc), "b": float(loc + scale)}
    n = len(arr)
    # Mode A: analytic MLE a = min, b = max (unique closed form).
    a_a = float(np.min(arr))
    b_a = float(np.max(arr))
    ll = log_lik(data, rv)  # independent scipy.fit LL (== analytic for uniform)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "closed-form",
            "params": {"a": a_a, "b": b_a},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_rayleigh(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    _loc, scale = stats.rayleigh.fit(arr, floc=0)
    rv = stats.rayleigh(loc=0, scale=scale)
    # HardFit: sigma = scale (IDENTITY — scipy's rayleigh `scale` IS the Rayleigh sigma; do NOT
    # invert like exponential/gamma).
    fixed = {"sigma": float(scale)}
    n = len(arr)
    # Mode A: analytic MLE sigma = sqrt(sum x^2 / 2n) — NOT scipy.fit's numerical scale.
    sigma_a = float(np.sqrt(np.sum(arr**2) / (2 * n)))
    ll = log_lik(data, rv)  # independent scipy.fit LL (numerical; <= analytic max)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=1),
        "modeA": {
            "form": "closed-form",
            "params": {"sigma": sigma_a},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 1, n),
        },
    }


def build_pareto(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    # Analytic MLE (closed form). scipy.pareto.fit is a fragile numerical optimizer, so the Mode-B
    # frozen dist + the Mode-A param reference are built from the ANALYTIC MLE, NOT scipy.pareto.fit.
    xm = float(np.min(arr))
    alpha = float(n / np.sum(np.log(arr / xm)))
    rv = stats.pareto(alpha, loc=0, scale=xm)
    fixed = {"shape": alpha, "scale": xm}
    # Independent LL cross-check: scipy's OWN numerical fit (does not share HardFit's closed form),
    # so a formula bug shared by the Python emit + the TS engine lands below this and fails the gate.
    b_s, _loc_s, scale_s = stats.pareto.fit(arr, floc=0)
    ll = log_lik(data, stats.pareto(b_s, loc=0, scale=scale_s))
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "closed-form",
            "params": {"shape": alpha, "scale": xm},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_laplace(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # NO floc — Laplace's location is a FREE parameter (median). Forcing floc=0 would corrupt it.
    loc, scale = stats.laplace.fit(arr)
    rv = stats.laplace(loc=loc, scale=scale)
    # Direct map (no inversion): mu = loc, b = scale (the Laplace diversity).
    fixed = {"mu": float(loc), "b": float(scale)}
    n = len(arr)
    ll = log_lik(data, rv)
    # LL-only gate (even-n median is non-unique): emit as "iterative" so the test gates LL, not params.
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"mu": float(loc), "b": float(scale)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_logistic(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    loc, scale = stats.logistic.fit(arr)  # NO floc (free location)
    rv = stats.logistic(loc=loc, scale=scale)
    fixed = {"mu": float(loc), "s": float(scale)}
    n = len(arr)
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"mu": float(loc), "s": float(scale)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_gumbel(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # gumbel_r = MAX / right-skewed (matches @stdlib gumbel: CDF exp(-exp(-(x-mu)/beta))). NO floc.
    loc, scale = stats.gumbel_r.fit(arr)
    rv = stats.gumbel_r(loc=loc, scale=scale)
    fixed = {"mu": float(loc), "beta": float(scale)}
    n = len(arr)
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"mu": float(loc), "beta": float(scale)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_cauchy(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # NO floc — the floc=0 trap would pin the median at 0 and corrupt the oracle on positive data.
    loc, scale = stats.cauchy.fit(arr)
    rv = stats.cauchy(loc=loc, scale=scale)
    fixed = {"x0": float(loc), "gamma": float(scale)}
    n = len(arr)
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"x0": float(loc), "gamma": float(scale)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_frechet(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # scipy invweibull == Frechet with m=0; floc=0 fixes the location at 0.
    c, _loc, scale = stats.invweibull.fit(arr, floc=0)
    rv = stats.invweibull(c, loc=0, scale=scale)
    # HardFit: shape = c, scale = scale (SCALE slot, passed directly — no inversion).
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


# --- M2.3 Batch B builders -------------------------------------------------


def build_levy(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    # Closed-form MLE with location fixed at 0: c = n / sum(1/x) (harmonic mean).
    c = float(n / np.sum(1.0 / arr))
    rv = stats.levy(loc=0, scale=c)
    # IDENTITY map: scipy levy `scale` == the @stdlib c (verified to 1e-14; do NOT invert).
    fixed = {"c": c}
    # Independent LL floor from scipy's own optimizer (scipy.levy.fit ~ analytic to 1e-5).
    _loc_s, scale_s = stats.levy.fit(arr, floc=0)
    ll = log_lik(data, stats.levy(loc=0, scale=scale_s))
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=1),
        "modeA": {
            "form": "closed-form",
            "params": {"c": c},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 1, n),
        },
    }


def build_chisquare(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # chi-squared has NO real loc/scale (scale is fixed at 2 via the df shape) -> fix BOTH.
    df, _loc, _scale = stats.chi2.fit(arr, floc=0, fscale=1)
    rv = stats.chi2(df, loc=0, scale=1)
    fixed = {"df": float(df)}
    n = len(arr)
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=1),
        "modeA": {
            "form": "iterative",
            "params": {"df": float(df)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 1, n),
        },
    }


def build_chi(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    df, _loc, _scale = stats.chi.fit(arr, floc=0, fscale=1)
    rv = stats.chi(df, loc=0, scale=1)
    fixed = {"k": float(df)}
    n = len(arr)
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=1),
        "modeA": {
            "form": "iterative",
            "params": {"k": float(df)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 1, n),
        },
    }


def build_invgamma(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # floc=0 ONLY: scale is a GENUINE parameter (= beta), unlike chi2/betaprime.
    a, _loc, scale = stats.invgamma.fit(arr, floc=0)
    rv = stats.invgamma(a, loc=0, scale=scale)
    # HardFit {shape, scale}: scale IDENTITY (scipy scale == @stdlib beta SCALE slot; do NOT invert).
    fixed = {"shape": float(a), "scale": float(scale)}
    n = len(arr)
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"shape": float(a), "scale": float(scale)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_betaprime(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # @stdlib betaprime is shape-only (no scale) -> fix BOTH loc and scale.
    a, b, _loc, _scale = stats.betaprime.fit(arr, floc=0, fscale=1)
    rv = stats.betaprime(a, b, loc=0, scale=1)
    fixed = {"alpha": float(a), "beta": float(b)}
    n = len(arr)
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"alpha": float(a), "beta": float(b)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_cosine(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    # cosine has REAL loc + scale (do NOT fix), but the default unseeded fit hits LL=-inf on
    # skewed data, so SEED loc/scale: MoM scale widened to contain all data inside (loc +- pi*scale).
    mu0 = float(np.mean(arr))
    mom_scale = float(np.std(arr) / np.sqrt(np.pi**2 / 3.0 - 2.0))
    need_scale = float(np.max(np.abs(arr - mu0)) / np.pi)
    scale0 = max(mom_scale, need_scale * 1.05)
    loc, scale = stats.cosine.fit(arr, loc=mu0, scale=scale0)
    rv = stats.cosine(loc=loc, scale=scale)
    # LOAD-BEARING: @stdlib support [mu-s, mu+s] == scipy [loc-pi*scale, loc+pi*scale], so the
    # @stdlib scale s = pi * scipy_scale. Emitting `scale` directly is a silent gate failure.
    fixed = {"mu": float(loc), "s": float(np.pi * scale)}
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"mu": float(loc), "s": float(np.pi * scale)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_beta(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    # Beta on (0,1): fix BOTH loc=0 and scale=1 so only the two shapes are estimated.
    a, b, _loc, _scale = stats.beta.fit(arr, floc=0, fscale=1)
    rv = stats.beta(a, b, loc=0, scale=1)
    fixed = {"alpha": float(a), "beta": float(b)}
    n = len(arr)
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"alpha": float(a), "beta": float(b)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


# --- M2.3 Batch C discrete builders (analytic MLEs; scipy discrete dists have NO .fit) -----


def build_poisson(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    lam = float(np.mean(arr))  # analytic MLE lambda = mean
    rv = stats.poisson(mu=lam, loc=0)  # IDENTITY: scipy mu == lambda, no shift
    ll = log_lik_discrete(data, rv)
    return {
        "fixedParams": {"lambda": lam},
        "modeB": gof_block_discrete(data, rv, n_params=1, support_min=0, support_max=math.inf),
        "modeA": {
            "form": "closed-form",
            "params": {"lambda": lam},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 1, n),
        },
    }


def build_geometric(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    p = float(1.0 / (1.0 + np.mean(arr)))  # analytic MLE for the {0,1,...} (failures) convention
    # LOAD-BEARING: scipy.geom is native {1,2,...}; loc=-1 shifts it to {0,1,...} to match @stdlib.
    rv = stats.geom(p, loc=-1)
    ll = log_lik_discrete(data, rv)
    return {
        "fixedParams": {"p": p},
        "modeB": gof_block_discrete(data, rv, n_params=1, support_min=0, support_max=math.inf),
        "modeA": {
            "form": "closed-form",
            "params": {"p": p},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 1, n),
        },
    }


def build_negative_binomial(data: list[float]) -> dict:
    from scipy.special import digamma, polygamma

    arr = np.asarray(data, dtype=float)
    n = len(arr)
    xbar = float(np.mean(arr))
    s2 = float(np.var(arr))  # population variance (ddof=0), matches the engine guard var > mean
    # Analytic MLE (scipy has no .fit): profile p = r/(r+xbar); solve r by 1-D Newton on the
    # profile score. Requires overdispersion (s2 > xbar) — else r -> inf (Poisson limit).
    r = max(1e-3, xbar * xbar / (s2 - xbar)) if s2 > xbar else float("nan")
    for _ in range(100):
        g = float(np.sum(digamma(arr + r) - digamma(r)) + n * math.log(r / (r + xbar)))
        gp = float(np.sum(polygamma(1, arr + r) - polygamma(1, r)) + n * (1.0 / r - 1.0 / (r + xbar)))
        step = g / gp
        nxt = r - step
        if not math.isfinite(nxt) or nxt <= 0:
            r = r / 2
            continue
        r = nxt
        if abs(step) < 1e-12 * r:
            break
    p = r / (r + xbar)
    rv = stats.nbinom(r, p, loc=0)  # IDENTITY: scipy n==r, p==p, 0-based failures (no shift)
    # Independent, formula-FREE LL floor: grid scan over (r, p) using scipy's own logpmf.
    r_grid = np.logspace(-1, 2, 40)
    p_grid = np.linspace(0.02, 0.98, 40)
    floor = grid_scan_loglik(data, lambda x, rr, pp: stats.nbinom.logpmf(x, rr, pp), [r_grid, p_grid])
    return {
        "fixedParams": {"r": float(r), "p": float(p)},
        "modeB": gof_block_discrete(data, rv, n_params=2, support_min=0, support_max=math.inf),
        "modeA": {
            "form": "iterative",
            "params": {"r": float(r), "p": float(p)},
            "logLik": floor,  # the engine's fitted LL must clear this independent grid floor
            "aicc": aicc_or_sentinel(float(np.sum(rv.logpmf(arr))), 2, n),
        },
    }


def build_discrete_uniform(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    a = int(np.min(arr))
    b = int(np.max(arr))
    # LOAD-BEARING: scipy.randint is half-open {low,...,high-1}; high=b+1 encodes @stdlib's inclusive b.
    rv = stats.randint(low=a, high=b + 1)
    ll = log_lik_discrete(data, rv)
    return {
        "fixedParams": {"a": float(a), "b": float(b)},
        "modeB": gof_block_discrete(data, rv, n_params=2, support_min=a, support_max=b),
        "modeA": {
            "form": "closed-form",
            "params": {"a": float(a), "b": float(b)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


# --- M2.3 Batch D builders (multi-parameter MLE via the vendored Nelder–Mead optimizer) ----


def build_student_t(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    # IDENTITY 1:1 mapping: scipy.stats.t.fit -> (df, loc, scale); HardFit {loc, scale, df} are the
    # SAME three free params (NO floc/fscale — loc and scale are genuine, free). df FIRST in scipy.
    df, loc, scale = stats.t.fit(arr)
    rv = stats.t(df, loc=loc, scale=scale)
    fixed = {"loc": float(loc), "scale": float(scale), "df": float(df)}
    # Mode A: iterative 3-D fit -> gate on log-likelihood (HardFit must reach >= scipy's; HardFit
    # may converge to a marginally BETTER optimum, so param equality is a loose diagnostic only).
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=3),
        "modeA": {
            "form": "iterative",
            "params": {"loc": float(loc), "scale": float(scale), "df": float(df)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 3, n),
        },
    }


def build_fisher_f(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    # scipy.stats.f is 4-param f(dfn, dfd, loc, scale); PIN floc=0, fscale=1 so it is the standard
    # two-parameter F. fit -> (dfn, dfd, 0, 1); map dfn -> d1, dfd -> d2 (VERIFIED).
    d1, d2, _loc, _scale = stats.f.fit(arr, floc=0, fscale=1)
    rv = stats.f(d1, d2, loc=0, scale=1)
    fixed = {"d1": float(d1), "d2": float(d2)}
    # Mode A: iterative 2-D fit -> gate on log-likelihood (HardFit must reach >= scipy's). scipy's
    # f.fit under-converges, so HardFit may reach a BETTER optimum and the params legitimately differ;
    # the parity skip-set silences the d1/d2 diagnostic while the LL cross-check stays the contract.
    ll = log_lik(data, rv)
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "iterative",
            "params": {"d1": float(d1), "d2": float(d2)},
            "logLik": ll,
            "aicc": aicc_or_sentinel(ll, 2, n),
        },
    }


def build_inverse_gaussian(data: list[float]) -> dict:
    arr = np.asarray(data, dtype=float)
    n = len(arr)
    # CLOSED-FORM MLE (emitted directly, levy-style — NOT scipy.fit's params): mu = sample mean,
    # lambda = n / (Σ(1/x) − n/mu). Emitting the closed form makes the 1e-9 param gate independent of
    # scipy.fit's convergence.
    mu = float(np.mean(arr))
    lam = float(n / (np.sum(1.0 / arr) - n / mu))
    # THE TRAP: scipy.stats.invgauss uses (mu_s = mu/lambda, loc=0, scale=lambda) — NOT mu directly.
    rv = stats.invgauss(mu / lam, loc=0, scale=lam)
    fixed = {"mu": mu, "lambda": lam}
    # Independent LL floor from scipy's OWN optimizer (preserves the formula-independence of the LL
    # gate): invgauss.fit(x, floc=0) -> (mu_s, 0, scale) ⇒ HardFit lambda = scale, mu = mu_s·scale.
    mu_s_fit, _loc_fit, scale_fit = stats.invgauss.fit(arr, floc=0)
    ll = log_lik(data, stats.invgauss(mu_s_fit, loc=0, scale=scale_fit))
    # Cross-check the closed form against scipy's independent fit (VERIFIED equal to ~1e-9): a bug in
    # either route fails this loudly at generation time, before it can reach the committed JSON.
    assert abs(mu - mu_s_fit * scale_fit) < 1e-6 * max(1.0, abs(mu)), "IG mu mismatch vs scipy.fit"
    assert abs(lam - scale_fit) < 1e-6 * max(1.0, abs(lam)), "IG lambda mismatch vs scipy.fit"
    return {
        "fixedParams": fixed,
        "modeB": gof_block(data, rv, n_params=2),
        "modeA": {
            "form": "closed-form",
            "params": {"mu": mu, "lambda": lam},
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
    # M2.3 Batch A
    "uniform": build_uniform,
    "rayleigh": build_rayleigh,
    "pareto": build_pareto,
    "laplace": build_laplace,
    "logistic": build_logistic,
    "gumbel": build_gumbel,
    "cauchy": build_cauchy,
    "frechet": build_frechet,
    # M2.3 Batch B
    "levy": build_levy,
    "chisquare": build_chisquare,
    "chi": build_chi,
    "invgamma": build_invgamma,
    "betaprime": build_betaprime,
    "cosine": build_cosine,
    "beta": build_beta,
    # M2.3 Batch C (discrete)
    "poisson": build_poisson,
    "geometric": build_geometric,
    "negative-binomial": build_negative_binomial,
    "discrete-uniform": build_discrete_uniform,
    # M2.3 Batch D (multi-parameter MLE)
    "student-t": build_student_t,
    "fisher-f": build_fisher_f,
    "inverse-gaussian": build_inverse_gaussian,
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
