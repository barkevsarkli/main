#!/usr/bin/env python3
"""Synthetic checks of the A4 critical-LR estimator (analysis/critical_lr.py): separation is
reported as a bracket or bound and never as a fitted number, a monotone logistic cell recovers
its LR50, a reversal is flagged, and the seed-bootstrap CI covers the true LR50 at about the
nominal rate.

    .venv/bin/python -m pytest tests/test_critical_lr.py
"""
import os
import sys

import numpy as np
import pytest

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V6, "analysis"))

from critical_lr import boot_ci, boot_lr50, fit_counts, ratio  # noqa: E402

LRS = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])


def weights(n_seeds, n_boot, seed):
    g = np.random.default_rng(seed)
    idx = g.integers(0, n_seeds, size=(n_boot, n_seeds))
    W = np.zeros((n_boot, n_seeds))
    np.add.at(W, (np.repeat(np.arange(n_boot), n_seeds), idx.ravel()), 1.0)
    return W


def simulate(lr50, slope, n_seeds, seed):
    """Outcome matrix (seeds x LRs) from P = logistic(slope * (log10 LR - log10 lr50))."""
    g = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-slope * (np.log10(LRS) - np.log10(lr50))))
    return (g.random((n_seeds, len(LRS))) < p).astype(float)


def test_complete_separation_is_a_bracket():
    r = fit_counts(LRS, [0, 0, 0, 10, 10, 10], [10] * 6)
    assert r["status"] == "bracket"
    assert np.isnan(r["lr50"])
    assert (r["lo"], r["hi"]) == (0.01, 0.03)


def test_quasi_separation_brackets_the_shared_lr():
    r = fit_counts(LRS, [0, 0, 0, 4, 10, 10], [10] * 6)
    assert r["status"] == "bracket" and np.isnan(r["lr50"])
    assert (r["lo"], r["hi"]) == (0.01, 0.1)


def test_one_sided_bounds():
    none = fit_counts(LRS, [0] * 6, [10] * 6)
    assert none["status"] == "above_grid" and none["lo"] == 0.3 and np.isinf(none["hi"])
    allf = fit_counts(LRS, [10] * 6, [10] * 6)
    assert allf["status"] == "below_grid" and allf["hi"] == 0.001


def test_monotone_fit_recovers_lr50():
    lr50, slope = 0.02, 2.0
    p = 1 / (1 + np.exp(-slope * (np.log10(LRS) - np.log10(lr50))))
    n = np.full(len(LRS), 10000.0)
    r = fit_counts(LRS, np.round(p * n), n)
    assert r["status"] == "fit"
    assert r["lr50"] == pytest.approx(lr50, rel=0.02)
    assert r["b"] == pytest.approx(slope, rel=0.02)


def test_reversal_is_flagged_not_fitted():
    r = fit_counts(LRS, [0, 0, 0, 30, 3, 30], [30] * 6)
    assert r["status"] == "non_monotone" and np.isnan(r["lr50"])
    assert (r["lo"], r["hi"]) == (0.01, 0.03)
    assert r["rev"][:2] == (0.03, 0.1)


def test_bootstrap_ci_coverage():
    """About 95% of conservative CIs from 30-seed cells contain the true LR50 (>= 90% required)."""
    lr50, slope, hits, reps = 0.02, 1.5, 0, 100
    W = weights(30, 400, 1)
    for rep in range(reps):
        F = simulate(lr50, slope, 30, 1000 + rep)
        k, n = F.sum(0), np.full(len(LRS), 30.0)
        if fit_counts(LRS, k, n)["status"] != "fit":
            continue
        lo, hi = boot_ci(boot_lr50(LRS, F, W))
        hits += lo <= np.log10(lr50) <= hi
    assert hits / reps >= 0.90


def test_bootstrap_deterministic_and_separated_resamples_open():
    F = np.zeros((6, len(LRS)))
    F[:, 3:] = 1
    lb = boot_lr50(LRS, F, weights(6, 50, 2))
    assert (lb[:, 0] == np.log10(0.01)).all() and (lb[:, 1] == np.log10(0.03)).all()
    F2 = simulate(0.02, 1.5, 30, 7)
    W = weights(30, 200, 3)
    assert np.array_equal(boot_lr50(LRS, F2, W), boot_lr50(LRS, F2, W))


def test_ratio_of_brackets_is_an_interval():
    a = (fit_counts(LRS, [0, 0, 0, 0, 10, 10], [10] * 6), None)   # [0.03, 0.1]
    b = (fit_counts(LRS, [0, 0, 10, 10, 10, 10], [10] * 6), None)  # [0.003, 0.01]
    r = ratio(a, b)
    assert r["status"] == "interval" and np.isnan(r["point"])
    assert r["lo"] == pytest.approx(3.0) and r["hi"] == pytest.approx(100 / 3)
    c = (fit_counts(LRS, [0] * 6, [10] * 6), None)                 # > 0.3
    assert ratio(c, b)["display"] == "> 30"
