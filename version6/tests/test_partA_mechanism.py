#!/usr/bin/env python3
"""Synthetic checks of the Part A building blocks: per-run events (mech_series.run_events),
the paired statistics and AUC (partA_common), the Q6 signal scores, the stats-file parser
(stats_io._read_one) and the model label of the Stage 1 mask control (load.model_label).

    .venv/bin/python -m pytest tests/test_partA_mechanism.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V6, "analysis"))

from load import model_label  # noqa: E402
from mech_series import run_events  # noqa: E402
from partA_common import auc, holm, paired  # noqa: E402
from stats_io import _read_one  # noqa: E402


def run_series(val, loss, dead2, dead1=None):
    """Series rows of one mlp2 run: epoch 0 (init, no curves) then epochs 1..len(val)."""
    n = len(val)
    e = np.arange(0, n + 1)
    dead1 = dead1 if dead1 is not None else [0.0] * n
    return pd.DataFrame({
        "epoch": e,
        "val": [np.nan] + list(val), "loss": [np.nan] + list(loss),
        "dead|hidden1|relu": [0.0] + list(dead1), "dead|hidden2|relu": [0.0] + list(dead2),
    })


def test_events_late_abrupt_collapse():
    val = [90, 95, 96, 97, 96, 50, 10, 10]
    loss = [0.5, 0.3, 0.2, 0.15, 0.2, 0.4, 2.3, 2.3]
    dead2 = [0, 0, 0, 0.05, 0.05, 0.5, 1.0, 1.0]
    ev = run_events(run_series(val, loss, dead2), "mlp2")
    assert ev["e_star"] == 7                  # first val < 15
    assert ev["e_fail"] == 6                  # 50 < 97 - 10
    assert ev["e_lossrise"] == 6              # 0.4 > 2 x 0.15
    assert ev["e_loss_min"] == 4
    assert ev["peak_before"] == 97
    assert ev["e10|hidden2|relu"] == 6 and ev["e50|hidden2|relu"] == 6 and ev["e90|hidden2|relu"] == 7
    assert ev["dmax|hidden2|relu"] == 1.0 and ev["d0|hidden2|relu"] == 0.0
    assert np.isnan(ev["dfinal|hidden2|relu"])     # dfinal is read at epoch 40 only
    assert np.isnan(ev["e10|hidden1|relu"])


def test_events_no_collapse():
    val = [90, 95, 96, 97]
    ev = run_events(run_series(val, [0.5, 0.3, 0.2, 0.1], [0, 0, 0, 0]), "mlp2")
    assert np.isnan(ev["e_star"]) and np.isnan(ev["e_fail"]) and np.isnan(ev["e_lossrise"])
    assert np.isnan(ev["peak_before"])


def test_events_collapse_at_epoch_one_has_no_peak():
    ev = run_events(run_series([10, 10, 11], [2.3, 2.3, 2.3], [0.9, 0.9, 0.9]), "mlp2")
    assert ev["e_star"] == 1 and np.isnan(ev["peak_before"])


def test_paired_known_values():
    gen = np.random.default_rng(1)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 1.0, 2.0, 5.0])
    r = paired(a, b, gen, n_boot=2000)
    assert r["n"] == 4 and r["wins"] == 3 and r["losses"] == 1
    assert r["mean_diff"] == pytest.approx(0.5)
    assert r["ci_lo"] <= 0.5 <= r["ci_hi"]
    # NaN pairs are dropped
    r2 = paired([1.0, np.nan, 3.0], [0.0, 1.0, 1.0], np.random.default_rng(1), n_boot=100)
    assert r2["n"] == 2 and r2["mean_diff"] == pytest.approx(1.5)


def test_paired_is_deterministic():
    a, b = np.arange(10.0), np.arange(10.0)[::-1]
    r1 = paired(a, b, np.random.default_rng(20261018))
    r2 = paired(a, b, np.random.default_rng(20261018))
    assert r1 == r2


def test_holm():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)      # monotone: max(0.04 x 1, 0.06)
    assert np.isnan(adj[3])


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert np.isnan(auc([], [1]))


def test_signal_scores_skip_initialisation_jump():
    from mechanism import signal_scores
    g = pd.DataFrame({"epoch": [0, 1, 2, 3, 4], "loss": [np.nan, 1.0, 0.8, 2.0, 2.0],
                      "zmax|hidden1": [1.0, 10.0, 11.0, 30.0, 31.0], "zmax|hidden2": [1.0, 5.0, 5.0, 5.0, 5.0],
                      "dead|hidden2|relu": [0.0, 0.0, 0.1, 0.5, 0.6],
                      "dw|hidden1": [np.nan, 1.0, 1.0, 1.0, 10.0]})
    sc, first = signal_scores(g, "mlp2")
    assert first["zmax_jump"] == 3             # the x10 growth in epoch 1 is not a signal
    assert first["loss_rise"] == 3             # 2.0 > 2 x 0.8
    assert first["dead_gt20"] == 3
    assert first["dw_spike"] == 4


def test_read_one(tmp_path):
    p = tmp_path / "abc123.csv"
    p.write_text("run_id,epoch,layer,group,metric,value\n"
                 "abc123,0,hidden1,relu,n_units,32\n"
                 "abc123,0,hidden1,relu,dead_units,3\n"
                 "abc123,1,hidden1,relu,dead_units,4\n"
                 "abc123,1,output,all,z_hist,1;2;3\n")
    out = _read_one((str(p), {"dead_units", "n_units"}))
    assert set(out) == {"dead_units", "n_units"}
    d = out["dead_units"]
    assert list(d.epoch) == [0, 1] and list(d.value) == [3.0, 4.0] and set(d.run_id) == {"abc123"}


def test_model_label_mask_offset():
    base = dict(arch="cnn", conv_act1="tanh", conv_act2="relu", conv_ratio="0.5",
                hid_act1="tanh", hid_act2="relu", hid_ratio="0.5")
    assert model_label({**base, "conv_mask_offset": 0}) == "tanh+relu"
    assert model_label({**base, "conv_mask_offset": 1}) == "tanh+relu/offset1"
    mlp = dict(arch="mlp2", conv_act1="none", conv_act2="none", conv_ratio="0", hid_act1="relu",
               hid_act2="relu", hid_ratio="0", conv_mask_offset=0)
    assert model_label(mlp) == "relu"
