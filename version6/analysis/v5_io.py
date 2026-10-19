"""Loader for the version5 mlp2 results used in Part A (A4 critical LR, A5 T7).

Sources (version5/results_aws/):
  main sweeps (30 seeds):   lrsweep_64.csv, lrsweep_128.csv (MNIST), c10_lrsweep_128.csv (CIFAR-10)
  width calibration (6 seeds, exploratory): calib_{12,32,64,128,256,512}.csv,
                            c10_calib_{12,32,64,128,256,512}.csv
version5 runs are deterministic, so the same (dataset, act1, act2, ratio, layout, hidden, lr,
epochs, loss, seed) in two files is the same run; it is kept once, in the first file of the
order above, and v5_duplicates() reports how many were dropped. Repeated header lines
(seed == "seed") are dropped.
"""
import functools
import os

import numpy as np
import pandas as pd

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5 = os.path.join(os.path.dirname(V6), "version5", "results_aws")
WIDTHS = (12, 32, 64, 128, 256, 512)
MAIN = ["lrsweep_64", "lrsweep_128", "c10_lrsweep_128"]
CALIB = [f"calib_{h}" for h in WIDTHS] + [f"c10_calib_{h}" for h in WIDTHS]
RUN_KEY = ["dataset", "act1", "act2", "ratio", "layout", "hidden", "lr", "epochs", "loss", "seed"]


def v5_model(act1, act2, ratio):
    return act1 if float(ratio) == 0 else f"{act1}+{act2}"


def _curve(s):
    return np.array([float(x) for x in s.split("|")]) if isinstance(s, str) and s else np.array([])


def _read(name):
    d = pd.read_csv(os.path.join(V5, f"{name}.csv"), dtype=str, keep_default_na=False)
    d = d[d["seed"] != "seed"].copy()
    d["file"] = name
    return d


@functools.lru_cache(maxsize=None)
def _load():
    raw = pd.concat([_read(f) for f in MAIN + CALIB], ignore_index=True)
    raw["_key"] = raw[RUN_KEY].astype(str).agg("|".join, axis=1)
    dup = raw[raw.duplicated("_key", keep="first")]
    d = raw.drop_duplicates("_key", keep="first").copy()
    d["hidden"] = d["hidden"].astype(int)
    d["seed"] = d["seed"].astype(int)
    d["lr"] = d["lr"].astype(float)
    v = d["epoch_val_acc"].map(_curve)
    d["final_val"] = v.map(lambda a: a[-1] if len(a) else np.nan)
    d["best_val"] = v.map(lambda a: a.max() if len(a) else np.nan)
    d["model"] = [v5_model(a, b, r) for a, b, r in zip(d.act1, d.act2, d.ratio)]
    d["source"] = "v5 " + d["file"]
    d["kind"] = np.where(d["file"].isin(MAIN), "main", "calib")
    d["seed_set"] = np.where(d["kind"] == "main", "v5 seeds 1-30 (30-seed sweep)",
                             "v5 calib seeds 101-106 (6 seeds, exploratory)")
    d["arch"] = "mlp2"
    d["arch_label"] = "mlp2"
    d["config"] = "mlp2"
    return d.reset_index(drop=True), dup.groupby(["file"]).size().to_dict()


def v5_runs():
    """One row per distinct version5 run, with model, outcome (relative rule) and outcome_abs."""
    from outcomes import classify
    d = _load()[0].copy()
    keys = ["source", "dataset", "hidden", "model"]
    d["outcome"] = classify(d, keys)
    d["outcome_abs"] = classify(d, keys, rule="absolute")
    return d


def v5_duplicates():
    """{file: number of rows dropped as duplicates of a run in an earlier file}."""
    return dict(_load()[1])
