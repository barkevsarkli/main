#!/usr/bin/env python3
"""Shared loader for the version6 per-run diagnostics (Part A).

The stats archives stay outside the repository (DECISIONS 45). They are extracted to

    $V6_ARCHIVES/extracted/stage<N>/stats/<run_id>.csv      (default ~/Desktop/v6_archives)

and every metric is cached once per stage as a parquet file under

    $V6_ARCHIVES/extracted/cache/stage<N>_<sha12>/<metric>.parquet

where <sha12> is the first 12 hex digits of the archive's sha256, so a changed archive
never reuses an old cache. The cache holds every layer and group of a metric; filtering by
layer or group happens after loading.

    from stats_io import extract_ok, load_metrics, runs, attach
    extract_ok(2)                                   # archive extracted and complete?
    st = load_metrics(2, ["dead_units", "n_units"], layers=["hidden1"], groups=["relu"])
    st = attach(st, runs(2), ["dataset", "config", "hidden", "model", "lr", "seed"])

    python3 analysis/stats_io.py --build 0 1 2 3    # extract (if needed) and fill the cache

Epoch 0 is the initialisation. `z_hist` (histogram strings) is not cached.
"""
import argparse
import functools
import hashlib
import json
import os
import sys
import tarfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_rows  # noqa: E402  (model_label / config_label are applied there)
from outcomes import classify  # noqa: E402

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVES = os.path.expanduser(os.environ.get("V6_ARCHIVES", "~/Desktop/v6_archives"))
EXTRACTED = os.path.join(ARCHIVES, "extracted")
CACHE = os.path.join(EXTRACTED, "cache")
STAGES = (0, 1, 2, 3)
WORKERS = int(os.environ.get("V6_WORKERS", os.cpu_count() or 4))

# Every numeric metric written by src/stats.cpp (z_hist excluded).
NUMERIC_METRICS = [
    "n_units", "z_mean", "z_std", "z_maxabs", "h_mean", "h_std", "frac_neg", "dead_units",
    "nearly_dead_units", "frac_sat", "sat_units", "z_p01", "z_p50", "z_p99",
    "logit_mean", "logit_std", "logit_between_std", "n_pred_classes", "max_class_share",
    "mean_max_prob", "mean_entropy", "probe_acc", "probe_loss",
    "w_norm", "dw_rel", "grad_mean_abs", "grad_max_abs", "clip_frac",
]
COLUMNS = ["run_id", "epoch", "layer", "group", "metric", "value"]
CACHED = ["run_id", "epoch", "layer", "group", "value"]
BATCH = 1000


def stats_dir(stage):
    return os.path.join(EXTRACTED, f"stage{stage}", "stats")


def archive_path(stage):
    return os.path.join(ARCHIVES, f"stats_stage{stage}.tar.gz")


@functools.lru_cache(maxsize=None)
def archive_sha256(stage):
    h = hashlib.sha256()
    with open(archive_path(stage), "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def runs(stage):
    """Merged result rows of a stage, with `model`, `config`, the outcome measures (load.py) and
    the `outcome` class (outcomes.py, reference = the stage's own LR grid)."""
    df = load_rows(os.path.join(V6, "results", f"stage{stage}"))
    df["seed_set"] = df["seed"].map(lambda s: seed_set(stage, s))
    df["arch_label"] = df.apply(arch_label, axis=1)
    df["outcome"] = classify(df, ["dataset", "config", "hidden", "model"])    # outcomes.py taxonomy
    return df


@functools.lru_cache(maxsize=None)
def stage1_selection():
    with open(os.path.join(V6, "results", "stage1", "selection.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=None)
def stage2_selection():
    with open(os.path.join(V6, "results", "stage2", "selection.json")) as f:
        return json.load(f)


def arch_label(r):
    """mlp1 / mlp2 / cfg1 / cfg2 (the Stage 1 selections), or the CNN config label otherwise."""
    if r["arch"] != "cnn":
        return r["arch"]
    sel = stage1_selection()[r["dataset"]]
    for lab in ("cfg1", "cfg2"):
        c = sel[lab]
        if (r["kernel_mode"], int(r["conv_k"]), r["kernel_family"]) == (c["kernel_mode"], c["conv_k"], c["kernel_family"]):
            return lab
    return r["config"]


def stage2_lrs(dataset, arch_lab, hidden):
    """Pre-registered Stage 2 LRs of a cell: dict with selected (per model), stress, below, above."""
    c = stage2_selection()[dataset][arch_lab][str(hidden)]
    return {"selected": c["selected_lr"], "stress": c["stress_lr"], "below": c["below_stress_lr"],
            "above": c["above_stress_lr"], "stress_found": c["stress_found"]}


def seed_set(stage, seed):
    """Pre-registered seed-set label (PLAN.md, Seed sets)."""
    seed = int(seed)
    if stage == 0:
        return "S0 seeds 1-10 (exploratory)"
    if stage == 1:
        return "S1 seeds 101-103 (screen)"
    if stage == 2:
        return "S2 sweep seeds 1-10 (exploratory)" if seed <= 10 else "S2 ext seeds 11-30 (confirmatory)"
    return "S3 seeds 31-45"


def extract(stage):
    """Extract stats_stage<N>.tar.gz into extracted/stage<N>/, refusing unexpected paths."""
    dest = os.path.join(EXTRACTED, f"stage{stage}")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive_path(stage)) as tar:
        members = tar.getmembers()
        for m in members:
            name = m.name.rstrip("/")
            ok = name == "stats" or (name.startswith("stats/") and "/" not in name[6:]
                                     and ".." not in name and (m.isfile() or m.isdir()))
            if not ok:
                raise RuntimeError(f"unexpected archive member {m.name!r} in {archive_path(stage)}")
        tar.extractall(dest, members=members, filter="data")


def extract_ok(stage, verbose=False):
    """True if extracted/stage<N>/stats holds exactly one file per run of the merged CSV."""
    d = stats_dir(stage)
    if not os.path.isdir(d):
        return False
    files = {f[:-4] for f in os.listdir(d) if f.endswith(".csv")}
    ids = set(runs(stage).run_id)
    ok = files == ids
    if verbose:
        print(f"stage{stage}: {len(files)} stats files, {len(ids)} runs, "
              f"missing {len(ids - files)}, extra {len(files - ids)} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def _cache_dir(stage):
    return os.path.join(CACHE, f"stage{stage}_{archive_sha256(stage)[:12]}")


def _read_one(args):
    """Parse one stats file; return {metric: DataFrame(epoch, layer, group, value)}."""
    path, metrics = args
    t = pd.read_csv(path, dtype={"run_id": str, "epoch": np.int16, "layer": str, "group": str,
                                 "metric": str, "value": str})
    t = t[t["metric"].isin(metrics)]
    rid = os.path.basename(path)[:-4]
    out = {}
    for m, g in t.groupby("metric", sort=False):
        out[m] = pd.DataFrame({"run_id": rid, "epoch": g["epoch"].to_numpy(), "layer": g["layer"].to_numpy(),
                               "group": g["group"].to_numpy(), "value": g["value"].astype(float).to_numpy()})
    return out


def build_cache(stage, metrics):
    """One pass over the stage's stats files; writes one parquet per metric not yet cached."""
    if not extract_ok(stage):
        extract(stage)
        if not extract_ok(stage, verbose=True):
            raise RuntimeError(f"stage{stage}: extracted stats do not match the merged CSV")
    cdir = _cache_dir(stage)
    os.makedirs(cdir, exist_ok=True)
    todo = [m for m in metrics if not os.path.exists(os.path.join(cdir, f"{m}.parquet"))]
    if not todo:
        return
    bad = [m for m in todo if m not in NUMERIC_METRICS]
    if bad:
        raise ValueError(f"unknown or non-numeric metrics {bad}")
    files = sorted(os.path.join(stats_dir(stage), f) for f in os.listdir(stats_dir(stage)) if f.endswith(".csv"))
    # Batches bound the memory: each batch becomes one categorical part per metric.
    parts = {m: [] for m in todo}
    with Pool(WORKERS) as pool:
        for b in range(0, len(files), BATCH):
            got = {m: [] for m in todo}
            for res in pool.imap(_read_one, [(f, set(todo)) for f in files[b:b + BATCH]], chunksize=16):
                for m, t in res.items():
                    got[m].append(t)
            for m in todo:
                if got[m]:
                    t = pd.concat(got[m], ignore_index=True)
                    for c in ("run_id", "layer", "group"):
                        t[c] = t[c].astype("category")
                    parts[m].append(t)
    for m in todo:
        t = (pd.concat(parts[m], ignore_index=True) if parts[m]
             else pd.DataFrame({c: pd.Series(dtype=float if c == "value" else object) for c in CACHED}))
        for c in ("run_id", "layer", "group"):
            t[c] = t[c].astype(str).astype("category")
        tmp = os.path.join(cdir, f".{m}.parquet.tmp")
        t.to_parquet(tmp, index=False)
        os.replace(tmp, os.path.join(cdir, f"{m}.parquet"))
        parts[m] = None


def load_metrics(stage, metrics, layers=None, groups=None):
    """Tidy long table run_id, epoch, layer, group, metric, value (float) for the given metrics."""
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    build_cache(stage, metrics)
    cdir = _cache_dir(stage)
    out = []
    for m in metrics:
        t = pd.read_parquet(os.path.join(cdir, f"{m}.parquet"))
        for c in ("run_id", "layer", "group"):
            t[c] = t[c].astype(str)
        if layers is not None:
            t = t[t["layer"].isin(layers)]
        if groups is not None:
            t = t[t["group"].isin(groups)]
        t.insert(4, "metric", m)
        out.append(t)
    t = pd.concat(out, ignore_index=True)
    t["epoch"] = t["epoch"].astype(int)
    return t[COLUMNS].sort_values(["run_id", "metric", "layer", "group", "epoch"], kind="stable").reset_index(drop=True)


def wide(st, index=("run_id", "epoch", "layer", "group")):
    """Long table -> one column per metric."""
    return st.pivot_table(index=list(index), columns="metric", values="value", aggfunc="first").reset_index()


def attach(st, rows, cols):
    """Join per-run columns of the merged CSV rows (e.g. runs(stage)) onto a stats table."""
    cols = [c for c in cols if c != "run_id"]
    return st.merge(rows[["run_id"] + cols], on="run_id", how="left", validate="many_to_one")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", nargs="*", type=int, default=[], help="stages whose full metric cache to build")
    ap.add_argument("--check", nargs="*", type=int, default=[], help="stages to check against the merged CSV")
    a = ap.parse_args()
    bad = 0
    for s in a.check:
        bad += not extract_ok(s, verbose=True)
    for s in a.build:
        build_cache(s, NUMERIC_METRICS)
        print(f"stage{s}: cache {_cache_dir(s)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
