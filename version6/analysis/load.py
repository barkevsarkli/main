"""Load merged stage CSVs and per-run stats files; derive the per-run outcome measures.

    from load import load_rows, load_stats
    df = load_rows("results/stage1")            # merged results/stage1/stage1.csv (or rows/)
    st = load_stats("results/stage1", df.run_id) # long stats table for those runs
    st = load_stats("results/stage1", df.run_id, stats_root="~/Desktop/v6_archives/extracted/stage1")

    python3 analysis/load.py results/stage0 [--stats-root DIR]   # overview of one stage

`stats_root` / `--stats-root` is the directory that holds `stats/` (default: the stage
directory itself), e.g. an extracted stats archive outside the repository.

Per-run outcome measures (PLAN.md, "Outcome measures"):
  final_val, best_val, drop (peak-to-final validation accuracy, pp), final_lt15,
  drop_gt10, ever_lt15 (any epoch below 15%), first_collapse_epoch (first epoch below
  15%, NaN if never), final_train_loss and loss_minus_ln10, test_acc_best / test_f1_best
  (checkpoint) and test_acc / test_f1 (final).
Labels: `model` (relu, tanh+relu, tanh+relu@conv, tanh+relu@0.25, ...) and `config`
(mlp1, mlp2, or "<mode>/<family>/K<k>" for the CNN).
"""
import glob
import math
import os

import numpy as np
import pandas as pd

LN10 = math.log(10.0)
CURVES = ["epoch_loss", "epoch_train_acc", "epoch_train_acc_clean", "epoch_val_acc", "epoch_val_f1", "epoch_val_loss"]


def model_label(r):
    def mask(a1, a2, ratio):
        return a1 if float(ratio) == 0 else f"{a1}+{a2}@{float(ratio):g}"

    hid = mask(r["hid_act1"], r["hid_act2"], r["hid_ratio"])
    if r["arch"] != "cnn":
        return hid.replace("@0.5", "")
    off = int(float(r.get("conv_mask_offset", 0) or 0))
    if off:                                  # Stage 1 mask control (DECISIONS 22)
        return model_label({**r, "conv_mask_offset": 0}) + f"/offset{off}"
    conv = mask(r["conv_act1"], r["conv_act2"], r["conv_ratio"])
    if conv == hid:
        return hid.replace("@0.5", "")
    if conv == "tanh+relu@0.5" and hid == "relu":
        return "tanh+relu@conv"
    if hid == "tanh+relu@0.5" and conv == "relu":
        return "tanh+relu@hidden"
    return f"conv:{conv}/hid:{hid}"


def config_label(r):
    if r["arch"] != "cnn":
        return r["arch"]
    fam = "" if r["kernel_family"] == "none" else f"/{r['kernel_family']}"
    return f"{r['kernel_mode']}{fam}/K{r['conv_k']}"


def _curve(s):
    if not isinstance(s, str) or not s:
        return np.array([])
    return np.array([float(x) for x in s.split("|")])


def load_rows(stage_dir_or_csv):
    p = stage_dir_or_csv
    if os.path.isdir(p):
        merged = os.path.join(p, os.path.basename(os.path.normpath(p)) + ".csv")
        if os.path.exists(merged):
            df = pd.read_csv(merged, dtype=str, keep_default_na=False)
        else:
            files = sorted(glob.glob(os.path.join(p, "rows", "*.csv")))
            df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    else:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df = df[df["run_id"] != "run_id"].copy()                    # never a repeated header
    df = df.drop_duplicates("run_id", keep="first")
    for c in ("seed", "epochs", "hidden", "conv_k", "conv_channels", "pool", "conv_mask_offset",
              "epochs_completed", "best_val_epoch", "status_epoch"):
        df[c] = df[c].astype(int)
    for c in ("lr", "conv_ratio", "hid_ratio", "seconds", "train_acc", "val_acc", "val_f1", "test_acc", "test_f1",
              "best_val_acc", "test_acc_best", "test_f1_best", "train_acc_clean"):
        df[c] = df[c].astype(float)
    for c in CURVES:
        df[c] = df[c].map(_curve)
    df["model"] = df.apply(model_label, axis=1)
    df["config"] = df.apply(config_label, axis=1)
    v = df["epoch_val_acc"]
    df["final_val"] = v.map(lambda a: a[-1] if len(a) else np.nan)
    df["best_val"] = v.map(lambda a: a.max() if len(a) else np.nan)
    df["drop"] = df["best_val"] - df["final_val"]
    df["final_lt15"] = df["final_val"] < 15
    df["drop_gt10"] = df["drop"] > 10
    df["ever_lt15"] = v.map(lambda a: bool(len(a)) and bool((a < 15).any()))
    df["first_collapse_epoch"] = v.map(lambda a: float(np.argmax(a < 15) + 1) if len(a) and (a < 15).any() else np.nan)
    df["final_train_loss"] = df["epoch_loss"].map(lambda a: a[-1] if len(a) else np.nan)
    df["loss_minus_ln10"] = df["final_train_loss"] - LN10
    return df.reset_index(drop=True)


def load_stats(stage_dir, run_ids=None, metrics=None, stats_root=None):
    """Long table run_id, epoch, layer, group, metric, value (histograms kept as strings).

    stats_root: directory holding `stats/` (default: stage_dir)."""
    root = os.path.expanduser(stats_root) if stats_root else stage_dir
    files = sorted(glob.glob(os.path.join(root, "stats", "*.csv")))
    if run_ids is not None:
        want = set(run_ids)
        files = [f for f in files if os.path.basename(f)[:-4] in want]
    parts = []
    for f in files:
        t = pd.read_csv(f, dtype={"run_id": str, "value": str})
        if metrics is not None:
            t = t[t["metric"].isin(metrics)]
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["run_id", "epoch", "layer", "group", "metric", "value"])
    st = pd.concat(parts, ignore_index=True)
    num = st["metric"] != "z_hist"
    st["num"] = np.nan
    st.loc[num, "num"] = st.loc[num, "value"].astype(float)
    return st


def final_stat(st, metric, layer, group=None, how="last"):
    """Per-run value of one stats metric at the last recorded epoch."""
    t = st[(st["metric"] == metric) & (st["layer"] == layer)]
    if group is not None:
        t = t[t["group"] == group]
    t = t.sort_values("epoch")
    return t.groupby("run_id")["num"].last() if how == "last" else t.groupby("run_id")["num"].agg(how)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Overview of one stage: runs, models and stats files found.")
    ap.add_argument("stage_dir")
    ap.add_argument("--stats-root", default=None, help="directory holding stats/ (default: stage_dir)")
    a = ap.parse_args()
    df = load_rows(a.stage_dir)
    root = os.path.expanduser(a.stats_root) if a.stats_root else a.stage_dir
    have = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(root, "stats", "*.csv"))}
    print(f"{len(df)} runs; statuses {df.status.value_counts().to_dict()}")
    print(f"models: {sorted(df.model.unique())}")
    print(f"configs: {sorted(df.config.unique())}")
    print(f"stats files in {root}/stats: {len(have & set(df.run_id))} of {len(df)} runs")


if __name__ == "__main__":
    main()
