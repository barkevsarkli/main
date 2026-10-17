#!/usr/bin/env python3
"""Tables of the PLAN.md outcome measures for one stage.

    python3 analysis/summarise.py results/stage1      -> results/stage1/summary/

Writes
  cells.csv    one row per (dataset, config, hidden, model, lr): n, final/best validation
               accuracy, peak-to-final drop, counts of final<15, drop>10 pp, ever<15,
               median first collapse epoch, final training loss - ln 10, checkpoint and
               final test accuracy / macro-F1, between-input logit std and dead-unit
               fractions at the last diagnostics epoch, LR relative to the model's own
               best LR (highest mean best-validation accuracy; ties -> lower LR).
  paired.csv   paired-by-seed differences (mixed model - reference) for final validation
               accuracy and checkpoint test accuracy: mean, 95% paired bootstrap CI
               (10,000 resamples, seed 20261018), wins, paired t-test p and Holm-adjusted
               p within each family = (dataset, metric, comparison).
  summary.md   the same, readable.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import final_stat, load_rows, load_stats  # noqa: E402

BOOT_N, BOOT_SEED = 10000, 20261018
COMPARISONS = [("tanh+relu", "relu"), ("tanh+relu", "tanh"), ("tanh+relu@conv", "relu"),
               ("tanh+relu@hidden", "relu"), ("tanh+relu@0.25", "relu"), ("tanh+relu@0.75", "relu"),
               ("leaky_relu", "relu")]
KEYS = ["dataset", "config", "hidden", "model", "lr"]


def paired_boot(d, rng):
    idx = rng.integers(0, len(d), size=(BOOT_N, len(d)))
    m = d[idx].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def per_run_diagnostics(stage_dir, df):
    st = load_stats(stage_dir, df.run_id, metrics=["logit_between_std", "dead_units", "n_units"])
    out = pd.DataFrame(index=df.run_id)
    if st.empty:
        return out
    out["logit_between_std"] = final_stat(st, "logit_between_std", "output")
    for layer in ("conv", "hidden1", "hidden2"):
        dead = final_stat(st, "dead_units", layer, "relu")
        n = final_stat(st, "n_units", layer, "relu")
        if len(dead):
            out[f"dead_frac_{layer}"] = dead / n
    return out


def cell_table(df, diag):
    d = df.join(diag, on="run_id")
    agg = {
        "n": ("seed", "size"), "final_val": ("final_val", "mean"), "final_val_sd": ("final_val", "std"),
        "best_val": ("best_val", "mean"), "drop": ("drop", "mean"),
        "n_final_lt15": ("final_lt15", "sum"), "n_drop_gt10": ("drop_gt10", "sum"),
        "n_ever_lt15": ("ever_lt15", "sum"), "first_collapse_epoch_median": ("first_collapse_epoch", "median"),
        "loss_minus_ln10": ("loss_minus_ln10", "mean"), "test_acc_best": ("test_acc_best", "mean"),
        "test_f1_best": ("test_f1_best", "mean"), "test_acc": ("test_acc", "mean"), "test_f1": ("test_f1", "mean"),
        "n_not_ok": ("status", lambda s: int((s != "ok").sum())),
    }
    for c in diag.columns:
        agg[c] = (c, "mean")
    t = d.groupby(KEYS).agg(**agg).reset_index()
    best = (t.sort_values(["best_val", "lr"], ascending=[False, True])
             .groupby(["dataset", "config", "hidden", "model"]).head(1)
             .set_index(["dataset", "config", "hidden", "model"])["lr"].rename("own_best_lr"))
    t = t.join(best, on=["dataset", "config", "hidden", "model"])
    t["lr_rel"] = t["lr"] / t["own_best_lr"]
    return t


def paired_table(df):
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for (ds, cfg, H, lr), g in df.groupby(["dataset", "config", "hidden", "lr"]):
        for a, b in COMPARISONS:
            ga, gb = g[g.model == a].set_index("seed"), g[g.model == b].set_index("seed")
            seeds = sorted(set(ga.index) & set(gb.index))
            if len(seeds) < 2:
                continue
            for metric in ("final_val", "test_acc_best"):
                d = (ga.loc[seeds, metric] - gb.loc[seeds, metric]).to_numpy(dtype=float)
                lo, hi = paired_boot(d, rng)
                p = sps.ttest_rel(ga.loc[seeds, metric], gb.loc[seeds, metric]).pvalue if d.std() > 0 else 1.0
                rows.append(dict(dataset=ds, config=cfg, hidden=H, lr=lr, metric=metric, comparison=f"{a} - {b}",
                                 n=len(seeds), mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, wins=int((d > 0).sum()),
                                 p_t=float(np.nan_to_num(p, nan=1.0))))
    t = pd.DataFrame(rows)
    if not t.empty:
        t["p_holm"] = np.nan
        for _, idx in t.groupby(["dataset", "metric", "comparison"]).groups.items():
            t.loc[idx, "p_holm"] = holm(t.loc[idx, "p_t"].to_numpy())
    return t


def md_table(t):
    """Markdown table without optional dependencies."""
    cols = list(t.columns)

    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.4g}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage_dir")
    a = ap.parse_args()
    df = load_rows(a.stage_dir)
    out = os.path.join(a.stage_dir, "summary")
    os.makedirs(out, exist_ok=True)
    cells = cell_table(df, per_run_diagnostics(a.stage_dir, df))
    paired = paired_table(df)
    cells.to_csv(os.path.join(out, "cells.csv"), index=False, float_format="%.4f")
    paired.to_csv(os.path.join(out, "paired.csv"), index=False, float_format="%.5g")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(f"# Summary of {os.path.basename(os.path.normpath(a.stage_dir))}\n\n")
        f.write(f"{len(df)} runs; statuses: {df.status.value_counts().to_dict()}\n\n")
        f.write("## Cells\n\n" + md_table(cells) + "\n\n")
        if not paired.empty:
            f.write("## Paired comparisons\n\n" + md_table(paired) + "\n")
    print(f"{len(df)} runs -> {len(cells)} cells, {len(paired)} paired comparisons; wrote {out}/")


if __name__ == "__main__":
    main()
