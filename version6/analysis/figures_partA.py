#!/usr/bin/env python3
"""A6: Part A figures -> analysis_out/partA/figures/ (PDF + 300-dpi PNG) and figures/CAPTIONS.md.

    python3 analysis/figures_partA.py

  lr_curves_<dataset>      Stage 2 final validation accuracy vs LR, panel per (arch, H), all models
  failure_map_<dataset>    Stage 2 share of runs with final < 15%, model x LR, panel per (arch, H)
  stage3_ratio             Stage 3 final validation and checkpoint test accuracy vs ReLU share rho
The mechanism figures (mech_*) come from mechanism.py and the critical-LR figures (crit_lr_*)
from critical_lr.py; this script only collects every caption into CAPTIONS.md, so it runs last.

Seeds: at each LR the Stage 2 extension seeds 11-30 (confirmatory, n = 20) are used where they
exist, else the sweep seeds 1-10 (exploratory, n = 10); filled markers = 11-30, open = 1-10.
Model colours are fixed (figures.COLORS, the validated reference palette) and markers give a
second, colour-independent encoding.
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from partA_common import COLORS, INK, MARKERS, MODEL_ORDER, MUTED, OUT, add_caption, savefig, style  # noqa: E402

ARCHS = ["cfg1", "cfg2", "mlp1"]
WIDTHS = [64, 128, 256]
DS_NAME = {"mnist": "MNIST", "cifar10": "CIFAR-10"}


def stage2_best_seeds():
    """Stage 2 runs, at each (cell, model, LR) seeds 11-30 if present, else 1-10; column `ext`."""
    from stats_io import runs
    r = runs(2)
    r["ext"] = r.seed > 10
    has = r.groupby(["dataset", "arch_label", "hidden", "model", "lr"]).ext.transform("any")
    return r[r.ext == has].copy()


def models_of(g):
    return [m for m in MODEL_ORDER if m in set(g.model)]


def legend_handles(models):
    h = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], ls="-", ms=4, lw=1.2, label=m) for m in models]
    h += [Line2D([], [], color=MUTED, marker="o", ls="none", ms=4, label="seeds 11-30 (n = 20)"),
          Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none", ms=4, label="seeds 1-10 (n = 10)")]
    return h


def lr_curves(r):
    for ds in ("mnist", "cifar10"):
        d = r[r.dataset == ds]
        fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharey=True, squeeze=False)
        used = set()
        for i, arch in enumerate(ARCHS):
            for j, H in enumerate(WIDTHS):
                ax = axes[i, j]
                g = d[(d.arch_label == arch) & (d.hidden == H)]
                for m in models_of(g):
                    q = g[g.model == m].groupby("lr").agg(mean=("final_val", "mean"), ext=("ext", "all"),
                                                          lo=("final_val", "min"), hi=("final_val", "max"))
                    used.add(m)
                    ax.plot(q.index, q["mean"], color=COLORS[m], lw=1.1)
                    for lr, row in q.iterrows():
                        ax.plot([lr], [row["mean"]], marker=MARKERS[m], ms=4, color=COLORS[m],
                                mfc=COLORS[m] if row.ext else "white", ls="none")
                ax.set_xscale("log")
                ax.set_title(f"{arch}  H = {H}", loc="left", color=INK)
                style(ax)
                if j == 0:
                    ax.set_ylabel("final validation accuracy (%)", color=MUTED)
                if i == 2:
                    ax.set_xlabel("learning rate", color=MUTED)
        fig.legend(handles=legend_handles([m for m in MODEL_ORDER if m in used]), loc="lower center",
                   ncol=5, frameon=False, fontsize=7)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        name = f"lr_curves_{ds}"
        savefig(fig, name)
        add_caption(name, f"{DS_NAME[ds]}, Stage 2: mean final validation accuracy versus learning rate per model, one "
                    "panel per architecture (cfg1 = standard CNN, cfg2 = selected kernel-bank CNN, mlp1) and width H. "
                    "Filled markers: seeds 11-30 (confirmatory set, n = 20), at the pre-registered selected, stress and "
                    "below-stress LRs; open markers: sweep seeds 1-10 (exploratory, n = 10). Mask controls "
                    "(tanh+relu@conv, @hidden) were run in cfg2 at H = 128 and 256 only.")


def failure_maps(r):
    for ds in ("mnist", "cifar10"):
        d = r[r.dataset == ds]
        fig, axes = plt.subplots(3, 3, figsize=(11, 8.5), squeeze=False)
        for i, arch in enumerate(ARCHS):
            for j, H in enumerate(WIDTHS):
                ax = axes[i, j]
                g = d[(d.arch_label == arch) & (d.hidden == H)]
                models = models_of(g)
                lrs = np.sort(g.lr.unique())
                M = np.full((len(models), len(lrs)), np.nan)
                for a, m in enumerate(models):
                    for b, lr in enumerate(lrs):
                        x = g[(g.model == m) & np.isclose(g.lr, lr)]
                        if len(x):
                            M[a, b] = (x.final_val < 15).mean()
                            k = int((x.final_val < 15).sum())
                            ax.text(b, a, f"{k}/{len(x)}", ha="center", va="center", fontsize=5.5,
                                    color="white" if M[a, b] > 0.55 else INK)
                ax.imshow(np.ma.masked_invalid(M), cmap="Blues", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(len(lrs)), [f"{v:g}" for v in lrs], rotation=45, fontsize=6)
                ax.set_yticks(range(len(models)), models, fontsize=6)
                ax.set_title(f"{arch}  H = {H}", loc="left", color=INK)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.tick_params(colors=MUTED, length=0)
        fig.tight_layout()
        name = f"failure_map_{ds}"
        savefig(fig, name)
        add_caption(name, f"{DS_NAME[ds]}, Stage 2: runs with final validation accuracy < 15% out of n, per model and "
                    "learning rate (shade = share, white = not run), one panel per architecture and width H. n = 20 "
                    "(seeds 11-30, confirmatory) at the pre-registered LRs, n = 10 (seeds 1-10, exploratory) elsewhere.")


def stage3_ratio():
    from stats_io import runs, stage2_lrs
    s3 = runs(3)
    rho = {"tanh": 0.0, "tanh+relu@0.25": 0.25, "tanh+relu": 0.5, "tanh+relu@0.75": 0.75, "relu": 1.0}
    s3["rho"] = s3.model.map(rho)
    shades = ["#9ec5f0", "#2a78d6", "#0b3a75"]   # one hue, light -> dark = low -> high LR
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True, squeeze=False)
    for j, ds in enumerate(("mnist", "cifar10")):
        s = stage2_lrs(ds, "cfg2", 128)
        g = s3[s3.dataset == ds]
        for k, lr in enumerate(np.sort(g.lr.unique())):
            role = "stress" if np.isclose(lr, s["stress"]) else ("below" if np.isclose(lr, s["below"]) else "above")
            q = g[np.isclose(g.lr, lr)].groupby("rho")
            for i, metric in enumerate(("final_val", "test_acc_best")):
                m, sd = q[metric].mean(), q[metric].std(ddof=1)
                ax = axes[i, j]
                ax.errorbar(m.index, m.values, yerr=sd.values, color=shades[k], marker="o", ms=4, lw=1.2,
                            capsize=2, label=f"LR {lr:g} ({role})")
        for i, lab in enumerate(("final validation accuracy (%)", "checkpoint test accuracy (%)")):
            ax = axes[i, j]
            style(ax)
            if j == 0:
                ax.set_ylabel(lab, color=MUTED)
            if i == 1:
                ax.set_xlabel("ReLU share rho (0 = tanh, 1 = relu)", color=MUTED)
                ax.set_xticks(sorted(rho.values()))
        axes[0, j].set_title(f"{DS_NAME[ds]}, cfg2 H = 128", loc="left", color=INK)
        axes[0, j].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    savefig(fig, "stage3_ratio")
    add_caption("stage3_ratio", "Stage 3 (seeds 31-45, n = 15 per point; pre-registered slice with fresh seeds): mean "
                "final validation accuracy (top) and test accuracy at the best-validation checkpoint (bottom), +- 1 SD, "
                "versus the share of ReLU units rho in both the conv and hidden layer, at Stage 2's below-stress, "
                "stress and above-stress LR (cfg2, H = 128).")


def captions_md():
    files = sorted(glob.glob(os.path.join(OUT, "figures", "captions", "*.txt")))
    lines = ["# Part A figure captions", "",
             "Generated by analysis/figures_partA.py from figures/captions/*.txt (written by the script that "
             "draws each figure). Every figure exists as PDF and PNG.", ""]
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f) as fh:
            lines += [f"## {name}", "", fh.read().strip(), ""]
    with open(os.path.join(OUT, "figures", "CAPTIONS.md"), "w") as fh:
        fh.write("\n".join(lines))
    return len(files)


def main():
    r = stage2_best_seeds()
    lr_curves(r)
    failure_maps(r)
    stage3_ratio()
    n = captions_md()
    print(f"figures_partA: {n} captions in CAPTIONS.md")


if __name__ == "__main__":
    main()
