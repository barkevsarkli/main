#!/usr/bin/env python3
"""Figures for one stage (PNG, results/<stage>/figures/).

    python3 analysis/figures.py results/stage1

  lr_curves_<dataset>_<config>_H<h>.png   final and best validation accuracy vs LR, one line
                                          per model (mean, with the seed range shaded)
  failure_map_<dataset>_<config>_H<h>.png share of seeds with final < 15% and with a
                                          peak-to-final drop > 10 pp, model x LR
  dynamics_<dataset>_<config>_H<h>_lr<lr>.png  validation accuracy, dead-unit fraction of
                                          ReLU units and between-input logit std over epochs
  kernel_heatmap_<dataset>.png            family x K (per kernel mode): ReLU mean best
                                          validation accuracy at its best LR, and mean
                                          (tanh+relu - relu) final validation accuracy at the
                                          configuration's stress LR (Stage 1 only)

Colours are fixed per model (never cycled), from the validated reference palette.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load import load_rows, load_stats  # noqa: E402

COLORS = {"relu": "#2a78d6", "tanh+relu": "#eb6834", "tanh": "#1baf7a", "leaky_relu": "#eda100",
          "sigmoid": "#e87ba4", "tanh+relu@conv": "#008300", "tanh+relu@hidden": "#6250d6",
          "tanh+relu@0.25": "#e34948", "tanh+relu@0.75": "#52514e"}
ORDER = list(COLORS)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FAMILIES = ["gauss", "sharpen", "sobel", "laplacian", "gabor", "random_frozen"]


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def models_in(g):
    return [m for m in ORDER if m in set(g.model)] + sorted(set(g.model) - set(ORDER))


def safe(s):
    return str(s).replace("/", "-")


def lr_curves(df, out):
    for (ds, cfg, H), g in df.groupby(["dataset", "config", "hidden"]):
        if g.lr.nunique() < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, metric, title in ((axes[0], "final_val", "final validation accuracy"),
                                  (axes[1], "best_val", "best validation accuracy (checkpoint)")):
            for m in models_in(g):
                t = g[g.model == m].groupby("lr")[metric].agg(["mean", "min", "max"]).sort_index()
                c = COLORS.get(m, MUTED)
                ax.plot(t.index, t["mean"], color=c, lw=2, marker="o", ms=5, label=m)
                ax.fill_between(t.index, t["min"], t["max"], color=c, alpha=0.12, lw=0)
            ax.set_xscale("log")
            lrs = sorted(g.lr.unique())
            ax.set_xticks(lrs, [f"{x:g}" for x in lrs])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_title(title, fontsize=9, color=INK, loc="left")
            ax.set_xlabel("learning rate", fontsize=8, color=MUTED)
            ax.axhline(15, color=MUTED, lw=0.8, ls=":")
            style(ax)
        axes[0].set_ylabel("accuracy (%)", fontsize=8, color=MUTED)
        axes[1].legend(fontsize=7, frameon=False, loc="lower left")
        fig.suptitle(f"{ds}  {cfg}  H={H}  (mean, band = seed range; dotted = 15%)", fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"lr_curves_{ds}_{safe(cfg)}_H{H}.png"), dpi=150)
        plt.close(fig)


def failure_maps(df, out):
    for (ds, cfg, H), g in df.groupby(["dataset", "config", "hidden"]):
        models, lrs = models_in(g), sorted(g.lr.unique())
        fig, axes = plt.subplots(1, 2, figsize=(1.4 + 0.75 * len(lrs) * 2, 0.9 + 0.38 * len(models)))
        for ax, col, title in ((axes[0], "final_lt15", "final < 15%"), (axes[1], "drop_gt10", "drop > 10 pp")):
            M = np.full((len(models), len(lrs)), np.nan)
            N = np.zeros_like(M)
            for i, m in enumerate(models):
                for j, lr in enumerate(lrs):
                    s = g[(g.model == m) & (g.lr == lr)][col]
                    if len(s):
                        M[i, j], N[i, j] = s.mean(), len(s)
            ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(len(models)):
                for j in range(len(lrs)):
                    if not np.isnan(M[i, j]):
                        k = int(round(M[i, j] * N[i, j]))
                        ax.text(j, i, f"{k}/{int(N[i, j])}", ha="center", va="center", fontsize=7,
                                color="white" if M[i, j] > 0.55 else INK)
            ax.set_xticks(range(len(lrs)), [f"{x:g}" for x in lrs], fontsize=7, rotation=45)
            ax.set_yticks(range(len(models)), models, fontsize=7)
            ax.set_title(title, fontsize=9, loc="left", color=INK)
        fig.suptitle(f"{ds}  {cfg}  H={H}: runs failing each rule", fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"failure_map_{ds}_{safe(cfg)}_H{H}.png"), dpi=150)
        plt.close(fig)


def dynamics(df, stage_dir, out, max_figs=12):
    st = load_stats(stage_dir, df.run_id, metrics=["dead_units", "n_units", "logit_between_std"])
    if st.empty:
        return
    st = st.merge(df[["run_id", "model", "dataset", "config", "hidden", "lr", "seed"]], on="run_id")
    made = 0
    for (ds, cfg, H, lr), g in df.groupby(["dataset", "config", "hidden", "lr"]):
        if made >= max_figs:
            break
        s = st[(st.dataset == ds) & (st.config == cfg) & (st.hidden == H) & (st.lr == lr)]
        if s.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for m in models_in(g):
            c = COLORS.get(m, MUTED)
            for _, r in g[g.model == m].iterrows():
                v = r["epoch_val_acc"]
                axes[0].plot(np.arange(1, len(v) + 1), v, color=c, lw=1, alpha=0.6)
            sm = s[s.model == m]
            dead = sm[(sm.metric == "dead_units") & (sm.group == "relu") & (sm.layer != "output")]
            nun = sm[(sm.metric == "n_units") & (sm.group == "relu")]
            if not dead.empty:
                f = dead.merge(nun, on=["run_id", "epoch", "layer", "group"], suffixes=("", "_n"))
                f["frac"] = f["num"] / f["num_n"]
                t = f.groupby(["layer", "epoch"])["frac"].mean().reset_index()
                for layer, ls in (("conv", "--"), ("hidden1", "-"), ("hidden2", ":")):
                    u = t[t.layer == layer]
                    if len(u):
                        axes[1].plot(u.epoch, u.frac, color=c, lw=2, ls=ls, label=f"{m} {layer}")
            lg = sm[sm.metric == "logit_between_std"].groupby("epoch")["num"].mean()
            axes[2].plot(lg.index, lg.values, color=c, lw=2, label=m)
        axes[0].set_title("validation accuracy (each seed)", fontsize=9, loc="left")
        axes[1].set_title("dead ReLU units (fraction, mean)", fontsize=9, loc="left")
        axes[2].set_title("between-input logit std (mean)", fontsize=9, loc="left")
        axes[2].set_yscale("symlog", linthresh=1e-3)
        for ax in axes:
            ax.set_xlabel("epoch", fontsize=8, color=MUTED)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            style(ax)
        axes[1].legend(fontsize=6, frameon=False)
        axes[2].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{ds}  {cfg}  H={H}  lr={lr:g}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"dynamics_{ds}_{safe(cfg)}_H{H}_lr{lr:g}.png"), dpi=150)
        plt.close(fig)
        made += 1


def kernel_heatmaps(df, out):
    cnn = df[(df.arch == "cnn") & (df.conv_mask_offset == 0)]
    if cnn.empty or cnn.lr.nunique() < 2:
        return
    for ds, g in cnn.groupby("dataset"):
        ks = sorted(g.conv_k.unique())
        modes = [m for m in ("frozen", "bank_init") if m in set(g.kernel_mode)]
        fams = [f for f in FAMILIES if f in set(g.kernel_family)]
        if not modes or not fams:
            continue
        mats = {}
        for mode in modes:
            A = np.full((len(fams), len(ks)), np.nan)
            B = np.full_like(A, np.nan)
            for i, fam in enumerate(fams):
                for k_i, K in enumerate(ks):
                    c = g[(g.kernel_mode == mode) & (g.kernel_family == fam) & (g.conv_k == K)]
                    relu = c[c.model == "relu"]
                    if relu.empty:
                        continue
                    A[i, k_i] = relu.groupby("lr").best_val.mean().max()
                    fv = relu.groupby("lr").final_val.mean().sort_index()
                    below = fv[fv < 0.5 * fv.max()]
                    s_lr = below.index[0] if len(below) else fv.index[-1]
                    a = c[(c.model == "tanh+relu") & (c.lr == s_lr)].set_index("seed").final_val
                    b = relu[relu.lr == s_lr].set_index("seed").final_val
                    common = a.index.intersection(b.index)
                    if len(common):
                        B[i, k_i] = (a[common] - b[common]).mean()
            mats[mode] = (A, B)
        # One colour scale per row, shared by the kernel modes, so panels compare directly.
        allA = np.concatenate([m[0].ravel() for m in mats.values()])
        allB = np.concatenate([m[1].ravel() for m in mats.values()])
        a_lo, a_hi = np.nanmin(allA), np.nanmax(allA)
        if a_hi - a_lo < 1:
            a_lo, a_hi = a_hi - 1, a_hi
        b_lim = max(1.0, np.nanmax(np.abs(allB))) if np.isfinite(allB).any() else 1.0
        fig, axes = plt.subplots(2, len(modes), figsize=(3.6 * len(modes), 6), squeeze=False)
        for j, mode in enumerate(modes):
            A, B = mats[mode]
            for row, (M, title, cmap, lo, hi) in enumerate(((A, "ReLU best val (%)", "Blues", a_lo, a_hi),
                                                            (B, "tanh+relu - relu, final val at stress LR (pp)",
                                                             "RdBu_r", -b_lim, b_lim))):
                ax = axes[row, j]
                ax.imshow(M, cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
                for i in range(M.shape[0]):
                    for k_i in range(M.shape[1]):
                        if np.isfinite(M[i, k_i]):
                            t = (M[i, k_i] - lo) / (hi - lo)
                            dark = t > 0.7 if row == 0 else abs(2 * t - 1) > 0.6
                            ax.text(k_i, i, f"{M[i, k_i]:.1f}", ha="center", va="center", fontsize=7,
                                    color="white" if dark else INK)
                ax.set_xticks(range(len(ks)), [f"K={k}" for k in ks], fontsize=7)
                ax.set_yticks(range(len(fams)), fams, fontsize=7)
                ax.set_title(f"{mode}: {title}", fontsize=8, loc="left")
        fig.suptitle(f"{ds}: kernel family x K", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"kernel_heatmap_{ds}.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage_dir")
    a = ap.parse_args()
    df = load_rows(a.stage_dir)
    out = os.path.join(a.stage_dir, "figures")
    os.makedirs(out, exist_ok=True)
    lr_curves(df, out)
    failure_maps(df, out)
    dynamics(df, a.stage_dir, out)
    kernel_heatmaps(df, out)
    print(f"wrote {len(os.listdir(out))} figures to {out}/")


if __name__ == "__main__":
    main()
