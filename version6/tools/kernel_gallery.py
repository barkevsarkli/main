#!/usr/bin/env python3
"""Draw every kernel bank as a PNG gallery: kernels/gallery_<dataset>.png.

Rows are families, column blocks are K = 3, 5, 7, 9, and each block shows the 8 filters
of that bank as the trainer uses them (L2-normalised filter x per-filter gain; for
CIFAR-10 the per-colour-channel weight, i.e. also divided by sqrt(3)). Each filter has
its own symmetric colour scale (blue negative, red positive) so that shapes are visible
whatever the gain; the gain is printed under the filter.

Usage: python3 tools/kernel_gallery.py [--kernels kernels]
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILIES = ["gauss", "sharpen", "sobel", "laplacian", "gabor", "random_frozen"]
KS = [3, 5, 7, 9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="kernels")
    a = ap.parse_args()
    for ds in ("mnist", "cifar10"):
        cin = 1 if ds == "mnist" else 3
        fig, axes = plt.subplots(len(FAMILIES), len(KS) * 8, figsize=(len(KS) * 8 * 0.62, len(FAMILIES) * 0.95))
        for r, fam in enumerate(FAMILIES):
            for kb, k in enumerate(KS):
                path = os.path.join(a.kernels, f"{ds}_{fam}_k{k}.json")
                with open(path) as f:
                    b = json.load(f)
                filt = np.array(b["filters"], dtype=np.float64).reshape(8, k, k)
                gains = np.array(b["gains"])
                for j in range(8):
                    ax = axes[r, kb * 8 + j]
                    w = filt[j] * gains[j] / math.sqrt(cin)
                    m = np.abs(w).max() or 1.0
                    ax.imshow(w, cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(f"{gains[j]:.2f}", fontsize=5, labelpad=1)
                    if j == 0 and kb == 0:
                        ax.set_ylabel(fam, fontsize=7, rotation=0, ha="right", va="center")
                    if r == 0 and j == 0:
                        ax.set_title(f"K={k}  (tau={b['tau']:.3f})", fontsize=7, loc="left")
        fig.suptitle(f"Kernel banks, {ds}: filter x gain (number = gain), per-filter colour scale", fontsize=9)
        fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.03, wspace=0.08, hspace=0.45)
        out = os.path.join(a.kernels, f"gallery_{ds}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
