#!/usr/bin/env python3
"""Pre-registered Stage 1 -> Stage 2 selection (PLAN.md). Validation metrics only.

    python3 sweep/select_stage2.py [results/stage1/stage1.csv] -> results/stage1/selection.json

Per dataset, over the main Stage 1 runs (H = 128, conv_mask_offset 0; the mask-control
and mlp1 runs are not used here):

1. Configuration 1 (standard CNN): the `random`-mode K whose ReLU model has the highest
   mean best-validation accuracy (screening seeds 101-103) at its best grid LR. Ties -> smaller K.
2. Configuration 2 (most informative bank): among the frozen and bank_init configurations
   whose ReLU mean best-validation accuracy (at its best grid LR) is within 2 pp of
   configuration 1's, the one with the largest mean (tanh+relu - relu) difference in
   *final* validation accuracy at that configuration's Stage 1 stress LR.
   Stress LR = the smallest grid LR at which ReLU's mean final validation accuracy is
   below 50% of its best grid mean final validation accuracy; if there is none, the
   largest grid LR. Ties -> smaller K, then family order gauss, sharpen, sobel,
   laplacian, gabor, random_frozen, then mode order frozen, bank_init.
3. Datasets are selected independently.

Every cell must have its full seed set; otherwise the script stops (a selection on
incomplete data is not the pre-registered rule).
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAM_ORDER = ["none", "gauss", "sharpen", "sobel", "laplacian", "gabor", "random_frozen"]
MODE_ORDER = ["random", "frozen", "bank_init"]
DEFAULT_SEEDS = "101,102,103"   # Stage 1 screening seeds (sweep/make_jobs.py STAGE1_SEEDS)


def model_of(r):
    if r["hid_ratio"] == "0" and r["conv_ratio"] == "0":
        return r["hid_act1"] if r["hid_act1"] == r["conv_act1"] else None
    if (r["conv_act1"], r["conv_act2"], r["conv_ratio"], r["hid_act1"], r["hid_act2"], r["hid_ratio"]) == \
            ("tanh", "relu", "0.5", "tanh", "relu", "0.5"):
        return "tanh+relu"
    return None


def stress_lr(final_by_lr):
    """final_by_lr: {lr: mean final val} for ReLU. Smallest LR below 50% of the best mean."""
    grid = sorted(final_by_lr)
    best = max(final_by_lr.values())
    for lr in grid:
        if final_by_lr[lr] < 0.5 * best:
            return lr, True
    return grid[-1], False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=os.path.join(V6, "results", "stage1", "stage1.csv"))
    ap.add_argument("--out", default=os.path.join(V6, "results", "stage1", "selection.json"))
    ap.add_argument("--seeds", default=DEFAULT_SEEDS,
                    help="comma-separated seeds every cell must have (default: the Stage 1 screening seeds)")
    a = ap.parse_args()
    SEEDS = {s.strip() for s in a.seeds.split(",") if s.strip()}

    # cells[(ds, mode, K, family, model, lr)] = {seed: (best_val, final_val)}
    cells = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.csv, newline="")):
        if r["arch"] != "cnn" or r["conv_mask_offset"] != "0" or r["hidden"] != "128":
            continue
        m = model_of(r)
        if m not in ("relu", "tanh+relu", "tanh"):
            continue
        key = (r["dataset"], r["kernel_mode"], int(r["conv_k"]), r["kernel_family"], m, float(r["lr"]))
        cells[key][r["seed"]] = (float(r["best_val_acc"]), float(r["val_acc"]), r["status"])

    missing = [k for k, v in cells.items() if set(v) != SEEDS]
    if missing:
        sys.exit(f"{len(missing)} Stage 1 cells lack seeds {sorted(SEEDS, key=int)}, e.g. {missing[:3]}; finish Stage 1 first")

    out = {}
    for ds in sorted({k[0] for k in cells}):
        def mean(mode, K, fam, model, lr, idx):
            v = cells[(ds, mode, K, fam, model, lr)]
            return statistics.mean(x[idx] for x in v.values())

        configs = sorted({(k[1], k[2], k[3]) for k in cells if k[0] == ds},
                         key=lambda c: (c[1], FAM_ORDER.index(c[2]), MODE_ORDER.index(c[0])))
        lrs = sorted({k[5] for k in cells if k[0] == ds})
        info = {}
        for mode, K, fam in configs:
            bv = {lr: mean(mode, K, fam, "relu", lr, 0) for lr in lrs}
            fv = {lr: mean(mode, K, fam, "relu", lr, 1) for lr in lrs}
            best_lr = max(lrs, key=lambda lr: (round(bv[lr], 9), -lr))   # ties -> lower LR
            s_lr, found = stress_lr(fv)
            diffs = [cells[(ds, mode, K, fam, "tanh+relu", s_lr)][s][1] - cells[(ds, mode, K, fam, "relu", s_lr)][s][1]
                     for s in sorted(SEEDS)]
            info[(mode, K, fam)] = {
                "relu_best_lr": best_lr, "relu_best_val_mean": bv[best_lr],
                "stress_lr": s_lr, "stress_found": found,
                "diff_final_val_at_stress": statistics.mean(diffs),
                "relu_final_val_by_lr": fv,
            }
        randoms = [c for c in configs if c[0] == "random"]
        cfg1 = max(randoms, key=lambda c: (round(info[c]["relu_best_val_mean"], 9), -c[1]))
        ref = info[cfg1]["relu_best_val_mean"]
        cands = [c for c in configs if c[0] != "random" and info[c]["relu_best_val_mean"] >= ref - 2.0]
        if not cands:
            sys.exit(f"{ds}: no bank configuration within 2 pp of configuration 1")
        cfg2 = max(cands, key=lambda c: (round(info[c]["diff_final_val_at_stress"], 9), -c[1],
                                         -FAM_ORDER.index(c[2]), -MODE_ORDER.index(c[0])))

        def as_cfg(c):
            return {"kernel_mode": c[0], "conv_k": c[1], "kernel_family": c[2]}

        out[ds] = {"cfg1": as_cfg(cfg1), "cfg2": as_cfg(cfg2), "cfg1_relu_best_val_mean": ref,
                   "table": [dict(as_cfg(c), **{k: v for k, v in info[c].items() if k != "relu_final_val_by_lr"},
                                  eligible=c in cands) for c in configs]}
        print(f"\n{ds}: configuration 1 = random K={cfg1[1]} (ReLU best-val {ref:.2f})")
        print(f"{ds}: configuration 2 = {cfg2[0]} {cfg2[2]} K={cfg2[1]} "
              f"(diff {info[cfg2]['diff_final_val_at_stress']:+.2f} pp at stress LR {info[cfg2]['stress_lr']:g})")
        print(f"  {'mode':9s} {'family':14s} K  reluBest  bestLR   stressLR  diff@stress  eligible")
        for c in configs:
            i = info[c]
            print(f"  {c[0]:9s} {c[2]:14s} {c[1]}  {i['relu_best_val_mean']:7.2f}  {i['relu_best_lr']:<7g}  "
                  f"{i['stress_lr']:<7g}{'' if i['stress_found'] else '*'}  {i['diff_final_val_at_stress']:+8.2f}     "
                  f"{'yes' if c in cands else ''}")
        print("  (* no grid LR met the 50% rule; the largest grid LR is used)")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
