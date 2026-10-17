#!/usr/bin/env python3
"""Pre-registered Stage 2 learning-rate selection (PLAN.md). Validation metrics only.

    python3 sweep/select_lr.py [results/stage2/stage2.csv] -> results/stage2/selection.json

Uses the Stage 2 sweep (seeds 1-10, full grid). Architectures are labelled by the Stage 1
selection (results/stage1/selection.json): cfg1, cfg2 (the two CNN configurations) and
mlp1.

* Selected LR, per (dataset, arch, H, model): the grid LR with the highest mean
  best-validation accuracy over seeds 1-10 (checkpoint-aligned: the test score quoted is
  the best-validation checkpoint's). Ties -> lower LR. A selection on the edge of the grid
  is flagged.
* Stress LR, per (dataset, arch, H): the smallest grid LR at which ReLU's mean final
  validation accuracy over seeds 1-10 is below 50% of its best grid mean final
  validation accuracy; if there is none, the largest grid LR. The grid LRs just below and
  just above it are recorded too (None at the grid edges).
"""
import argparse
import collections
import csv
import json
import os
import statistics
import sys

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_stage2 import model_of as base_model_of  # noqa: E402

SWEEP_SEEDS = {str(s) for s in range(1, 11)}


def model_of(r):
    m = base_model_of(r)
    if m:
        return m
    if r["arch"] == "cnn":
        conv_mixed = (r["conv_act1"], r["conv_act2"], r["conv_ratio"]) == ("tanh", "relu", "0.5")
        hid_mixed = (r["hid_act1"], r["hid_act2"], r["hid_ratio"]) == ("tanh", "relu", "0.5")
        if conv_mixed and r["hid_ratio"] == "0" and r["hid_act1"] == "relu":
            return "tanh+relu@conv"
        if hid_mixed and r["conv_ratio"] == "0" and r["conv_act1"] == "relu":
            return "tanh+relu@hidden"
    elif (r["hid_act1"], r["hid_act2"], r["hid_ratio"]) == ("tanh", "relu", "0.5"):
        return "tanh+relu"
    elif r["hid_ratio"] == "0":
        return r["hid_act1"]
    return None


def label_of(r, sel1):
    if r["arch"] == "mlp1":
        return "mlp1"
    for lab in ("cfg1", "cfg2"):
        c = sel1[r["dataset"]][lab]
        if (r["kernel_mode"], int(r["conv_k"]), r["kernel_family"]) == (c["kernel_mode"], c["conv_k"], c["kernel_family"]):
            return lab
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=os.path.join(V6, "results", "stage2", "stage2.csv"))
    ap.add_argument("--stage1", default=os.path.join(V6, "results", "stage1", "selection.json"))
    ap.add_argument("--out", default=os.path.join(V6, "results", "stage2", "selection.json"))
    a = ap.parse_args()
    sel1 = json.load(open(a.stage1))

    cells = collections.defaultdict(dict)    # (ds, label, H, model, lr) -> {seed: (best, final)}
    for r in csv.DictReader(open(a.csv, newline="")):
        if r["seed"] not in SWEEP_SEEDS:
            continue
        lab, m = label_of(r, sel1), model_of(r)
        if lab is None or m is None:
            continue
        cells[(r["dataset"], lab, int(r["hidden"]), m, float(r["lr"]))][r["seed"]] = \
            (float(r["best_val_acc"]), float(r["val_acc"]))
    incomplete = [k for k, v in cells.items() if set(v) != SWEEP_SEEDS]
    if incomplete:
        sys.exit(f"{len(incomplete)} cells lack seeds 1-10, e.g. {incomplete[:3]}")

    out = {}
    groups = sorted({k[:3] for k in cells})
    for ds, lab, H in groups:
        lrs = sorted({k[4] for k in cells if k[:3] == (ds, lab, H)})
        models = sorted({k[3] for k in cells if k[:3] == (ds, lab, H)})
        sel = {}
        for m in models:
            bv = {lr: statistics.mean(v[0] for v in cells[(ds, lab, H, m, lr)].values()) for lr in lrs}
            best = max(lrs, key=lambda lr: (round(bv[lr], 9), -lr))
            sel[m] = best
            edge = " <-- grid edge" if best in (lrs[0], lrs[-1]) else ""
            print(f"{ds:8s} {lab:5s} H={H:<4d} {m:18s} selected lr {best:<7g}" + edge)
        fv = {lr: statistics.mean(v[1] for v in cells[(ds, lab, H, "relu", lr)].values()) for lr in lrs}
        best_f = max(fv.values())
        stress = next((lr for lr in lrs if fv[lr] < 0.5 * best_f), None)
        found = stress is not None
        if not found:
            stress = lrs[-1]
        i = lrs.index(stress)
        rec = {"selected_lr": sel, "stress_lr": stress, "stress_found": found,
               "below_stress_lr": lrs[i - 1] if i > 0 else None,
               "above_stress_lr": lrs[i + 1] if i + 1 < len(lrs) else None,
               "relu_final_val_by_lr": {f"{lr:g}": fv[lr] for lr in lrs}}
        out.setdefault(ds, {}).setdefault(lab, {})[str(H)] = rec
        print(f"{ds:8s} {lab:5s} H={H:<4d} stress lr {stress:g}{'' if found else ' (none met the rule: largest grid LR)'}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
