#!/usr/bin/env python3
"""Pick the best learning rate per config from a calibration CSV (by mean val_acc)
and merge it into a best-lr json.
Usage: select_lr.py [--out results/best_lr.json] results/calib_12.csv [more.csv ...]

The key format carries no dataset field, so CIFAR-10 calibration must be written to its
own file (--out results/best_lr_c10.json) or it would overwrite the MNIST rates."""
import sys, csv, json, os, collections, statistics

argv = sys.argv[1:]
out_path = "results/best_lr.json"
if argv and argv[0] == "--out":
    out_path = argv[1]
    argv = argv[2:]

best = json.load(open(out_path)) if os.path.exists(out_path) else {}
edges = []
for path in argv:
    groups = collections.defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["seed"] == "seed" or row["status"] != "ok":
                continue
            k = f'{row["act1"]}/{row["act2"]}/{float(row["ratio"]):g}/{row["layout"]}/{row["hidden"]}/{row["loss"]}'
            groups[(k, float(row["lr"]))].append(float(row["val_acc"]))
    per_cfg = collections.defaultdict(dict)
    for (k, lr), v in groups.items():
        per_cfg[k][lr] = statistics.mean(v)
    for k, d in per_cfg.items():
        lr = max(d, key=d.get)
        best[k] = lr
        grid = sorted(d)
        edge = ""
        if len(grid) > 1 and lr in (grid[0], grid[-1]):
            # The selected rate sits on the boundary of the grid, so the real optimum may
            # lie outside it and every config could be clipped to the same edge value.
            # That is a clipped calibration, not a calibrated one -- extend the grid.
            edge = f"   <-- AT GRID {'LOW' if lr == grid[0] else 'HIGH'} EDGE, extend the grid"
            edges.append((k, lr, "low" if lr == grid[0] else "high"))
        print(f"{k:45s} best lr={lr:<8g}  " + "  ".join(f"{l:g}:{m:.2f}" for l, m in sorted(d.items())) + edge)
json.dump(best, open(out_path, "w"), indent=1)
print("wrote", out_path)
if edges:
    print(f"\nWARNING: {len(edges)} of {len(best)} selections sit on the edge of the tested grid.")
    for k, lr, side in edges:
        print(f"  {k}  ->  {lr:g} ({side} edge)")
    print("A boundary selection is not a calibrated learning rate: the optimum may lie outside\n"
          "the grid, and if every config clips to the same edge the comparison is confounded.\n"
          "Extend the grid past that edge and re-run this stage before using these rates.")
    sys.exit(2)          # the caller must not start an evaluation grid on these rates
