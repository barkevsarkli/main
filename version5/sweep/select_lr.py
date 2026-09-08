#!/usr/bin/env python3
"""Pick the best learning rate per config from a calibration CSV (by mean val_acc)
and merge it into results/best_lr.json.   Usage: select_lr.py results/calib_12.csv [more.csv ...]"""
import sys, csv, json, os, collections, statistics

out_path = "results/best_lr.json"
best = json.load(open(out_path)) if os.path.exists(out_path) else {}
for path in sys.argv[1:]:
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
        print(f"{k:45s} best lr={lr:<6g}  " + "  ".join(f"{l:g}:{m:.2f}" for l, m in sorted(d.items())))
json.dump(best, open(out_path, "w"), indent=1)
print("wrote", out_path)
