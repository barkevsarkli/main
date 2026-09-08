#!/usr/bin/env python3
"""Print the job lines (from stdin) whose result row is not yet in the CSV each job writes to.
A run is identified by (seed, act1, act2, ratio, layout, hidden, lr, loss)."""
import sys, csv, os, shlex

def parse(line):
    t = shlex.split(line)
    d = {t[i][2:]: t[i + 1] for i in range(len(t) - 1) if t[i].startswith("--") and not t[i + 1].startswith("--")}
    return d

done = {}
jobs = [l.strip() for l in sys.stdin if l.strip()]
for line in jobs:
    d = parse(line)
    out = d["out"]
    if out not in done:
        done[out] = set()
        if os.path.exists(out):
            with open(out) as f:
                for r in csv.DictReader(f):
                    if r["seed"] == "seed" or not r.get("test_confusion") or r["test_confusion"].count("|") != 99:
                        continue  # header repeat or truncated row (killed mid-write)
                    done[out].add((r["seed"], r["act1"], r["act2"], f'{float(r["ratio"]):g}', r["layout"],
                                   r["hidden"], f'{float(r["lr"]):g}', r["loss"]))
    k = (d["seed"], d["act1"], d["act2"], f'{float(d["ratio"]):g}', d["layout"], d["hidden"], f'{float(d["lr"]):g}', d["loss"])
    if k not in done[out]:
        print(line)
