#!/usr/bin/env python3
"""Smoke-run check (PROMPT.md section 9, test 12): every (arch, kernel mode, family, K)
on MNIST and CIFAR-10 at H = 64 with tanh+relu for 2 epochs must end with status ok and,
for learned modes, validation accuracy above chance (10%). Frozen modes are reported.

    python3 sweep/make_jobs.py smoke && python3 sweep/run_jobs.py results/smoke/jobs.jsonl
    python3 sweep/merge.py results/smoke && python3 tests/check_smoke.py
"""
import csv
import os
import sys

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(V6, "results", "smoke", "smoke.csv")
rows = list(csv.DictReader(open(path, newline="")))
bad = 0
print(f"{'dataset':8s} {'arch':5s} {'mode':9s} {'family':14s} K  status  val1   val2   s/epoch")
for r in sorted(rows, key=lambda r: (r["dataset"], r["arch"], r["kernel_mode"], r["kernel_family"], int(r["conv_k"]))):
    v = [float(x) for x in r["epoch_val_acc"].split("|")] if r["epoch_val_acc"] else []
    learned = r["kernel_mode"] != "frozen"
    ok = r["status"] == "ok" and len(v) == 2 and (v[-1] > 10.0 or not learned)
    bad += not ok
    print(f"{r['dataset']:8s} {r['arch']:5s} {r['kernel_mode']:9s} {r['kernel_family']:14s} {r['conv_k']}  "
          f"{r['status']:6s} {v[0]:6.2f} {v[-1]:6.2f}  {float(r['seconds']) / 2:6.1f}" + ("" if ok else "   <-- FAIL"))
print(f"check_smoke: {len(rows)} runs, {bad} failures -> {'PASS' if not bad else 'FAIL'}")
sys.exit(1 if bad else 0)
