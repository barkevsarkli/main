#!/usr/bin/env python3
"""Version5 regression (PROMPT.md section 9, test 2, POSIX variant).

Builds version5 *unchanged* from ../version5 with the same compiler and the same flags as
this version6 build, runs the same configurations through both binaries
(version6: --arch mlp2 --rng-mode legacy) and requires every metric column and curve to be
bit-identical (string-equal as printed; `seconds` ignored).

Configurations: MNIST H=12 relu, MNIST H=12 tanh+relu, CIFAR-10 H=12 tanh+relu, plus
MNIST H=12 tanh+relu with the random layout (exercises the legacy RNG order through the
mask shuffle); 3 epochs, seeds 1 and 2, lr 0.01.

Usage: python3 tests/regress_v5.py [--build build] [--data ..]
"""
import argparse
import csv
import os
import re
import subprocess
import sys

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5 = os.path.join(V6, "..", "version5")

COMPARE = ["status", "train_acc", "val_acc", "val_f1", "test_acc", "test_f1", "best_val_acc",
           "best_val_epoch", "epoch_loss", "epoch_train_acc", "epoch_val_acc", "epoch_val_f1",
           "test_confusion", "train_acc_clean", "test_acc_best", "test_f1_best",
           "epoch_train_acc_clean", "test_confusion_best"]

CASES = [  # dataset, act1, act2, ratio, layout
    ("mnist", "relu", "relu", 0.0, "interleave"),
    ("mnist", "tanh", "relu", 0.5, "interleave"),
    ("cifar10", "tanh", "relu", 0.5, "interleave"),
    ("mnist", "tanh", "relu", 0.5, "random"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=os.path.join(V6, "build"))
    ap.add_argument("--data", default=os.path.join(V6, ".."))
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()

    cache = open(os.path.join(a.build, "CMakeCache.txt")).read()
    cxx = re.search(r"^CMAKE_CXX_COMPILER:\w+=(.*)$", cache, re.M).group(1)
    prov = open(os.path.join(a.build, "generated", "provenance.h")).read()
    flags = re.search(r'V6_BUILD_FLAGS "(.*)"', prov).group(1).split()
    work = os.path.join(a.build, "v5ref")
    os.makedirs(work, exist_ok=True)
    v5bin = os.path.join(work, "main_v5")
    srcs = [os.path.join(V5, f) for f in ("main.cpp", "net.cpp", "mnist_data.cpp", "cifar10_data.cpp")]
    cmd = [cxx, "-std=c++17", *flags, "-o", v5bin, *srcs]
    print("building version5:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    v6bin = os.path.join(V6, "bin", "main")

    fails = 0
    for ds, a1, a2, r, layout in CASES:
        for seed in (1, 2):
            o5 = os.path.join(work, f"v5_{ds}_{a1}_{a2}_{layout}_{seed}.csv")
            o6 = os.path.join(work, f"v6_{ds}_{a1}_{a2}_{layout}_{seed}.csv")
            if os.path.exists(o5):
                os.remove(o5)            # version5 appends
            subprocess.run([v5bin, "--act1", a1, "--act2", a2, "--ratio", str(r), "--layout", layout,
                            "--hidden", "12", "--seed", str(seed), "--epochs", str(a.epochs), "--lr", "0.01",
                            "--loss", "ce", "--dataset", ds, "--data", a.data, "--out", o5, "--quiet"], check=True)
            subprocess.run([v6bin, "--arch", "mlp2", "--rng-mode", "legacy", "--hid-act1", a1, "--hid-act2", a2,
                            "--hid-ratio", str(r), "--layout", layout, "--hidden", "12", "--seed", str(seed),
                            "--epochs", str(a.epochs), "--lr", "0.01", "--loss", "ce", "--dataset", ds,
                            "--data", a.data, "--out", o6, "--quiet"], check=True)
            r5 = next(csv.DictReader(open(o5)))
            r6 = next(csv.DictReader(open(o6)))
            diff = [k for k in COMPARE if r5[k] != r6[k]]
            tag = "IDENTICAL" if not diff else "DIFFERENT: " + ", ".join(diff)
            print(f"{ds:8s} {a1}+{a2} r={r} {layout:10s} seed {seed}: {tag}  (val {r6['epoch_val_acc']})")
            fails += bool(diff)
    print(f"regress_v5: {len(CASES) * 2 - fails}/{len(CASES) * 2} bit-identical -> {'PASS' if not fails else 'FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
