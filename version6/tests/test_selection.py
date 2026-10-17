#!/usr/bin/env python3
"""Synthetic check of the pre-registered selection rules (sweep/select_stage2.py and
sweep/select_lr.py): ties, the 2 pp eligibility window, the stress-LR rule including the
"no collapse -> largest LR" fallback, and the lower-LR tie-break.

    python3 tests/test_selection.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LRS = [0.01, 0.03, 0.1]
COLS = ["arch", "dataset", "seed", "lr", "hidden", "conv_k", "kernel_mode", "kernel_family", "conv_act1",
        "conv_act2", "conv_ratio", "conv_mask_offset", "hid_act1", "hid_act2", "hid_ratio", "best_val_acc",
        "val_acc", "status"]


def row(mode, K, fam, model, lr, seed, best, final, arch="cnn", H=128):
    a1, a2, r = ("tanh", "relu", "0.5") if model == "tanh+relu" else (model, model, "0")
    return dict(arch=arch, dataset="mnist", seed=str(seed), lr=f"{lr:g}", hidden=str(H), conv_k=str(K),
                kernel_mode=mode, kernel_family=fam, conv_act1=a1, conv_act2=a2, conv_ratio=r,
                conv_mask_offset="0", hid_act1=a1, hid_act2=a2, hid_ratio=r, best_val_acc=f"{best}",
                val_acc=f"{final}", status="ok")


def stage1_rows():
    # (mode, K, family): relu best per LR, relu final per LR, tanh+relu final per LR
    spec = {
        ("random", 3, "none"): ([98, 97, 50], [97, 60, 10], [97, 90, 80]),
        ("random", 5, "none"): ([98, 96, 50], [97, 60, 10], [97, 90, 80]),      # ties K=3 -> K=3 wins
        ("frozen", 3, "sobel"): ([97, 90, 40], [97, 40, 10], [97, 90, 20]),     # stress 0.03, diff +50
        ("bank_init", 5, "gabor"): ([95.5, 90, 40], [95, 30, 10], [95, 90, 20]),  # not eligible (2 pp)
        ("frozen", 3, "gauss"): ([97, 97, 97], [97, 96, 95], [97, 96, 145]),    # no collapse: stress 0.1, diff +50
    }
    rows = []
    for (mode, K, fam), (rb, rf, tf) in spec.items():
        for i, lr in enumerate(LRS):
            for s in (1, 2, 3):
                rows.append(row(mode, K, fam, "relu", lr, s, rb[i], rf[i]))
                rows.append(row(mode, K, fam, "tanh+relu", lr, s, tf[i], tf[i]))
                rows.append(row(mode, K, fam, "tanh", lr, s, 90, 90))
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def main():
    tmp = tempfile.mkdtemp(prefix="v6sel_")
    s1 = os.path.join(tmp, "stage1.csv")
    write(s1, stage1_rows())
    sel1 = os.path.join(tmp, "sel1.json")
    p = subprocess.run([sys.executable, os.path.join(V6, "sweep", "select_stage2.py"), s1, "--out", sel1],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    j = json.load(open(sel1))["mnist"]
    assert j["cfg1"] == {"kernel_mode": "random", "conv_k": 3, "kernel_family": "none"}, j["cfg1"]
    # sobel K3 and gauss K3 tie at +50; same K -> family order puts gauss first.
    assert j["cfg2"] == {"kernel_mode": "frozen", "conv_k": 3, "kernel_family": "gauss"}, j["cfg2"]
    t = {(r["kernel_mode"], r["conv_k"], r["kernel_family"]): r for r in j["table"]}
    assert t[("frozen", 3, "sobel")]["stress_lr"] == 0.03
    assert t[("frozen", 3, "gauss")]["stress_lr"] == 0.1 and not t[("frozen", 3, "gauss")]["stress_found"]
    assert not t[("bank_init", 5, "gabor")]["eligible"]

    # Stage 2 LR selection: ties -> lower LR; stress rule on 10 seeds.
    rows = []
    for i, lr in enumerate(LRS):
        for s in range(1, 11):
            rows.append(row("random", 3, "none", "relu", lr, s, [96, 96, 50][i], [96, 30, 10][i]))
            rows.append(row("random", 3, "none", "tanh+relu", lr, s, [95, 97, 97][i], [95, 97, 97][i]))
    s2 = os.path.join(tmp, "stage2.csv")
    write(s2, rows)
    sel2 = os.path.join(tmp, "sel2.json")
    p = subprocess.run([sys.executable, os.path.join(V6, "sweep", "select_lr.py"), s2, "--stage1", sel1,
                        "--out", sel2], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr + p.stdout
    c = json.load(open(sel2))["mnist"]["cfg1"]["128"]
    assert c["selected_lr"]["relu"] == 0.01, c          # 96 at 0.01 and 0.03 -> lower
    assert c["selected_lr"]["tanh+relu"] == 0.03, c     # tie 0.03/0.1 -> lower
    assert c["stress_lr"] == 0.03 and c["below_stress_lr"] == 0.01 and c["above_stress_lr"] == 0.1, c
    print("test_selection: PASS")


if __name__ == "__main__":
    main()
