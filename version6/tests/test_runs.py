#!/usr/bin/env python3
"""End-to-end tests that drive the trainer binary (PROMPT.md section 9, tests 7-10 and
the frozen-mode and non-finite checks). Runnable with pytest or directly:

    python3 tests/test_runs.py          (or: pytest tests/test_runs.py)

  test_determinism          same job twice -> identical rows (except `seconds`)
  test_workers              results do not depend on --workers (1 vs 4)
  test_stats_invariance     rows identical with --stats-out on and off; stats size
  test_rng_alignment        dense init hash identical across kernel modes and activation
                            configurations; epoch-1 data order identical across models
  test_frozen               conv weights/biases bit-identical before and after training
                            (frozen), changed (bank_init)
  test_nonfinite            an exploding run stops with status=nan and still writes its row
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V6, "sweep"))
import common  # noqa: E402

BIN = common.default_binary()
DATA = os.path.join(V6, "..")
KDIR = os.path.join(V6, "kernels")
TMP = tempfile.mkdtemp(prefix="v6tests_")
VOLATILE = {"seconds"}


def run(args, name):
    out = os.path.join(TMP, name + ".csv")
    cmd = [BIN, "--quiet", "--data", DATA, "--kernel-dir", KDIR, "--out", out] + args
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode in (0, 2), f"{cmd}\n{p.stderr}"
    with open(out, newline="") as f:
        return next(csv.DictReader(f)), p.stdout


def same(r1, r2, ignore=VOLATILE):
    d = [k for k in r1 if k not in ignore and r1[k] != r2.get(k)]
    return d


SMALL = ["--arch", "cnn", "--dataset", "mnist", "--hidden", "16", "--conv-k", "3", "--act1", "tanh",
         "--act2", "relu", "--ratio", "0.5", "--epochs", "1", "--lr", "0.01", "--seed", "3"]


def test_determinism():
    a, _ = run(SMALL, "det_a")
    b, _ = run(SMALL, "det_b")
    assert not same(a, b), same(a, b)


def test_stats_invariance():
    a, _ = run(SMALL + ["--epochs", "2"], "stats_off")
    sp = os.path.join(TMP, "stats.csv")
    b, _ = run(SMALL + ["--epochs", "2", "--stats-out", sp], "stats_on")
    assert not same(a, b), same(a, b)
    size = os.path.getsize(sp)
    with open(sp) as f:
        epochs = {line.split(",")[1] for line in f.readlines()[1:]}
    assert epochs == {"0", "1", "2"}, epochs
    print(f"    stats file: {size} bytes for 3 measurement points ({size / 3:.0f} B per epoch)")


def test_workers():
    jobs = [{"stage": "wk", "arch": "cnn", "dataset": "mnist", "hidden": 16, "conv_k": 3, "seed": s,
             "epochs": 1, "lr": 0.01, "conv_act1": "tanh", "conv_act2": "relu", "conv_ratio": 0.5,
             "stats_every": 0} for s in (1, 2)]
    jobs += [{"stage": "wk", "arch": "mlp2", "dataset": "mnist", "hidden": 16, "seed": s, "epochs": 1,
              "lr": 0.01, "hid_act1": "tanh", "hid_act2": "relu", "hid_ratio": 0.5, "stats_every": 0}
             for s in (1, 2)]
    jf = os.path.join(TMP, "jobs.jsonl")
    with open(jf, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    rows = {}
    for w in (1, 4):
        root = os.path.join(TMP, f"res_w{w}")
        p = subprocess.run([sys.executable, os.path.join(V6, "sweep", "run_jobs.py"), jf, "--workers", str(w),
                            "--results", root], capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        d = os.path.join(root, "wk", "rows")
        rows[w] = {fn: common.read_row(os.path.join(d, fn)) for fn in sorted(os.listdir(d))}
    assert rows[1].keys() == rows[4].keys() and len(rows[1]) == 4
    for k in rows[1]:
        assert not same(rows[1][k], rows[4][k]), (k, same(rows[1][k], rows[4][k]))
    # Resumption: a second call finds everything finished.
    p = subprocess.run([sys.executable, os.path.join(V6, "sweep", "run_jobs.py"), jf, "--workers", "2",
                        "--results", os.path.join(TMP, "res_w1"), "--dry-run"], capture_output=True, text=True)
    assert "4 finished, 0 to run" in p.stdout, p.stdout


def test_rng_alignment():
    base = ["--arch", "cnn", "--dataset", "mnist", "--hidden", "32", "--conv-k", "5", "--seed", "5",
            "--epochs", "1", "--max-steps", "1", "--debug-order", "20"]
    variants = []
    for mode, fam in (("random", None), ("bank_init", "gabor"), ("frozen", "gabor"), ("frozen", "random_frozen")):
        for acts in (["--act1", "relu"], ["--act1", "tanh", "--act2", "relu", "--ratio", "0.5"],
                     ["--conv-act1", "tanh", "--conv-act2", "relu", "--conv-ratio", "0.5"],
                     ["--act1", "tanh", "--act2", "relu", "--ratio", "0.25", "--layout", "random"],
                     ["--act1", "sigmoid"]):
            a = base + ["--kernel-mode", mode] + (["--kernel-family", fam] if fam else []) + acts
            variants.append(a)
    hashes, orders, conv_hashes = set(), set(), {}
    for i, a in enumerate(variants):
        r, out = run(a, f"align_{i}")
        hashes.add(r["init_hash_dense"])
        orders.add(next(l for l in out.splitlines() if l.startswith("ORDER")))
        conv_hashes.setdefault(r["kernel_mode"] + "/" + r["kernel_family"], set()).add(r["kernel_hash"])
    # An MLP sees the same data order too (separate stream).
    _, out = run(["--arch", "mlp1", "--hidden", "32", "--seed", "5", "--epochs", "1", "--max-steps", "1",
                  "--debug-order", "20"], "align_mlp")
    orders.add(next(l for l in out.splitlines() if l.startswith("ORDER")))
    assert len(hashes) == 1, hashes
    assert len(orders) == 1, orders
    # Conv init is shared by every activation config of a kernel mode; bank_init and frozen
    # with the same family load the same bank; different families / random differ.
    assert all(len(v) == 1 for v in conv_hashes.values()), conv_hashes
    h = {k: next(iter(v)) for k, v in conv_hashes.items()}
    assert h["bank_init/gabor"] == h["frozen/gabor"], h
    assert len({h["random/none"], h["frozen/gabor"], h["frozen/random_frozen"]}) == 3, h
    print(f"    {len(variants)} configurations share dense init {hashes.pop()}; order: {orders.pop()[:60]}...")


def read_dump(path):
    t, cur = {}, None
    with open(path) as f:
        for line in f:
            if line.startswith("tensor "):
                cur = line.split()[1]
            else:
                t[cur] = line.split()
    return t


def test_frozen():
    for mode, expect_same in (("frozen", True), ("bank_init", False)):
        di, df = os.path.join(TMP, f"{mode}_init.txt"), os.path.join(TMP, f"{mode}_final.txt")
        r, _ = run(["--arch", "cnn", "--hidden", "16", "--conv-k", "5", "--kernel-mode", mode, "--kernel-family",
                    "sobel", "--act1", "tanh", "--act2", "relu", "--ratio", "0.5", "--epochs", "1",
                    "--max-steps", "3000", "--lr", "0.01", "--dump-init", di, "--dump-final", df], f"frz_{mode}")
        a, b = read_dump(di), read_dump(df)
        for k in ("conv.W", "conv.b"):
            if expect_same or k == "conv.W":
                assert (a[k] == b[k]) == expect_same, (mode, k)
        assert a["hidden1.W"] != b["hidden1.W"]
        if mode == "frozen":
            assert int(r["n_params_trainable"]) == int(r["n_params_total"]) - (8 * 25 + 8)


def test_nonfinite():
    r, _ = run(["--arch", "mlp1", "--hidden", "16", "--act1", "relu", "--lr", "1e30", "--epochs", "3"], "nan")
    assert r["status"] == "nan" and r["status_epoch"] == "1" and r["epochs_completed"] == "0", \
        (r["status"], r["status_epoch"], r["epochs_completed"])


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"test_runs: {'PASS' if not fails else f'{fails} FAILED'}  (scratch: {TMP})")
    sys.exit(1 if fails else 0)
