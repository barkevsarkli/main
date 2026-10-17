#!/usr/bin/env python3
"""PyTorch equivalence (PROMPT.md section 9, test 5). Optional; needs torch (CPU).

Rebuilds the `cnn` in float64 PyTorch, loads the exact initial weights the trainer dumps
(--dump-init), runs the same 200 batch-size-1 SGD steps on the same images in the same
order (--debug-order), with the dL/dz clip at 10 implemented as a gradient hook on every
pre-activation tensor, and compares all weights after training against the trainer's
float64 build of the same path (--dtype float64 --dump-final).

Pass criterion: max |torch - ours| / max |ours| < 1e-5 for every tensor (both are float64,
so the observed error is expected near 1e-12).

Cases: kernel modes random, bank_init, frozen; MNIST and CIFAR-10; mixed tanh+relu masks
in both layers; K = 5 and 3; pool 2.

Usage: .venv/bin/python tests/torch_equiv.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V6, "sweep"))
import common  # noqa: E402

DATA = os.path.join(V6, "..")
STEPS = 200
torch.set_default_dtype(torch.float64)


def read_dump(path):
    t, cur = {}, None
    with open(path) as f:
        for line in f:
            if line.startswith("tensor "):
                cur = line.split()[1]
            else:
                t[cur] = np.array([float(x) for x in line.split()], dtype=np.float64)
    return t


class Images:
    """Raw bytes -> float32(b) / float32(255) -> float64, exactly like the trainer's LUT."""

    def __init__(self, ds):
        self.ds = ds
        if ds == "mnist":
            raw = np.fromfile(os.path.join(DATA, "train-images-idx3-ubyte", "train-images-idx3-ubyte"), np.uint8)
            self.x = raw[16:].reshape(-1, 1, 28, 28)
            self.y = np.fromfile(os.path.join(DATA, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte"), np.uint8)[8:]
        else:
            parts = [os.path.join(DATA, "CIFAR10", "train_images", f"data_batch_{i}.bin") for i in range(1, 6)]
            parts.append(os.path.join(DATA, "CIFAR10", "test_images", "test_batch.bin"))
            rec = np.concatenate([np.fromfile(p, np.uint8).reshape(-1, 3073) for p in parts])
            self.y = rec[:, 0]
            self.x = rec[:, 1:].reshape(-1, 3, 32, 32)

    def get(self, i):
        x = (self.x[i].astype(np.float32) / np.float32(255.0)).astype(np.float64)
        return torch.from_numpy(x)[None], int(self.y[i])


def clip_hook(z):
    if z.requires_grad:
        z.register_hook(lambda g: g.clamp(-10.0, 10.0))
    return z


def act(z, kinds):
    """Per-unit activation along dim 1; kinds is a list of 'tanh'/'relu' per unit."""
    t = torch.tensor([k == "tanh" for k in kinds])
    shape = [1, len(kinds)] + [1] * (z.dim() - 2)
    t = t.view(shape)
    return torch.where(t, torch.tanh(z), torch.relu(z))


def run_case(ds, mode, family, K, tmp):
    cin, side = (1, 28) if ds == "mnist" else (3, 32)
    H, C, P, lr = 32, 8, 2, 0.01
    di, df = os.path.join(tmp, "init.txt"), os.path.join(tmp, "final.txt")
    args = [common.default_binary(), "--arch", "cnn", "--dataset", ds, "--data", DATA, "--hidden", str(H),
            "--conv-k", str(K), "--pool", str(P), "--kernel-mode", mode, "--act1", "tanh", "--act2", "relu",
            "--ratio", "0.5", "--lr", str(lr), "--seed", "4", "--epochs", "1", "--max-steps", str(STEPS),
            "--debug-order", str(STEPS), "--dtype", "float64", "--dump-init", di, "--dump-final", df,
            "--kernel-dir", os.path.join(V6, "kernels"), "--out", os.path.join(tmp, "row.csv")]
    if family:
        args += ["--kernel-family", family]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    order = [int(v) for v in next(l for l in out.splitlines() if l.startswith("ORDER")).split()[1:]]
    w0, w1 = read_dump(di), read_dump(df)

    # interleave at ratio 0.5: odd units ReLU (act2), even units tanh (act1)
    conv_kinds = ["relu" if i % 2 else "tanh" for i in range(C)]
    hid_kinds = ["relu" if i % 2 else "tanh" for i in range(H)]
    F_in = C * (side // P) ** 2
    p = {
        "conv.W": torch.tensor(w0["conv.W"]).view(C, cin, K, K),
        "conv.b": torch.tensor(w0["conv.b"]),
        "hidden1.W": torch.tensor(w0["hidden1.W"]).view(H, F_in),
        "hidden1.b": torch.tensor(w0["hidden1.b"]),
        "output.W": torch.tensor(w0["output.W"]).view(10, H),
        "output.b": torch.tensor(w0["output.b"]),
    }
    # The trainer parses --lr into a float32 (as version5 did): 0.01 -> 0.0099999997764825820.
    lr32 = float(np.float32(lr))
    trainable = [k for k in p if not (mode == "frozen" and k.startswith("conv"))]
    for k in trainable:
        p[k].requires_grad_(True)
    imgs = Images(ds)
    for i in order:
        x, y = imgs.get(i)
        z1 = clip_hook(F.conv2d(x, p["conv.W"], p["conv.b"], padding=(K - 1) // 2))
        h1 = F.max_pool2d(act(z1, conv_kinds), P).flatten(1)
        z2 = clip_hook(h1 @ p["hidden1.W"].T + p["hidden1.b"])
        z3 = clip_hook(act(z2, hid_kinds) @ p["output.W"].T + p["output.b"])
        loss = F.cross_entropy(z3, torch.tensor([y]))
        grads = torch.autograd.grad(loss, [p[k] for k in trainable])
        with torch.no_grad():
            for k, g in zip(trainable, grads):
                p[k] -= lr32 * g
    worst = 0.0
    for k, v in p.items():
        ours = w1[k]
        theirs = v.detach().numpy().ravel()
        err = np.abs(ours - theirs).max() / max(np.abs(ours).max(), 1e-300)
        worst = max(worst, err)
        if mode == "frozen" and k.startswith("conv"):
            assert np.array_equal(ours, w0[k]), "frozen conv changed"
    return worst


def main():
    fails = 0
    cases = [("mnist", "random", None, 5), ("mnist", "bank_init", "sobel", 5), ("mnist", "frozen", "gabor", 5),
             ("cifar10", "random", None, 3), ("cifar10", "bank_init", "laplacian", 3), ("cifar10", "frozen", "gauss", 5)]
    with tempfile.TemporaryDirectory() as tmp:
        for c in cases:
            err = run_case(*c, tmp)
            ok = err < 1e-5
            fails += not ok
            print(f"{c[0]:8s} {c[1]:9s} {str(c[2]):10s} K={c[3]}: max rel weight error after {STEPS} steps "
                  f"{err:.3g} -> {'PASS' if ok else 'FAIL'}")
    print(f"torch_equiv: {'PASS' if not fails else 'FAIL'} (torch {torch.__version__})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
