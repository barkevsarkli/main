#!/usr/bin/env python3
"""Generate the JSON-lines jobs file of a stage (PLAN.md). Generates; never runs anything.

    python3 sweep/make_jobs.py stage0            -> results/stage0/jobs.jsonl
    python3 sweep/make_jobs.py stage1            -> results/stage1/jobs.jsonl
    python3 sweep/make_jobs.py stage2 [--part sweep|ext]
    python3 sweep/make_jobs.py stage3
    python3 sweep/make_jobs.py smoke             -> results/smoke/jobs.jsonl (2 epochs, tests)

Stage 2 and 3 depend on earlier results:
  * results/stage1/selection.json  (sweep/select_stage2.py) fixes CNN configurations 1, 2;
  * results/stage2/selection.json  (sweep/select_lr.py) fixes selected and stress LRs.
Without them the jobs are written with placeholders (<CFG2_K>, <STRESS_LR>, ...), which
sweep/run_jobs.py refuses to run; counts and time estimates then use ASSUMPTION below.

All runs: --rng-mode split, 40 epochs, CE, diagnostics every epoch.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

V6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(V6, "results")

DATASETS = ("mnist", "cifar10")
FAMILIES = ("gauss", "sharpen", "sobel", "laplacian", "gabor")       # designed banks, PROMPT order
KS = (3, 5, 7, 9)
EVAL_SEEDS = list(range(1, 31))

STAGE1_LRS = {"mnist": [0.001, 0.003, 0.01, 0.03, 0.1], "cifar10": [0.0003, 0.001, 0.003, 0.01, 0.03]}
STAGE1_MID_LR = {"mnist": 0.01, "cifar10": 0.003}
STAGE2_LRS = {"mnist": [0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1],
              "cifar10": [0.0003, 0.001, 0.003, 0.01, 0.02, 0.03, 0.05]}

# Used for counts and time estimates while the selections do not exist yet.
ASSUMPTION = {
    "cfg1": {"kernel_mode": "random", "conv_k": 3, "kernel_family": "none"},
    "cfg2": {"kernel_mode": "bank_init", "conv_k": 5, "kernel_family": "gabor"},
}
PLACEHOLDER = {
    "cfg1": {"kernel_mode": "random", "conv_k": "<CFG1_K>", "kernel_family": "none"},
    "cfg2": {"kernel_mode": "<CFG2_MODE>", "conv_k": "<CFG2_K>", "kernel_family": "<CFG2_FAMILY>"},
}


def model_fields(model, arch):
    """Activation fields of a named model.

    relu, leaky_relu, tanh, sigmoid   homogeneous in every layer
    tanh+relu                         conv and hidden mixed, ratio 0.5
    tanh+relu@conv / @hidden          only that layer mixed, the other all ReLU (CNN only)
    tanh+relu@0.25 (@0.5, @0.75)      both layers mixed at that share of ReLU
    """
    if model in common.ACTS:
        f = {"hid_act1": model, "hid_act2": model, "hid_ratio": 0.0}
        if arch == "cnn":
            f.update(conv_act1=model, conv_act2=model, conv_ratio=0.0)
        return f
    pair, _, where = model.partition("@")
    a1, a2 = pair.split("+")
    mixed = {"act1": a1, "act2": a2, "ratio": 0.5}
    homo = {"act1": "relu", "act2": "relu", "ratio": 0.0}
    conv, hid = mixed, mixed
    if where == "conv":
        hid = homo
    elif where == "hidden":
        conv = homo
    elif where:
        r = float(where)
        conv = hid = {"act1": a1, "act2": a2, "ratio": r}
    if arch != "cnn" and where in ("conv", "hidden"):
        raise ValueError(f"{model} needs a conv layer")
    f = {"hid_act1": hid["act1"], "hid_act2": hid["act2"], "hid_ratio": hid["ratio"]}
    if arch == "cnn":
        f.update(conv_act1=conv["act1"], conv_act2=conv["act2"], conv_ratio=conv["ratio"])
    return f


def job(stage, arch, dataset, hidden, model, lr, seed, kcfg=None, **extra):
    j = {"stage": stage, "arch": arch, "dataset": dataset, "hidden": hidden, "lr": lr, "seed": seed,
         "epochs": 40, "loss": "ce", "rng_mode": "split", "stats_every": 1}
    if arch == "cnn":
        j.update(conv_channels=8, pool=2, **(kcfg or {}))
    j.update(model_fields(model, arch))
    j.update(extra)
    return j


# ---- stages ---------------------------------------------------------------------------

def stage0():
    jobs = []
    for H in (64, 128):
        for lr in (0.02, 0.03, 0.05):
            for m in ("relu", "leaky_relu", "tanh", "tanh+relu"):
                for s in range(1, 11):
                    jobs.append(job("stage0", "mlp2", "mnist", H, m, lr, s))
    return jobs


def stage1_kernel_configs():
    cfgs = [{"kernel_mode": "random", "conv_k": k, "kernel_family": "none"} for k in KS]
    for fam in FAMILIES:
        for k in KS:
            for mode in ("frozen", "bank_init"):
                cfgs.append({"kernel_mode": mode, "conv_k": k, "kernel_family": fam})
    cfgs += [{"kernel_mode": "frozen", "conv_k": k, "kernel_family": "random_frozen"} for k in KS]
    assert len(cfgs) == 48
    return cfgs


def stage1(seeds=(1, 2, 3), ks=KS, drop_low_lr=False):
    jobs = []
    for ds in DATASETS:
        lrs = STAGE1_LRS[ds][1:] if drop_low_lr else STAGE1_LRS[ds]
        for kc in stage1_kernel_configs():
            if kc["conv_k"] not in ks:
                continue
            for m in ("relu", "tanh", "tanh+relu"):
                for lr in lrs:
                    for s in seeds:
                        jobs.append(job("stage1", "cnn", ds, 128, m, lr, s, kc))
            if kc["kernel_mode"] == "frozen":           # mask control: complementary conv mask
                for s in seeds:
                    jobs.append(job("stage1", "cnn", ds, 128, "tanh+relu", STAGE1_MID_LR[ds], s, kc,
                                    conv_mask_offset=1))
        for m in ("relu", "tanh", "tanh+relu"):         # no-conv reference
            for lr in lrs:
                for s in seeds:
                    jobs.append(job("stage1", "mlp1", ds, 128, m, lr, s))
    return jobs


STAGE2_MODELS = ("relu", "leaky_relu", "tanh", "sigmoid", "tanh+relu")
CONTROLS = ("tanh+relu@conv", "tanh+relu@hidden")


def stage2_cells(sel1, ds, controls_h=(128, 256), drop_sigmoid_c10_256=False):
    """(arch, label, kernel cfg, H, model) cells of stage 2 for one dataset."""
    cells = []
    archs = [("cnn", "cfg1", sel1[ds]["cfg1"]), ("cnn", "cfg2", sel1[ds]["cfg2"]), ("mlp1", "mlp1", None)]
    for arch, label, kc in archs:
        for H in (64, 128, 256):
            for m in STAGE2_MODELS:
                if drop_sigmoid_c10_256 and ds == "cifar10" and H == 256 and m == "sigmoid":
                    continue
                cells.append((arch, label, kc, H, m))
            if label == "cfg2" and H in controls_h:
                for m in CONTROLS:
                    cells.append((arch, label, kc, H, m))
    return cells


def stage2_sweep(sel1, seeds=range(1, 11), **kw):
    jobs = []
    for ds in DATASETS:
        for arch, label, kc, H, m in stage2_cells(sel1, ds, **kw):
            for lr in STAGE2_LRS[ds]:
                for s in seeds:
                    jobs.append(job("stage2", arch, ds, H, m, lr, s, kc))
    return jobs


def stage2_ext(sel1, sel2, seeds=range(11, 31), **kw):
    """Seeds 11-30 at the selected LR, the stress LR and the grid LR just below it."""
    jobs = []
    for ds in DATASETS:
        for arch, label, kc, H, m in stage2_cells(sel1, ds, **kw):
            lrs = ext_lrs(sel2, ds, label, H, m)
            for lr in lrs:
                for s in seeds:
                    jobs.append(job("stage2", arch, ds, H, m, lr, s, kc))
    return jobs


def ext_lrs(sel2, ds, label, H, m):
    if sel2 is None:
        return ["<SELECTED_LR>", "<STRESS_LR>", "<BELOW_STRESS_LR>"]
    cell = sel2[ds][label][str(H)]
    out = []
    for lr in (cell["selected_lr"][m], cell["stress_lr"], cell["below_stress_lr"]):
        if lr is not None and lr not in out:
            out.append(lr)
    return out


def stage3(sel1, sel2, seeds=range(1, 16)):
    jobs = []
    for ds in DATASETS:
        if sel2 is None:
            lrs = ["<BELOW_STRESS_LR>", "<STRESS_LR>", "<ABOVE_STRESS_LR>"]
        else:
            c = sel2[ds]["cfg2"]["128"]
            lrs = [x for x in (c["below_stress_lr"], c["stress_lr"], c["above_stress_lr"]) if x is not None]
        for m in ("tanh+relu@0.25", "tanh+relu@0.5", "tanh+relu@0.75", "relu", "tanh"):
            for lr in lrs:
                for s in seeds:
                    jobs.append(job("stage3", "cnn", ds, 128, m, lr, s, sel1[ds]["cfg2"]))
    return jobs


def smoke(epochs=2):
    """Every (arch, kernel mode, family, K) on MNIST and CIFAR-10, H = 64, tanh+relu."""
    jobs = []
    for ds in DATASETS:
        lr = STAGE1_MID_LR[ds]
        for arch in ("mlp1", "mlp2"):
            jobs.append(job("smoke", arch, ds, 64, "tanh+relu", lr, 1, epochs=epochs))
        for kc in stage1_kernel_configs():
            jobs.append(job("smoke", "cnn", ds, 64, "tanh+relu", lr, 1, kc, epochs=epochs))
        # random_frozen also in bank_init mode, for completeness of (mode, family)
        for k in KS:
            jobs.append(job("smoke", "cnn", ds, 64, "tanh+relu", lr, 1,
                            {"kernel_mode": "bank_init", "conv_k": k, "kernel_family": "random_frozen"},
                            epochs=epochs))
    return jobs


# ---- helpers ----------------------------------------------------------------------------

def dedupe(jobs):
    """Drop jobs whose canonical run key repeats (stage-local)."""
    out, seen = [], set()
    for j in jobs:
        if any(isinstance(v, str) and v.startswith("<") for v in j.values()):
            out.append(j)
            continue
        k = (j["stage"], common.run_key(common.canonical(j)))
        if k not in seen:
            seen.add(k)
            out.append(j)
    return out


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def selections(assume=False):
    sel1 = load_json(os.path.join(RESULTS, "stage1", "selection.json"))
    sel2 = load_json(os.path.join(RESULTS, "stage2", "selection.json"))
    if sel1 is None:
        sel1 = {ds: (ASSUMPTION if assume else PLACEHOLDER) for ds in DATASETS}
    return sel1, sel2


def write(jobs, stage, name="jobs.jsonl"):
    d = os.path.join(RESULTS, stage)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, sort_keys=True) + "\n")
    ph = sum(1 for j in jobs if any(isinstance(v, str) and v.startswith("<") for v in j.values()))
    print(f"{p}: {len(jobs)} jobs" + (f" ({ph} with placeholders -- not runnable yet)" if ph else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("stage0", "stage1", "stage2", "stage3", "smoke"))
    ap.add_argument("--part", choices=("sweep", "ext"), default="sweep", help="stage 2 only")
    ap.add_argument("--epochs", type=int, default=2, help="smoke only")
    a = ap.parse_args()
    if a.stage == "stage0":
        write(dedupe(stage0()), "stage0")
    elif a.stage == "stage1":
        write(dedupe(stage1()), "stage1")
    elif a.stage == "stage2":
        sel1, sel2 = selections()
        if a.part == "sweep":
            write(dedupe(stage2_sweep(sel1)), "stage2")
        else:
            write(dedupe(stage2_ext(sel1, sel2)), "stage2", "jobs_ext.jsonl")
    elif a.stage == "stage3":
        sel1, sel2 = selections()
        write(dedupe(stage3(sel1, sel2)), "stage3")
    else:
        write(dedupe(smoke(a.epochs)), "smoke")


if __name__ == "__main__":
    main()
