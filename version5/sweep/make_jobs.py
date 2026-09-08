#!/usr/bin/env python3
"""Emit one ./main command per line for a sweep stage, at any hidden width.

Stages
  calib    LR grid x calibration seeds, for a config set (feeds select_lr.py)
  main     paired evaluation seeds per config, each at its calibrated LR
  lrsweep  LR grid x evaluation seeds (seed-replicated learning-rate robustness)
  layout   tanh+relu / tanh+leaky_relu at layouts {interleave, block, random} x ratio 0.5
           plus interleave at ratios {0.25, 0.75}; LR taken from interleave/0.5 of the pair

Examples
  make_jobs.py --stage calib   --hidden 64  --seeds 101,102,103 --lrs 0.001,0.003,0.01,0.03 \
               --configs all  --out results/calib_64.csv
  make_jobs.py --stage main    --hidden 64  --seeds 1-20 --configs core --dataset cifar10 \
               --lr-json results/best_lr_c10.json --out results/c10_main_64.csv
  make_jobs.py --stage main    --hidden 64  --seeds 1-20 --configs all \
               --lr-json results/best_lr.json --out results/main_64.csv
  make_jobs.py --stage lrsweep --hidden 128 --seeds 1-10 --lrs 0.001,0.003,0.01,0.03,0.1 \
               --configs core --out results/lrsweep_128.csv
  make_jobs.py --stage layout  --hidden 128 --seeds 1-10 --lr-json results/best_lr.json \
               --out results/layout_128.csv

--configs accepts a named set (all, core, mse) or a comma list of config names
(homogeneous: relu tanh leaky_relu sigmoid; hybrid: tanh+relu tanh+leaky_relu sigmoid+relu).
--seeds accepts a comma list, an a-b range, or a mix ("1-10,101").
--loss defaults to ce, --epochs to 10, --data to "..", --dataset to mnist.
CIFAR-10 runs must use their own --lr-json and --out: the best_lr key format carries no
dataset field, so sharing results/best_lr.json would overwrite the MNIST learning rates.
"""
import sys, json, argparse

HOMOGENEOUS = ("relu", "tanh", "leaky_relu", "sigmoid")

CONFIG_SETS = {
    "all":  ["relu", "tanh", "leaky_relu", "sigmoid",
             "tanh+relu", "tanh+leaky_relu", "sigmoid+relu"],
    # "all" minus the sigmoid entries -- used at 256/512 where sigmoid is not run
    "core": ["relu", "tanh", "leaky_relu", "tanh+relu", "tanh+leaky_relu"],
    "mse":  ["relu", "tanh", "tanh+relu"],
}

# The pairs whose layout/ratio variants stage "layout" explores.
LAYOUT_PAIRS = ["tanh+relu", "tanh+leaky_relu"]
LAYOUT_VARIANTS = [("interleave", 0.5), ("block", 0.5), ("random", 0.5),
                   ("interleave", 0.25), ("interleave", 0.75)]


def parse_config(name):
    """'relu' -> (relu, relu, 0.0);  'tanh+relu' -> (tanh, relu, 0.5)."""
    if "+" in name:
        a1, a2 = name.split("+", 1)
        return (a1, a2, 0.5)
    if name not in HOMOGENEOUS:
        sys.exit(f"unknown config '{name}' (expected one of {', '.join(HOMOGENEOUS)} or 'a+b')")
    return (name, name, 0.0)


def parse_configs(spec):
    if spec in CONFIG_SETS:
        names = CONFIG_SETS[spec]
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    return [parse_config(n) for n in names]


def parse_seeds(spec):
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        sys.exit("--seeds produced no seeds")
    return seeds


def parse_lrs(spec):
    if not spec:
        sys.exit("this stage needs --lrs")
    return [float(s) for s in spec.split(",") if s.strip()]


def key(act1, act2, ratio, hidden, loss, layout="interleave"):
    """The best_lr.json key; unchanged from the first-look sweep."""
    return f"{act1}/{act2}/{ratio:g}/{layout}/{hidden}/{loss}"


def cmd(act1, act2, ratio, hidden, seed, lr, loss, out, layout="interleave", epochs=10, data="..",
        dataset="mnist"):
    return (f"./main --act1 {act1} --act2 {act2} --ratio {ratio:g} --layout {layout} --hidden {hidden} "
            f"--seed {seed} --epochs {epochs} --lr {lr:g} --loss {loss} --dataset {dataset} "
            f"--quiet --data {data} --out {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("calib", "main", "lrsweep", "layout"))
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lrs", default=None)
    ap.add_argument("--configs", default="all")
    ap.add_argument("--lr-json", default="results/best_lr.json")
    ap.add_argument("--loss", default="ce", choices=("ce", "mse"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--data", default="..")
    ap.add_argument("--dataset", default="mnist", choices=("mnist", "cifar10"))
    a = ap.parse_args()

    seeds = parse_seeds(a.seeds)
    best = {}
    try:
        best = json.load(open(a.lr_json))
    except FileNotFoundError:
        pass

    def lr_for(k):
        if k not in best:
            sys.exit(f"no calibrated LR for {k} in {a.lr_json}; run the calibration stage first")
        return best[k]

    def emit(a1, a2, r, sd, lr, layout="interleave"):
        print(cmd(a1, a2, r, a.hidden, sd, lr, a.loss, a.out,
                  layout=layout, epochs=a.epochs, data=a.data, dataset=a.dataset))

    if a.stage in ("calib", "lrsweep"):
        lrs = parse_lrs(a.lrs)
        for a1, a2, r in parse_configs(a.configs):
            for lr in lrs:
                for sd in seeds:
                    emit(a1, a2, r, sd, lr)

    elif a.stage == "main":
        for a1, a2, r in parse_configs(a.configs):
            lr = lr_for(key(a1, a2, r, a.hidden, a.loss))
            for sd in seeds:
                emit(a1, a2, r, sd, lr)

    elif a.stage == "layout":
        for pair in LAYOUT_PAIRS:
            a1, a2, base_r = parse_config(pair)
            lr = lr_for(key(a1, a2, base_r, a.hidden, a.loss))   # lr of interleave/0.5
            for layout, r in LAYOUT_VARIANTS:
                for sd in seeds:
                    emit(a1, a2, r, sd, lr, layout=layout)


if __name__ == "__main__":
    main()
