#!/usr/bin/env python3
"""Step-4 sanity checks over the extended sweep CSVs.

  1. every stage file holds exactly the intended config x seed (x lr) grid
  2. no rejected rows (status != ok, or truncated) -- any are listed in full
  3. duplicate run keys are reported (analysis keeps the first occurrence)

Exit status is 0 when every check passes, 1 otherwise.  Prints a markdown report
on stdout so it can be pasted straight into REPORT_EXTENDED.md.
"""
import csv, os, sys, collections

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

ALL  = ["relu", "tanh", "leaky_relu", "sigmoid", "tanh+relu", "tanh+leaky_relu", "sigmoid+relu"]
CORE = ["relu", "tanh", "leaky_relu", "tanh+relu", "tanh+leaky_relu"]
MSE3 = ["relu", "tanh", "tanh+relu"]
LAYOUT_VARIANTS = [(c, l, r) for c in ("tanh+relu", "tanh+leaky_relu")
                   for l, r in (("interleave", 0.5), ("block", 0.5), ("random", 0.5),
                                ("interleave", 0.25), ("interleave", 0.75))]
CAL = [101, 102, 103]
CAL2 = [101, 102]
GRID4 = [0.001, 0.003, 0.01, 0.03]
GRID3 = [0.001, 0.003, 0.01]

# file -> (kind, configs, seeds, lrs or None)
EXPECT = {
    "calib_12_v2.csv":   ("calib", ALL,  CAL,             GRID4),
    "calib_32.csv":      ("calib", ALL,  CAL,             GRID4),
    "calib_64.csv":      ("calib", ALL,  CAL,             GRID4),
    "calib_128_v2.csv":  ("calib", ALL,  CAL,             GRID4),
    "calib_256.csv":     ("calib", CORE, CAL,             GRID4),
    "calib_mse_128.csv": ("calib", MSE3, CAL,             [0.003, 0.01, 0.03]),
    "main_12_v2.csv":    ("main",  ALL,  range(1, 31),    None),
    "main_32.csv":       ("main",  ALL,  range(1, 31),    None),
    "main_64.csv":       ("main",  ALL,  range(1, 21),    None),
    "main_128_v2.csv":   ("main",  ALL,  range(1, 21),    None),
    "main_256.csv":      ("main",  CORE, range(1, 21),    None),
    "main_512.csv":      ("main",  CORE, range(1, 21),    None),
    "calib_512.csv":     ("calib", CORE, [101, 102],      [0.001, 0.003, 0.01]),   # lr 0.03 dropped by design
    "mse_128.csv":       ("main",  MSE3, range(1, 11),    None),
    "lrsweep_12.csv":    ("calib", CORE, range(1, 11),    [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]),
    "lrsweep_64.csv":    ("calib", CORE, range(1, 11),    [0.001, 0.003, 0.01, 0.03, 0.1]),
    "lrsweep_128.csv":   ("calib", CORE, range(1, 11),    [0.001, 0.003, 0.01, 0.03, 0.1]),
    "layout_128.csv":    ("layout", LAYOUT_VARIANTS, range(1, 11), None),
    "layout_256.csv":    ("layout", LAYOUT_VARIANTS, range(1, 7),  None),

    # CIFAR-10 replication.  Same protocol, same core config set, n = 20 at every width.
    # Calibration is trimmed to 3 rates x 2 seeds at 256/512, as it was for MNIST at 512.
    "c10_calib_12.csv":  ("calib", CORE, CAL,          GRID4),
    "c10_calib_32.csv":  ("calib", CORE, CAL,          GRID4),
    "c10_calib_64.csv":  ("calib", CORE, CAL,          GRID4),
    "c10_calib_128.csv": ("calib", CORE, CAL,          GRID4),
    "c10_calib_256.csv": ("calib", CORE, CAL2,         GRID3),
    "c10_calib_512.csv": ("calib", CORE, CAL2,         GRID3),
    "c10_main_12.csv":   ("main",  CORE, range(1, 21), None),
    "c10_main_32.csv":   ("main",  CORE, range(1, 21), None),
    "c10_main_64.csv":   ("main",  CORE, range(1, 21), None),
    "c10_main_128.csv":  ("main",  CORE, range(1, 21), None),
    "c10_main_256.csv":  ("main",  CORE, range(1, 21), None),
    "c10_main_512.csv":  ("main",  CORE, range(1, 21), None),
}

# Stages dropped from the plan when the machine turned out to be 2.6x slower than the
# Step-0 projection; absence of these files is expected, not a failure.
# Deliberate, documented reductions — reported separately, not as failures.
NOTES = [
    ("calib_512.csv", "lr 0.03 dropped from the width-512 grid by design; the 4 runs completed at "
                      "0.03 before the trim remain in the file and are counted as extra"),
    ("lrsweep_64.csv", "optional stage, past the time budget; run last, now complete at 250/250"),
]
DROPPED = []

def cfg_name(r):
    ratio = float(r["ratio"])
    if ratio == 0: return r["act1"]
    if ratio == 1: return r["act2"]
    return f'{r["act1"]}+{r["act2"]}'

def read(path):
    ok, bad = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("seed") == "seed":
                continue
            if r.get("status") != "ok":
                bad.append((cfg_name(r), r.get("seed"), r.get("lr"), r.get("layout"), f'status={r.get("status")}'))
            elif not r.get("test_confusion") or r["test_confusion"].count("|") != 99:
                bad.append((cfg_name(r), r.get("seed"), r.get("lr"), r.get("layout"), "truncated row"))
            else:
                ok.append(r)
    return ok, bad

def main():
    problems = []
    out = ["### Run-inventory check", "",
           "| file | expected runs | ok rows | unique keys | missing | extra | duplicates | rejected |",
           "|---|---|---|---|---|---|---|---|"]
    all_bad = []
    for fname, (kind, cfgs, seeds, lrs) in EXPECT.items():
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            out.append(f"| `{fname}` | – | *file absent* | – | – | – | – | – |")
            problems.append(f"{fname}: file absent")
            continue
        ok, bad = read(path)
        all_bad += [(fname,) + b for b in bad]
        seeds = list(seeds)
        if kind == "layout":
            want = {(c, l, f"{r:g}", s) for (c, l, r) in cfgs for s in seeds}
            have = collections.Counter((cfg_name(r), r["layout"], f'{float(r["ratio"]):g}', int(r["seed"])) for r in ok)
        elif lrs is None:
            want = {(c, s) for c in cfgs for s in seeds}
            have = collections.Counter((cfg_name(r), int(r["seed"])) for r in ok)
        else:
            want = {(c, f"{l:g}", s) for c in cfgs for l in lrs for s in seeds}
            have = collections.Counter((cfg_name(r), f'{float(r["lr"]):g}', int(r["seed"])) for r in ok)
        missing = want - set(have)
        extra = set(have) - want
        if fname == "calib_512.csv":
            # The 4 runs completed at lr 0.03 before that rate was dropped from the grid.
            # They are kept in the file (nothing in results/ is deleted) and are expected.
            extra = {e for e in extra if e[1] != "0.03"}
        dups = {k: v for k, v in have.items() if v > 1}
        out.append(f"| `{fname}` | {len(want)} | {len(ok)} | {len(have)} | {len(missing)} | {len(extra)} "
                   f"| {len(dups)} | {len(bad)} |")
        if missing: problems.append(f"{fname}: {len(missing)} missing, e.g. {sorted(missing)[:4]}")
        if extra:   problems.append(f"{fname}: {len(extra)} unexpected, e.g. {sorted(extra)[:4]}")
        if dups:    problems.append(f"{fname}: {len(dups)} duplicated keys, e.g. {list(dups)[:4]}")
        if bad:     problems.append(f"{fname}: {len(bad)} rejected rows")
    out.append("")

    out += ["### Rejected rows (status != ok, or truncated)", ""]
    if all_bad:
        out += ["| file | config | seed | lr | layout | reason |", "|---|---|---|---|---|---|"]
        for f, cfg, seed, lr, lay, why in all_bad:
            out.append(f"| `{f}` | {cfg} | {seed} | {lr} | {lay} | {why} |")
    else:
        out.append("None. Every row in every extended-sweep CSV has `status=ok` and a complete 100-entry confusion matrix.")
    out.append("")

    out += ["### Deliberate reductions (documented, not failures)", ""]
    for f, why in NOTES:
        out.append(f"- `{f}` — {why}")
    out.append("")

    print("\n".join(out))
    if problems:
        print("### Problems\n")
        for p in problems:
            print(f"- {p}")
        return 1
    print("### Problems\n\nNone — every stage file holds exactly its intended grid.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
