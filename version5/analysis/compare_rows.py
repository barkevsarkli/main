#!/usr/bin/env python3
"""Diff re-run rows against the stored ones, on every CSV field except wall-clock `seconds`.

  compare_rows.py <rerun.csv> <stored.csv> [more_stored.csv ...]

For each row in the re-run file, finds the row in the stored files with the same run key
(seed, act1, act2, ratio, layout, hidden, lr, loss) and compares all remaining columns as
exact strings.  This is the acceptance gate for any change that is supposed to be
numerically inert: a changed learning curve shows up in `epoch_loss` long before it shows
up in `test_acc`, and a changed data split shows up in the `test_confusion` row sums.

Exit status 0 when every re-run row matched a stored row exactly, 1 otherwise.
"""
import csv, sys

KEY = ("seed", "act1", "act2", "ratio", "layout", "hidden", "lr", "loss")
IGNORE = {"seconds"}


def norm(row):
    return tuple(f'{float(row[k]):g}' if k in ("ratio", "lr") else row[k] for k in KEY)


def load(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("seed") == "seed":
                continue
            out[norm(r)] = r
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    rerun = load(sys.argv[1])
    stored = {}
    for p in sys.argv[2:]:
        stored.update(load(p))

    fields = [c for c in next(iter(rerun.values())).keys() if c not in IGNORE]
    checked = mismatched = missing = 0

    for k, new in sorted(rerun.items()):
        old = stored.get(k)
        label = " ".join(f"{n}={v}" for n, v in zip(KEY, k))
        if old is None:
            print(f"NO STORED ROW  {label}")
            missing += 1
            continue
        diffs = [c for c in fields if new.get(c) != old.get(c)]
        checked += 1
        if diffs:
            mismatched += 1
            print(f"MISMATCH  {label}")
            for c in diffs:
                a, b = str(old.get(c)), str(new.get(c))
                if len(a) > 90:
                    a, b = a[:90] + "…", b[:90] + "…"
                print(f"    {c}\n      stored: {a}\n      rerun:  {b}")
        else:
            print(f"identical in all {len(fields)} fields  {label}")

    print(f"\n{checked} row(s) compared, {mismatched} mismatched, {missing} with no stored counterpart")
    return 1 if (mismatched or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
