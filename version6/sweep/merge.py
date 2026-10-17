#!/usr/bin/env python3
"""Concatenate a stage's per-run row files into one CSV with exactly one header.

    python3 sweep/merge.py results/stage1            -> results/stage1/stage1.csv
    python3 sweep/merge.py results/stage1 --stats    -> also results/stage1/stats_stage1.csv

Rows are sorted by run_id so the merged file is the same whatever order runs finished in.
All row files must share one header; a mismatch is an error, not a silent mix.
"""
import argparse
import os
import sys


def merge(src_dir, out_path, kind):
    files = sorted(f for f in os.listdir(src_dir) if f.endswith(".csv")) if os.path.isdir(src_dir) else []
    header, n = None, 0
    tmp = out_path + ".tmp"
    with open(tmp, "w", newline="") as out:
        for fn in files:
            with open(os.path.join(src_dir, fn), newline="") as f:
                h = f.readline()
                body = f.read()
            if header is None:
                header = h
                out.write(h)
            elif h != header:
                os.remove(tmp)
                sys.exit(f"{kind} header of {fn} differs from the first file's; refusing to merge")
            if body and not body.endswith("\n"):
                body += "\n"
            out.write(body)
            n += 1
    os.replace(tmp, out_path)
    print(f"{out_path}: {n} {kind} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage_dir")
    ap.add_argument("--stats", action="store_true", help="also merge the stats files")
    a = ap.parse_args()
    stage = os.path.basename(os.path.normpath(a.stage_dir))
    merge(os.path.join(a.stage_dir, "rows"), os.path.join(a.stage_dir, f"{stage}.csv"), "row")
    if a.stats:
        merge(os.path.join(a.stage_dir, "stats"), os.path.join(a.stage_dir, f"stats_{stage}.csv"), "stats")


if __name__ == "__main__":
    main()
