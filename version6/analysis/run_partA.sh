#!/usr/bin/env bash
# Part A driver: rebuilds everything under analysis_out/partA from the run logs and stats archives.
#
#   bash analysis/run_partA.sh
#
# Order matters: later scripts read the tables and claims written by earlier ones
# (figures_partA.py collects all captions, checks_partA.py compares T2 with A2, and
# report_partA.py merges every claims/*.json and renders the two reports).
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
START=$(date +%s)

for step in outcomes mechanism critical_lr paper_tables figures_partA checks_partA report_partA; do
    t0=$(date +%s)
    echo "== $step"
    "$PY" "analysis/$step.py"
    echo "== $step done in $(( $(date +%s) - t0 )) s"
done

echo "== Part A finished in $(( $(date +%s) - START )) s"
