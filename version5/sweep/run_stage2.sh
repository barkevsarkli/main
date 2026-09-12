#!/bin/bash
# version5 stage 2: the three sections the extended sweep left empty --
#   section 4  learning-rate robustness on the evaluation seeds (MNIST 64/128, CIFAR-10 128)
#   section 6  layout / ratio variants (MNIST 128)
#   section 7  the MSE + sigmoid-output replication (MNIST 128)
#
#   P=190 EPOCHS=40 bash sweep/run_stage2.sh [first_stage]
#
# Resumable exactly like run_aws.sh: pending.py skips runs already present in the target
# CSV, so this is safe to interrupt, safe to re-run after a spot interruption, and safe to
# restart after a reboot.  Results go to $V5_RESULTS (default results_aws/), beside the
# extended-sweep files, under the names analyze.py already discovers.
#
# Ordering is cheapest-first so a mistake surfaces in minutes, and the single most expensive
# stage (CIFAR-10 LR sweep, ~2/3 of the total) runs last where an interruption costs least.
cd "$(dirname "$0")/.." || exit 1

P=${P:-190}
EPOCHS=${EPOCHS:-40}
export V5_RESULTS=${V5_RESULTS:-$PWD/results_aws}
FROM=${1:-}
STARTED=0
LOG=$V5_RESULTS/stage2.log
LRS=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3
CHUNK=$((P*2))

mkdir -p "$V5_RESULTS" || exit 1
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
M(){ python3 sweep/make_jobs.py "$@"; }

stage(){   # stage <label> <jobs file>
  local name=$1
  if [ -n "$FROM" ] && [ "$STARTED" -eq 0 ]; then
    if [ "$name" = "$FROM" ]; then STARTED=1; else say "SKIP $name (starting at $FROM)"; return 0; fi
  fi
  say "=== $name ==="
  bash sweep/run_stage.sh "$name" "$2" "$P" "$CHUNK" || { say "STAGE $name FAILED -- stopping"; exit 1; }
}

say "START stage2 | P=$P | epochs=$EPOCHS | results=$V5_RESULTS"
say "cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu) | $(uname -sm)"
say "binary sha256=$(sha256sum main 2>/dev/null | cut -c1-16 || shasum -a 256 main | cut -c1-16)"

# ---------- section 4: LR robustness on the evaluation seeds ----------------
# 30 evaluation seeds per point instead of the 6 calibration seeds, so the LR-window claim
# rests on the same sample the accuracy tables do.
M --stage lrsweep --hidden 64  --seeds 1-30 --lrs "$LRS" --configs all --epochs "$EPOCHS" \
  --out "$V5_RESULTS/lrsweep_64.csv"  > sweep/jobs_lrsweep_64.txt
stage lrsweep_64  sweep/jobs_lrsweep_64.txt

M --stage lrsweep --hidden 128 --seeds 1-30 --lrs "$LRS" --configs all --epochs "$EPOCHS" \
  --out "$V5_RESULTS/lrsweep_128.csv" > sweep/jobs_lrsweep_128.txt
stage lrsweep_128 sweep/jobs_lrsweep_128.txt

# ---------- section 6: layout and ratio -------------------------------------
# Takes the calibrated LR of the interleave/0.5 variant of each pair, so layout and ratio are
# the only things that move.
M --stage layout --hidden 128 --seeds 1-30 --lr-json "$V5_RESULTS/best_lr.json" --epochs "$EPOCHS" \
  --out "$V5_RESULTS/layout_128.csv" > sweep/jobs_layout_128.txt
stage layout_128 sweep/jobs_layout_128.txt

# ---------- section 7: MSE + sigmoid output replication ---------------------
# MSE needs its own calibration: the key format carries the loss, so these rates merge into
# best_lr.json beside the CE ones without overwriting them.
M --stage calib --hidden 128 --seeds 101-106 --lrs "$LRS" --configs mse --loss mse --epochs "$EPOCHS" \
  --out "$V5_RESULTS/calib_mse_128.csv" > sweep/jobs_calib_mse_128.txt
stage calib_mse_128 sweep/jobs_calib_mse_128.txt
python3 sweep/select_lr.py --out "$V5_RESULTS/best_lr.json" "$V5_RESULTS/calib_mse_128.csv" 2>&1 | tee -a "$LOG"

M --stage main --hidden 128 --seeds 1-30 --configs mse --loss mse \
  --lr-json "$V5_RESULTS/best_lr.json" --epochs "$EPOCHS" \
  --out "$V5_RESULTS/mse_128.csv" > sweep/jobs_mse_128.txt
stage mse_128 sweep/jobs_mse_128.txt

# ---------- section 4, CIFAR-10: the cross-dataset half of the claim --------
# ~2/3 of this script's total cost.  analyze.py does not discover c10_lrsweep_*, so this file
# is read by analysis/lr_window.py rather than appearing in summary_extended.md.
M --stage lrsweep --hidden 128 --seeds 1-30 --lrs "$LRS" --configs all --dataset cifar10 --epochs "$EPOCHS" \
  --out "$V5_RESULTS/c10_lrsweep_128.csv" > sweep/jobs_c10_lrsweep_128.txt
stage c10_lrsweep_128 sweep/jobs_c10_lrsweep_128.txt

say "STAGE 2 COMPLETE"
