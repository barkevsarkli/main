#!/bin/bash
# Drive the whole plan: balance the MNIST main grid to n = 20, then replicate it on CIFAR-10.
#
#   bash sweep/run_balance_and_cifar10.sh [workers] [first_stage]
#
# Every stage is resumable -- pending.py skips runs already present in the target CSV -- so
# this is safe to interrupt with ^C, safe to re-run after an OOM kill, and safe to restart
# after a reboot.  Pass a stage name as the second argument to skip straight to it, e.g.
#   bash sweep/run_balance_and_cifar10.sh 6 c10_64
#
# Stage order is deliberate: MNIST top-ups first (13 core-hours, finishes the width axis the
# article already depends on), then CIFAR-10 in ascending width.  Width 512 is roughly half
# the CIFAR-10 budget, so it runs last and can be abandoned without costing anything else.
cd "$(dirname "$0")/.." || exit 1

P=${1:-6}
FROM=${2:-}
LOG=results/sweep_v6.log
STARTED=0

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

# Run a stage unless we were told to start at a later one.
stage(){
  local name=$1; shift
  if [ -n "$FROM" ] && [ "$STARTED" -eq 0 ]; then
    if [ "$name" = "$FROM" ]; then STARTED=1; else say "SKIP $name (starting at $FROM)"; return 0; fi
  fi
  say "=== $name ==="
  "$@" || { say "STAGE $name FAILED -- stopping"; exit 1; }
}

topup(){   # topup <label> <jobs file> <chunk>
  bash sweep/run_stage.sh "$1" "$2" "$P" "$3"
}

c10(){     # c10 <width> <calib chunk> <main chunk>
  local W=$1 CHUNK_C=$2 CHUNK_M=$3
  bash sweep/run_stage.sh "c10calib$W" "sweep/jobs_c10_calib_$W.txt" "$P" "$CHUNK_C" || return 1
  # select_lr.py exits 2 when a chosen rate sits on the edge of the tested grid.  That is a
  # clipped calibration, not a calibrated one, so stop rather than spend the evaluation grid
  # on it -- widen --lrs for this width and re-run.
  set -o pipefail
  python3 sweep/select_lr.py --out results/best_lr_c10.json "results/c10_calib_$W.csv" | tee -a "$LOG"
  local rc=$?
  set +o pipefail
  [ "$rc" -eq 0 ] || { say "calibration at width $W is not usable (select_lr exit $rc)"; return 1; }
  python3 sweep/make_jobs.py --stage main --dataset cifar10 --hidden "$W" --seeds 1-20 \
      --configs core --lr-json results/best_lr_c10.json --out "results/c10_main_$W.csv" \
      > "sweep/jobs_c10_main_$W.txt" || return 1
  bash sweep/run_stage.sh "c10main$W" "sweep/jobs_c10_main_$W.txt" "$P" "$CHUNK_M"
}

say "START | workers=$P | $(sw_vers -productVersion) | $(sysctl -n hw.memsize | awk '{printf "%.0f GB", $1/1073741824}')"

# --- phase 1: balance the MNIST main grid to n = 20 -------------------------
stage mnist_256 topup main256topup sweep/jobs_main_256_topup.txt 32
stage mnist_512 topup main512topup sweep/jobs_main_512_topup.txt 16

# --- phase 3: CIFAR-10, ascending width ------------------------------------
stage c10_12  c10 12  60 50
stage c10_32  c10 32  60 50
stage c10_64  c10 64  60 50
stage c10_128 c10 128 30 25
stage c10_256 c10 256 15 20
stage c10_512 c10 512 15 10

say "ALL STAGES COMPLETE"
python3 analysis/sanity.py | tee -a "$LOG"
