#!/bin/bash
# Full version5 sweep: both datasets, six widths, seven configurations, 40 epochs with
# early stopping.  Written for a single large Linux host; runs anywhere.
#
#   P=190 EPOCHS=40 bash sweep/run_aws.sh [first_stage]
#
# Every stage is resumable: pending.py skips runs already present in the target CSV, so
# this is safe to interrupt, safe to re-run after a spot interruption, and safe to restart
# after a reboot.  Pass a stage name to skip ahead, e.g.
#   P=190 bash sweep/run_aws.sh c10_256
#
# Results go to $V5_RESULTS (default results_aws/), deliberately NOT results/.  The stored
# results/ tree came from a different protocol -- 10 epochs, no early stopping, and
# std::shuffle rather than the deterministic shuffle -- and mixing the two generations in
# one directory is how the previous sweep lost two widths.
cd "$(dirname "$0")/.." || exit 1

P=${P:-8}
EPOCHS=${EPOCHS:-40}
export V5_RESULTS=${V5_RESULTS:-$PWD/results_aws}
FROM=${1:-}
STARTED=0
LOG=$V5_RESULTS/sweep.log

mkdir -p "$V5_RESULTS" || exit 1

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

stage(){
  local name=$1; shift
  if [ -n "$FROM" ] && [ "$STARTED" -eq 0 ]; then
    if [ "$name" = "$FROM" ]; then STARTED=1; else say "SKIP $name (starting at $FROM)"; return 0; fi
  fi
  say "=== $name ==="
  "$@" || { say "STAGE $name FAILED -- stopping"; exit 1; }
}

# One grid for both datasets, deliberately wider than any single configuration needs, because
# select_lr.py stops the run when a selection lands on the boundary and an unattended sweep
# should not halt because one rate drifted a step.  It has to span two families at once:
# relu/tanh settle at 0.001-0.01, but sigmoid needs roughly ten times more, since its
# derivative peaks at 0.25 against ReLU's 1.  The macOS sweep never saw this -- sigmoid chose
# the top rate of its grid at widths 64 and 128 and nothing checked.
LRS=0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3

# Six calibration seeds, paired across rates.  Two seeds were what produced the two false
# results in the macOS sweep: seed-to-seed sd is ~1.0 point, so the standard error on two
# seeds is ~0.7 and six of seven candidate margins were smaller than that.  Six paired seeds
# put the standard error on a rate-to-rate difference near 0.25, well under the ~0.4 margins
# that actually need resolving, without making calibration cost more than the evaluation
# grid it feeds.
CAL_SEEDS=101-106
EVAL_SEEDS=1-30
CONFIGS=all                # all seven, including the sigmoid pair at every width

run(){   # run <label> <jobs file> <chunk>
  bash sweep/run_stage.sh "$1" "$2" "$P" "$3"
}

sweep_width(){   # sweep_width <dataset> <prefix> <lr grid> <width> <calib chunk> <main chunk>
  local DS=$1 PFX=$2 LRS=$3 W=$4 CC=$5 MC=$6
  local CAL="$V5_RESULTS/${PFX}calib_$W.csv"
  local MAIN="$V5_RESULTS/${PFX}main_$W.csv"
  local LRJSON="$V5_RESULTS/${PFX}best_lr.json"

  python3 sweep/make_jobs.py --stage calib --dataset "$DS" --hidden "$W" --seeds "$CAL_SEEDS" \
      --lrs "$LRS" --configs "$CONFIGS" --epochs "$EPOCHS" --out "$CAL" \
      > "sweep/jobs_${PFX}calib_$W.txt" || return 1
  run "${PFX}calib$W" "sweep/jobs_${PFX}calib_$W.txt" "$CC" || return 1

  # Exits 2 when a chosen rate sits on the edge of the grid, and writes nothing in that
  # case.  Widen --lrs for this width and re-run rather than proceeding.
  set -o pipefail
  python3 sweep/select_lr.py --out "$LRJSON" "$CAL" | tee -a "$LOG"
  local rc=$?
  set +o pipefail
  [ "$rc" -eq 0 ] || { say "calibration at width $W ($DS) is not usable (select_lr exit $rc)"; return 1; }

  python3 sweep/make_jobs.py --stage main --dataset "$DS" --hidden "$W" --seeds "$EVAL_SEEDS" \
      --configs "$CONFIGS" --lr-json "$LRJSON" --epochs "$EPOCHS" --out "$MAIN" \
      > "sweep/jobs_${PFX}main_$W.txt" || return 1
  run "${PFX}main$W" "sweep/jobs_${PFX}main_$W.txt" "$MC"
}

say "START | P=$P | epochs=$EPOCHS | results=$V5_RESULTS"
say "cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu) | $(uname -sm)"
say "binary sha256=$(sha256sum main 2>/dev/null | cut -c1-16 || shasum -a 256 main | cut -c1-16)"

# Ascending width on both datasets: the widest cells cost the most, so an interruption or a
# decision to stop early loses the least informative work.  MNIST first because it is ~4x
# cheaper per run and surfaces any problem in a fraction of the time.
for W in 12 32 64 128 256 512; do
  stage "mnist_$W" sweep_width mnist   ""      "$LRS" "$W" $((P*2)) $((P*2))
done
for W in 12 32 64 128 256 512; do
  stage "c10_$W"   sweep_width cifar10 "c10_" "$LRS" "$W" $((P*2)) $((P*2))
done

say "ALL STAGES COMPLETE"
python3 analysis/sanity.py | tee -a "$LOG"
