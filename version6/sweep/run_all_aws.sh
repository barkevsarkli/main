#!/usr/bin/env bash
# Stages 0-3 on one machine, in the pre-registered order, stopping at the first failure.
#
#   W=192 bash sweep/run_all_aws.sh        (inside tmux, from version6/)
#
# Safe to re-run after an interruption (spot stop, reboot, Ctrl-C): run_jobs.py skips every
# finished run, merge/selection steps are deterministic and simply repeat, and the job
# files regenerated from the same selections are identical. Stats archives go to
# ~/v6_archives (outside the repository; DECISIONS 45).
set -euo pipefail
cd "$(dirname "$0")/.."
W=${W:-192}
ARCH_DIR=${ARCH_DIR:-$HOME/v6_archives}
LOG=results/run_all_aws.log
mkdir -p results "$ARCH_DIR"

say() { echo "[$(date -u '+%F %T') UTC] $*" | tee -a "$LOG"; }
step() { say "STEP $*"; "$@" 2>&1 | tee -a "$LOG"; }
run() { step python3 sweep/run_jobs.py "$1" --workers "$W"; }
archive() {   # archive <stage>: stats of one stage -> $ARCH_DIR/stats_<stage>.tar.gz
  if [ -d "results/$1/stats" ]; then
    step tar czf "$ARCH_DIR/stats_$1.tar.gz" -C "results/$1" stats
  fi
}

say "START W=$W commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) $(uname -sm) cores=$(nproc)"
grep -q 'V6_GIT_DIRTY "0"' build/generated/provenance.h || { say "build is dirty or missing: run ./build.sh test first"; exit 1; }

# Stage 0 - MLP mechanism check (240 runs)
run results/stage0/jobs.jsonl
step python3 sweep/merge.py results/stage0
archive stage0

# Stage 1 - kernel screen (4,554 runs), then the pre-registered selection
run results/stage1/jobs.jsonl
step python3 sweep/merge.py results/stage1
step python3 sweep/select_stage2.py
archive stage1

# Stage 2 - sweep on seeds 1-10, LR / stress-LR selection, extension on seeds 11-30
step python3 sweep/make_jobs.py stage2
run results/stage2/jobs.jsonl
step python3 sweep/merge.py results/stage2
step python3 sweep/select_lr.py
step python3 sweep/make_jobs.py stage2 --part ext
run results/stage2/jobs_ext.jsonl
step python3 sweep/merge.py results/stage2
archive stage2

# Stage 3 - ratio x LR on seeds 31-45
step python3 sweep/make_jobs.py stage3
run results/stage3/jobs.jsonl
step python3 sweep/merge.py results/stage3
archive stage3

say "ALL STAGES COMPLETE"
