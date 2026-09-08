#!/bin/bash
# Restored stages, reordered on request: both width-256 stages, then width 512, then the
# width-64 LR sweep last.  Width 512 runs ungated (explicitly requested); only the width-64
# LR robustness sweep, which was never in the original priority list, is budget-gated.
cd "$(dirname "$0")/.." || exit 1
LOG=results/sweep_extended.log
P=${P:-4}
SWEEP_START=${SWEEP_START:-1788521537}
BUDGET_H=${BUDGET_H:-14}
M(){ python3 sweep/make_jobs.py "$@"; }
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
elapsed_h(){ python3 -c "print(f'{($(date +%s)-$SWEEP_START)/3600:.2f}')"; }
stage(){ bash sweep/run_stage.sh "$1" "$2" "$P" "$3" || { echo "[$(ts)] stage $1 FAILED, stopping" | tee -a $LOG; exit 1; }; }
optional(){
  local left; left=$(python3 -c "print(f'{$BUDGET_H-$(elapsed_h):.2f}')")
  if python3 -c "import sys; sys.exit(0 if $left > $4 else 1)"; then
    echo "[$(ts)] budget: ${left}h left, $1 needs ~$4h -> running" | tee -a $LOG; stage "$1" "$2" "$3"
  else
    echo "[$(ts)] budget: ${left}h left, $1 needs ~$4h -> SKIPPED (budget)" | tee -a $LOG
  fi
}

echo "[$(ts)] ===== RESTORED STAGES (reordered: 256 -> 512 -> lrsweep64) | elapsed $(elapsed_h)h =====" | tee -a $LOG

# ---- width 256 ----------------------------------------------------------
M --stage main --hidden 256 --seeds 1-10 --configs core --out results/main_256.csv > sweep/jobs_main_256.txt
stage "B-main-256-topup" sweep/jobs_main_256.txt 24
M --stage layout --hidden 256 --seeds 1-6 --out results/layout_256.csv > sweep/jobs_layout_256.txt
stage "D-layout-256" sweep/jobs_layout_256.txt 24

# ---- width 512 (requested; not budget-gated) ----------------------------
# lr 0.03 dropped at 512: already measured divergent there (tanh 81.3/88.5, relu 95.9/96.5 against
# 97.9-98.3 at 0.003/0.01) and catastrophic at 256; no core config at any width ever selected it.
M --stage calib --hidden 512 --seeds 101-102 --lrs 0.001,0.003,0.01 --configs core --out results/calib_512.csv > sweep/jobs_calib_512.txt
stage "A-calib-512" sweep/jobs_calib_512.txt 16
python3 sweep/select_lr.py results/calib_512.csv 2>&1 | tee -a $LOG
M --stage main --hidden 512 --seeds 1-5 --configs core --out results/main_512.csv > sweep/jobs_main_512.txt
stage "B-main-512" sweep/jobs_main_512.txt 16

# ---- width-64 LR robustness (lowest priority, budget-gated) -------------
M --stage lrsweep --hidden 64 --seeds 1-10 --lrs 0.001,0.003,0.01,0.03,0.1 --configs core --out results/lrsweep_64.csv > sweep/jobs_lrsweep_64.txt
optional "C-lrsweep-64" sweep/jobs_lrsweep_64.txt 64 0.6

echo "[$(ts)] ===== RESTORED STAGES COMPLETE | elapsed $(elapsed_h)h =====" | tee -a $LOG
