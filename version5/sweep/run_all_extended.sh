#!/bin/bash
# Drive the extended sweep.  Resumable: re-running skips runs already present in the CSVs.
#
# Scope was cut after stage A-12/32/64 measured the machine's sustained throughput at 1.81x
# over the Step-0 serial timings (not the 5.05x a short benchmark suggested), which puts the
# full plan at ~24 h.  Cuts follow the prompt's priority order, from the bottom:
#   dropped : width 512 entirely, layout at 256, LR robustness at 64
#   reduced : main at 256 from 10 seeds to 6
#   optional: layout at 128 (P4) and the MSE replication at 128 (P6) run only if the
#             13-hour budget still has room when the priority 1-3 stages are done.
cd "$(dirname "$0")/.." || exit 1
LOG=results/sweep_extended.log
P=${P:-8}
SWEEP_START=${SWEEP_START:-1788521537}      # 2026-09-04 14:32:17, when the sweep first started
BUDGET_H=${BUDGET_H:-13}
M(){ python3 sweep/make_jobs.py "$@"; }
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
elapsed_h(){ python3 -c "print(f'{($(date +%s)-$SWEEP_START)/3600:.2f}')"; }
stage(){ # stage <label> <jobs file> <chunk>
  bash sweep/run_stage.sh "$1" "$2" "$P" "$3" || { echo "[$(ts)] stage $1 FAILED, stopping" | tee -a $LOG; exit 1; }
}
optional(){ # optional <label> <jobs file> <chunk> <estimated hours>
  local left; left=$(python3 -c "print(f'{$BUDGET_H-$(elapsed_h):.2f}')")
  if python3 -c "import sys; sys.exit(0 if $left > $4 else 1)"; then
    echo "[$(ts)] budget check: ${left}h left, stage $1 needs ~$4h -> running" | tee -a $LOG
    stage "$1" "$2" "$3"
  else
    echo "[$(ts)] budget check: ${left}h left, stage $1 needs ~$4h -> SKIPPED (budget)" | tee -a $LOG
  fi
}

echo "[$(ts)] ===== EXTENDED SWEEP (revised scope) RESUME | workers=$P budget=${BUDGET_H}h elapsed=$(elapsed_h)h =====" | tee -a $LOG

# ---------- LR selection for the calibration grids already finished ------
echo "[$(ts)] --- select_lr on calib_12_v2 / calib_32 / calib_64 ---" | tee -a $LOG
python3 sweep/select_lr.py results/calib_12_v2.csv results/calib_32.csv results/calib_64.csv 2>&1 | tee -a $LOG

# ---------- priority 1: the crossover at the cheap widths ----------------
M --stage main --hidden 12 --seeds 1-30 --configs all --out results/main_12_v2.csv > sweep/jobs_main_12v2.txt
M --stage main --hidden 32 --seeds 1-30 --configs all --out results/main_32.csv    > sweep/jobs_main_32.txt
M --stage main --hidden 64 --seeds 1-20 --configs all --out results/main_64.csv    > sweep/jobs_main_64.txt
stage "B-main-12v2" sweep/jobs_main_12v2.txt 96
stage "B-main-32"   sweep/jobs_main_32.txt   96
stage "B-main-64"   sweep/jobs_main_64.txt   64

# ---------- priority 2: LR robustness at width 12 (cheap) ----------------
M --stage lrsweep --hidden 12 --seeds 1-10 --lrs 0.001,0.003,0.01,0.03,0.1,0.3 --configs core --out results/lrsweep_12.csv > sweep/jobs_lrsweep_12.txt
stage "C-lrsweep-12" sweep/jobs_lrsweep_12.txt 96

# ---------- priority 1+3: width 128 --------------------------------------
M --stage calib --hidden 128 --seeds 101-103 --lrs 0.001,0.003,0.01,0.03 --configs all --out results/calib_128_v2.csv > sweep/jobs_calib_128v2.txt
stage "A-calib-128" sweep/jobs_calib_128v2.txt 48
python3 sweep/select_lr.py results/calib_128_v2.csv 2>&1 | tee -a $LOG
M --stage main --hidden 128 --seeds 1-20 --configs all --out results/main_128_v2.csv > sweep/jobs_main_128v2.txt
stage "B-main-128v2" sweep/jobs_main_128v2.txt 48

# ---------- priority 1: width 256 (seeds reduced 10 -> 6) ----------------
M --stage calib --hidden 256 --seeds 101-103 --lrs 0.001,0.003,0.01,0.03 --configs core --out results/calib_256.csv > sweep/jobs_calib_256.txt
stage "A-calib-256" sweep/jobs_calib_256.txt 24
python3 sweep/select_lr.py results/calib_256.csv 2>&1 | tee -a $LOG
M --stage main --hidden 256 --seeds 1-6 --configs core --out results/main_256.csv > sweep/jobs_main_256.txt
stage "B-main-256" sweep/jobs_main_256.txt 24

# ---------- priority 2: LR robustness at width 128 -----------------------
M --stage lrsweep --hidden 128 --seeds 1-10 --lrs 0.001,0.003,0.01,0.03,0.1 --configs core --out results/lrsweep_128.csv > sweep/jobs_lrsweep_128.txt
stage "C-lrsweep-128" sweep/jobs_lrsweep_128.txt 48

echo "[$(ts)] ===== PRIORITY 1-3 COMPLETE | elapsed $(elapsed_h)h =====" | tee -a $LOG

# ---------- priority 4: layout / ratio at 128 (budget permitting) --------
M --stage layout --hidden 128 --seeds 1-10 --out results/layout_128.csv > sweep/jobs_layout_128.txt
optional "D-layout-128" sweep/jobs_layout_128.txt 48 1.4

# ---------- priority 6: MSE replication at 128 (budget permitting) -------
M --stage calib --hidden 128 --seeds 101-103 --lrs 0.003,0.01,0.03 --configs mse --loss mse --out results/calib_mse_128.csv > sweep/jobs_calib_mse_128.txt
optional "E-calib-mse-128" sweep/jobs_calib_mse_128.txt 27 0.4
if [ -s results/calib_mse_128.csv ] && [ "$(python3 sweep/pending.py < sweep/jobs_calib_mse_128.txt | wc -l)" -eq 0 ]; then
  python3 sweep/select_lr.py results/calib_mse_128.csv 2>&1 | tee -a $LOG
  M --stage main --hidden 128 --seeds 1-10 --configs mse --loss mse --out results/mse_128.csv > sweep/jobs_mse_128.txt
  optional "E-mse-128" sweep/jobs_mse_128.txt 30 0.45
fi

echo "[$(ts)] ===== EXTENDED SWEEP COMPLETE | elapsed $(elapsed_h)h =====" | tee -a $LOG
