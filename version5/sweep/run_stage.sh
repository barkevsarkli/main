#!/bin/bash
# Drive one sweep stage to completion with resumable chunks and progress logging.
#   run_stage.sh <label> <jobs file> [workers] [chunk size]
# Repeatedly calls run_chunk.sh until pending == 0.  Safe to interrupt and re-run:
# finished runs are detected from the result CSVs by pending.py and skipped.
cd "$(dirname "$0")/.." || exit 1
LABEL=$1; JOBS=$2; P=${3:-8}; CHUNK=${4:-64}
LOG=${V5_RESULTS:-results}/sweep_extended.log
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
pend(){ python3 sweep/pending.py < "$JOBS" | wc -l | tr -d ' '; }

TOTAL=$(wc -l < "$JOBS" | tr -d ' ')
START_LEFT=$(pend)
T0=$(date +%s)
echo "[$(ts)] STAGE $LABEL start | jobs=$TOTAL pending=$START_LEFT workers=$P" | tee -a "$LOG"

while :; do
  LEFT=$(pend)
  [ "$LEFT" -eq 0 ] && break
  bash sweep/run_chunk.sh "$JOBS" "$CHUNK" "$P" >> "$LOG" 2>&1
  NEW=$(pend)
  NOW=$(date +%s); EL=$((NOW-T0)); DONE=$((START_LEFT-NEW))
  if [ "$DONE" -gt 0 ]; then
    ETA=$(( EL * NEW / DONE ))
    printf '[%s] STAGE %s | done %d/%d  remaining %d  elapsed %dm%02ds  ETA %dh%02dm\n' \
      "$(ts)" "$LABEL" "$DONE" "$START_LEFT" "$NEW" $((EL/60)) $((EL%60)) $((ETA/3600)) $(((ETA%3600)/60)) | tee -a "$LOG"
  fi
  if [ "$NEW" -eq "$LEFT" ]; then
    echo "[$(ts)] STAGE $LABEL ABORT: no progress in a full chunk ($LEFT still pending) -- check for failing jobs" | tee -a "$LOG"
    exit 1
  fi
done
NOW=$(date +%s); EL=$((NOW-T0))
printf '[%s] STAGE %s COMPLETE | %d jobs run in %dh%02dm%02ds\n' \
  "$(ts)" "$LABEL" "$START_LEFT" $((EL/3600)) $(((EL%3600)/60)) $((EL%60)) | tee -a "$LOG"
