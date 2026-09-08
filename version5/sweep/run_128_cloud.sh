#!/bin/bash
# Full 128-hidden pipeline: LR calibration -> select -> main sweep.  2 workers.
cd "$(dirname "$0")/.." || exit 1
python3 sweep/make_jobs.py calib128 --data data > sweep/jobs_calib128.txt
while [ "$(python3 sweep/pending.py < sweep/jobs_calib128.txt | wc -l)" -gt 0 ]; do
  bash sweep/run_chunk.sh sweep/jobs_calib128.txt 10 2
done
python3 sweep/select_lr.py results/calib_128.csv
python3 sweep/make_jobs.py main128 --data data > sweep/jobs_main128.txt
while [ "$(python3 sweep/pending.py < sweep/jobs_main128.txt | wc -l)" -gt 0 ]; do
  bash sweep/run_chunk.sh sweep/jobs_main128.txt 30 2
done
echo "[$(date +%H:%M:%S)] 128-hidden pipeline DONE"
