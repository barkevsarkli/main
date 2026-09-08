#!/bin/bash
# Wait for run_all_extended.sh to exit, then run the restored stages, so nothing idles overnight.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "sweep/run_all_extended.sh" > /dev/null 2>&1; do sleep 20; done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] master script finished; starting restored stages" >> results/sweep_extended.log
exec caffeinate -i bash sweep/run_restored.sh
