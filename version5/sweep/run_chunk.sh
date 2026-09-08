#!/bin/bash
# Run up to $2 pending jobs from job file $1 with $3 parallel workers (defaults: 28 jobs, 4 workers).
# Safe to call repeatedly: finished runs are detected from the results CSVs and skipped.
cd "$(dirname "$0")/.." || exit 1
JOBS=$1; N=${2:-28}; P=${3:-4}
python3 sweep/pending.py < "$JOBS" > /tmp/pending_$$.txt
TOTAL=$(wc -l < "$JOBS"); LEFT=$(wc -l < /tmp/pending_$$.txt)
echo "[$(date +%H:%M:%S)] $JOBS: $LEFT of $TOTAL pending; running up to $N with $P workers"
# The job line is passed as an argument (not substituted twice into the command
# string): BSD xargs caps a -I-constructed argument at 255 bytes, and a doubled
# job line exceeds that.  Behaviour is identical on GNU xargs.
head -n "$N" /tmp/pending_$$.txt | xargs -P "$P" -I{} bash -c 'eval "$0" > /dev/null 2>&1 || echo "FAILED: $0"' {}
rm -f /tmp/pending_$$.txt
python3 sweep/pending.py < "$JOBS" | wc -l | xargs -I{} echo "[$(date +%H:%M:%S)] remaining: {}"
