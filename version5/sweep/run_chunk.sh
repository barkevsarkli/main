#!/bin/bash
# Run up to $2 pending jobs from job file $1 with $3 parallel workers (defaults: 28 jobs, 4 workers).
# Safe to call repeatedly: finished runs are detected from the results CSVs and skipped.
cd "$(dirname "$0")/.." || exit 1
JOBS=$1; N=${2:-28}; P=${3:-4}
PENDING=$(mktemp "${TMPDIR:-/tmp}/pending_XXXXXX")
trap 'rm -f "$PENDING" "$PENDING.chunk"' EXIT

python3 sweep/pending.py < "$JOBS" > "$PENDING"
TOTAL=$(wc -l < "$JOBS" | tr -d ' '); LEFT=$(wc -l < "$PENDING" | tr -d ' ')
echo "[$(date +%H:%M:%S)] $JOBS: $LEFT of $TOTAL pending; running up to $N with $P workers"

head -n "$N" "$PENDING" > "$PENDING.chunk"
COUNT=$(wc -l < "$PENDING.chunk" | tr -d ' ')

# xargs is fed line NUMBERS, not the job lines themselves, and each worker looks its own
# line up with sed.  BSD xargs caps a -I-constructed argument at 255 bytes, and a job line
# grows with the length of the --out path: at 278 bytes (an --out under a long directory)
# every chunk dies with "command line cannot be assembled, too long" and the stage aborts
# reporting no progress.  Passing an index keeps the constructed argument a handful of
# bytes no matter how long the job line or the results path is.
if [ "$COUNT" -gt 0 ]; then
  seq 1 "$COUNT" | xargs -P "$P" -I{} bash -c \
    'job=$(sed -n "$0p" "$1"); [ -n "$job" ] || exit 0; eval "$job" >/dev/null 2>&1 || echo "FAILED: $job"' \
    {} "$PENDING.chunk"
fi

python3 sweep/pending.py < "$JOBS" | wc -l | xargs -I{} echo "[$(date +%H:%M:%S)] remaining: {}"
