#!/usr/bin/env bash
# Resilient confirmatory-battery runner for a memory-constrained box (14 GB).
# Runs each remaining task in its OWN python process so RSS is fully reclaimed
# between tasks (the first all-in-one-process run OOM-killed mid-T2 after ~5h
# of cumulative transformer index builds). T1 already completed; its outputs
# exist, so the single-shot guard skips it. Each task appends to battery.log.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=reports/confirmatory/battery.log
echo "=== sequential battery restart $(date '+%F %T %z') (T1 already done) ===" >> "$LOG"
for task in t2 t3 t4 ssb; do
  echo "=== launching $task in fresh process $(date '+%T') ===" >> "$LOG"
  python3 -u scripts/confirmatory_battery.py --task "$task" >> "$LOG" 2>&1
  rc=$?
  echo "=== $task exited rc=$rc $(date '+%T') ===" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "=== $task FAILED (rc=$rc); continuing to next task ===" >> "$LOG"
  fi
done
echo "=== sequential battery complete $(date '+%F %T %z') ===" >> "$LOG"
