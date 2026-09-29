#!/bin/bash
# One target on Cora and one on CiteSeer every 30 minutes, until 05:30 Asia/Shanghai.
set -u
cd /workspace
LOG=/opt/cursor/artifacts/multistart_mcts_log.jsonl
SUM=/opt/cursor/artifacts/multistart_mcts_summary.txt
mkdir -p Python/results

seconds_until_next() {
  python3 - "$LOG" << 'PY'
import json, sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from Python.multistart_mcts_attribution import deadline_530

tz = ZoneInfo("Asia/Shanghai")
now = datetime.now(tz)
stop = deadline_530()
log = Path(sys.argv[1])
last = None
if log.exists():
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stamp = json.loads(line).get("time")
        if stamp:
            last = datetime.fromisoformat(stamp)
if last is None:
    wait = 0
else:
    wait = (last + timedelta(seconds=1800) - now).total_seconds()
remain = (stop - now).total_seconds()
if remain <= 0:
    print(-1)
else:
    print(int(max(0, min(wait, remain))))
PY
}

while true; do
  wait_s=$(seconds_until_next)
  if [ "$wait_s" -lt 0 ]; then
    echo "past 05:30 Asia/Shanghai, stop"
    break
  fi
  if [ "$wait_s" -gt 0 ]; then
    echo "sleep ${wait_s}s"
    sleep "$wait_s"
  fi
  python3 Python/multistart_mcts_attribution.py --log "$LOG"
  cp "$LOG" Python/results/multistart_mcts_log.jsonl
  cp "$SUM" Python/results/multistart_mcts_summary.txt
  git add Python/results/multistart_mcts_log.jsonl Python/results/multistart_mcts_summary.txt
  if ! git diff --cached --quiet; then
    git commit -m "Record the latest half-hour multi-start MCTS attribution round."
    git push -u origin HEAD
  fi
done
