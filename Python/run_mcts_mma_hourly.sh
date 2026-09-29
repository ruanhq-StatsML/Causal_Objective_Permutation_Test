#!/bin/bash
# One new draw of the four-dataset MCTS attribution every hour.
set -u
cd /workspace
LOG=/opt/cursor/artifacts/mcts_mma_hourly.jsonl
mkdir -p Python/results /opt/cursor/artifacts
if [ ! -f "$LOG" ] && [ -f Python/results/mcts_mma_hourly.jsonl ]; then
  cp Python/results/mcts_mma_hourly.jsonl "$LOG"
fi

seconds_until_next() {
  python3 - "$LOG" << 'PY'
import json, sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

tz = ZoneInfo("Asia/Shanghai")
now = datetime.now(tz)
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
    print(0)
else:
    wait = (last + timedelta(seconds=3600) - now).total_seconds()
    print(int(max(0, wait)))
PY
}

while true; do
  wait_s=$(seconds_until_next)
  if [ "$wait_s" -gt 0 ]; then
    echo "sleep ${wait_s}s until the next MCTS multimodal round"
    sleep "$wait_s"
  fi
  python3 Python/mcts_mma_four_datasets.py --hourly --log "$LOG"
  git add \
    Python/results/mcts_mma_hourly.jsonl \
    Python/results/mcts_mma_hourly_summary.txt \
    Python/results/mcts_mma_four.json
  if ! git diff --cached --quiet; then
    git commit -m "Record the latest hourly MCTS multimodal attribution round."
    git push -u origin HEAD || echo "push failed; the round is saved in Python/results"
  fi
done
