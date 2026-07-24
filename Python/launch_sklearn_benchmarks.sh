#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m pip install -q -r requirements_sklearn_benchmark.txt

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/sklearn_realdata_${STAMP}.log"

echo "[launch] starting sklearn real-data benchmarks -> $LOG"
nohup python3 -u run_sklearn_realdata_benchmarks.py --benchmark all \
  > "$LOG" 2>&1 &
echo $! > "$LOG_DIR/sklearn_realdata.pid"
echo "[launch] pid=$(cat "$LOG_DIR/sklearn_realdata.pid") log=$LOG"
