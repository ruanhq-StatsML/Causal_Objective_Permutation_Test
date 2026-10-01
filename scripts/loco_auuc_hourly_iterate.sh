#!/usr/bin/env bash
# Hourly re-run benchmark (cron-friendly). Logs append to artifacts/loco_auuc_hourly.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/Python"
echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >> "$ROOT/artifacts/loco_auuc_hourly.log"
python3 run_loco_auuc_benchmark.py --quick >> "$ROOT/artifacts/loco_auuc_hourly.log" 2>&1
