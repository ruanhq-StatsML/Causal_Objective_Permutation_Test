#!/usr/bin/env bash
# One-shot rerun at your Mac Python/ folder. No git pull required.
#
#   cd "/Users/heqiaoruan/Library/Mobile Documents/com~apple~CloudDocs/Documents/GitHub 2/Causal_Objective_Permutation_Test/Python"
#   chmod +x rerun_now.sh
#   ./rerun_now.sh              # resume (skip finished rows in CSV)
#   RERUN_FRESH=1 ./rerun_now.sh  # backup CSVs then rerun from scratch
#
set -euo pipefail
cd "$(dirname "$0")"
export RERUN_FRESH="${RERUN_FRESH:-0}"
exec ./rerun_realdata_benchmarks.sh
