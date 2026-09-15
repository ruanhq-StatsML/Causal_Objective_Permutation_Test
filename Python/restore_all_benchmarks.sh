#!/usr/bin/env bash
# Recover CSVs from .tmp, fix empty helper files, audit status, then resume all jobs.
#
# Usage (on your Mac, in the Python/ folder):
#   cd ".../Causal_Objective_Permutation_Test/Python"
#   chmod +x restore_all_benchmarks.sh resume_all_benchmarks.sh
#   RESTORE_YES=1 ./restore_all_benchmarks.sh          # recover + resume (no prompt)
#   ./restore_all_benchmarks.sh                        # recover + ask before resume
#   RESTORE_ONLY=1 ./restore_all_benchmarks.sh         # recover + audit only
#
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [[ -x /opt/anaconda3/bin/python ]]; then
  PY="${PYTHON:-/opt/anaconda3/bin/python}"
fi

echo "========== 1) recover CSV .tmp + helper files (iCloud corruption) =========="
"$PY" restore_benchmark_helpers.py

echo ""
echo "========== 2) audit CSV checkpoints =========="
"$PY" check_benchmark_status.py

if [[ "${RESTORE_ONLY:-}" == "1" ]]; then
  echo ""
  echo "RESTORE_ONLY=1 — skipped job launch."
  exit 0
fi

echo ""
echo "========== 3) resume incomplete benchmarks =========="
if [[ "${RESTORE_YES:-}" == "1" ]]; then
  ./resume_all_benchmarks.sh
else
  read -r -p "Start background jobs now? [y/N] " ans
  if [[ "${ans,,}" == "y" ]]; then
    ./resume_all_benchmarks.sh
  else
    echo "Skipped launch. Run: RESTORE_YES=1 ./restore_all_benchmarks.sh"
  fi
fi

echo ""
echo "========== iCloud / Time Machine manual recovery =========="
echo "If CSVs are still missing:"
echo "  1. Finder → right-click *.csv → Browse All Versions"
echo "  2. Time Machine → restore Python/*.csv and *.csv.tmp"
echo "  3. Re-run: RESTORE_ONLY=1 ./restore_all_benchmarks.sh"
echo ""
echo "Completed rows in CSV are NOT re-run; resume scripts skip finished jobs."
