#!/usr/bin/env bash
# Recover CSVs from .tmp, fix empty helper files, audit status, then resume all jobs.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-/opt/anaconda3/bin/python}"

echo "========== 1) restore helper files (iCloud corruption) =========="
"$PY" restore_benchmark_helpers.py

echo ""
echo "========== 2) audit CSV checkpoints =========="
"$PY" check_benchmark_status.py

echo ""
echo "========== 3) resume incomplete benchmarks =========="
if [[ "${RESTORE_YES:-}" == "1" ]]; then
  ./resume_all_benchmarks.sh
else
  read -r -p "Start background jobs now? [y/N] " ans
  if [[ "${ans,,}" == "y" ]]; then
    ./resume_all_benchmarks.sh
  else
    echo "Skipped launch. Run ./resume_all_benchmarks.sh when ready."
  fi
fi

echo ""
echo "========== iCloud manual recovery =========="
echo "- Finder: right-click CSV -> Browse All Versions"
echo "- Time Machine: restore Python/*.csv"
echo "- Completed data is NOT re-generated; resume skips finished jobs in CSV."
