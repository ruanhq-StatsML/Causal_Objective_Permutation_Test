#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-/opt/anaconda3/bin/python}"
echo "[resume] starting benchmarks..."
nohup "$PY" -u run_time_profile_vimpDS.py >> time_profile_benchmark_vimpDS.log 2>&1 & echo "time n=$!"
nohup "$PY" -u run_time_profile_vimpDS_by_p.py >> time_profile_benchmark_vimpDS_by_p.log 2>&1 & echo "time p=$!"
nohup "$PY" -u FMoW_dataset_benchmark.py >> FMoW_dataset_benchmark.log 2>&1 & echo "FMoW=$!"
nohup "$PY" -u whyshift_benchmark_run.py --whyshift-linear-only >> whyshift_benchmark_run.log 2>&1 & echo "whyshift=$!"
nohup "$PY" -u run_whyshift_adv_cd_benchmark.py >> whyshift_adv_cd_benchmark.log 2>&1 & echo "advcd=$!"
echo "[resume] all launched"
