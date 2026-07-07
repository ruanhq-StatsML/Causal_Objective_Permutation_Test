#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [[ -x /opt/anaconda3/bin/python ]]; then
  PY="${PYTHON:-/opt/anaconda3/bin/python}"
fi
echo "[resume] starting benchmarks..."
nohup "$PY" -u run_time_profile_vimpDS.py >> time_profile_benchmark_vimpDS.log 2>&1 & echo "time n=$!"
nohup "$PY" -u run_time_profile_vimpDS_by_p.py >> time_profile_benchmark_vimpDS_by_p.log 2>&1 & echo "time p=$!"
nohup "$PY" -u FMoW_dataset_benchmark.py >> FMoW_dataset_benchmark.log 2>&1 & echo "FMoW=$!"
nohup "$PY" -u whyshift_benchmark_run.py --whyshift-linear-only >> whyshift_benchmark_run.log 2>&1 & echo "whyshift=$!"
nohup "$PY" -u run_whyshift_adv_cd_benchmark.py >> whyshift_adv_cd_benchmark.log 2>&1 & echo "advcd=$!"
nohup "$PY" -u video1234_benchmark.py >> video1234_benchmark.log 2>&1 & echo "video=$!"
nohup "$PY" -u audio1234_benchmark.py >> audio1234_benchmark.log 2>&1 & echo "audio=$!"
nohup "$PY" -u ChronoBerg_benchmark.py >> ChronoBerg_benchmark.log 2>&1 & echo "chronoberg_adv=$!"
nohup "$PY" -u ChronoBerg_covariate_shift_benchmark.py >> ChronoBerg_cs30_benchmark.log 2>&1 & echo "chronoberg_cs30=$!"
nohup "$PY" -u wilds_data_benchmark.py >> wilds_data_benchmark.log 2>&1 & echo "wilds=$!"
nohup "$PY" -u run_fgsm30_benchmark.py >> fgsm30_benchmark.log 2>&1 & echo "fgsm30=$!"
echo "[resume] all launched"
