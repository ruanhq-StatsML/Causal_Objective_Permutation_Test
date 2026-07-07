#!/usr/bin/env bash
# Fresh rerun for video / audio / ChronoBerg / WILDS / tabular FGSM benchmarks.
#
# On Mac:
#   cd ".../Causal_Objective_Permutation_Test/Python"
#   git pull origin cursor/rerun-realdata-benchmarks-5ef9
#   chmod +x rerun_realdata_benchmarks.sh
#   RERUN_FRESH=1 ./rerun_realdata_benchmarks.sh
#
# RERUN_FRESH=1  -> backup existing CSV/JSON, then rerun from scratch
# RERUN_FRESH=0  -> resume only (skip rows already in scores CSV)
#
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [[ -x /opt/anaconda3/bin/python ]]; then
  PY="${PYTHON:-/opt/anaconda3/bin/python}"
fi

backup_group () {
  local tag="$1"; shift
  local files=("$@")
  local bk="backup_rerun_${tag}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$bk"
  local moved=0
  for f in "${files[@]}"; do
    if [[ -f "$f" ]]; then
      mv "$f" "$bk/"
      moved=1
    fi
    if [[ -f "${f}.tmp" ]]; then
      mv "${f}.tmp" "$bk/"
      moved=1
    fi
  done
  if [[ "$moved" == "1" ]]; then
    echo "[rerun] backed up ${tag} -> ${bk}/"
  else
    rmdir "$bk" 2>/dev/null || true
    echo "[rerun] nothing to backup for ${tag}"
  fi
}

start_job () {
  local script="$1"
  local log="$2"
  if [[ ! -f "$script" ]]; then
    echo "[rerun] skip missing script: $script"
    return
  fi
  if pgrep -f "[Pp]ython.*-u ${script}" >/dev/null 2>&1; then
    echo "[rerun] already running: $script"
    return
  fi
  nohup "$PY" -u "$script" >> "$log" 2>&1 &
  echo "[rerun] started ${script} pid=$! log=${log}"
}

if [[ "${RERUN_FRESH:-}" == "1" ]]; then
  echo "========== backup old checkpoints (fresh rerun) =========="
  backup_group video \
    video1234_benchmark_scores.csv video1234_benchmark_metrics.csv video1234_benchmark_results.json
  backup_group audio \
    audio1234_benchmark_scores.csv audio1234_benchmark_metrics.csv audio1234_benchmark_results.json
  backup_group chronoberg_adv \
    ChronoBerg_benchmark_scores.csv ChronoBerg_benchmark_metrics.csv ChronoBerg_benchmark_results.json
  backup_group chronoberg_cs30 \
    ChronoBerg_cs30_benchmark_scores.csv ChronoBerg_cs30_benchmark_metrics.csv ChronoBerg_cs30_benchmark_results.json
  backup_group wilds \
    wilds_vimp_scores.csv wilds_vimp_metrics.csv wilds_vimp_results.json
  backup_group fgsm30 \
    fgsm30_vimp_scores.csv fgsm30_vimp_metrics.csv fgsm30_benchmark_results.json
  backup_group realdata_whyshift \
    realdata_vimp_scores.csv realdata_vimp_metrics.csv realdata_vimp_results.json
fi

echo ""
echo "========== launch benchmarks (background) =========="
start_job video1234_benchmark.py video1234_benchmark.log
start_job audio1234_benchmark.py audio1234_benchmark.log
start_job ChronoBerg_benchmark.py ChronoBerg_benchmark.log
start_job ChronoBerg_covariate_shift_benchmark.py ChronoBerg_cs30_benchmark.log
start_job wilds_data_benchmark.py wilds_data_benchmark.log
start_job run_fgsm30_benchmark.py fgsm30_benchmark.log
start_job whyshift_benchmark_run.py whyshift_benchmark_run.log

echo ""
echo "========== monitor =========="
echo "  python monitor_benchmarks.py"
echo "  python check_benchmark_status.py"
echo "  tail -f video1234_benchmark.log ChronoBerg_benchmark.log"
echo ""
echo "Note: audio1234 skips pairs if real_data/audio*_pca30.npy is missing."
echo "Jobs keep running after this script exits (nohup)."
