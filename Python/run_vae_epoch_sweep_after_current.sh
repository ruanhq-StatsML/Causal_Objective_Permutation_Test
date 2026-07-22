#!/usr/bin/env bash
# Wait for the current 50-epoch fmow VAE run, then launch epoch sweep.
set -euo pipefail

cd /workspace/Python
LOG=/workspace/Python/vae_epoch_sweep_queue.log

echo "[$(date -Is)] Waiting for current vae_fmow_outputs training to finish..." | tee -a "$LOG"

while pgrep -f "run_vae_domain_shift.py.*vae_fmow_outputs" >/dev/null 2>&1; do
  sleep 60
done

echo "[$(date -Is)] Current run finished. Starting epoch sweep (25/50/75/100 + early-stop)..." | tee -a "$LOG"

python3 -u run_vae_epoch_sweep.py \
  --train-dir /workspace/data/fmow_subset/train \
  --eval-dir /workspace/data/fmow_subset/eval \
  --output-root vae_fmow_epoch_sweep \
  --epochs-grid 25 50 75 100 \
  --batch-size 32 \
  --num-workers 4 \
  --cpu \
  --early-stop-max-epochs 100 \
  --early-stop-patience 5 \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] Epoch sweep done." | tee -a "$LOG"
