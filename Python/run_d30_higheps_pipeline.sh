#!/usr/bin/env bash
# Background pipeline: epsilon-greedy robustness at d=30 with higher exploration traffic.
set -u
cd "$(dirname "$0")"
LOG_DIR="mab_results_d30"
mkdir -p "$LOG_DIR" /opt/cursor/artifacts
MASTER_LOG="$LOG_DIR/pipeline_master.log"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER_LOG"; }

run_one() {
  local name="$1"; shift
  local out_log="$LOG_DIR/${name}.log"
  log "START $name: $*"
  if python3 "$@" 2>&1 | tee "$out_log"; then
    log "OK $name"
    # Snapshot key outputs into results + artifacts
    for f in "${name}"_*.csv "${name}"_*.png; do
      [[ -f "$f" ]] || continue
      cp -f "$f" "$LOG_DIR/" 2>/dev/null || true
      cp -f "$f" /opt/cursor/artifacts/ 2>/dev/null || true
    done
    return 0
  else
    log "FAIL $name (exit $?)"
    return 1
  fi
}

log "===== d=30 high-eps robustness pipeline begin ====="

# 1) Main mid-stream shift, high-ε grid, 3 repeats (idempotent re-run if needed)
run_one mab_scale_d30_higheps \
  run_mab_scale_eps.py \
  --feature-dim 30 \
  --total-samples 20000 \
  --ref-samples 5000 \
  --batch-size 25 \
  --n-repeats 3 \
  --epsilon-grid 0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.60 0.70 \
  --output-prefix mab_scale_d30_higheps

# 2) More repeats for stabler variance estimates
run_one mab_scale_d30_higheps_r5 \
  run_mab_scale_eps.py \
  --feature-dim 30 \
  --total-samples 20000 \
  --ref-samples 5000 \
  --batch-size 25 \
  --n-repeats 5 \
  --epsilon-grid 0.05 0.10 0.15 0.20 0.30 0.40 0.50 0.60 0.70 \
  --output-prefix mab_scale_d30_higheps_r5

# 3) Dense high-exploration focus (30%–70%)
run_one mab_scale_d30_dense_higheps \
  run_mab_scale_eps.py \
  --feature-dim 30 \
  --total-samples 20000 \
  --ref-samples 5000 \
  --batch-size 25 \
  --n-repeats 5 \
  --epsilon-grid 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 \
  --output-prefix mab_scale_d30_dense_higheps

# 4) Longer stream, still d=30
run_one mab_scale_d30_n40k_higheps \
  run_mab_scale_eps.py \
  --feature-dim 30 \
  --total-samples 40000 \
  --ref-samples 8000 \
  --batch-size 50 \
  --n-repeats 3 \
  --epsilon-grid 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 \
  --output-prefix mab_scale_d30_n40k_higheps

# 5) Summarize available variance tables
python3 - <<'PY' 2>&1 | tee -a "$LOG_DIR/pipeline_master.log"
import glob
import os
import pandas as pd

paths = sorted(glob.glob("mab_scale_d30*_variance_vs_explore.csv"))
print("\n===== variance_vs_explore summaries =====")
for p in paths:
    df = pd.read_csv(p)
    print(f"\n--- {p} ---")
    cols = [
        "epsilon",
        "explore_traffic_ratio",
        "final_regret_mean",
        "final_regret_std",
        "final_regret_cv",
        "path_cumregret_std_mean",
        "n_repeats",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    best = df.loc[df["final_regret_mean"].idxmin()]
    # lowest CV among eps>=0.2 as high-explore robustness proxy
    hi = df[df["epsilon"] >= 0.2]
    if len(hi):
        rob = hi.loc[hi["final_regret_cv"].idxmin()]
        print(
            f"best_mean_eps={best['epsilon']} regret={best['final_regret_mean']:.4f} | "
            f"most_robust_high_eps(cv)={rob['epsilon']} cv={rob['final_regret_cv']:.4f}"
        )
print("\nDONE")
PY

log "===== pipeline complete ====="
