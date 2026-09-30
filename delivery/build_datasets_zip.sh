#!/usr/bin/env bash
# Bundle datasets for GitHub (each zip < 100MB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEL="$ROOT/delivery"
DGP="$ROOT/Python/datasets/DGP_Simulation_BothCSCD"
FS="$DEL/data/feature_store"

rm -f "$DEL/fsds_datasets_feature_store.zip" \
      "$DEL/fsds_datasets_dgp_meta.zip" \
      "$DEL/fsds_datasets_dgp_outputs_comprehensive.zip" \
      "$DEL/fsds_datasets_dgp_outputs_onesided.zip"

# Feature-store relational + pre-aggregated features
(cd "$DEL/data" && zip -rq "$DEL/fsds_datasets_feature_store.zip" feature_store README_DATA.md DATASETS_MANIFEST.json)

# DGP simulation: meta CSVs (small)
(cd "$DGP" && zip -rq "$DEL/fsds_datasets_dgp_meta.zip" DGP_whole_meta_comprehensive.csv DGP_whole_meta_onesided_37.csv)

# DGP simulation: one .npy per zip (GitHub 100MB file limit)
zip -j -q "$DEL/fsds_datasets_dgp_outputs_comprehensive.zip" "$DGP/DGP_whole_outputs.npy"
zip -j -q "$DEL/fsds_datasets_dgp_outputs_onesided.zip" "$DGP/DGP_whole_outputs_onesided.npy"

for f in "$DEL"/fsds_datasets_*.zip; do
  echo "$(basename "$f") $(du -h "$f" | cut -f1)"
done
