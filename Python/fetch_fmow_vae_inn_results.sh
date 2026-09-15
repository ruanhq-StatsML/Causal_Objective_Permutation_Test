#!/usr/bin/env bash
# Download VAE/INN result bundles produced by the cloud agent.
# Prefer the Artifacts panel on:
#   https://cursor.com/agents/bc-2fb3ca4b-d04a-46b8-b039-656735ce9eb2
# Save the .tar.gz next to this script, then:
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/fmow_freia_results}"
mkdir -p "$OUT"
if [[ -f "$ROOT/fmow_vae_inn_results_lite.tar.gz" ]]; then
  tar -xzf "$ROOT/fmow_vae_inn_results_lite.tar.gz" -C "$OUT"
  echo "Extracted lite bundle into $OUT"
elif [[ -f "$ROOT/fmow_vae_inn_results_full.tar.gz" ]]; then
  tar -xzf "$ROOT/fmow_vae_inn_results_full.tar.gz" -C "$OUT"
  echo "Extracted full bundle into $OUT"
else
  echo "Place fmow_vae_inn_results_lite.tar.gz (or _full) in $ROOT first."
  echo "Get it from: https://cursor.com/agents/bc-2fb3ca4b-d04a-46b8-b039-656735ce9eb2  (Artifacts)"
  exit 1
fi
