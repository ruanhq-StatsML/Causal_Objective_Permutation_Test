#!/usr/bin/env bash
# Rebuild delivery/fsds_sot_delivery.zip from repo tree.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/delivery/fsds_sot_package"
STAGE="$ROOT/delivery/.fsds_sot_staging"
OUT="$ROOT/delivery/fsds_sot_delivery.zip"

rm -rf "$STAGE"
mkdir -p "$STAGE/Python" "$STAGE/docs" "$STAGE/artifacts"

cp -a "$ROOT/Python/fsds_sot" "$STAGE/Python/"
cp "$ROOT/Python/demo_fsds_modality_attribution.py" "$STAGE/Python/"
cp "$ROOT/Python/demo_fsds_token_perturbation.py" "$STAGE/Python/"
cp "$ROOT/Python/DRPerm.py" "$STAGE/Python/"
cp "$ROOT/Python/model_registry_class.py" "$STAGE/Python/"
cp "$ROOT/docs/FSDS_TWO_LAYER_MODALITY_ATTRIBUTION.md" "$STAGE/docs/"
cp "$PKG/README_PACKAGE.md" "$STAGE/README.md"
cp "$PKG/PACKAGE_INVENTORY.md" "$STAGE/PACKAGE_INVENTORY.md"
cp "$PKG/MODULE_LIST.txt" "$STAGE/MODULE_LIST.txt"
cp "$PKG/MANIFEST.json" "$STAGE/MANIFEST.json"
touch "$STAGE/artifacts/.gitkeep"

rm -f "$OUT"
(cd "$STAGE" && zip -rq "$OUT" . -x '*__pycache__*' '*.pyc')
rm -rf "$STAGE"
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
