# FSDS-SoT portable package (modality + two-layer attribution + token perturb)

## Layout

- `Python/fsds_sot/` — full `fsds_sot` Python package
- `Python/demo_fsds_modality_attribution.py` — main demo
- `Python/demo_fsds_token_perturbation.py` — token mask/replace audit
- `Python/feature_store/` — **feature-store** multi-layer attribution (merchant / user / cross-level)
- `Python/demo_fsds_feature_store_attribution.py` — exports `data/feature_store/*.csv` + JSON report
- `data/feature_store/` — bundled relational dataset (users, orders, merchants, feature matrices)
- `Python/DRPerm.py`, `Python/model_registry_class.py` — PO-risk / Model Registry deps
- `docs/FSDS_TWO_LAYER_MODALITY_ATTRIBUTION.md` — technical notes

## Setup

```bash
cd Python
pip install numpy scikit-learn xgboost pandas scipy statsmodels
# Text encoder (production): copy fsds_sot/qwen_episode_encoder.py.example → qwen_episode_encoder.py
```

## Run

```bash
cd Python
python3 demo_fsds_modality_attribution.py
python3 demo_fsds_token_perturbation.py
python3 demo_fsds_feature_store_attribution.py
```

Outputs: `../artifacts/modality_attribution_report.json`, `../artifacts/token_perturbation_report.json` (create `artifacts/` at repo root if missing).

Branch: `cursor/fsds-sot-tot-manuscript-7451`.

**Full module list & provenance (extract vs new):** see `PACKAGE_INVENTORY.md` and `MODULE_LIST.txt` in this folder (also inside the zip).
