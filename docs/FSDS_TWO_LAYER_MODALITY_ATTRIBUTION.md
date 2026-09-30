# Two-layer modality attribution (delivery)

## Layer 1 — leave-one-group-out (LOGO)

On concatenated ref/live features, drop entire **modality blocks** (text / structured / embedding) and measure:

- `mmd_logo_delta = MMD²(full) − MMD²(without group)` (larger ⇒ group drives covariate shift)
- `domain_logo_delta = AUC(full) − AUC(without group)` (larger ⇒ group drives batch separability)

Text-only track uses sub-groups `text_low` / `text_high` (first/second half of text dims).

Optional: within **embedding**, branch blocks `emb_b0`, … get the same LOGO treatment.

## Layer 2 — within-group feature attribution

Per group, rank features by global concat **permutation VIMP** and **MMD-LOCO** (fast path), or refit domain RF on the block only (`refit_layer2_blocks=True`).

## Entrypoint

```bash
cd Python && python3 demo_fsds_modality_attribution.py
```

Modules:

- `fsds_sot/modality_dataloaders.py` — upstream loaders + concat
- `fsds_sot/modality_attribution.py` — full pipeline + report JSON
- `fsds_sot/hierarchical_attribution.py` — two-layer LOGO core
- `fsds_sot/attribution.py` — `covariate_attribution`, `mmd_logo_group_delta`, `domain_logo_group_delta`

Report keys: `two_layer_concat`, `two_layer_text`, `two_layer_embedding_branches`.

## Token perturbation → re-calculate MMD / PO-risk

Module `fsds_sot/token_perturbation_audit.py` (with `text_tokens.py`):

1. Episode → token sequence (`BR_*`, `Q_*`, tool/latency tags).
2. Perturb live batch: **mask**, **replace**, **shuffle**, **drop** (global fraction), or **mask token position i** / **mask prefix group** (`BR_retrieve`, …).
3. Re-encode via **your Qwen script**: copy `qwen_episode_encoder.py.example` → `qwen_episode_encoder.py` with `encode_episode_texts(texts) -> (n,d)`. Optional: `FSDS_TEXT_ENCODER_MODULE=...`. Hash fallback is demo-only.
4. Stack `X_live'` with ref → **MMD²** and **PO-risk** (`prediction_deltas_on_window`).

Demo: `python3 demo_fsds_token_perturbation.py` → `artifacts/token_perturbation_report.json`.

Enable in full modality report: `run_modality_attribution(..., include_token_perturbation=True)` → JSON key `token_perturbation`.
