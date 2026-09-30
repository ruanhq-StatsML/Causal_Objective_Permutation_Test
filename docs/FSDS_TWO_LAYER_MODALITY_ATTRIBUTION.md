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
