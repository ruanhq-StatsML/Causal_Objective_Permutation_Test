# Multimodal attribution — upstream dataloaders only

There is **no separate “multimodal FSDS math”**. Upstream you plug in **a few dataloaders** that each return a matrix `X` (n episodes × p_m). Downstream is the **same** ref vs live batch pipeline as everywhere else in this repo: MMD², domain AUC, permutation VIMP, overlap, two-layer LOGO, optional registry PO-risk and optional text token audit.

```text
  ref episodes ──┐
                 ├──► dataloader_text       ──► ModalityBatch("text", X, names)
  live episodes ─┤    dataloader_structured ──► ModalityBatch("structured", …)
                 │    dataloader_embedding  ──► ModalityBatch("embedding", …)
                 │
                 └──► concatenate_modalities (optional) ──► X_concat, slices
                              │
                              ▼
                 covariate_attribution(X_ref, X_live)   # batch FSDS
                 run_two_layer_hierarchical (LOGO + fine-grain)
```

## Upstream contract (all you customize)

Each loader returns `ModalityBatch`:

| Field | Meaning |
|-------|---------|
| `modality` | Label, e.g. `"text"`, `"structured"`, `"embedding"`, or your own `"tool_json"` |
| `X` | `(n_episodes, p_m)` float matrix for **one** batch (ref or live — call loaders per batch) |
| `feature_names` | Length `p_m`, for reports |

**Production**: replace demo loaders with your warehouse / feature store / encoder outputs. Keep episode alignment: row `i` is the same episode across modalities.

### Default three loaders (demo / agentic DGP)

| Loader | Source (demo) | Typical production |
|--------|---------------|-------------------|
| `text_dataloader` | `text_tokens` → Qwen hook or hash fallback | Episode text → your `encode_episode_texts` |
| `structured_dataloader` | `trace_features[-4:]` | tool_calls, tool_fail, latency, n_steps, … |
| `embedding_dataloader` | `branch_embeddings` flattened | Any fixed embedding per episode or per branch block |

`load_all_modalities(episodes)` returns a dict; `concatenate_modalities(...)` horizontal stack + `modality_slices` for LOGO.

Adding a fourth modality: implement `def my_loader(episodes) -> ModalityBatch` and pass it in your fork of `load_all_modalities` or build the dict manually before `run_modality_attribution`.

## Downstream (unchanged batch FSDS)

1. **Per-modality** ref vs live: MMD², domain AUC, VIMP mass (same as `covariate_attribution` on each block).
2. **Concat** ref/live: global attribution + slice VIMP back to modalities.
3. **Layer 1 LOGO** on concat: leave out whole groups (text / structured / embedding, or `emb_b*`, or text halves).
4. **Layer 2** within group: VIMP + MMD-LOCO on columns (fast path on global model slices).

Nothing else is required for “multimodal” beyond **building X per modality**.

## Layer 1 — LOGO (reminder)

On concatenated ref/live features, drop entire **modality blocks** and measure:

- `mmd_logo_delta = MMD²(full) − MMD²(without group)`
- `domain_logo_delta = AUC(full) − AUC(without group)`

Text-only track: sub-groups `text_low` / `text_high`. Embedding: optional `emb_b0`, … blocks.

## Layer 2 — within-group

Per group, rank features by concat **VIMP** and **MMD-LOCO**, or refit domain RF on the block only (`refit_layer2_blocks=True`).

## Optional (not part of core dataloaders)

| Add-on | When |
|--------|------|
| `include_registry_deltas=True` | You have outcome `Y` and want PO-risk / prediction delta on stacked X |
| `include_token_perturbation=True` | Text causal audit: perturb tokens → re-encode → MMD / PO-risk |
| Online PFI stream | Rolling ref/live on **same** trace matrices; see `docs/FSDS_ONLINE_PFI.md` |

## Entrypoint

```bash
cd Python && python3 demo_fsds_modality_attribution.py
```

## Modules

| File | Role |
|------|------|
| `fsds_sot/modality_dataloaders.py` | **Upstream only** — loaders + concat |
| `fsds_sot/modality_attribution.py` | Wire loaders → reports |
| `fsds_sot/hierarchical_attribution.py` | Two-layer LOGO |
| `fsds_sot/attribution.py` | `covariate_attribution`, LOGO deltas |

Report keys: `per_modality`, `global_concat`, `two_layer_concat`, `two_layer_text`, `two_layer_embedding_branches`, optional `registry_deltas`, `token_perturbation`.

## What this is *not*

- Not a new training loop for agents or SoT budgets (that is `FSDSSoT` / `pipeline.py`).
- Not LOCO-AUUC (uplift ranking — see `docs/FSDS_UPLIFT_LOCO_AUUC_MONITORING.md`).
- Not mandatory token perturbation or Qwen — structured-only attribution works with two loaders if you omit text.

## Token perturbation (text lane only)

See `token_perturbation_audit.py` + `demo_fsds_token_perturbation.py`. Requires your `qwen_episode_encoder.py` for production; hash fallback is demo-only.
