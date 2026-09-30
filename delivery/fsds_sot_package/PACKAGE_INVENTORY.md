# FSDS-SoT delivery — inventory & provenance

This zip bundles **`Python/fsds_sot/`** plus the **minimum repo-root glue** needed to run the **multimodal ref/live attribution** demos. Much of the stack is **extracted or aligned** with the upstream project [Causal_Objective_Permutation_Test](https://github.com/ruanhq-StatsML/Causal_Objective_Permutation_Test) (PO-risk / DRPerm / Model Registry); the **SoT agent application layer** and **modality delivery** were added on branch `cursor/fsds-sot-tot-manuscript-7451`.

---

## 1. What is in `fsds_sot_delivery.zip`

| Path | Role |
|------|------|
| `README.md` | Quick start (copy of `README_PACKAGE.md`) |
| `Python/fsds_sot/*` | Full `fsds_sot` package (28 `.py` files) |
| `Python/demo_fsds_modality_attribution.py` | **Primary demo** — 3 dataloaders, two-layer LOGO, token audit, JSON report |
| `Python/demo_fsds_token_perturbation.py` | Token mask/replace → MMD + PO-risk only |
| `Python/DRPerm.py` | **Upstream repo** — DRPerm / PO-risk permute-refit test |
| `Python/model_registry_class.py` | **Upstream repo** — RF/XGB/MLP registry for nuisances & τ |
| `docs/FSDS_TWO_LAYER_MODALITY_ATTRIBUTION.md` | Two-layer + token perturb notes |
| `Python/feature_store/*` | **Feature-store use case** (from merchant prototype branch): user/order/merchant layers |
| `Python/demo_fsds_feature_store_attribution.py` | Export CSV dataset + LOGO-MMD + multilevel tree + cross-level L1/L2 |
| `data/feature_store/*.csv` | Bundled relational tables + merchant/user feature matrices |
| `Python/feature_store/LAYERS.md` | Schema: user → order → item → merchant |
| `artifacts/` | Empty placeholder for demo JSON output |

**Not in zip (same GitHub repo, other folders):** `Python/R_risk_loco.py`, `Python/RRPerm.py`, `R/*`, remaining merchant-branch scripts (`fsds_metric_graph`, …), Online PFI under `delivery/online_pfi/`.

---

## 2. `fsds_sot` module map (by layer)

### A. Multimodal attribution delivery (this thread — **new**)

| Module | Responsibility |
|--------|----------------|
| `modality_dataloaders.py` | Upstream loaders: **text** / **structured** / **embedding** → `concatenate_modalities` + column slices |
| `modality_attribution.py` | **`run_modality_attribution`**: ref vs live, per-modality, concat global, fine-grain, optional Registry + token audit |
| `hierarchical_attribution.py` | **Layer 1** group LOGO (modality / branch blocks); **Layer 2** feature VIMP + MMD-LOCO within group |
| `text_tokens.py` | Episode → token sequence; **mask / replace / shuffle / drop**; prefix groups |
| `text_encoder.py` | Pluggable encode: **`qwen_episode_encoder.encode_episode_texts`** (your script); hash = demo fallback only |
| `qwen_episode_encoder.py.example` | **You copy → `qwen_episode_encoder.py`** and paste your Qwen embedding code |
| `token_perturbation_audit.py` | Perturb live tokens → re-encode → **MMD²** + **PO-risk** (`prediction_deltas_on_window`) |

### B. Distribution-shift core (adapted from **upstream repo** patterns)

| Module | Responsibility | Upstream analogue |
|--------|----------------|-------------------|
| `attribution.py` | Ref/live domain RF, permutation VIMP, MMD², feature LOCO, **group LOGO** | Same spirit as covariate-shift / OOD-VIMP in paper repo; MMD RBF as in FSDS writeups |
| `online_pfi_registry.py` | Cross-fit μ̂, ê → τ̂ → **prediction deltas** + optional **`DRPerm`** stream | Wraps `DRPerm.py` + `model_registry_class.py` |
| `online_pfi.py` | Lightweight rolling **RF-PFI** (sklearn shortcut, not Registry) | B-track dashboard; optional for monitoring |
| `plot_online_pfi.py` | Online PFI heatmap / dashboard | Delivery with Online PFI pack |

### C. Feature-store multi-layer attribution (**extracted** from `cursor/fsds-merchant-prototype-1905`)

| Module | Responsibility |
|--------|----------------|
| `feature_store/merchant_prototype.py` | Relational gen; **merchant** + **user** rollups; PO-risk + LOCO (concept drift path) |
| `feature_store/logo_mmd.py` | **LOGO-MMD** on merchant grain (covariate shift P(X)) |
| `feature_store/multilevel_localization.py` | **L1** raw attr → **L2** agg family → **L3** leaf (BB-FDR tree) |
| `feature_store/cross_level.py` | **L1** merchant attribute + **L2** product vertical (conditional) |

Run: `python3 demo_fsds_feature_store_attribution.py` → `delivery/data/feature_store/*.csv` + `artifacts/feature_store/feature_store_attribution_report.json`.

### D. SoT / agent scheduling (application — **new on SoT branch**)

| Module | Responsibility |
|--------|----------------|
| `agentic_dgp.py` | Synthetic ReAct-style episodes (ref vs live drift on retrieve/act) |
| `pipeline.py` | `FSDSSoT`: covariate attribution + decomposability + budget plan |
| `decompose.py` | Embedding topology / coupling for Object 1 |
| `budget.py`, `entropy_alloc.py`, `entropy_reg.py` | Object 2 token/tier/check budget |
| `economics.py`, `pricing.py` | $/episode, ROI alignment |
| `incremental_value.py`, `pareto_poc.py`, `plan_metrics.py` | Boss / Pareto demos |
| `mcts_search.py`, `mcts_local.py` | Budget search for success |
| `closed_loop.py` | Measure → act → re-measure, reference decay |
| `applications.py` | ReAct / RAG / self-consistency trace builders |
| `viz.py` | SoT four-panel figures |

---

## 3. Provenance (extract vs new)

| Source | What you get | Where it lives in this package |
|--------|----------------|--------------------------------|
| **This repo (canonical)** | `DRPerm`, Model Registry, R/RRPerm theory | `Python/DRPerm.py`, `Python/model_registry_class.py`; called from `online_pfi_registry.py` |
| **This repo — SoT branch** | Agentic DGP, SoT pipeline, online PFI, closed loop | `fsds_sot/*` except modality block |
| **This repo — SoT branch (modality delivery)** | 3 dataloaders, two-layer LOGO, token perturb audit | §2A modules + demos |
| **Branch `cursor/fsds-merchant-prototype-1905`** | Merchant rollup, LOGO-MMD, multilevel tree, cross-level | **Vendored** into `Python/feature_store/` + CSV dataset in zip |
| **Your external script** | Qwen (or other) text embeddings | `qwen_episode_encoder.py` (you add; not in public repo) |

---

## 4. Demos in repo but **outside** this zip

Still on the same branch; add to zip manually if needed:

| Demo | Focus |
|------|--------|
| `demo_fsds_sot.py`, `demo_fsds_sot_viz.py` | Core SoT plan + viz |
| `demo_fsds_online_pfi_viz.py`, `demo_fsds_online_pfi_registry.py` | Online PFI B / A track |
| `demo_fsds_boss_six_points.py` | Six-point boss metrics |
| `demo_fsds_closed_loop.py`, `demo_fsds_react_agent.py`, … | Agent pattern loops |

---

## 5. JSON outputs (modality demo)

| File | Main keys |
|------|-----------|
| `artifacts/modality_attribution_report.json` | `two_layer_concat`, `two_layer_text`, `fine_grain`, `token_perturbation`, `registry_deltas`, … |
| `artifacts/token_perturbation_report.json` | `baseline`, `global_perturbations`, `token_position_loco`, `prefix_group_perturbations` |

---

## 6. Dependencies

```text
numpy, scikit-learn, xgboost, pandas   # DRPerm / Registry
# optional: torch, transformers, sentence-transformers  # your Qwen encoder
```

---

## 7. Rebuild zip

From repo root:

```bash
bash delivery/build_fsds_sot_zip.sh
```

Output: `delivery/fsds_sot_delivery.zip`.

---

## 8. Suggested README sections (for you)

1. **Problem** — ref vs live agent traces; multimodal features.  
2. **Stack** — upstream PO-risk (DRPerm) + FSDS attribution (MMD/VIMP/LOGO) + your Qwen text encoder.  
3. **Two-layer attribution** — modality LOGO → in-modality features.  
4. **Token perturb** — counterfactual masks → re-embed → MMD / PO-risk.  
5. **What we extracted** — table from §3 above.  
6. **Run** — §1 demos; point to zip on GitHub branch.
