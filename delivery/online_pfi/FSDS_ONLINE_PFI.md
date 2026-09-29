# Online PFI — where the logic lives

## Name map

| You said | In this repo |
|----------|----------------|
| **onlinePFI** / streaming attribution | `Python/fsds_sot/online_pfi.py` |
| **Batch PFI** (one window) | `Python/fsds_sot/attribution.py` → `permutation_importance` + RF domain AUC |
| **Panel 4** (single snapshot bar chart) | `Python/fsds_sot/viz.py` |
| **onlineRFPerm** (RAP re-route p-value) | `online_drift_detectors.py` on branch/commit `9a6149a` (clever-covariate RAP; **not** on SoT branch) |
| **Reference decay** (warm moving baseline) | `Python/fsds_sot/closed_loop.py` → `decay_reference` |

## Online PFI procedure

For each traffic window \(t = 1,2,\ldots\):

1. **Live batch** \(\mathcal{D}_t\) vs **reference** \(\mathcal{D}_{\mathrm{old}}\) (initial ref or decayed).
2. Fit RF \(W \sim X\) on stacked batches; **OOB permutation VIMP** = online PFI vector \(\phi_t\).
3. **MMD²**, domain **AUC**, overlap ESS (same as batch FSDS).
4. **C2ST p-value**: label-permutation on RF OOB accuracy (onlineRFPerm recipe) — drift trigger when \(p < \alpha\).
5. **Update** \(\mathcal{D}_{\mathrm{old}} \leftarrow \alpha \mathcal{D}_{\mathrm{old}} + (1-\alpha)\mathcal{D}_t\).

## Run visualization

```bash
cd Python
python3 demo_fsds_online_pfi_viz.py
```

Outputs:

- `artifacts/07_online_pfi_dashboard.png` — 4 panels: AUC, MMD², p-value stream, **PFI heatmap over time**
- `artifacts/online_pfi_summary.json`

## How this guides SoT / agents

- Rising **tool_fail** / **latency** rows in the heatmap → batch covariate shift on trace scalars.
- **emb_*** blocks shifting → embedding drift on retrieve/act (tie to Object 2 budget).
- **p-value** crossing 0.05 → closed-loop ladder in `closed_loop.py` (reallocate → re-topology → …).
