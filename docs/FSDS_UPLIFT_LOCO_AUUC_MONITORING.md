# Uplift model monitoring: LOCO × AUUC under batch shift

## Why not reuse batch FSDS alone?

Batch FSDS (MMD, domain AUC, overlap on **X**) answers: *did the input population change?*

An uplift/CATE model adds a second object: **ranking quality** — does \(\hat\tau(X)\) still put high-increment users first?

| Signal | What breaks | Typical cause |
|--------|-------------|---------------|
| MMD↑ on **X**, AUUC\_ref OK, **AUUC\_live↓**, overlap OK | **Ranking on LIVE** | Covariate shift: new regions/channels; propensity \(e(X)\) drift; support mismatch |
| MMD flat, **AUUC\_ref↓** and **AUUC\_live↓** | **True ranking** | Concept drift: \(P(Y\mid T,X)\) or CATE changed |
| LOCO drop profile **REF vs LIVE** (Spearman ↓) | **Which features drive targeting** | Concept or policy change; need Group LOCO |
| overlap\_ess ↓ | Attribution / LOCO unreliable | Mix shift — stratify ref before LOCO |

**LOCO-AUUC monitoring** = after each leave-one-feature (or group) out: **retrain \(\hat\tau\) on REF only**, then **recompute AUUC** on REF holdout **and** LIVE. No shortcut importance on a frozen model.

## Procedure (production window)

1. **SRM** on REF train and LIVE (treatment balance).
2. Train uplift learner on **REF train** (T-/X-/R-learner or Model Registry DR path).
3. **Baseline AUUC**: REF eval + LIVE (same frozen train set).
4. **LOCO loop** (parallel optional): for each feature or group \(g\):
   - Drop \(g\), retrain on REF train
   - Predict on REF eval + LIVE
   - `drop_ref = AUUC_full_ref − AUUC_loco_ref`
   - `drop_live = AUUC_full_live − AUUC_loco_live`
5. **Diagnostics**: MMD on X (ref vs live), batch overlap ESS, `auuc_gap = AUUC_ref − AUUC_live`, Spearman(`drop_ref`, `drop_live`).
6. **Act**: see diagnosis labels in `loco_auuc.monitor.diagnose_shift`.

## Link to FSDS-SoT stack

- **L0 batch FSDS** on trace/score features including \(\hat\tau\) summaries → early warning.
- **Group LOCO** mirrors **modality LOGO** (drop whole blocks).
- **Registry PO-risk** (`online_pfi_registry`) when \(Y\) is business outcome (conversion, SAR hit).
- **Closed-loop ladder**: REALLOCATE targeting rules before RELEARN CATE model.

## Code

```bash
cd Python && python3 demo_loco_auuc_uplift_monitor.py
```

Modules: `Python/loco_auuc/` — `metrics.auuc` (standard cumulative gain), `loco.loco_auuc_monitor`, `monitor.run_uplift_monitor`.

## AML / ops example

- **Treatment**: extra KYC step, model send-to-review, or incentive.
- **Outcome**: SAR flag, loss avoided (lagging).
- **Monitor**: weekly LIVE vs REF; if MMD↑ + AUUC\_gap↑ + overlap OK → recalibrate propensity / stratify; if LOCO profile flips → concept drift → retrain \(\tau\) with new labels, not only refresh rules.
