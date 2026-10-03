# Uplift monitor playbook: FSDS + LOCO–AUUC → actions when AUUC drops

**Two layers:** Layer 1 = AUUC / LOCO / pairwise AUUC (ranking). Layer 2 = empirical ATE + mean τ̂ per slice (effect). When to use which: [FSDS_UPLIFT_TWO_LAYERS.md](./FSDS_UPLIFT_TWO_LAYERS.md).

**English flow figure:** `artifacts/uplift_fsds_monitor_flow_en.png` (alias: `uplift_fsds_monitor_flow.png`).  
Two-batch REF/LIVE · FSDS gates (MMD, **domain RF AUC**, ESS) · post-hoc L1 AUUC · L2 uplift slices · PO-risk → compound business rules.  
Generate: `cd Python && python3 plot_uplift_fsds_monitor_flow.py`.

## Post-hoc subset localization

After global AUUC/MMD fire: **LOCO** localizes feature groups; **domain quintile AUUC** localizes which LIVE slice lost ranking; **group MMD** localizes which $X$ blocks shifted; **overlap-band AUUC** checks the comparable-support core. See `loco_auuc.posthoc_localization.posthoc_from_monitor`.

## SRM, ESS overlap, and domain RF AUC (support gate)

| Check | What it guards | Typical use |
|-------|----------------|-------------|
| **SRM on T** | Treatment ratio in REF train and LIVE matches design (e.g. 50/50) | **Before AUUC**: imbalanced T invalidates uplift curves; fix assignment/logging first. |
| **Domain RF AUC(X)** | Can a random forest separate REF vs LIVE using **X alone**? | Standard **batch FSDS** covariate signal (with MMD²). AUC ≈ 0.5 → batches look alike on X; AUC → 1 → strong shift (same story as high MMD / Online PFI domain AUC). |
| **ESS overlap** on \(\hat e(W{=}1\mid X)\) | Effective sample size after batch propensity weights | **Before LOCO–AUUC / PO-LOCO**: low ESS → ref and live sit on different X supports; attributions are unstable even if AUUC moves. |

You can gate Layer 2 attributions with **either or both**:

- **ESS overlap high** (e.g. above team threshold ~0.15–0.3 rule-of-thumb in `diagnose_shift`) → proceed with LOCO–AUUC.
- **Domain RF AUC(X) not extreme** (e.g. &lt; 0.95 after calibration) together with MMD → covariate shift is real but you may still compare rankings on overlap.
- **ESS low or domain AUC ≈ 1** → treat as **mix shift**: refresh REF, stratify, do **not** trust LOCO drops until the gate passes.

SRM is independent: pass SRM on **T** even when X overlap looks fine.

## How to run the monitor (one window)

```mermaid
flowchart TD
  A[REF + LIVE batches] --> B[FSDS on X: MMD, domain AUC, ESS_ovlp, SRM on T]
  B --> C{ESS_ovlp OK?}
  C -->|no| D[Refresh/match REF — defer LOCO interpretation]
  C -->|yes| E[Meta-learner LOCO–AUUC on REF train]
  E --> F[Optional: PO-risk + DRPerm on Y|X]
  F --> G[diagnose_shift + AUUC bootstrap CI]
  G --> H[Action ladder below]
```

**ESS overlap** (`overlap_ess_batch` in `run_uplift_monitor`): batch propensity \(\hat e(W{=}1\mid X)\) must have enough effective sample size. When ESS is **high** (e.g. Hillstrom ~0.98 in uplift overlap checks, ~0.69 in online PFI stream — always report both X-batch and T-overlap where relevant), LOCO–AUUC attributions are **credible**. When ESS is **low**, fix REF/LIVE composition before acting on LOCO ranks.

## When AUUC_live falls: adapt in order

General rule: **REALLOCATE targeting / rules before RELEARN** the full prod uplift model. Use LOCO–AUUC to see *which feature groups* matter; use PO-LOCO when PO-risk fires.

| Diagnosis (`monitor.diagnose_shift`) | Signals | First actions (hours–days) | If unresolved (days–weeks) |
|--------------------------------------|---------|----------------------------|----------------------------|
| **mix_shift_or_new_population** | MMD↑, **ESS_ovlp low** | Stop trusting LOCO; **refresh REF** or build matched REF; segment LIVE (new channel vs legacy) | Domain adaptation on X; block targeting on unsupported X until REF updated |
| **covariate_shift_ranking_degradation** | MMD↑, \|AUUC gap\|↑, **ESS_ovlp OK** | **Group LOCO–AUUC** → drop/ cap high-MMD blocks; **recalibrate** \(\hat e(T\mid X)\) in X-learner path; restrict scores to overlap support | Partial REF extension; stratified AUUC targets; avoid full τ relearn until PO-risk clear |
| **concept_drift_or_tau_change** | MMD flat, AUUC_ref & AUUC_live ↓ | Run **DRPerm / PO-risk**; **PO-LOCO** on outcome drivers; audit treatment policy & label delay | **Retrain τ** on extended REF+labels; prod relearn (DragonNet) after meta monitor confirms |
| **concept_drift_feature_reallocation** | LOCO **Spearman** ↓, modest MMD | Compare `drop_ref` vs `drop_live`; shift targeting weights to new LOCO-top groups; feature refresh | Retrain τ with new features; retire stale blocks |
| **stable** | — | Continue monitor; watch CI width on AUUC_live | — |

## AUUC-specific knobs (not generic retrain)

1. **Ranking-only fix (covariate path)**  
   - Re-score with updated propensity; use **percentile caps** on LIVE segments where MMD is largest.  
   - AUUC can recover without new labels if \(\hat\tau\) rank is still informative on the **overlap** subset — re-evaluate AUUC on trimmed LIVE.

2. **Label / surface fix (concept path)**  
   - PO-risk reject → outcome mechanism moved; **new Y** or relabeled window before τ relearn.  
   - LOCO–AUUC top groups differ from PO-LOCO top → split work: targeting features vs outcome features.

3. **Probe vs prod**  
   - Meta-learner monitor fires alert but prod AUUC OK → shadow mismatch; schedule **prod LOCO** once.  
   - Meta bad, prod bad → relearn prod; meta drives **which groups** to ablate in ablation test.

## Code entry

```bash
cd Python && python3 demo_loco_auuc_uplift_monitor.py
python3 plot_uplift_fsds_monitor_flow.py
```

`run_uplift_monitor(..., fit_propensity_for_overlap=...)` must be set so **ESS_ovlp** is populated.
