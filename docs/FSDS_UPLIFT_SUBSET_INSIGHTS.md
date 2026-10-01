# Uplift subset insights (no groups required)

**Question:** Global AUUC dropped — **on which subset?**

## Two intuitive slice errors (ranking layer)

| Type | What it is | Read when |
|------|------------|-----------|
| **LOCO–AUUC high `drop_live`** | Remove feature group (or single feature if no groups), retrain on REF, AUUC on LIVE falls | **Which inputs** drive ranking on LIVE |
| **Domain-quintile AUUC** | AUUC on LIVE slices by $P(W{=}1\mid X)$ | **Which users / region of $X$** drive the global AUUC move |

Optional third: **overlap-band AUUC** vs global — comparable-support core only.

**Pairwise AUUC:** compare quintile A vs B (or overlap vs global); largest $|\Delta|$ → direct cap/continue rules.

Groups are optional: without groups, LOCO runs per-feature; LOGO-MMD is skipped; quintile + pairwise still run.

## Regimes (where to look first)

| Pattern | Meaning | Subset tools |
|---------|---------|--------------|
| **$X$ shifts, gap large** | Covariate / mixture on inputs | LOGO-MMD (if groups), quintile AUUC, pairwise |
| **$X$ stable, gap large** | User **mixture on $X$** similar but ranking/concept changed | LOCO–AUUC, PO-risk; quintiles still show *who* hurts if scores drift |
| **ESS low** | Not comparable REF/LIVE | Refresh REF before any subset rule |

“$X$ stable + gap large” is when **mixture on covariates looks flat** but **treatment effect ranking** broke — LOCO names features, PO-risk names $Y\mid X$, quintiles name slices.

## Code

`run_uplift_subset_localization(..., groups=None)` → `pairwise_auuc_quintiles`, `loco_high_drop_live`, `subset_insight_regime`, `business_rules`.
