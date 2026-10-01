# Uplift subset insights (no groups required)

**Two layers (read first):** [FSDS_UPLIFT_TWO_LAYERS.md](./FSDS_UPLIFT_TWO_LAYERS.md) — Layer 1 = AUUC/ranking; Layer 2 = empirical ATE + mean τ̂ (effect). Same quintiles, different questions.

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

## Concept drift (PO-risk): pairwise slice uplift strategy

**Trigger:** ESS OK, MMD/domain AUC flat on $X$, large $\Delta_{\mathrm{AUUC}}$, PO-risk / DRPerm **reject** on stacked REF∪LIVE.

**Two pairwises (different questions):**

| Layer | Metric | Question |
|-------|--------|----------|
| Ranking | Pairwise **AUUC** on quintiles | Where does **today’s ranker** sort badly? |
| Effect | Pairwise **empirical ATE** $\bar Y_{T=1}-\bar Y_{T=0}$ on LIVE quintiles | **Which user slice’s observed uplift moved today?** |

**Order:** confirm concept → LOCO–AUUC (features) → optional PO-LOCO (outcome drivers) → per-quintile AUUC + empirical ATE + mean $\hat\tau$ → sort pairs by $|\Delta\mathrm{ATE}|$ → **REALLOCATE** budget toward high-ATE slice; **RELEARN** only if mean $\hat\tau$ vs empirical ATE diverge across slices.

Implementation: `slice_uplift_on_live`, `pairwise_slice_uplift_compare(..., metric="empirical_ate")`; rules when `po_risk_reject=True` in `business_rules_from_localization`.

## Other subset tools (still post-hoc, no feature tree)

- **Overlap-band AUUC** — global red but band green → narrow targeting, not relearn.
- **LOGO-MMD** — only when **groups** exist and **$X$** shifted (which blocks moved $P(X)$).
- **Pairwise mean $\hat\tau$** — model-side slice contrast; sanity check vs empirical ATE under concept.
- **SRM on $T$**, bootstrap global AUUC — gates before any slice rule.

## Code

`run_uplift_subset_localization(..., groups=None)` → `pairwise_auuc_quintiles`, `loco_high_drop_live`, `subset_insight_regime`, `business_rules`.
