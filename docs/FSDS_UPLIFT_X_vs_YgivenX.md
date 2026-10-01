# Two axes: shift on X vs shift on Y|X

Uplift monitoring hinges on **two independent questions**. AUUC moves under either; **which question is “yes”** changes the logic, which subset results you trust, and what ops should do.

## Axis A — Did X (covariate mixture) change?

**Tools:** MMD²(ref, live), domain RF AUC(X), LOGO-MMD on feature groups, domain-quintile scores \(\hat P(W{=}1\mid X)\).

**If yes:**

- Global AUUC_live can drop even when the **production τ̂** was never wrong on the old support—LIVE is a **different population**.
- Quintile AUUC and pairwise slices are **primary** (who to cap).
- LOGO-MMD (if groups exist) says **which feature blocks** moved \(P(X)\).
- LOCO–AUUC still runs but must pass **ESS overlap**; otherwise attributions confound mix shift with ranking.
- PO-risk may stay **non-significant** (pure covariate shift does not require Y|X to move).

**Business rules:** REALLOCATE (segments), recalibrate propensity, refresh/match REF—not immediate full τ relearn.

## Axis B — Did Y|X (outcome / concept) change?

**Tools:** PO-risk, DRPerm on batch W, PO-LOCO, optional Δμ on \(\mathbb E[Y\mid X]\).

**If yes (often with flat MMD on X):**

- **Mixture on X looks stable** but AUUC gap is large → ranking or CATE surface changed.
- **LOCO–AUUC first** (per-feature or groups): which inputs still drive τ ranking on LIVE.
- Quintiles still matter: **same X-mixture**, different τ ranking by slice.
- PO-LOCO: which features drive **outcome** shift vs **targeting** shift (LOCO).

**Business rules:** extend labels, RELEARN prod uplift after REALLOCATE trial; PO-risk confirms concept before expensive retrain.

**Slice localization under concept:** partition LIVE by domain quintiles; report empirical ATE per slice; **pairwise empirical ATE** (largest $|\Delta|$) identifies which subset gained vs lost **observed uplift today**—then REALLOCATE before full RELEARN. Pairwise AUUC on the same slices still describes **ranking** failure of the fixed REF model, not effect movement alone.

## How they combine (results interpretation)

| X shift | Y\|X shift | AUUC | Trust most | Misread if ignored |
|---------|------------|------|------------|-------------------|
| No | No | OK | Monitor only | — |
| No | Yes | Down | LOCO + PO-risk | Blaming “new users” |
| Yes | No | Down | Quintile + LOGO-MMD | Full τ relearn |
| Yes | Yes | Down | ESS → stratify → both LOCO and PO-LOCO | LOCO on unmatched REF/LIVE |

**Key insight:** AUUC is one alarm bell; **X tools and Y|X tools disambiguate**. Subset localization (LOCO vs quintile vs pairwise) sits **after** you know which axis fired—or explicitly under “mixed” with REF refresh first.

LaTeX: `docs/latex/uplift_fsds_subset_insights.tex` (2×2 table at top).
