# Two layers: AUUC (ranking) vs uplift (effect)

**Scope:** both layers are defined for **two-batch** monitoring (REF vs LIVE). See [FSDS_UPLIFT_TWO_BATCH_LANDING.md](./FSDS_UPLIFT_TWO_BATCH_LANDING.md) for the full ops playbook.

Uplift monitoring uses **two layers**. They share the same LIVE slices (domain quintiles, optional overlap band) but answer **different questions**. Do not merge them into one metric.

## Layer 1 — AUUC movement (ranking / targeting quality)

**Object:** the **fixed REF meta-learner** \(\hat\tau(x)\). You never retrain on LIVE for this layer’s global read (LOCO is the exception: retrain on REF only after dropping a feature).

| Metric | What moved | Subset tools |
|--------|------------|--------------|
| \(\mathrm{AUUC}_{\mathrm{live}}\), \(\Delta_{\mathrm{AUUC}}=\mathrm{AUUC}_{\mathrm{ref}}-\mathrm{AUUC}_{\mathrm{live}}\) | **Overall**: “Is today’s traffic **ranked** well by yesterday’s \(\hat\tau\)?” | Global SLA alarm |
| LOCO `drop_live` | **Features**: “Which inputs does LIVE ranking **depend on**?” | Per-column or group drop → retrain on REF |
| Quintile \(\mathrm{AUUC}_k\) | **Users**: “On which slice is **sorting** bad?” | Domain quintiles on LIVE |
| Pairwise \(\mathrm{AUUC}_A-\mathrm{AUUC}_B\) | **Users A vs B**: “Where is the **rank gap** largest?” | Cap worse slice; keep better slice |

**AUUC reacts to:** covariate shift (support/ranking breaks), concept drift (true order changes but model unchanged), and plain model staleness. **It does not measure** the level of treatment effect—only whether **ordering users by \(\hat\tau\)** matches realized uplift curves.

**Use Layer 1 when:**

- Daily/hourly **monitor SLA** on the prod or shadow meta-learner.
- \(\Delta_{\mathrm{AUUC}}\) or \(\mathrm{AUUC}_{\mathrm{live}}\) fails threshold → always run **LOCO–AUUC**, then **quintile + pairwise AUUC**.
- **X shifted** (MMD/domain AUC ↑): Layer 1 is **primary** for ops (who to cap, overlap-only targeting). Effect layer is secondary until ESS OK.
- You suspect **targeting / ranker** problem: model might still be wrong on **who to treat first**, even if average effect is fine.

**Typical actions from Layer 1 only:** REALLOCATE by **rank** (cap low-AUUC quintile), HOLD_FEATURE (high LOCO), overlap-band targeting, recalibrate propensity—**before** full τ relearn.

---

## Layer 2 — Uplift movement (effect level)

**Object:** **how large** uplift is on each slice today—not whether \(\hat\tau\) sorts well.

Two complementary views (both on the **same quintiles**):

| Metric | Symbol | What moved | Needs model? |
|--------|--------|------------|--------------|
| **Empirical ATE** | \(\hat\tau^{\mathrm{obs}}_k = \bar Y_{T=1,k}-\bar Y_{T=0,k}\) on LIVE slice \(k\) | **Observed** treatment–control gap **today** | No (SRM + enough T=0/1 per slice) |
| **Mean estimated uplift** | \(\bar{\hat\tau}_k = \mathbb E[\hat\tau(X)\mid Q_k]\) on LIVE | What the **REF model says** uplift is on that slice | Yes (same \(\hat\tau\) as Layer 1) |

| Subset tool | Layer |
|-------------|-------|
| `concept_slice_uplift_live` | Per-quintile AUUC + \(\hat\tau^{\mathrm{obs}}\) + \(\bar{\hat\tau}\) |
| Pairwise on `empirical_ate` | **Effect**: who gained/lost **observed** uplift vs another slice |
| Pairwise on `mean_tau_hat` | **Model effect map**: where \(\hat\tau\) **level** differs across slices |

**Use Layer 2 when:**

- **PO-risk / DRPerm reject** (X stable, Y|X shifted) → **must** add Layer 2; pairwise **empirical ATE** answers “where did uplift jump today?”
- Global AUUC OK but **ops question is budget**: “Which segment actually earns more per treat?” → empirical ATE by quintile (Layer 2), not AUUC alone.
- **Compare model vs reality**: \(\bar{\hat\tau}_k\) vs \(\hat\tau^{\mathrm{obs}}_k\) by slice → **align** → REALLOCATE toward high observed uplift; **diverge** → RELEARN (model level wrong, not just order).
- After REALLOCATE trial: did **observed** uplift move in the slice you shifted? (Layer 2 follow-up)

**Do not use Layer 2 alone as the only alarm:** empirical ATE is noisy in small slices; it does not replace AUUC for **ranking SLA**. Always keep Layer 1 as the primary monitor; Layer 2 **explains and localizes effect** once Layer 1 (and FSDS/PO-risk) set the regime.

**Typical actions from Layer 2:** REALLOCATE budget toward high \(\hat\tau^{\mathrm{obs}}\) quintile (after SRM); RELEARN when PO-risk + systematic \(\bar{\hat\tau}\) vs \(\hat\tau^{\mathrm{obs}}\) mismatch.

---

## When to use what (decision table)

| Situation | Layer 1 (AUUC) | Layer 2 (uplift) | First read |
|-----------|----------------|------------------|------------|
| Routine monitor, all green | Global AUUC | Optional spot-check | Monitor |
| \(\Delta_{\mathrm{AUUC}}\) ↑, ESS OK, MMD ↑ | LOCO + quintile/pairwise **AUUC** | Optional \(\hat\tau^{\mathrm{obs}}\) (support may confound) | **Ranking on new support** → cap slices, refresh REF |
| \(\Delta_{\mathrm{AUUC}}\) ↑, ESS OK, MMD flat | LOCO + quintile/pairwise **AUUC** | Add \(\hat\tau^{\mathrm{obs}}\) + \(\bar{\hat\tau}\); run PO-risk | **Ranking and/or concept** → LOCO first, then PO-risk |
| PO-risk **reject**, MMD flat | Still run (ranker may be stale) | **Pairwise empirical ATE** + pairwise \(\bar{\hat\tau}\) | **Effect moved** → REALLOCATE; rank gap → pairwise AUUC |
| AUUC OK, PO-risk reject | Watch LOCO | **Layer 2 primary** for ops | **Silent concept**: effect moved, order still OK |
| ESS low | Defer LOCO interpretation | Defer slice ATE | **REFRESH_REF** |
| Overlap band AUUC ≫ global AUUC | Layer 1 | Layer 2 on overlap only | **Narrow targeting**, not relearn |

---

## One workflow (both layers)

1. **Gates:** SRM(T), MMD/domain AUC(X), ESS overlap.
2. **Layer 1:** \(\mathrm{AUUC}_{\mathrm{ref/live}}\), \(\Delta_{\mathrm{AUUC}}\) → if alert: LOCO–AUUC → quintile AUUC → pairwise AUUC.
3. **Regime:** X shift vs X stable (FSDS); if stable and gap large → PO-risk.
4. **Layer 2:** `slice_uplift_on_live` → pairwise empirical ATE; optional pairwise mean \(\hat\tau\).
5. **Act:** Layer 1 → rank-based REALLOCATE / HOLD_FEATURE; Layer 2 → effect-based REALLOCATE; PO-risk + mismatch → RELEARN.

Benchmark run (4 datasets, business rules JSON + narrative):

```bash
cd Python && python3 run_uplift_subset_benchmark.py
```

Outputs: `artifacts/uplift_two_layer_benchmark.json`, `artifacts/uplift_two_layer_business_insights.md`.

Code: `run_uplift_subset_localization` returns both layers in one dict (`auuc_*`, `pairwise_auuc_*`, `concept_slice_uplift_live`, `pairwise_empirical_ate`, `pairwise_mean_tau_by_slice`). Set `po_risk_reject=True` to emit REALLOCATE rules on the top ATE pair.

LaTeX: `docs/latex/uplift_fsds_subset_insights.tex`, subsection *Two monitoring layers*.
