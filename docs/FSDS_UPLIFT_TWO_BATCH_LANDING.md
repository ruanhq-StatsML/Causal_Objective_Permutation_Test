# Two-batch landing playbook (REF vs LIVE)

This document is the **deployment story** when you commit to **exactly two batches**:

- **REF** — calibration window: fit τ̂, define “normal” ranking and (optionally) outcome baselines.
- **LIVE** — deployment window: score with **REF-frozen** τ̂, measure whether targeting still works **today**.

Everything in `loco_auuc` is built for this pair. Batch indicator **W** (0=REF, 1=LIVE) is **not** treatment **T**; uplift experiments still use **T** inside each batch.

**Why this scope is enough (and huge):**

- Same contract as **batch FSDS** on predictive models (MMD, domain AUC, ESS, permute tests) — ops already runs REF vs LIVE.
- **Two-layer localization** (AUUC vs uplift) both **define slices on LIVE** but use **REF∪LIVE** only where batch comparison is required (domain scores, PO-risk).
- Outputs are **plain business rules**, not a second serving stack — easy to wire to caps, holdouts, and relearn tickets.

**LaTeX (full formulation):** compile `docs/latex/uplift_fsds_two_batch_standalone.tex` (inputs `uplift_fsds_two_batch_formulation.tex` + benchmark + subset tables).

See also: [two layers](./FSDS_UPLIFT_TWO_LAYERS.md), [X vs Y|X](./FSDS_UPLIFT_X_vs_YgivenX.md), benchmark narratives in `artifacts/uplift_two_layer_business_insights.md`.

---

## 1. Batch roles and refresh policy

| Role | REF | LIVE |
|------|-----|------|
| **Train τ̂** | Yes (train split inside REF) | **No** (frozen REF model scores LIVE) |
| **AUUC eval** | Holdout inside REF (`AUUC_ref`) | Full LIVE window (`AUUC_live`) |
| **LOCO retrain** | Yes, per dropped feature/group | Never train on LIVE; only **score** LIVE |
| **SRM** | On **T** in REF train + LIVE | Same |
| **FSDS on X** | REF eval (or pool) vs LIVE | MMD², domain RF AUC |
| **PO-risk / DRPerm** | Stacked **REF∪LIVE**, **W** as batch | Same stack |
| **Subset quintiles** | Used to train **P(W=1\|X)** | **Slice LIVE rows** by live-likeness |

**REF refresh triggers (two-batch gates):**

- ESS overlap on **P(W=1|X)** below team floor → **REFRESH_REF** before LOCO-driven cuts.
- Domain AUC → 1 or sustained MMD↑ with new channel mix → extend or replace REF cohort.
- After successful **RELEARN**, new τ̂ **becomes** the next REF calibration (LIVE rolls into history).

**LIVE window:** one day, one campaign week, or one model-scoring batch — only changes **statistical power**, not the logic.

---

## 2. End-to-end monitor (one REF, one LIVE)

```text
  REF                          LIVE
  ├─ train τ̂                   ├─ score τ̂ (REF weights)
  ├─ eval AUUC_ref              ├─ AUUC_live, ΔAUUC
  └─ LOCO: retrain on REF only  └─ LOCO: drop_* on LIVE scores

        └────────── FSDS: MMD(X_ref, X_live), domain AUC, ESS(W|X) ──────────┘
        └────────── PO-risk: DRPerm on (X,Y) with W=batch ───────────────────┘
        └────────── Subset: quintiles on LIVE via P(W=1|X) trained on REF∪LIVE ─┘
```

**Order (do not skip):**

1. **Validity** — SRM(T) on REF and LIVE; ESS overlap; MMD² + domain AUC on **X**.
2. **Layer 1 global** — `ΔAUUC = AUUC_ref − AUUC_live`; bootstrap CI if needed.
3. **Layer 1 localize** — LOCO–AUUC (`drop_live`, `drop_ref`, `drop_gap`); domain-quintile AUUC on LIVE; pairwise AUUC; optional overlap-band AUUC on LIVE.
4. **Regime** — X shifted vs X stable (FSDS); if stable and gap large → PO-risk.
5. **Layer 2 localize** — per-quintile empirical ATE and mean τ̂ on LIVE; pairwise ATE (and mean τ̂); **REALLOCATE on ATE only if PO-risk reject**.
6. **Business rules** — REALLOCATE / HOLD_FEATURE / REFRESH_REF / RELEARN / MONITOR.

Code: `run_uplift_monitor` → `run_uplift_subset_localization`; batch demo `run_uplift_subset_benchmark.py`.

---

## 3. Two-batch localization map (where each tool lives)

### 3.1 Feature axis — LOCO–AUUC (cross-batch by design)

For each dropped feature/group **g**:

- Retrain τ̂ on **REF train** without **g**.
- AUUC on **REF holdout** → `drop_ref`.
- Same model scores **LIVE** → `drop_live`.

| Pattern | Read | Two-batch action |
|---------|------|------------------|
| `drop_live` large, `drop_ref` small | Ranking on LIVE depends on **g**; REF eval OK | Cap LIVE segments where **g** is extreme; do not drop **g** from prod rules without replacement |
| `drop_ref ≈ drop_live` | **g** structurally important for rank | HOLD_FEATURE; ablation in REF before prod change |
| `drop_gap` large (Spearman LOCO REF vs LIVE low) | **Which features drive rank changed** between batches | Concept / reallocation → LOCO first, then PO-risk; consider RELEARN |
| All drops tiny, `ΔAUUC` large | Global rank break not tied to single **g** | Quintile AUUC; overlap band; not single-feature blame |

LOCO answers: **“Under REF calibration, which inputs explain LIVE ranking failure?”**

### 3.2 User axis — domain quintiles (cross-batch slice definition, LIVE metrics)

Train **ŝ(x) = P(W=1|X)** on **REF pool ∪ LIVE** (RF on **X**, label **W**).

Partition **LIVE only** into quintiles Q1–Q5 by **ŝ** (low → “REF-like”, high → “LIVE-like / shift tail”).

On each **Q_k ⊆ LIVE**:

| Layer | Metric | Two-batch question |
|-------|--------|-------------------|
| L1 | `AUUC_k` | Where does **REF τ̂** sort badly among users who **look most like LIVE** (or REF-like core)? |
| L1 | pairwise `AUUC_a − AUUC_b` | Which **pair of LIVE slices** should we cap vs continue? |
| L2 | `empirical ATE_k` on LIVE | Where did **observed** treat−control move **in today’s LIVE traffic**? |
| L2 | `mean τ̂_k` on LIVE | Where does **REF model** think uplift is high vs low on that slice? |

**Covariate shift:** AUUC often collapses in **high-ŝ** quintiles first (LIVE-like tail). Action: **REALLOCATE** — cap high-ŝ low-AUUC slices; use **overlap-band AUUC** if global red but core green.

**Concept (PO-risk reject, MMD flat):** quintiles still on LIVE; **pairwise ATE** names budget moves; check **τ̂ vs ATE** alignment for RELEARN vs REALLOCATE.

Quintiles answer: **“On LIVE, which users (relative to REF) drive the alert?”**

### 3.3 Overlap band (two-batch support)

**ê(W|X)** on LIVE from a classifier trained on REF∪LIVE.

AUUC on LIVE rows with **ê ∈ [0.15, 0.85]** (comparable batch support).

If **AUUC_overlap ≫ AUUC_global** → global ΔAUUC is partly **mix/support**; **narrow targeting** to overlap, refresh REF, defer full RELEARN.

---

## 4. Two axes × two batches (decision spine)

| X (REF vs LIVE on **X**) | Y\|X (PO-risk on stack) | Primary two-batch tools | First ops move |
|--------------------------|-------------------------|-------------------------|----------------|
| Stable | Stable | Monitor `ΔAUUC` | None |
| Stable | Reject | LOCO + L2 ATE pairwise | REALLOCATE (L1+L2); RELEARN if τ̂≠ATE |
| Shift | Stable | LOGO-MMD (if groups), quintile AUUC, overlap | REALLOCATE / REFRESH_REF; recalibrate e(T\|X) |
| Shift | Reject | ESS → stratify → LOCO + quintile + PO-LOCO | REFRESH_REF or match; then relearn |
| ESS low | any | Gates only | **REFRESH_REF** — no LOCO/ATE policy |

AUUC alone cannot split the first column; **FSDS + PO-risk** do. Subset tools run **after** gates.

---

## 5. Landing scenarios (product-shaped)

### A. Daily uplift SLA (most common)

- **REF:** last 14–28 days stable traffic used to train monitor meta-learner (prod may be DragonNet; monitor can be X-learner).
- **LIVE:** yesterday or rolling 24h.
- **Alert:** `AUUC_live` or `ΔAUUC` vs SLA.
- **Run:** full pipeline §2; emit `business_rules` JSON to policy engine.
- **Default action ladder:** REALLOCATE (quintile) → HOLD_FEATURE (LOCO) → RELEARN ticket if diagnosis `concept_drift_or_tau_change`.

### B. New channel / campaign launch

- Expect **X shift** (MMD↑, domain AUC↑).
- **Trust:** quintile AUUC + overlap band; **skeptical:** global LOCO until ESS OK.
- **Do not** full RELEARN day one — **cap high-ŝ segments**, expand REF with new channel once stable.

### C. Promo or label lag (concept)

- **X flat**, **ΔAUUC**↑, **PO-risk reject**.
- **L1:** cap low-AUUC quintile; **L2:** shift spend to high **empirical ATE** quintile (same day holdout rules).
- **RELEARN** after REALLOCATE trial if τ̂ vs ATE still diverge.

### D. Shadow monitor vs prod

- **REF** fixed; **LIVE** same window; meta-learner fires, prod AUUC OK → shadow mismatch; run **LOCO once on prod features**.
- Both bad → prod relearn; meta drives **which groups to ablate** in experiment.

### E. Rolling LIVE as next REF

- When monitor **stable** N days, append LIVE to REF pool and slide window — still **two-batch** at each daily step; avoids “single-day LOCO only” confusion (train always multi-day REF).

---

## 6. What makes two-batch localization “wide” but coherent

1. **One frozen ranker, two eval surfaces** — REF holdout sanity + LIVE deployment truth (`ΔAUUC`).
2. **Same LIVE slices, two layers** — pairwise **AUUC** (sort) vs pairwise **ATE** (level); PO-risk gates aggressive L2 moves.
3. **Feature vs user** — LOCO (cross-batch feature) + quintiles (cross-batch-defined user slices on LIVE).
4. **Same stack as FSDS** — batch **W** for domain/PO-risk; treatment **T** for uplift curves inside LIVE.
5. **No hierarchical tree** — flat LOCO, flat quintiles, pairwise tables → **direct caps and budgets**.

**Explicit non-goals in two-batch v1:** intra-LIVE-only τ̂ deciles without REF (valid for intraday ops, different semantics); 3+ batch tensor (extend by chaining REF←LIVE₁, LIVE₁←LIVE₂).

---

## 7. Minimal contract for eng / policy

**Inputs per run:**

- `X_ref, t_ref, y_ref`, `X_live, t_live, y_live`
- Optional feature **groups** for LOGO-MMD / group LOCO
- Thresholds: ESS floor, MMD high, AUUC SLA, PO α=0.05

**Outputs:**

- Gates: SRM, MMD², domain AUC, ESS, PO p-value
- L1: global AUUC, LOCO rows, quintile AUUC, top pairwise AUUC
- L2: slice table (AUUC, ATE, mean τ̂), top pairwise ATE
- `business_rules[]`: `{ action, priority, rule, evidence }`

**Run command:**

```bash
cd Python && python3 run_uplift_subset_benchmark.py   # four canonical scenarios
# or integrate run_uplift_subset_localization(...) in your daily job
```

---

## 8. Reading benchmark runs (two-batch sanity)

| Dataset | Two-batch story |
|---------|-----------------|
| `hillstrom` | Natural REF/LIVE split; L1 quintile cap (Q3 vs Q4); L2 ATE high on Q3 but AUUC worst → **do not treat ATE alone** |
| `hillstrom_covariate_drift` | Injected X drift on LIVE; global AUUC red; tune MMD threshold or LOGO for prod |
| `synthetic_covariate` | Overlap AUUC > global → **narrow targeting** |
| `synthetic_concept` | PO reject; L1 and L2 **align** on Q2 → strong REALLOCATE + relearn path |

Full text: `artifacts/uplift_two_layer_business_insights.md`.
