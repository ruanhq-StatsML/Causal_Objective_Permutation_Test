# Federated FSDS on feature blocks (communication-minimal)

Research note: connect **vertical / block-partitioned** features with the same **two-batch FSDS** contract (REF vs LIVE, batch label **W**, optional outcome **Y** for PO-risk). Goal: **attribute drift and decide retain/retire features** without centralizing raw **X**.

## 0. What this repo already is (without the word “federated”)

| Pattern | Module | Federated reading |
|---------|--------|-------------------|
| Feature held by “site” | `modality_dataloaders` + LOGO groups | Each modality / merchant block = **client** |
| Leave-one-block-out | `hierarchical_attribution`, `logo_mmd`, uplift `LOGO-MMD` | **No raw cross-block X**; only drop-block recomputed scores |
| Two-batch monitor | uplift `run_uplift_subset_localization`, `online_pfi_registry` | Same REF/LIVE **W**; **T** only inside client for uplift |
| PO-risk / concept | `DRPerm`, `prediction_deltas_on_window` | Needs **Y** on stack; can run **per block** + merge **p-values** |
| Online stream | `online_pfi.py` | OFS-like **when to refresh** reference; pairs with drift scores |
| FDR on trees | `multilevel_localization.py` (BB-FDR) | Not online FDR; different from alpha-wealth OFS |

**Gap:** no explicit **client/server protocol**, **drift-type label per feature/block**, or **online FDR budget** across clients. The prototype below adds a **minimal message schema** and **FedORA-style typing** on merged summaries.

---

## 1. Online feature selection (OFS) × FSDS — closed loop

**OFS constraint:** decide retain/retire as features arrive; no undo. **FSDS output:** ranked drift evidence (MMD/VIMP/LOCO/PO-LOCO), not a retain set.

**Interface (proposed):**

```text
  Client k:  local FSDS(ref_k, live_k) → scores s_j, drift_type_j
  Policy:    if s_j > τ_retire and type ∈ {covariate, compound} → mark "candidate retire"
             if type = concept only → mark "retrain driver", do not auto-drop X_j
  Server:    aggregate ranks; resolve cross-client same semantic feature via secure ID map (future)
```

**Evolutionary / OSSFS-style grouping:** treat each **block** as a “group arrival”; server runs **incremental LOGO** when a new block joins (only aggregated MMD deltas uploaded). Aligns with `run_two_layer_hierarchical` Layer-1 without concat raw data on server if each client sends **block MMD** and **block domain AUC**.

---

## 2. Online FDR and alpha-wealth (federated sketch)

**Constraint:** decisions at time **t** cannot use future hypotheses; **no retroactive** acceptance.

**Two-level wealth (your proposal, formalized):**

- **Local wealth** \(W_k(t)\): client **k** spends on **within-block** sequential tests (column order or stream order). Normalization (z-score, rank) stays local → **does not debit global wealth**.
- **Global wealth** \(W_G(t)\): server spends when **merging** cross-block hypotheses (same feature name, or block-level LOGO entries).

**Merge rule:** for block-level null “block **k** carries no shift”, combine local **C2ST p-values** with Fisher or Stouffer; apply **online FDR** (e.g. LORD, SAFFRON) on the **stream of merged block p-values** only.

**Open problem:** align **PO-risk p-values** (one per stack) with **per-feature** streams — use PO-risk as **global gate**; per-feature VIMP/LOCO as **local** ranking that does not consume \(W_G\) until promoted to cross-client claim.

---

## 3. Drift-type diagnosis (FedORA-style) on federated summaries

Per client **k**, on REF\(\cup\)LIVE with batch **W**:

| Signal | High? | Interpretation |
|--------|-------|----------------|
| MMD\(_k\), domain AUC\(_k\) on **X** | yes | **Feature / covariate drift** (block **k**) |
| PO-risk / DRPerm on **(X_k, Y)** stacked | yes, MMD\(_k\) low | **Concept** (outcome mechanism; may need other blocks’ **Y** only at server) |
| Both | yes | **Compound** |
| Label **Y** shift only | (future) | **Label drift** — not in current uplift code |

**Server decision tree (monitoring → aggregation policy):**

- Covariate-heavy → weight **LOGO-MMD** blocks; OFS **retire** only off-support columns; **RefreshRef** on overlap.
- Concept-heavy → run **PO-LOCO** locally where **Y** exists; global **retrain** ticket; do not blame block **k** with flat MMD\(_k\).
- Compound → **stratify REF** per block, then re-run local FSDS.

Prototype: `fsds_sot/federated_block_fsds.py` → `classify_block_drift_type`.

---

## 4. TFIDD-style dynamics (monitoring polish)

Embed on **local** score stream \(s_j(t)\):

- **EWMA:** \(\tilde s_j(t) = \lambda \tilde s_j(t-1) + (1-\lambda) s_j(t)\)
- **Freeze:** if bootstrap CI of \(\tilde s_j\) crosses 0, hold rank until CI clears
- **Adaptive top-k:** full refresh on top-k only; tail features every \(K\) windows

These are **Layer-2 monitoring** on attribution outputs; reuse `online_pfi` decayed reference as **t**.

---

## 5. Stability as health metric

- **Cross-fold Jaccard** on selected features (central) ↔ **cross-client Jaccard** on **binary “drift flagged”** for features registered in overlap catalog.
- **Cross-base attribution correlation:** for feature **j** observed at clients **k, k'**, low Spearman(\(s_{j,k}, s_{j,k'}\)) → **site-specific mechanism** or **biased sample** — surface as **MONITOR**, not auto-retire.

---

## 6. Economics (sketch)

- **Ranking quality:** \(\rho(\widehat R, R^\star)\) from synthetic DGP or logged A/B → map to \(\Delta\text{Precision} = g(\rho)\cdot \Delta\text{Precision}_{\max}\) (calibrate **g** offline).
- **Vertical complementarity:**  
  \(\mathrm{Comp}(k,k') = \frac{\mathrm{AUC}(F_k \cup F_{k'}) - \max(\mathrm{AUC}(F_k),\mathrm{AUC}(F_{k'}))}{\max(\cdot)}\)  
  requires **fed evaluation** or public validation set at server — FSDS ranks **which blocks to include** in the union.

---

## 7. Block granularity

Tradeoff: large **|F_k|** → stable local normalization, higher upload (top-k scores); small **|F_k|** → cheap comm, noisy MMD. Optimization:

\[
\min_{\{|F_k|\}} \mathbb E[\ell(R,R^\star)] \quad \text{s.t.} \quad \sum_k |F_k| = d,\ |F_k|\ge m_{\min}
\]

**m_min** from ESS overlap on local \(\hat e(W\mid X_k)\). Uplift monitor already exposes **ESS** gate — reuse per block.

**Cross-block redundancy:** after global top-k, run **fed correlation** protocol on pairs (k,k') for flagged columns only (not full **X**).

---

## 8. Priority roadmap (this repository)

| Priority | Deliverable | Status |
|----------|-------------|--------|
| **P0** | Block-local REF/LIVE FSDS + server merge + **drift type** | `federated_block_fsds.py` + demo |
| **P0** | Wire LOGO-MMD groups = clients in uplift subset run | reuse `logo_mmd_groups` |
| **P1** | Online FDR on merged block **p-value** stream | new module / SAFFRON stub |
| **P1** | OFS policy hook: FSDS score → retire candidate | config JSON on business rules |
| **P2** | \(\rho \to\) ROI calibration | simulation only |
| **P2** | Fed Comp(k,k') | needs eval harness |

---

## 9. Run the federated-block demo

```bash
cd Python && python3 demo_federated_block_fsds.py
```

Outputs: `artifacts/federated_block_fsds_report.json` — per-block local stats, server merged ranking, drift **type** per block, suggested **OFS/monitor** actions.

Communication envelope (per round, per client): `{mmd2, domain_auc, ess, n_ref, n_live, top_features[], po_pvalue optional}` — **O(p_k)** floats, not **O(n p)** rows.

## 10. Comprehensive LaTeX + five-dataset benchmark

```bash
cd Python && python3 demo_federated_protocol_datasets.py
cd docs/latex && pdflatex federated_fsds_comprehensive_standalone.tex
```

- **Full document (communication efficiency first):** `docs/latex/federated_fsds_comprehensive_standalone.tex`
- **Generated results section:** `docs/latex/federated_fsds_protocol_results.tex`
- **Methodology (static):** `docs/latex/federated_fsds_comprehensive_preamble.tex`
- **JSON:** `artifacts/federated_protocol_benchmark.json`
- **Datasets:** WDBC, Wine, Diabetes, Ionosphere, Heart Statlog (vertical blocks + injected LIVE shift)
