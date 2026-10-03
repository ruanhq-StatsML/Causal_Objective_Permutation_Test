# Uplift monitoring + adaptation + joint refit (unified framework)

Bridges **two-batch FSDS monitor** (`run_uplift_subset_localization`, `business_rules_from_localization`) with **online joint refit** (temporal split of e, μ₀, μ₁, τ) and streaming constraints.

---

## 0. Two layers (do not merge)

| Layer | Latency | Frozen? | Purpose |
|-------|---------|---------|---------|
| **A. Monitor + policy adapt** | hours–days | **τ̂ frozen** on LIVE trial | AUUC SLA, caps, narrow support, REF refresh |
| **B. Joint refit (model)** | days–weeks | new e, μ, τ after holdout | Multi-plane drift, post-trial failure |

**Principle:** Layer A exhausts **where to treat** before Layer B changes **how to score**. Economic: OPEX (caps) before CAPEX (retrain).

---

## 1. Layer A — Two-batch monitor (what we implemented)

**Contract:** REF calibrates; LIVE deploys; batch **W** for FSDS/PO-risk; treatment **T** for uplift curves.

**Pipeline:** Gates (SRM, MMD², domain AUC, ESS on ê(W|X)) → L1 τ ranking → L2 level → DRPerm → `business_rules` sorted P1→P5.

**After localization, strategy sort:**

1. **Phase A:** ESS low → REFRESH_REF only.  
2. **Phase B:** `diagnose_shift.label` picks tables (LOGO/quintile/overlap vs LOCO/PO).  
3. **Phase C:** P1 caps + narrow support ∥ P2 segment → P3 overlap → L2 if PO reject → RELEARN ticket deferred.

**MMD vs AUUC_overlap → RELEARN?**

- MMD high, **AUUC_ovlp ≫ global** → narrow + cap + REF; **no** immediate joint refit.  
- MMD low, **both AUUC bad**, **PO reject**, τ̂ ≠ ATE after trial → escalate to **Layer B**.

See PNG: `artifacts/uplift_fsds_localization_adapt_decision_en.png`.

---

## 2. Layer B — When joint refit (aligned with your triggers)

### 2.1 Joint refit **YES** (any + trial failure where applicable)

| # | Condition | Monitor analogue |
|---|-----------|------------------|
| 1 | **PSI(e), PSI(μ₀), PSI(μ₁)** all high | Batch: MMD↑ + domain AUC↑ + ESS OK; treat as **multi-plane covariate + outcome**; not fixed by cap alone |
| 2 | **Local adapt (1–3) done, AUUC still fails** on **deployed support** | `AUUC_ovlp` and post-cap quintile still < SLA after 1–2 LIVE windows |
| 3 | **Structural break** (new market / category / cohort) | `mix_shift_or_new_population` or sustained domain AUC→1; REFRESH_REF insufficient |
| 4 | **Feedback loop pollution** (policy→e→pseudo→τ) | Monitor **e(T|X) extremity**, ESS on T overlap; pause adapt; need joint refit + exploration holdout |
| 5 | **τ independent drift** (sync test fails) | MMD flat, PO reject, LOCO ρ collapse, τ̂ vs ATE mismatch → not fixed by recalibrate e alone |

### 2.2 Joint refit **NO** (use Layer A only)

| # | Condition | Layer A action |
|---|-----------|----------------|
| 1 | **Single-plane drift** | e-only → recalibrate **ê(T|X)**; μ-only → refresh labels / μ heads; ranking-only → caps / narrow support |
| 2 | **Pure mixture shift** | Threshold / band / REF match; **AUUC_ovlp green** |
| 3 | **Insufficient n** (e.g. n_t, n_c < 1k per window) | Online summary + conservative treat; **wait** |
| 4 | **Gradual drift, τ synced with e/μ** | Fix e/μ first; keep τ frozen through monitor trial |

---

## 3. Joint refit procedure (temporal split — online safe)

**No random K-fold on streams.** Use **past window → nuisances**, **current window → pseudo → τ**.

```
past   = [-2w : -w)     # train e, μ₀, μ₁ (optional exp decay on past only)
current = [-w : now)    # predict e, μ on current; build DR pseudo; train τ
holdout = independent  # never used in fit; DR-AUUC compare old vs new
```

**Deploy gate:** `AUUC_new > AUUC_old + δ` on holdout **and** bin/subgroup checks (calibration, top-k overlap).

**Sample gates:** `check_sample_size` per T and key subgroups before fit.

---

## 4. Streaming reality: feature replay

**Problem:** Hard time-split needs **X at t−2w..t**; pure streaming without storage cannot joint-refit.

| Architecture | Monitor (Layer A) | Joint refit (Layer B) |
|--------------|-------------------|------------------------|
| **Feature snapshot** (warehouse / feature store) | REF/LIVE batches from snapshots | Full temporal split OK |
| **Summary only** (Welford, t-digest, online PSI/MMD) | Drift **alarms** only | Refit **blocked** until backfill |
| **Hybrid** (stream → Snowpipe → raw + online FS) | Daily two-batch on slices | Weekly joint refit from raw |

**Two-batch monitor** maps naturally to **rolling windows**: each day REF = last 14–28d, LIVE = last 24h — still Layer A; joint refit uses **deeper history** from warehouse.

---

## 5. τ vs e/μ synchronization (when to refit what)

| Sync read | Signal | Action |
|-----------|--------|--------|
| **e-driven** | Δe large, R-learner / pseudo stable, ranking fix after recalibrate e | Refit **e** only; monitor trial |
| **μ-driven** | Outcome shift, PO-risk, flat τ rank after cap | Refresh **labels / μ**; optional μ heads |
| **τ independent** | Δτ large, Δe/Δμ small, PO reject, τ̂≠ATE | **Joint refit** or τ-only after e/μ frozen check |
| **Desync multi-plane** | PSI(e), PSI(μ₀), PSI(μ₁) high | **Full joint refit** |

**Batch monitor proxies:** PO-risk (Y|X batch), LOCO Spearman (ranking drivers moved), `prediction_deltas_on_window` (Δμ, Δτ_po).

---

## 6. Periodic drift (fixed period)

**Do not** use heavy exp decay on τ if **dow/hour** seasonality dominates.

- Add **calendar features** (sin/cos dow, hour) or **phase-aligned windows** (train on same weekday history).  
- **STL** on AUUC/τ series: adapt **trend**; treat seasonal as stable.  
- Layer A: compare **same phase** REF vs LIVE (e.g. Mon REF vs Mon LIVE) before RELEARN.

---

## 7. Cross-fitting substitutes (online)

| Method | When | vs two-batch FSDS |
|--------|------|-------------------|
| Temporal split | Features persisted | Joint refit |
| Exp-decay single pool | Soft past/current | Approximate; tune on holdout |
| Frozen e/μ, partial_fit τ | e/μ slow drift | Between daily monitor and full joint |
| Online ensemble / bandit | Fast concept drift | Shadow alongside monitor meta-learner |
| Conformal on pseudo | Uncertainty guard | Gate deploy if interval blows up |

**DRPerm on stacked REF∪LIVE** remains batch **concept test**; not a substitute for joint refit, but **gates L2** and **RELEARN ticket**.

---

## 8. Master decision table (merged)

| Scenario | Time decay | Time split | e handling | Layer A | Layer B joint refit |
|----------|------------|------------|------------|---------|---------------------|
| Gradual drift, big n | optional | if stored | refit e if PSI(e) | caps / narrow | if trial fails |
| Abrupt shift, big n | no | yes | refit e | REFRESH + cap | **yes** if multi PSI |
| Small n | no | yes | clip e stable | conservative | **no** |
| Periodic fixed | no | phase-aligned | calendar feats | phase-wise monitor | rare |
| Pure e drift | maybe | yes | **refit e** | recalibrate | **no** |
| Pure μ drift | maybe | yes | freeze e | REF / labels | **no** (μ refresh) |
| Pure τ drift | yes | yes | freeze e | LOCO trial | τ-only or joint if PO+ |
| Multi-plane | yes | yes | refit e | full L1/L2 | **yes** |
| Overlap collapse | no | yes | entropy balance / match | REFRESH_REF | **no** first |
| Feedback pollution | no | yes | + exploration | **pause** adapt | **yes** after clean window |

---

## 9. Other essentials (easy to miss)

1. **Exploration traffic** (5–10%) — without it, e collapses and pseudo-outcomes lie.  
2. **Latency tiers:** detect (s) → policy adapt (h) → e/μ refresh (h–d) → joint refit (d).  
3. **Versioning:** shadow → canary → promote; rollback tied to monitor SLA.  
4. **Feature parity:** train/serve same defs; monitor missingness.  
5. **Federated blocks:** same REF/LIVE on **summaries**; refit still local with snapshot policy.

---

## 10. Code / artifact map

| Piece | Path |
|-------|------|
| Subset localization + rules | `Python/loco_auuc/subset_localization.py` |
| Shift diagnosis | `Python/loco_auuc/monitor.py` → `diagnose_shift` |
| PO-risk | `Python/DRPerm.py` |
| Decision PNG | `artifacts/uplift_fsds_localization_adapt_decision_en.png` |
| Adapt ladder PNG | `artifacts/uplift_fsds_adapt_economics_methodology_en.png` |
| Playbook | `docs/FSDS_UPLIFT_TWO_BATCH_LANDING.md`, `FSDS_UPLIFT_MONITOR_ACTIONS.md` |

**Joint refit** (`joint_refit(...)` in user spec) is **Layer B** — implement adjacent to model registry; **entry** only when Section 2.1 satisfied and holdout passes Section 3.
