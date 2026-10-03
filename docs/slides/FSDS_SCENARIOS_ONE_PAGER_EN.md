# FSDS scenario one-pagers (English) — data digestion & economic justification

One slide per scenario. Numbers marked **(sim)** unless from boss demo. Reproduce: `python3 run_fsds_business_impact_pack.py`.

---

## Slide 1 · Cross-border sales copilot (federated Legal / Finance / Product)

**Data digestion**

| Stage | What happens |
|--------|----------------|
| **Partition** | Vertical blocks: legal clause embeds, finance margin features, product personalization — each silo holds raw **X_k** locally. |
| **REF / LIVE** | Batch **W** on stacked windows; per block local MMD², domain AUC(**X**,**W**), ESS, top VIMP. |
| **Uplink** | **O(\|F_k\|)** envelope per round (not **(n_ref+n_live)×p** matrix). |
| **Server** | Merge block scores → `drift_type` (feature / concept / compound) → OFS or narrow **legal** agent path only. |
| **Multi-agent** | Role segments (pro/con/judge) get FSDS budget; **legal block drift ≠ shutdown** of finance/product agents. |

**Economic justification**

| Type | Claim | Evidence |
|------|--------|----------|
| **Direct OPEX** | Fewer full re-debates & less token on stable roles when routing + segment budget apply | ~**39%** $/debate, ~**20%** round cost (demos) |
| **Incremental (sim)** | **Controlled rollout** vs blanket copilot ban when only legal shifts | **~$125k/quarter** unlock narrative |
| **Audit** | Block-level **impact receipt** (MMD, AUC, rule_id) | `federated_legal_agent_blocks.json` |

**Further justify:** Replace sim unlock with one paused market’s quarterly GMV; A/B legal-narrow vs global-off.

---

## Slide 2 · Bank / insurance federated monitoring

**Data digestion**

| Stage | What happens |
|--------|----------------|
| **Constraint** | Legal entities **cannot pool X**; may share **Y**, batch **W**, and aggregated stats. |
| **Local** | Each node runs batch FSDS on **F_k**; optional DRPerm on stacked REF∪LIVE for PO-risk. |
| **Protocol** | Upload **3\|F_k\|+3** scalars per round (Eff vs centralized matrix). |
| **Server** | Merge p-values → **online FDR**; actions: keep / candidate_retire / alert_recalibrate. |
| **Validation** | Injected shift on known columns → **top-1 block hit** in bench. |

**Economic justification**

| Type | Claim | Evidence |
|------|--------|----------|
| **Direct (infra)** | Lower egress & storage vs centralizing full matrices | **48×–1410×** Eff; Spambase **1.45 KiB** vs **~2 MiB**/window |
| **Incremental** | Monitoring **exists** where central FSDS was **non-compliant** | Enables programs that were $0 before |
| **Risk** | Typed drift avoids auto-dropping features on concept-only PO | `classify_block_drift_type` |

**Further justify:** Cloud bill × windows/year; compliance FTE for data-pooling reviews avoided.

---

## Slide 3 · Group uplift / campaign (two-batch)

**Data digestion**

| Stage | What happens |
|--------|----------------|
| **Contract** | REF trains **τ̂**; LIVE scores **frozen τ̂**; **W** = batch, **T** = treatment. |
| **Gates** | MMD²(**X**), ESS on **ê(W\|X)**, domain AUC, SRM. |
| **L1** | AUUC_ref/live, LOCO–AUUC, quintile pairwise AUUC; **AUUC_ovlp vs global** separates mix vs ranking. |
| **L2** | Slice ATE; PO-risk gates concept REALLOCATE. |
| **Output** | `business_rules` P1–P5 + `evidence.economics` + **impact_receipts**; RELEARN = **CAPEX bucket**. |

**Economic justification**

| Type | Claim | Evidence |
|------|--------|----------|
| **Direct OPEX (sim)** | Cap worst quintiles → avoided treat spend | **~$818/window** avg; **~$3.3k** total (4 benches) |
| **CAPEX discipline** | RELEARN tickets **not** in OPEX net | **~$10k** tickets listed separately |
| **Incremental** | Avoid weekly **τ** retrains on promo mix alone | OPEX adapt ladder first |

**Further justify:** Plug real **margin_usd**, **treatment_cost**, weekly LIVE n.

---

## Slide 4 · Agent platform (multi-agent + LangGraph)

**Data digestion**

| Stage | What happens |
|--------|----------------|
| **Segments** | Debate roles **or** LangGraph checkpoint state embeddings. |
| **Batch monitor** | REF/LIVE on trace features (tools, length, mix). |
| **Budget** | shift × quality × need → **L**, tier, checks per segment. |
| **Debate macro** | Round-centroid **dispersion** → **early-stop** before max rounds. |
| **Graph micro** | Re-execute **drifted nodes** (e.g. tool_call) only. |
| **Receipt** | `suggest_intervention` attaches **impact_receipt** JSON. |

**Economic justification**

| Type | Claim | Evidence |
|------|--------|----------|
| **Direct OPEX** | SoT critical-path budget | **~34%** $/ep (**~$0.012/ep**) |
| **Direct OPEX** | Multi-agent + early-stop | **~39%** $/debate + **~20%** rounds |
| **Scale (illustr.)** | 500k ep/mo | **~$72k/yr** SoT + **~$16k/yr** debate (exec JSON) |

**Further justify:** Platform SKU “$/1k tasks with receipt”; shadow A/B on prod traffic.

---

## Slide 5 · BPO / human–agent hybrid QA

**Data digestion**

| Stage | What happens |
|--------|----------------|
| **Unit** | Handoff message embeddings (agent→human or worker→worker). |
| **REF / LIVE** | Shift on handoff feature distribution. |
| **Localization** | FSDS flags **which segments** moved — not “review everything”. |
| **Federated option** | Site/tenant = **block**; uplink stats only, no cross-site raw dialog. |
| **Receipt** | Same schema as agent loop for audit sampling policy. |

**Economic justification**

| Type | Claim | Evidence |
|------|--------|----------|
| **Direct OPEX** | Supervisor hours ∝ review count | Target **shift-only** sampling |
| **Formula** | (full_rate − fsds_rate) × **$/review** × volume | Fill from BPO contract |
| **Incremental** | Defensible QA to regulators vs ad-hoc spot checks | receipt + statistic |

**Further justify:** One month of review logs → measure shift coverage vs random sample defect catch rate.

---

## Slide 6 · Feature store / online OFS (federated blocks)

**Data digestion**

| Stage | What happens |
|--------|----------------|
| **Stream** | New feature **groups/blocks** arrive over time (OFS). |
| **Local FSDS** | Per-block REF/LIVE → scores + **drift_type**. |
| **Policy** | Covariate/compound → **candidate retire**; concept-only → **retrain driver**, no auto-drop. |
| **Server** | Incremental LOGO when blocks join; merged rank. |
| **Link** | ESS gate shared with uplift overlap monitor. |

**Economic justification**

| Type | Claim | Evidence |
|------|--------|----------|
| **Direct** | Retire bad groups → less train/score/storage downstream | Model $ + compute (site-specific) |
| **Direct** | Minimal uplink vs central FSDS on full store | Same Eff story as Scenario 2 |
| **Incremental** | Faster **safe** OFS in multi-team regulated stores | Time-to-production for features |

**Further justify:** Count retired candidates × avg feature pipeline cost; FDR precision on injection tests.

---

*PNG deck: `artifacts/fsds_scenario_slides/slide_*_{en|zh}.png`*
