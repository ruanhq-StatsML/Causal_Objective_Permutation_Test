# FSDS × multi-agent collaboration — profit, audit trail, and where to push next

**Purpose:** Connect **multi-agent** and **orchestrator–worker** patterns to the same FSDS spine already used for SoT budgeting, uplift two-batch monitoring, and federated feature blocks — with **P&L language** finance and product can defend, not only ML metrics.

**Related:** `docs/FSDS_AGENT_REASONING_APPLICATIONS.md`, `docs/FSDS_SoT_LANDING.md`, `docs/FSDS_UPLIFT_TWO_BATCH_LANDING.md`, `docs/FSDS_FEDERATED_FEATURE_BLOCKS.md`, `Python/fsds_sot/economics.py`, `Python/fsds_sot/applications.py` (`build_multi_agent_trace`).

---

## 1. One P&L spine (three surfaces, same ledger)

Treat every deployment as optimizing **expected contribution margin per decision episode** (user, ticket, claim, campaign send):

\[
\Pi = \underbrace{p_{\mathrm{succ}}\, v_{\mathrm{succ}}}_{\text{value when the workflow succeeds}}
- \underbrace{c_{\mathrm{var}}}_{\text{tokens, tools, checks, human minutes}}
- \underbrace{c_{\mathrm{fix}}}_{\text{FSDS sidecar, storage, federated uplink}}
\]

| Surface | What \(p_{\mathrm{succ}}\) and \(v_{\mathrm{succ}}\) mean | What FSDS moves first (cheap) | What escalates (expensive) |
|--------|-----------------------------------------------------------|-------------------------------|----------------------------|
| **Agent SoT / multi-agent** | Task success, SLA met, human escalation avoided | Per-segment **budget** (L, tier, checks); prune redundant roles; route to stable worker | Re-topology (merge agents); re-ground tools; relearn router |
| **Uplift / treatment** | Incremental conversion or margin on treated users | **Caps, narrow support, REALLOCATE** on bad AUUC slices | Joint refit of \(e,\mu,\tau\) after PO-risk + trial failure |
| **Federated blocks** | Campaign or model still **legal to run** cross-silo | Block-level monitor → retire bad features; minimal uplink | Cross-block correlation; global OFS / relearn |

**Code alignment:** agent demos use `net_economic_gain_usd()` in `economics.py` (success lift + variable cost delta − overhead). Uplift playbooks already encode **OPEX before CAPEX** (`FSDS_UPLIFT_MONITOR_AND_JOINT_REFIT_FRAMEWORK.md`). Federated roadmap lists **ρ → ROI calibration** as P2 — this doc is the narrative bridge.

**Justifiable impact (audit):** each action should ship with a **monitor receipt**: `{window, MMD² or segment shift, test p-value, quality metric (AUUC / success), rule_id, estimated $}` so risk and finance can tie spend change to a **pre-registered statistic**, not a one-off prompt tweak.

---

## 2. Multi-agent patterns → money (not only latency)

| Pattern | Where margin leaks today | FSDS lever | Typical $ mechanism |
|---------|--------------------------|------------|---------------------|
| **Debate (pro/con/judge)** | 3× redundant reasoning on stable facts | Segment shift on utterance embeddings; **down-weight** low-shift roles | −30–50% tokens/debate when one role adds no information |
| **Orchestrator–workers** | Wrong worker → rework loops | Route to worker with **lowest shift** on skill embedding vs REF | −rework minutes; +first-pass success |
| **Supervisor–worker (human)** | Supervisor reviews everything | FSDS flags **which handoffs** shifted; supervisor samples those | Human FTE hours → measurable \(c_{\mathrm{var}}\) |
| **Specialist swarm (legal/pricing/tech)** | All specialists always invoked | Topology merge when coupling low; parallel only when decomposable | Same as SoT critical-path: save on **non-critical** specialists |
| **Tool orchestra (MCP)** | Duplicate API calls | Covariate plane on tool args + results; skip duplicate segments | API list price × call reduction |
| **RAG + multi-agent** | Retrieve then 2 agents re-summarize same chunks | Chunk-level budget; debate only on **high-shift** hops | Retrieval + generation both drop |

**Repo gap (high leverage):** `build_multi_agent_trace` exists but there is **no** `demo_fsds_multi_agent_economics.py` pairing debate traces with `estimate_economics()` — mirror `demo_fsds_boss_six_points.py`. That demo becomes the **sales artifact** for agent ROI.

---

## 3. Broader scenarios (divergent map — iteration pass)

Below: **scenario → batch geometry → profit hook → FSDS product hook**. These are intentionally wider than the current codebase; they show where the **same** REF/LIVE + localization ladder sells.

### 3.1 Revenue and growth

| Scenario | REF vs LIVE | Primary \(Y\) | Profit story |
|----------|-------------|---------------|--------------|
| **Outbound sales copilot** (SDR drafts, legal redlines, pricing approves) | REF = won-opportunity traces; LIVE = current quarter | Opportunity advanced / meeting booked | Fewer legal cycles; faster time-to-quote → pipeline velocity |
| **Lifecycle marketing + uplift** | REF = calibration cohort; LIVE = send window | Conversion / incremental margin | **Same math as uplift monitor:** cap bad quintiles before retraining content model |
| **Personalized agent storefront** | REF = stable catalog epoch; LIVE = promo week | AOV, attach rate | Mixture shift (traffic) vs concept shift (preference) separated → don’t relearn recommender on every promo |
| **B2B renewal desk** | REF = last renewal season; LIVE = this week | Renewal probability | REALLOCATE effort to accounts where **ranking** (propensity to churn) still works on overlap core |

### 3.2 Risk, compliance, and cost avoidance

| Scenario | Blocks / segments | Primary \(Y\) | Profit story |
|----------|-------------------|---------------|--------------|
| **Prior auth (clinical multi-agent)** | Role = specialty reviewer | Approval without escalation | Denial overturn cost avoided; measurable \(v_{\mathrm{succ}}\) per approved case |
| **Fraud investigation swarm** | Investigator + rules + narrative LLM | Confirmed fraud $ / false positive rate | Shift on **evidence segment** → don’t rerun full LLM narrative when rules plane stable |
| **KYC / AML orchestration** | Federated **blocks** = product / geo / partner | SAR quality, auto-clear rate | Federated FSDS: run cross-border model without moving raw PII — **revenue unlock** (markets you couldn’t serve) |
| **Insurance claims** | Vision + policy + fraud agents | Loss ratio, leakage | PO-risk on stacked batches → concept drift on fraud pattern before payout drift hits LR |

### 3.3 Operations and platform

| Scenario | Monitor object | Primary \(Y\) | Profit story |
|----------|----------------|---------------|--------------|
| **Tier-2 support swarm** | Handoff embeddings | CSAT, reopen rate | Cap automated agents on slices where AUUC-like **resolution ranking** fails |
| **SWE release train** | Agent = file region / CI step | Green build, incident count | Re-run only **shifted hunks** (already in application map) |
| **Internal knowledge ops** | RAG hops × curator agent | Answer accuracy, time-to-answer | Adaptive retrieve depth (SoT + RAG loop demos exist) |
| **Vendor / procurement multi-stakeholder** | Segment = stakeholder memo | Cycle time, compliance pass | Topology: sequential legal → finance only when coupling high |

### 3.4 “Wide canvas” — agent fleet as a **market**

| Idea | Why FSDS fits | Business impact angle |
|------|---------------|-------------------------|
| **Agent marketplace** (buy/sell sub-agents) | REF = seller baseline; LIVE = buyer traffic; shift → **reputation / SLA tier** | Platform take rate on **verified** stable agents only |
| **Cross-tenant federated monitor** | Each tenant = block; server merges drift | SaaS: sell “compliance monitor” without seeing tenant raw data |
| **Human–agent hybrid BPO** | Batch W = shift week; segments = agent vs human spans | Labor arbitrage with **defensible** sampling of human QA on shifted segments only |
| **Regulatory stress episodes** | LIVE = shock window (rate change, policy update) | Document that caps were triggered by **AUUC_ovlp vs global**, not discretionary |

---

## 4. Cross-link: when multi-agent *is* uplift (and vice versa)

Many “multi-agent” products are secretly **uplift systems**:

- **Treatment** \(T\): agent policy variant (full swarm vs cap to one specialist vs human-only).
- **Outcome** \(Y\): conversion, margin, ticket resolved, claim decision.
- **Batch** \(W\): REF calibration window vs LIVE deployment window.

Then the **full uplift ladder** applies: REALLOCATE before RELEARN; PO-risk gates concept drift; overlap band separates mix from ranking failure.

**Positioning sentence for GTM:** *FSDS gives you one monitor for **who** you touch (uplift), **how hard** you compute (SoT budget), and **which silo** moved (federated blocks) — with the same receipts for audit.*

---

## 5. Three iteration rounds (what we refined)

### Round A — Widen (scenarios)

- Added **federated × multi-agent** (department blocks + role segments).
- Added **human-in-the-loop** as first-class \(c_{\mathrm{var}}\) (supervisor minutes, not only tokens).
- Added **marketplace / multi-tenant** as monetization of **stability certificates** (low shift = premium listing).

### Round B — Sharpen (measurable impact)

| Metric bucket | Agent multi-agent | Uplift | Federated |
|---------------|-------------------|--------|-----------|
| **Efficiency** | \( \Delta c_{\mathrm{var}} \) ($/ep, checks) | Spend avoided on bad slices | Uplink bytes × rounds |
| **Effectiveness** | \( \Delta p_{\mathrm{succ}} \) | Incremental conversions × margin | Top-1 block hit rate |
| **Risk** | Escalation rate, tool error | AUUC SLA breaches | FDR on block alerts |
| **Audit** | Segment shift table | LOCO + quintile + PO p | Per-block envelope JSON |

**Impact receipt (minimal schema):**

```json
{
  "episode_id": "...",
  "pattern": "multi_agent_debate",
  "ref_window": "2026-09-01/2026-09-14",
  "live_window": "2026-09-15/2026-09-16",
  "shift_summary": {"max_segment_shift": 0.42, "po_pvalue": 0.04},
  "action": "downweight_role:con",
  "kpi_before_after": {"success_rate": [0.71, 0.74], "usd_per_ep": [0.0356, 0.0236]},
  "estimated_net_gain_usd": 0.012,
  "rule_ids": ["P1_cap", "segment_budget_L"]
}
```

### Round C — Prioritize (repo + GTM opportunities)

| Priority | Opportunity | Why now | Effort |
|----------|-------------|---------|--------|
| **P0** | `demo_fsds_multi_agent_economics.py` | Closes the only **quantified** gap vs SoT boss demo | Small: reuse `build_multi_agent_trace` + `estimate_economics` |
| **P0** | **Impact receipt** emitter in `closed_loop.suggest_intervention` | Unifies agent + uplift ops logs for finance | Small JSON schema + docs |
| **P1** | **Uplift $ model** next to `business_rules_from_localization` | Maps cap quintile → expected saved spend / saved bad conversions | Medium: needs margin + cost inputs |
| **P1** | **LangGraph / checkpoint** hook doc + sample | Wide adoption; segment = node state | Doc + 1 example |
| **P1** | Federated demo narrative: **“legal agent block”** synthetic shift | Tells cross-silo story with existing 7-dataset bench | Medium |
| **P2** | Multi-agent **debate early-stop** when dispersion ↓ (link self-consistency demo) | Direct token ROI | Medium |
| **P2** | **A/B**: uniform swarm vs FSDS-routed swarm | Proof for enterprise procurement | Needs production traffic |
| **P2** | ρ → ROI calibration (federated P2) | Tie attribution accuracy to campaign $ | Research + sim |

---

## 6. Talk tracks (polished)

**For CFO / risk:** “We don’t cut model spend blindly. We cap **segments and slices** whose ranking or success rate failed a **pre-registered** two-batch test, and we log the statistic that triggered the cap.”

**For product / growth:** “When LIVE traffic mixes change, we narrow deployment before we retrain — so promos don’t trigger a week-long model project. When **outcome** law changes (PO-risk rejects), we relearn with holdout — same gate for agents and uplift.”

**For platform / AI:** “Multi-agent isn’t free parallelism. FSDS merges roles when embeddings show **no incremental information**, and pushes budget to the segment that actually shifted — same Object 1 + 2 as SoT, different \(\phi\) (utterance embeddings vs skeleton points).”

**For federated / enterprise:** “Each department block runs local REF/LIVE; the server only sees **O(|F_k|)** envelopes. You monetize **markets you can enter** without centralizing raw features, not just saving bandwidth.”

---

## 7. Suggested next commit themes (agent team)

1. Ship **multi-agent economics demo** + screenshot in walkthrough artifacts.  
2. Add **§ Multi-agent business impact** `\input` to `fsds_monitoring_full_standalone.tex` or agent manuscript (`FSDS_agent_reasoning_applications.tex`).  
3. Wire one **uplift business rule** line to optional `margin_usd` / `treatment_cost_usd` in benchmark JSON (even if simulated).  
4. Publish **impact receipt** schema in `docs/FSDS_UPLIFT_MONITOR_ACTIONS.md` appendix.

---

*Generated as a strategic iteration doc; numbers in SoT demos are reproducible via `demo_fsds_boss_six_points.py`; uplift rules via `run_uplift_subset_benchmark.py`.*
