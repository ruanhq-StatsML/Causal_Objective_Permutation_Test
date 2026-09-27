# FSDS × Skeleton-of-Thought — Landing Guide

## What FSDS does for SoT

SoT splits reasoning into (1) a **skeleton** of points and (2) **parallel expansion** of each point. FSDS adds a control layer **before and during expansion**:

| Stage | FSDS role |
|--------|-----------|
| Pre-expansion | **Decomposability gate**: is the skeleton weakly coupled enough to parallelize? |
| Batch monitoring | **Covariate attribution**: which trace/node features shifted between reference and live traffic? |
| Expansion | **Critical-path budget**: assign expansion length, check budget, and model tier per branch |
| Post-step | **Economics**: latency/cost vs uniform SoT baseline |

This is **not** next-token gradient attribution; it attributes **nonstationarity in embeddings and trace features** (`P(X|W)`), matching the FSDS paper’s continuous-inference framing.

## Search width + depth (ToT) vs SoT

- **ToT**: “width” = branching factor per node; “depth” = search horizon. FSDS allocates **search budget** to nodes/regions with high drift share.
- **SoT**: there is no tree search; the analogue is **parallel width** (= number of skeleton points / clusters) and **expansion depth** (= tokens per branch). FSDS controls **width** via the decomposability gate (merge clusters → sequential) and **depth** via per-branch `expansion_tokens`.

## Expansion length + check budget + model capacity

These three levers are tied:

1. **Expansion length** — caps critical-path latency (`max_b L_b`). High shift × high importance → longer expansion; stable branches → terse (`L_min`).
2. **Check budget** — self-verify / PRM / tool-validity calls scale with **risk** `shift × (1 - quality)`. Cheap guardrail before committing long generations.
3. **Model capacity** — route **large** tier only where drift and risk are jointly high; **small** tier on stationary branches. Capacity and length multiply cost, so FSDS uses **tier × tokens** jointly rather than maxing both everywhere.

Implementation: `Python/fsds_sot/budget.py` (`allocate_branch_budgets`).

## Economic justification (sketch)

Uniform SoT assigns every branch the same token budget and often the same model → critical path = longest branch, cost ∝ `B × L_uniform`.

FSDS-SoT:

- Shortens low-shift branches (↓ span latency).
- Spends checks only on high-risk branches (↓ wasted verification).
- Uses small models on stable slices (↓ $/token).

The demo reports **latency reduction %**, **cost savings %**, **net savings USD**, and **ROI vs FSDS overhead** (`demo_fsds_sot.py`).

Typical production narrative:

- **Latency SLA**: p99 bounded by `latency_cap_tokens` per branch instead of unbounded uniform expansion.
- **Unit economics**: 15–40% token/compute reduction on mixed workloads (stable + drifting branches) before quality lift from targeted checks.
- **Reliability**: check budget on concept-risk regions reduces expensive full re-runs.

## Business vs data: how many groups?

**Business** should set hard constraints: max parallel width, latency SLA, cost cap, compliance rules (e.g. always run checks on financial claims). **FSDS** recommends a data-driven granularity \(K\) and merge/prune suggestions from coupling + redundancy. Production policy is usually:

\[
K_{\mathrm{final}} = \mathrm{clip}(K_{\mathrm{FSDS}},\ K_{\min}^{\mathrm{biz}},\ K_{\max}^{\mathrm{biz}})
\]

So “选几个” is a **joint** decision: business caps the envelope; FSDS fills in the partition inside the envelope.

## Tree-search (ToT) vs branch count (SoT)

| | **SoT** | **ToT** |
|---|---------|---------|
| Width | Number of skeleton points / parallel clusters | **Branching factor** per expanded node (how many children) |
| Depth | **Expansion tokens** per branch (critical path) | **Search depth** (how many reasoning layers) |
| FSDS knob | Decomposability \(K\), then \(L_b\) / tier / checks | Prune/expand **which nodes**, adjust \(b\) and max depth where drift concentrates |

Same budget executor (tokens, model tier, checks); only the **topology** differs.

## Dashboard panel 3 — actionable budget executor

Panel 3 is **not** just embedding geometry; it is the **control output** of procedure (2).

| Signal | Meaning | Action emitted |
|--------|---------|----------------|
| **shift** (bar) | Branch embedding moved vs skeleton/reference → new information | Scales **expansion tokens** \(L_b\) |
| **quality** (bar) | Verifier / PRM / pass-rate proxy for this branch | Down-weights spend when already confident |
| **risk** (line) | \(\propto \mathrm{shift}\times(1-\mathrm{quality})\) | Scales **check budget**; with shift, selects **model tier** |
| **L** (purple bars) | `expansion_tokens` after cap + total budget | Scheduler: `max decode = L_b` |
| **M·Nc** label | tier initial + check count | Route model; run N self-verify/tool checks |

**How to read for ops**

- **High shift + high quality**: long expansion OK (content-rich, trustworthy) — watch **critical path** (longest purple bar).
- **High shift + low quality**: high risk → more checks, prefer **large** tier; may trigger re-retrieval before expand.
- **Low shift + high quality**: terse **L**, **small** tier, minimal checks — reclaim budget (zero-sum mode).
- **Low shift + low quality**: suspicious stale/low signal — keep minimal checks; do not waste large model.

Panels 1–2 act on **topology**; panel 3 acts on **resources**; panel 4 acts on **batch-level monitoring** (when to re-plan defaults, not per-request decode).

## Quick start

```bash
cd Python
pip install numpy scikit-learn scipy matplotlib
python3 demo_fsds_sot.py
python3 demo_fsds_sot_viz.py   # writes artifacts/fsds_sot_dashboard.png
```

Programmatic use:

```python
from fsds_sot import FSDSSoT

controller = FSDSSoT(seed=2026)
report, plan, econ = controller.fit_plan(X_old, X_new, branch_embeddings, branch_quality)
```

## Other levers (beyond length + checks + tier)

- Retrieval budget per branch (RAG SoT)
- KV re-encode vs reuse per modality block
- Straggler-first scheduling under finite parallel slots
- Cache hit on concept-stationary branches (zero expansion)
- Fallback to CoT when decomposability fails
