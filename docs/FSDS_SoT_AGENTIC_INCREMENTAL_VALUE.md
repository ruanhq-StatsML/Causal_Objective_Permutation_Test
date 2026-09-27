# Incremental Value on an Agentic SoT Dataset (Demo)

## What we justify (two objects only)

Everything else in FSDS-SoT supports these two **actionable** objects:

1. **Grouping granularity (decomposability gate)** — suggested cluster count \(K\), merge/sequentialize coupled skeleton branches before parallel expand.
2. **Critical-path budget executor** — per-branch expansion tokens \(L_b\), model tier, and check budget from embedding shift and verifier quality.

Business caps \(K_{\min/max}^{\mathrm{biz}}\), SLA \(C_{\mathrm{lat}}\), and total token budget clip the suggestions; FSDS supplies **data-driven recommendations**, not product policy.

## Agentic dataset (synthetic demo)

We model **ReAct-style agent tasks** as SoT skeletons with five branches:

| Branch | Agentic role | Latent budget need (simulator prior) |
|--------|----------------|-------------------------------------|
| plan | decomposition | low |
| retrieve | tool/RAG | **high** |
| reason | CoT chunk | medium |
| act | tool execution | **high** |
| verify | guardrail | medium |

**Reference batch** (`n=200`): stable tool failure rate and embeddings.  
**Live batch** (`n=200`): **covariate drift** on retrieve/act embedding blocks + higher tool failures (distribution shift in agent traces).  
**Eval rollouts** (`n=150`): same live regime; compare policies.

Trace features \(\X\): mean branch embedding + scalars `(B, tool_calls, tool_failures, latency_proxy)`.  
Outcome \(Y\): episode success (for monitoring / future concept plane).

## Incremental value definition

Compare:

- **Uniform SoT** — every branch gets the same \(L\), medium tier, fixed checks (status quo parallel agent).
- **FSDS-SoT** — `fit_plan()` after \(\Dold\) vs \(\Dnew\): gate + budget executor.

Simulator (transparent, not a production LLM): success probability rises when token mass **aligns** with latent branch `need` and when quality is high; span penalizes long critical paths. This makes incremental value **measurable** offline before wiring a live agent.

Run:

```bash
cd Python
python3 demo_fsds_sot_agentic_incremental.py
```

Metrics reported:

- `success_delta` — task success rate lift
- `span_reduction_pct` — critical-path token reduction (latency proxy under parallel decode)
- `cost_reduction_pct` — $/episode (tier-weighted tokens + checks)
- `checks_delta` — targeted verification spend

## How to read results for stakeholders

- **If span and cost drop with ≥ neutral success** → incremental value is **efficiency** (same agent, cheaper/faster).
- **If success rises with ≤ neutral cost** → incremental value is **reliability under drift** (retrieve/act branches funded when embeddings shift).
- **Domain AUC / VIMP on live batch** → explains *why* the executor moved budget (covariate monitoring object).

## Limitations (explicit)

- Demo uses a **simulator**, not API calls to GPT/tools; replace `simulate_episode_outcome` with logged production outcomes when available.
- Two objects only; merge/fallback/AGOD are out of scope for this justification slide.

## LaTeX pointer

Appendix `FSDS_SoT_ToT_supplement.tex` §\ref{sec:sot-landing}; add empirical row from JSON artifact when publishing.
