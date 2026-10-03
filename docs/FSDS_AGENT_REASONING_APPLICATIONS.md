# FSDS on Agent Reasoning — Application Map

Beyond **SoT** (skeleton branches) and **ToT** (search nodes), any agent loop that produces **segmentable states** can use the same two objects:

1. **Topology** — suggested groups \(K\), merge, order (Object 1)  
2. **Budget** — tokens / tier / checks per segment (Object 2)  

Plus **(0) batch covariate monitoring** on trace features \(\Dold\) vs \(\Dnew\).

## High-priority agent patterns

| Pattern | Segments (intervals) | Reference vs live batch | Typical \(Y\) | FSDS action |
|---------|----------------------|-------------------------|---------------|-------------|
| **ReAct** | Thought / Action / Observation spans | Step embeddings before vs after tool | Tool success, grounding | Skip re-tool on noise obs; budget on high-shift steps |
| **Plan-and-Execute** | Plan steps vs execution traces | Plan node vs executed sub-trace | Subgoal success | Re-plan which step; sequential budget inside failed step |
| **RAG agent** | One segment per retrieved chunk | Chunk embedding in answer context | Citation hit | top-\(k\) per segment; drop redundant chunks |
| **Self-Consistency** | Each sampled chain | Chain final-state embedding | Vote margin | Adaptive \(N\); stop when dispersion low |
| **Reflexion** | Draft vs critique vs revise | Iteration embeddings | Critic score | Stop early; rewrite only shifted spans |
| **Multi-agent debate** | Round × role (pro/con/judge) | Utterance embeddings | Agreement / task success | Merge weights; prune redundant agents; **early-stop** when round centroids converge (`docs/FSDS_DEBATE_EARLY_STOP.md`) |
| **Orchestrator–workers** | Worker handoff messages | Worker output embedding | Downstream success | Route to worker with lowest shift on needed skill |
| **SWE / code agent** | File region or edit hunk | Diff-context embedding | Tests pass | Re-run only hot files; cap edits on stable hunks |
| **LangGraph / state machine** | Graph nodes | Node state vector | Edge success | Re-execute drifted nodes only — see `docs/FSDS_LANGGRAPH_INTEGRATION.md` |
| **MCP / tool orchestra** | Per-tool call block | Tool args + result embed | Valid JSON / API OK | Tool budget; skip duplicate calls |

## Same math, different \(\phi\)

- **Covariate plane**: RF \(W \sim \X\) on trace features (tool counts, step length, role one-hot, node depth).  
- **Topology**: coupling on **segment embeddings** (not only SoT skeleton points).  
- **Budget**: \(L_b, \mathrm{tier}, \mathrm{checks}\) per segment or per tool call.

## Implementation in this repo

| Module | Role |
|--------|------|
| `fsds_sot/pipeline.py` | Generic `FSDSSoT.fit_plan` |
| `fsds_sot/applications.py` | Adapters: ReAct, plan–execute, multi-agent |
| `demo_fsds_react_agent.py` | ReAct-style segment demo |
| `demo_fsds_sot_agentic_incremental.py` | SoT agentic economics |
| `demo_fsds_closed_loop.py` | Two-window SoT loop + intervention JSON |
| `demo_fsds_rag_multihop_loop.py` | Multi-hop RAG until stabilize |
| `demo_fsds_self_consistency_loop.py` | Dispersion ↓ → adaptive \(N^\star\) |

## Closed loop (self-iteration)

1. `iterate_once` / `run_self_iteration` in `closed_loop.py`
2. Ladder: Reallocate → Re-topology → Re-ground → Re-configure → Re-learn
3. On accept: `decay_reference(X_old, X_new, alpha≈0.85)`
4. Log `suggest_intervention(plan, report, rung)` for ops

## When *not* to force FSDS

- Single-shot prompts with no intermediate structure.  
- Overlap near zero between batches (use covariate-only + abstain).  
- Segments fewer than 2 (no topology gain).

## Next production hooks

- Log `{segment_embedding, quality, tool_ok}` per step → rolling \(\Dold,\Dnew\).  
- Sidecar calls `fit_plan` async; scheduler reads `BranchBudget`-like outputs per segment type.

## Business impact and multi-agent GTM

For **profit generation**, audit receipts, and a divergent map of enterprise scenarios (sales swarms, federated compliance, uplift-as-agent-policy), see **`docs/FSDS_MULTI_AGENT_BUSINESS_IMPACT.md`** and LaTeX fragment **`docs/latex/fsds_multi_agent_business_impact.tex`**.
