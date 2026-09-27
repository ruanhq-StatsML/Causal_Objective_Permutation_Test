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
| **Multi-agent debate** | Round × role (pro/con/judge) | Utterance embeddings | Agreement / task success | Merge weights; prune redundant agents |
| **Orchestrator–workers** | Worker handoff messages | Worker output embedding | Downstream success | Route to worker with lowest shift on needed skill |
| **SWE / code agent** | File region or edit hunk | Diff-context embedding | Tests pass | Re-run only hot files; cap edits on stable hunks |
| **LangGraph / state machine** | Graph nodes | Node state vector | Edge success | Re-execute drifted nodes only |
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

## When *not* to force FSDS

- Single-shot prompts with no intermediate structure.  
- Overlap near zero between batches (use covariate-only + abstain).  
- Segments fewer than 2 (no topology gain).

## Next production hooks

- Log `{segment_embedding, quality, tool_ok}` per step → rolling \(\Dold,\Dnew\).  
- Sidecar calls `fit_plan` async; scheduler reads `BranchBudget`-like outputs per segment type.
