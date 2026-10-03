# FSDS × LangGraph checkpoints

Map each **executed graph node** to one FSDS segment (state embedding at checkpoint).
Rolling **REF** vs **LIVE** windows on trace features drive the same monitor as uplift two-batch.

## Production wiring

1. On each checkpoint commit, append `{node_id, embedding, tool_ok, W}` to the batch store.
2. Sidecar runs `fit_agent_plan(build_langgraph_node_trace(...), X_ref, X_live)`.
3. Scheduler reads `intervention.impact_receipt` — re-execute only nodes with high shift or failed overlap.

## Demo

```bash
cd Python && python3 demo_fsds_langgraph_hook.py
```

Artifact: `artifacts/langgraph_fsds_hook.json` (generated 2026-10-03T02:18:24.486501+00:00).

## Conditional edge example

```python
# after sidecar returns receipt:
if receipt['shift_summary']['mmd2'] > 0.012:
    return 'retry_tool_call'  # not full graph replay
```
