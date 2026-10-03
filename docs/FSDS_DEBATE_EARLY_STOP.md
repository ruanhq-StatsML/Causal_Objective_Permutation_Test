# Multi-agent debate early-stop (FSDS + dispersion)

When **pro / con / judge** embeddings stabilize across rounds, continuing debate burns tokens without new information.

## Signal

`debate_inter_round_dispersion(round_role_embs)` on shape `(R, B, d)` — same geometry as self-consistency `chain_dispersion`.

`debate_early_stop_round(..., dispersion_target=0.14)` returns stop round ≥ `min_rounds`.

## Demo

```bash
cd Python && python3 demo_fsds_multi_agent_debate_early_stop.py
```

Output: `artifacts/multi_agent_debate_early_stop.json` with `debate_cost_savings_pct` and `impact_receipt`.

## Combine with segment budget

1. Early-stop reduces **rounds** (macro).
2. `fit_agent_plan` on final-round roles reduces **tokens per role** (micro, FSDS).

`combined_story_usd` in the JSON adds both levers for narrative.
