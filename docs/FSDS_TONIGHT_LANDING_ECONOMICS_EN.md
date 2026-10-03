# FSDS — Scenarios you can land tonight with direct economic benefit

After `docs/latex/fsds_concrete_pipeline_justification.tex` (objects, algorithms, JSON deltas), this note is **action-only**: what to wire **tonight** so spend moves on real bills—not slide-only narrative.

Checklist JSON: `artifacts/fsds_tonight_landing_checklist.json` via `python3 export_fsds_tonight_landing.py`.

## Direct vs deferred (summary)

| Tier | Scenario | Tonight OPEX? | $ source | Demo unit | Evidence |
|------|----------|---------------|----------|-----------|----------|
| **T0** | Agent SoT + impact receipt | Yes | LLM tokens/tiers | ~$0.012/ep | `boss_delivery_six_points.json` |
| **T0** | Multi-agent FSDS routing | Yes | Same | ~$0.0081/debate | `multi_agent_fsds_economics.json` |
| **T0** | Debate early-stop | Yes | Fewer rounds | ~$0.0088/debate | `multi_agent_debate_early_stop.json` |
| **T0** | LangGraph checkpoints | Yes | Partial replay | SoT-style receipt | `langgraph_fsds_hook.json` |
| **T1** | Uplift REALLOCATE caps | Yes (real config) | Treatment spend | ~$818/window (sim avg) | `uplift_two_layer_benchmark.json` |
| **T2** | Federated legal blocks | Indirect | Avoid full shutdown | Receipt tonight; $125k/q sim | `federated_legal_agent_blocks.json` |

**Not tonight OPEX:** default **RELEARN** retrain path (CAPEX tickets; excluded from OPEX net in bench).

## T0 — 2–4h on existing API billing or traces

Wire REF/LIVE windows, sidecar plan, log **`impact_receipt`** for finance.

- SoT: `suggest_intervention(..., attach_impact_receipt=True)` in `fsds_sot/closed_loop.py`
- Debate: `demo_fsds_multi_agent_economics.py` + `demo_fsds_multi_agent_debate_early_stop.py`
- LangGraph: `demo_fsds_langgraph_hook.py` + `FSDS_LANGGRAPH_INTEGRATION.md`

24h metrics: sum of `estimated_net_gain_usd`; cost per episode/debate; receipt coverage.

Scale: edit `config/fsds_economics_assumptions.json` traffic tiers; see `FSDS_BUSINESS_IMPACT_EXECUTIVE_SUMMARY.md`.

## T1 — First uplift window tonight

1. Set margin, treatment cost, treat rate in config.
2. Export REF/LIVE with row-level `X, Y, T`.
3. Run `run_uplift_subset_benchmark.py` (or production CSV through the same localization path).
4. Apply only **opex** bucket rules; RELEARN → CAPEX ticket.

## T2 — Receipts tonight; revenue story is strategic

`demo_federated_legal_agent_blocks.py` — block-level receipts and targeted pause vs global kill-switch.

## Reproduce everything

```bash
cd Python
python3 run_fsds_business_impact_pack.py
python3 export_fsds_tonight_landing.py
```

**Recommended order:** debate FSDS + early-stop → SoT → uplift window → federated receipts in parallel.

中文：`docs/FSDS_TONIGHT_LANDING_ECONOMICS_ZH.md`.
