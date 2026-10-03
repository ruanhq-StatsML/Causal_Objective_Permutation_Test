#!/usr/bin/env python3
"""
LangGraph-style checkpoint hook → FSDS segment plan + impact receipt.

No langgraph dependency: simulates StateGraph node embeddings + trace features.
Integrate in prod by logging checkpoint state vectors to the sidecar batch (REF/LIVE).

Run: cd Python && python3 demo_fsds_langgraph_hook.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fsds_sot.applications import AgentPattern, build_langgraph_node_trace, fit_agent_plan, pattern_playbook
from fsds_sot.closed_loop import LoopState, iterate_once
from fsds_sot.impact_receipt import build_agent_impact_receipt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"


def _checkpoint_batches(n: int, d: int, *, live: bool, seed: int) -> np.ndarray:
    nodes = ("router", "retrieve", "reason", "tool_call", "respond")
    rows = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        embs = rng.normal(size=(len(nodes), d))
        if live:
            embs[3] += rng.normal(scale=2.2, size=d)
        seg = build_langgraph_node_trace(
            nodes,
            embs,
            node_success=np.array([0.9, 0.75, 0.8, 0.55, 0.85]),
        )
        rows.append(seg.trace_features)
    return np.stack(rows)


def main() -> int:
    d = 24
    X0 = _checkpoint_batches(50, d, live=False, seed=1)
    X1 = _checkpoint_batches(50, d, live=True, seed=2)

    rng = np.random.default_rng(7)
    nodes = ("router", "retrieve", "reason", "tool_call", "respond")
    embs = rng.normal(size=(len(nodes), d))
    embs[3] += rng.normal(scale=2.5, size=d)
    segment = build_langgraph_node_trace(
        nodes,
        embs,
        node_success=np.array([0.92, 0.78, 0.81, 0.52, 0.88]),
    )
    need = np.array([0.4, 0.7, 0.85, 1.0, 0.75], dtype=float)

    report, plan, econ = fit_agent_plan(
        segment,
        X0,
        X1,
        need_weights=need,
        total_token_budget=2600,
        latency_cap_tokens=520,
        min_tokens_by_branch=np.array([0, 0, 0, 360, 280], dtype=int),
        budget_kappa=0.9,
        entropy_lambda=0.05,
    )

    state = LoopState()
    out, _ = iterate_once(
        X0,
        X1,
        segment.embeddings,
        segment.quality,
        state,
        need_weights=need,
        total_token_budget=2600,
        latency_cap_tokens=520,
        min_tokens_by_branch=np.array([0, 0, 0, 360, 280], dtype=int),
        budget_kappa=0.9,
        entropy_lambda=0.05,
    )

    integration_snippet = {
        "log_per_checkpoint": {
            "node_id": "str",
            "state_embedding": "float[d] or hash→lookup",
            "tool_ok": "bool",
            "latency_ms": "int",
            "batch_label_W": "0=REF window, 1=LIVE window",
        },
        "sidecar_schedule": "hourly fit_plan on rolling REF vs LIVE trace matrices",
        "action_hook": "read intervention.impact_receipt → LangGraph conditional edge / skip node",
        "example_conditional": "if receipt.shift_summary.mmd2 > tau: re-run tool_call only",
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pattern": AgentPattern.LANGGRAPH.value,
        "playbook": pattern_playbook()[AgentPattern.LANGGRAPH.value],
        "scenario": "tool_call node drifted on LIVE; FSDS raises budget on tool_call, trims stable nodes",
        "gates": {
            "mmd2": float(report.mmd2),
            "domain_auc": float(report.domain_auc),
            "overlap_ok": bool(report.overlap_ok),
        },
        "per_node_budget": [
            {
                "node": segment.segment_names[i],
                "L": plan.branch_budgets[i].expansion_tokens,
                "tier": plan.branch_budgets[i].model_tier,
                "shift": float(plan.branch_budgets[i].shift_score),
            }
            for i in range(len(nodes))
        ],
        "economics": econ.to_dict(),
        "impact_receipt": build_agent_impact_receipt(
            intervention=out.intervention,
            report=out.report,
            econ=econ,
            pattern=AgentPattern.LANGGRAPH.value,
            ref_window="ref_checkpoints_50",
            live_window="live_checkpoints_50",
        ),
        "langgraph_integration": integration_snippet,
    }

    ART.mkdir(parents=True, exist_ok=True)
    json_path = ART / "langgraph_fsds_hook.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = DOCS / "FSDS_LANGGRAPH_INTEGRATION.md"
    md_path.write_text(
        "\n".join(
            [
                "# FSDS × LangGraph checkpoints",
                "",
                "Map each **executed graph node** to one FSDS segment (state embedding at checkpoint).",
                "Rolling **REF** vs **LIVE** windows on trace features drive the same monitor as uplift two-batch.",
                "",
                "## Production wiring",
                "",
                "1. On each checkpoint commit, append `{node_id, embedding, tool_ok, W}` to the batch store.",
                "2. Sidecar runs `fit_agent_plan(build_langgraph_node_trace(...), X_ref, X_live)`.",
                "3. Scheduler reads `intervention.impact_receipt` — re-execute only nodes with high shift or failed overlap.",
                "",
                "## Demo",
                "",
                "```bash",
                "cd Python && python3 demo_fsds_langgraph_hook.py",
                "```",
                "",
                f"Artifact: `artifacts/langgraph_fsds_hook.json` (generated {payload['generated_at_utc']}).",
                "",
                "## Conditional edge example",
                "",
                "```python",
                "# after sidecar returns receipt:",
                "if receipt['shift_summary']['mmd2'] > 0.012:",
                "    return 'retry_tool_call'  # not full graph replay",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"written": str(json_path), "md": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
