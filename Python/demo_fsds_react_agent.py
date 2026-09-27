#!/usr/bin/env python3
"""Demo: FSDS on ReAct-style segments (not only SoT skeleton)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fsds_sot.applications import build_react_trace, fit_agent_plan, pattern_playbook
from fsds_sot.agentic_dgp import generate_agentic_batch, episodes_to_arrays


def main() -> int:
    rng = np.random.default_rng(2026)
    d = 16

    # Reference / live batches for covariate monitoring (trace-level)
    ref = generate_agentic_batch(80, seed=1, live=False)
    live = generate_agentic_batch(80, seed=2, live=True, drift_strength=1.2)
    X0, _, _, _ = episodes_to_arrays(ref)
    X1, _, _, _ = episodes_to_arrays(live)

    # One ReAct episode: observation drifted (tool noise)
    thought = rng.normal(size=d)
    action = rng.normal(size=d)
    observation = rng.normal(size=d)
    observation[:6] += 1.5  # live-style shift on obs embedding

    seg = build_react_trace(
        thought,
        action,
        observation,
        tool_ok=[0.85, 0.8, 0.45],
    )
    report, plan, econ = fit_agent_plan(
        seg, X0, X1, total_token_budget=2400, latency_cap_tokens=400
    )

    out = {
        "pattern": seg.pattern.value,
        "segments": list(seg.segment_names),
        "playbook": pattern_playbook(),
        "attribution": {
            "domain_auc": report.domain_auc,
            "clusters": report.decomposability.sequential_clusters,
            "allow_parallel": bool(report.decomposability.allow_parallel),
        },
        "segment_budgets": [
            {
                "segment": seg.segment_names[b.branch_id],
                "L": b.expansion_tokens,
                "tier": b.model_tier,
                "checks": b.check_budget,
            }
            for b in plan.branch_budgets
        ],
        "economics": econ.to_dict(),
        "hint": "High shift on 'observation' → more checks / tier on obs segment; "
        "do not re-expand 'thought' if shift low (workflow efficiency).",
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
