#!/usr/bin/env python3
"""
Agentic dataset demo: incremental value of FSDS-SoT vs uniform SoT.

Synthetic ReAct-style skeleton (plan/retrieve/reason/act/verify) with
injected drift on tool branches in the live batch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fsds_sot.agentic_dgp import generate_agentic_batch, episodes_to_arrays, DEFAULT_BRANCH_NAMES
from fsds_sot.agentic_dgp import _branch_need_vector
from fsds_sot.incremental_value import evaluate_policies_on_agentic
from fsds_sot import FSDSSoT


def main() -> int:
    ref_eps = generate_agentic_batch(200, seed=11, live=False)
    live_eps = generate_agentic_batch(200, seed=22, live=True, drift_strength=1.4)
    eval_eps = generate_agentic_batch(150, seed=33, live=True, drift_strength=1.4)

    X0, Y0, _, _ = episodes_to_arrays(ref_eps)
    X1, Y1, _, _ = episodes_to_arrays(live_eps)
    _, _, Z_eval, Q_eval = episodes_to_arrays(eval_eps)

    names = eval_eps[0].branch_names
    need = _branch_need_vector(names)

    report = evaluate_policies_on_agentic(
        Z_eval, Q_eval, need, X0, X1, seed=2026
    )

    # Attribution snapshot on one live episode (justify monitoring object)
    ctrl = FSDSSoT(seed=2026)
    att, plan, econ = ctrl.fit_plan(
        X0, X1, Z_eval[0], branch_quality=Q_eval[0], total_token_budget=2800, latency_cap_tokens=520
    )

    out = {
        "dataset": {
            "type": "synthetic_agentic_sot",
            "branches": list(names),
            "n_ref": len(ref_eps),
            "n_live": len(live_eps),
            "n_eval": len(eval_eps),
            "drift": "retrieve/act embedding + tool-failure rate (live batch)",
        },
        "incremental_value_vs_uniform_sot": report,
        "example_single_episode": {
            "domain_auc": att.domain_auc,
            "mmd2": att.mmd2,
            "overlap_ok": bool(att.overlap_ok),
            "suggested_clusters": att.decomposability.sequential_clusters,
            "plan": [
                {
                    "branch": names[b.branch_id],
                    "L": b.expansion_tokens,
                    "tier": b.model_tier,
                    "checks": b.check_budget,
                }
                for b in plan.branch_budgets
            ],
            "economics_vs_uniform": econ.to_dict(),
        },
        "justification_one_liner": (
            "Two objects drive incremental value: (1) topology gate suggests safe parallel "
            "groups; (2) budget executor aligns tokens/checks/tier with shift×quality so "
            "agentic high-need branches (retrieve/act) get resources under drift without "
            "paying uniform SoT cost on every branch."
        ),
    }

    print(json.dumps(out, indent=2))
    print("\n=== Incremental value (FSDS-SoT vs uniform SoT) ===")
    inc = report["incremental"]
    print(f"  Success delta:        {inc['success_delta']:+.3f}")
    print(f"  Span reduction:       {inc['span_reduction_pct']:.1f}%")
    print(f"  Cost reduction:       {inc['cost_reduction_pct']:.1f}%")
    print(f"  Checks delta:         {inc['checks_delta']:+.2f}")

    out_path = Path("/opt/cursor/artifacts/fsds_sot_agentic_incremental.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
