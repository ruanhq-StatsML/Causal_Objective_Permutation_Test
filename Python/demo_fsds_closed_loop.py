#!/usr/bin/env python3
"""Demo: two-window closed-loop self-iteration (synthetic)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from demo_fsds_sot import synthetic_sot_batch
from fsds_sot.closed_loop import LadderRung, LoopState, iterate_once


def main() -> int:
    B, p = 6, 32
    X0, _, _ = synthetic_sot_batch(100, B, p, seed=1, drift_strength=0.0)
    X1, B1, Q1 = synthetic_sot_batch(100, B, p, seed=2, drift_strength=0.0)
    X1[:, :6] += np.random.default_rng(3).normal(scale=1.2, size=(X1.shape[0], 6))

    state = LoopState()
    interventions = []

    def apply(plan, rung):
        interventions.append({"rung": rung.name, "branches": len(plan.branch_budgets)})

    out1, state = iterate_once(
        X0, X1, B1.mean(axis=0), Q1[0], state, apply_intervention=apply
    )
    # Simulate post-intervention window (slightly less drift)
    X1p = X1.copy()
    X1p[:, :6] *= 0.5
    out2, state = iterate_once(
        X0, X1p, B1.mean(axis=0), Q1[0], state, apply_intervention=apply
    )

    result = {
        "window1": {
            "accepted": out1.accepted,
            "escalated": out1.escalated,
            "drift_proxy": out1.drift_proxy,
            "rung": out1.rung_used.name,
        },
        "window2_after_intervention": {
            "accepted": out2.accepted,
            "escalated": out2.escalated,
            "drift_proxy": out2.drift_proxy,
            "rung": out2.rung_used.name,
        },
        "history": state.history,
        "interventions_applied": interventions,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
