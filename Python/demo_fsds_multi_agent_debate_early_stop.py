#!/usr/bin/env python3
"""
Multi-agent debate early-stop: self-consistency dispersion → fewer rounds → $/debate.

Run: cd Python && python3 demo_fsds_multi_agent_debate_early_stop.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fsds_sot.applications import (
    AgentPattern,
    build_multi_agent_trace,
    debate_early_stop_round,
    debate_inter_round_dispersion,
    fit_agent_plan,
)
from fsds_sot.impact_receipt import build_agent_impact_receipt
from fsds_sot.pricing import DEFAULT_CHECK_COST_USD, DEFAULT_COST_PER_1K_MEDIUM_USD

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def _simulate_debate(max_rounds: int = 5, d: int = 24, seed: int = 0) -> np.ndarray:
    """Roles converge over rounds (dispersion falls)."""
    rng = np.random.default_rng(seed)
    roles = 3
    base = rng.normal(size=(roles, d))
    rounds = []
    for r in range(max_rounds):
        noise = rng.normal(scale=max(0.02, 0.55 - 0.14 * r), size=(roles, d))
        rounds.append(base + noise)
    return np.stack(rounds, axis=0)


def _cost_per_round(tokens_per_role: int = 380, roles: int = 3, checks: int = 1) -> float:
    gen = roles * tokens_per_role * (DEFAULT_COST_PER_1K_MEDIUM_USD / 1000.0)
    return gen + checks * DEFAULT_CHECK_COST_USD


def main() -> int:
    rounds_emb = _simulate_debate()
    max_r = rounds_emb.shape[0]
    disp_series = [
        float(debate_inter_round_dispersion(rounds_emb[: r + 1])) for r in range(max_r)
    ]
    stop_r = debate_early_stop_round(
        rounds_emb, max_rounds=max_r, dispersion_target=0.38, min_rounds=2
    )

    cost_full = max_r * _cost_per_round()
    cost_early = stop_r * _cost_per_round()
    savings_pct = 100.0 * (1.0 - cost_early / max(cost_full, 1e-9))

    # FSDS on final round roles vs REF/LIVE trace batches
    rng = np.random.default_rng(42)
    n = 40
    d = rounds_emb.shape[-1]
    X0 = np.stack(
        [
            build_multi_agent_trace(
                ("pro", "con", "judge"),
                rng.normal(size=(3, d)),
            ).trace_features
            for _ in range(n)
        ]
    )
    def _live_embs() -> np.ndarray:
        e = rng.normal(size=(3, d))
        e[1] += 0.5
        return e

    X1 = np.stack(
        [build_multi_agent_trace(("pro", "con", "judge"), _live_embs()).trace_features for _ in range(n)]
    )
    final_roles = rounds_emb[stop_r - 1]
    segment = build_multi_agent_trace(
        ("pro", "con", "judge"),
        final_roles,
        role_quality=np.array([0.7, 0.65, 0.88]),
    )
    report, plan, econ = fit_agent_plan(
        segment,
        X0,
        X1,
        need_weights=np.array([0.5, 0.5, 0.95]),
        total_token_budget=1600,
        latency_cap_tokens=420,
    )

    receipt = build_agent_impact_receipt(
        intervention={
            "type": "debate_early_stop",
            "rung": "REALLOCATE",
            "stop_round": stop_r,
            "max_rounds": max_r,
        },
        report=report,
        econ=econ,
        pattern=AgentPattern.MULTI_AGENT.value,
        ref_window="ref_debate",
        live_window="live_debate",
    )
    receipt["debate"] = {
        "dispersion_by_round": disp_series,
        "stop_round": stop_r,
        "rounds_avoided": max_r - stop_r,
        "usd_per_debate_full": round(cost_full, 4),
        "usd_per_debate_early_stop": round(cost_early, 4),
        "debate_cost_savings_pct": round(savings_pct, 1),
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pattern": "multi_agent_debate_early_stop",
        "mechanism": "debate_inter_round_dispersion <= target → stop before max_rounds",
        "dispersion_series": disp_series,
        "stop_round": stop_r,
        "cost": {
            "full_debate_usd": cost_full,
            "early_stop_usd": cost_early,
            "savings_pct": savings_pct,
        },
        "fsds_segment_budget": [
            {"role": segment.segment_names[i], "L": plan.branch_budgets[i].expansion_tokens}
            for i in range(3)
        ],
        "impact_receipt": receipt,
        "combined_story_usd": round((cost_full - cost_early) + econ.net_economic_gain_usd, 4),
    }

    ART.mkdir(parents=True, exist_ok=True)
    path = ART / "multi_agent_debate_early_stop.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(path), "stop_round": stop_r, "savings_pct": savings_pct}, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
