#!/usr/bin/env python3
"""
Multi-agent debate × FSDS economics (ROI artifact).

Run:  cd Python && python3 demo_fsds_multi_agent_economics.py
Writes: ../artifacts/multi_agent_fsds_economics.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fsds_sot.applications import AgentPattern, build_multi_agent_trace, fit_agent_plan, pattern_playbook
from fsds_sot.closed_loop import LadderRung, LoopState, iterate_once
from fsds_sot.impact_receipt import build_agent_impact_receipt, merge_receipt_into_intervention
from fsds_sot.incremental_value import simulate_episode_outcome, uniform_plan
from fsds_sot.pipeline import FSDSSoT

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def _trace_batch(n: int, *, live: bool, seed: int, d: int = 32) -> np.ndarray:
    rows = []
    for i in range(n):
        rng = np.random.default_rng(seed + 1000 * i)
        embs = rng.normal(size=(3, d))
        if live:
            embs[1] += rng.normal(scale=1.8, size=d)
        seg = build_multi_agent_trace(
            ("pro", "con", "judge"),
            embs,
            role_quality=np.array([0.68, 0.62, 0.82]),
        )
        rows.append(seg.trace_features)
    return np.stack(rows, axis=0)


def _eval_debate_episodes(n: int, seed: int, *, use_fsds: bool, X0: np.ndarray, X1: np.ndarray) -> dict:
    rng = np.random.default_rng(seed)
    ctrl = FSDSSoT(seed=2026)
    need = np.array([0.5, 0.55, 0.95], dtype=float)
    succ, costs, totals, checks = [], [], [], []
    d = 32
    for i in range(n):
        rng_i = np.random.default_rng(seed + i)
        embs = rng_i.normal(size=(3, d))
        embs[1] += rng_i.normal(scale=0.9, size=d)
        seg = build_multi_agent_trace(("pro", "con", "judge"), embs, role_quality=np.array([0.68, 0.62, 0.82]))
        Z, q = seg.embeddings, seg.quality
        ref = Z.mean(axis=0, keepdims=True)
        shift = np.linalg.norm(Z - ref, axis=1)
        shift = shift / (shift.max() + 1e-9)
        if use_fsds:
            _, plan, _ = ctrl.fit_plan(
                X0,
                X1,
                Z,
                branch_quality=q,
                total_token_budget=1800,
                latency_cap_tokens=480,
                need_weights=need,
                min_tokens_by_branch=np.array([0, 0, 320], dtype=int),
                budget_kappa=0.88,
                entropy_lambda=0.06,
            )
        else:
            plan = uniform_plan(shift, q, latency_cap_tokens=480, total_token_budget=1800)
        s, _, tot, c, ch = simulate_episode_outcome(plan, need, q, uniform_L=480, rng=rng)
        succ.append(s)
        costs.append(c)
        totals.append(tot)
        checks.append(ch)
    return {
        "mean_success": float(np.mean(succ)),
        "mean_cost_usd": float(np.mean(costs)),
        "mean_total_tokens": float(np.mean(totals)),
        "mean_checks": float(np.mean(checks)),
    }


def main() -> int:
    n_ref, n_live, n_eval = 60, 60, 50
    X0 = _trace_batch(n_ref, live=False, seed=11)
    X1 = _trace_batch(n_live, live=True, seed=22)

    rng = np.random.default_rng(99)
    d = 32
    embs = rng.normal(size=(3, d))
    embs[1] += rng.normal(scale=2.0, size=d)
    segment = build_multi_agent_trace(
        ("pro", "con", "judge"),
        embs,
        role_quality=np.array([0.68, 0.58, 0.85]),
    )

    report, plan, econ = fit_agent_plan(
        segment,
        X0,
        X1,
        need_weights=np.array([0.5, 0.55, 0.95]),
        total_token_budget=1800,
        latency_cap_tokens=480,
        min_tokens_by_branch=np.array([0, 0, 320], dtype=int),
        budget_kappa=0.88,
        entropy_lambda=0.06,
    )

    uni = _eval_debate_episodes(n_eval, 300, use_fsds=False, X0=X0, X1=X1)
    fsds = _eval_debate_episodes(n_eval, 301, use_fsds=True, X0=X0, X1=X1)

    state = LoopState()
    out, state = iterate_once(
        X0,
        X1,
        segment.embeddings,
        segment.quality,
        state,
        total_token_budget=1800,
        latency_cap_tokens=480,
        need_weights=np.array([0.5, 0.55, 0.95]),
        min_tokens_by_branch=np.array([0, 0, 320], dtype=int),
        budget_kappa=0.88,
        entropy_lambda=0.06,
    )
    receipt = build_agent_impact_receipt(
        intervention=out.intervention,
        report=out.report,
        econ=econ,
        pattern=AgentPattern.MULTI_AGENT.value,
        ref_window="ref_debate_traces_60",
        live_window="live_debate_traces_60",
    )
    intervention_with_receipt = merge_receipt_into_intervention(out.intervention, receipt)

    per_role = [
        {
            "role": segment.segment_names[i],
            "L_tokens": plan.branch_budgets[i].expansion_tokens,
            "tier": plan.branch_budgets[i].model_tier,
            "checks": plan.branch_budgets[i].check_budget,
            "shift_score": float(plan.branch_budgets[i].shift_score),
        }
        for i in range(len(segment.segment_names))
    ]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pattern": AgentPattern.MULTI_AGENT.value,
        "playbook_hint": pattern_playbook()[AgentPattern.MULTI_AGENT.value],
        "scenario": "3-role debate (pro/con/judge); LIVE drift on con utterance embedding",
        "gates": {
            "mmd2": float(report.mmd2),
            "domain_auc": float(report.domain_auc),
            "overlap_ok": bool(report.overlap_ok),
        },
        "economics_from_plan": econ.to_dict(),
        "monte_carlo_eval": {
            "uniform_debate_swarm": uni,
            "fsds_routed_debate": fsds,
            "success_delta": fsds["mean_success"] - uni["mean_success"],
            "cost_reduction_pct": 100.0
            * (1.0 - fsds["mean_cost_usd"] / max(uni["mean_cost_usd"], 1e-9)),
            "checks_delta": fsds["mean_checks"] - uni["mean_checks"],
        },
        "per_role_budget": per_role,
        "closed_loop_window": {
            "accepted": out.accepted,
            "drift_proxy": out.drift_proxy,
            "rung": out.rung_used.name,
            "intervention": intervention_with_receipt,
        },
        "business_readout": {
            "headline_usd_per_episode_save": econ.baseline_total_cost_usd - econ.fsds_total_cost_usd,
            "net_economic_gain_usd_plan_proxy": econ.net_economic_gain_usd,
            "roi_multiple_vs_impl": econ.roi_multiple,
            "justification": "Receipt ties REALLOCATE segment budgets to MMD/domain AUC; "
            "down-weight redundant con when shift localizes to judge/pro.",
        },
    }

    ART.mkdir(parents=True, exist_ok=True)
    out_path = ART / "multi_agent_fsds_economics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(out_path), "net_gain_usd": econ.net_economic_gain_usd}, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
