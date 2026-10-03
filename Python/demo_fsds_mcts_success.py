#!/usr/bin/env python3
"""MCTS + local search budget tuning for success-rate (JSON)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fsds_sot.agentic_dgp import _branch_need_vector, generate_agentic_batch, episodes_to_arrays
from fsds_sot.economics import estimate_economics
from fsds_sot.incremental_value import success_probability, uniform_plan
from fsds_sot.mcts_search import local_search_refine, mcts_search_success, plan_from_hyperparams, BudgetHyperparams
from fsds_sot.pipeline import FSDSSoT


def main() -> int:
    eval_eps = generate_agentic_batch(1, seed=33, live=True, drift_strength=1.4)
    Z = eval_eps[0].branch_embeddings
    Q = eval_eps[0].branch_quality
    need = _branch_need_vector(eval_eps[0].branch_names)

    ref = Z.mean(axis=0, keepdims=True)
    shift = ((Z - ref) ** 2).sum(axis=1) ** 0.5
    shift = shift / (shift.max() + 1e-9)

    hp0 = BudgetHyperparams(1.0, 0.15, 320)
    hp_m, mcts_score, _ = mcts_search_success(shift, Q, need, n_simulations=80, seed=2026)
    hp, ls_score, plan = local_search_refine(shift, Q, need, hp_m)

    uni = uniform_plan(shift, Q, latency_cap_tokens=520, total_token_budget=2800)
    ref_b = generate_agentic_batch(40, seed=1, live=False)
    live_b = generate_agentic_batch(40, seed=2, live=True, drift_strength=1.2)
    X0, _, _, _ = episodes_to_arrays(ref_b)
    X1, _, _, _ = episodes_to_arrays(live_b)
    ctrl = FSDSSoT()
    _, plan_ent, econ_ent = ctrl.fit_plan(
        X0,
        X1,
        Z,
        branch_quality=Q,
        need_weights=need,
        total_token_budget=2800,
        latency_cap_tokens=520,
    )

    p_mcts = success_probability(plan, need, Q, uniform_L=520)
    p_uni = success_probability(uni, need, Q, uniform_L=520)
    p_ent = success_probability(plan_ent, need, Q, uniform_L=520)
    econ_m = estimate_economics(
        plan,
        branches=len(plan.branch_budgets),
        uniform_tokens_per_branch=520,
        success_rate=p_mcts,
        baseline_success_rate=p_uni,
    )

    out = {
        "mcts_local_search": {
            "hyperparams": {"kappa": hp.kappa, "entropy_lambda": hp.entropy_lambda, "need_floor": hp.need_floor},
            "mcts_score": mcts_score,
            "local_search_score": ls_score,
            "success_rate_est": p_mcts,
            "uniform_success_rate_est": p_uni,
            "entropy_default_success_rate_est": p_ent,
            "success_lift_vs_uniform": p_mcts - p_uni,
        },
        "economics_mcts_plan": econ_m.to_dict(),
        "economics_entropy_default": econ_ent.to_dict(),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
