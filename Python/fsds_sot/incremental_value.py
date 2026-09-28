"""
Simulate incremental value of FSDS-SoT vs uniform SoT on agentic episodes.

Two core objects justified:
  (1) Grouping / decomposability (topology gate)
  (2) Critical-path budget executor (L_b, tier, checks)

Outcome simulator: task success probability increases when token/check budget
is aligned with latent branch ``need`` and branch quality is low (agentic).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np

from .budget import BranchBudget, SoTPlan, allocate_branch_budgets
from .pipeline import FSDSSoT
from .pricing import DEFAULT_CHECK_COST_USD, DEFAULT_COST_PER_1K_MEDIUM_USD, plan_variable_cost_usd


@dataclass
class PolicyMetrics:
    name: str
    mean_success: float
    mean_span_tokens: float
    mean_total_tokens: float
    mean_cost_usd: float
    mean_checks: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def success_probability(
    plan: SoTPlan,
    need: np.ndarray,
    quality: np.ndarray,
    *,
    uniform_L: int = 520,
) -> float:
    """Deterministic success proxy (same as simulate_episode_outcome expectation path)."""
    span = max(b.expansion_tokens for b in plan.branch_budgets)
    L = np.array([b.expansion_tokens for b in plan.branch_budgets], dtype=float)
    align = (L / (L.sum() + 1e-9) * need).sum()
    q_pen = float(np.mean(quality))
    u = np.ones_like(need) / len(need)
    align_uniform = (u * need).sum()
    lift = (align - align_uniform) / (align_uniform + 1e-9)
    return float(np.clip(0.52 + 0.28 * align + 0.15 * q_pen + 0.08 * lift - 0.0003 * span, 0.05, 0.98))


def simulate_episode_outcome(
    plan: SoTPlan,
    need: np.ndarray,
    quality: np.ndarray,
    *,
    uniform_L: int,
    rng: np.random.Generator,
) -> tuple[float, int, int, float, float]:
    """
    Returns success (0/1), span_tokens, total_tokens, cost_usd, n_checks.
    """
    span = max(b.expansion_tokens for b in plan.branch_budgets)
    total = sum(b.expansion_tokens for b in plan.branch_budgets)
    checks = sum(b.check_budget for b in plan.branch_budgets)
    _, _, cost = plan_variable_cost_usd(
        [b.expansion_tokens for b in plan.branch_budgets],
        [b.model_tier for b in plan.branch_budgets],
        [b.check_budget for b in plan.branch_budgets],
        cost_per_1k_medium=DEFAULT_COST_PER_1K_MEDIUM_USD,
        check_cost=DEFAULT_CHECK_COST_USD,
    )
    p_succ = success_probability(plan, need, quality, uniform_L=uniform_L)
    success = float(rng.random() < p_succ)
    return success, int(span), int(total), float(cost), float(checks)


def uniform_plan(
    shift_scores: np.ndarray,
    quality: np.ndarray,
    *,
    latency_cap_tokens: int,
    total_token_budget: int,
) -> SoTPlan:
    B = len(shift_scores)
    L = min(latency_cap_tokens, total_token_budget // max(B, 1))
    branches = [
        BranchBudget(
            branch_id=i,
            expansion_tokens=int(L),
            check_budget=2,
            model_tier="medium",
            shift_score=float(shift_scores[i]),
            quality_score=float(quality[i]),
        )
        for i in range(B)
    ]
    return SoTPlan(
        branch_budgets=branches,
        latency_cap_tokens=latency_cap_tokens,
        total_token_budget=total_token_budget,
        use_parallel=True,
        clusters=[],
    )


def evaluate_policies_on_agentic(
    episodes_Z: np.ndarray,
    episodes_Q: np.ndarray,
    need: np.ndarray,
    X_old: np.ndarray,
    X_new: np.ndarray,
    *,
    seed: int = 2026,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    controller = FSDSSoT(seed=seed)
    n = episodes_Z.shape[0]

    rows: List[PolicyMetrics] = []
    for policy_name, use_fsds in [("uniform_sot", False), ("fsds_sot", True)]:
        succ, spans, totals, costs, checks = [], [], [], [], []
        for i in range(n):
            Z = episodes_Z[i]
            q = episodes_Q[i]
            ref = Z.mean(axis=0, keepdims=True)
            shift = np.linalg.norm(Z - ref, axis=1)
            shift = shift / (shift.max() + 1e-9)

            if use_fsds:
                plan, _ = controller.fit_plan_branches(
                    Z,
                    q,
                    total_token_budget=2800,
                    latency_cap_tokens=520,
                    need_weights=need,
                    min_tokens_by_branch=np.where(need >= 0.95, 320, 0).astype(int),
                    budget_kappa=0.85,
                    entropy_lambda=0.05,
                )
            else:
                plan = uniform_plan(shift, q, latency_cap_tokens=520, total_token_budget=2800)

            s, sp, tot, c, ch = simulate_episode_outcome(plan, need, q, uniform_L=520, rng=rng)
            succ.append(s)
            spans.append(sp)
            totals.append(tot)
            costs.append(c)
            checks.append(ch)

        rows.append(
            PolicyMetrics(
                name=policy_name,
                mean_success=float(np.mean(succ)),
                mean_span_tokens=float(np.mean(spans)),
                mean_total_tokens=float(np.mean(totals)),
                mean_cost_usd=float(np.mean(costs)),
                mean_checks=float(np.mean(checks)),
            )
        )

    u, f = rows[0], rows[1]
    return {
        "policies": [r.to_dict() for r in rows],
        "incremental": {
            "success_delta": f.mean_success - u.mean_success,
            "span_reduction_pct": 100.0 * (1.0 - f.mean_span_tokens / max(u.mean_span_tokens, 1)),
            "cost_reduction_pct": 100.0 * (1.0 - f.mean_cost_usd / max(u.mean_cost_usd, 1e-9)),
            "checks_delta": f.mean_checks - u.mean_checks,
        },
        "core_objects": [
            "Decomposability gate (suggested K, parallel vs sequential)",
            "Critical-path budget executor (L_b, tier, checks from shift×quality)",
        ],
    }
