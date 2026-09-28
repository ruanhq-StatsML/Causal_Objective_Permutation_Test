"""Shared unit economics for simulators, MCTS, and ROI reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .budget import SoTPlan

ModelTier = Literal["small", "medium", "large"]

# $ per 1k output tokens by tier (relative to medium = 1.0)
TIER_PRICE_MULT: dict[str, float] = {"small": 0.25, "medium": 1.0, "large": 2.5}

DEFAULT_COST_PER_1K_MEDIUM_USD = 0.006
DEFAULT_CHECK_COST_USD = 0.002
DEFAULT_FSDS_OVERHEAD_USD = 0.0005
DEFAULT_VALUE_PER_SUCCESS_USD = 0.45
DEFAULT_IMPL_COST_USD = 0.002


def generation_cost_usd(
    tokens: int,
    tier: str,
    *,
    cost_per_1k_medium: float = DEFAULT_COST_PER_1K_MEDIUM_USD,
) -> float:
    return tokens * TIER_PRICE_MULT.get(tier, 1.0) * (cost_per_1k_medium / 1000.0)


def plan_variable_cost_usd(
    branch_tokens: list[int],
    branch_tiers: list[str],
    branch_checks: list[int],
    *,
    cost_per_1k_medium: float = DEFAULT_COST_PER_1K_MEDIUM_USD,
    check_cost: float = DEFAULT_CHECK_COST_USD,
) -> tuple[float, float, float]:
    gen = sum(
        generation_cost_usd(t, tier, cost_per_1k_medium=cost_per_1k_medium)
        for t, tier in zip(branch_tokens, branch_tiers)
    )
    chk = sum(branch_checks) * check_cost
    return float(gen), float(chk), float(gen + chk)


def plan_variable_cost_from_plan(
    plan: "SoTPlan",
    *,
    cost_per_1k_medium: float = DEFAULT_COST_PER_1K_MEDIUM_USD,
    check_cost: float = DEFAULT_CHECK_COST_USD,
) -> tuple[float, float, float]:
    return plan_variable_cost_usd(
        [b.expansion_tokens for b in plan.branch_budgets],
        [b.model_tier for b in plan.branch_budgets],
        [b.check_budget for b in plan.branch_budgets],
        cost_per_1k_medium=cost_per_1k_medium,
        check_cost=check_cost,
    )


def uniform_baseline_variable_cost_usd(
    n_branches: int,
    *,
    tokens_per_branch: int = 520,
    checks_per_branch: int = 2,
    cost_per_1k_medium: float = DEFAULT_COST_PER_1K_MEDIUM_USD,
    check_cost: float = DEFAULT_CHECK_COST_USD,
) -> float:
    gen = n_branches * generation_cost_usd(
        tokens_per_branch, "medium", cost_per_1k_medium=cost_per_1k_medium
    )
    chk = n_branches * checks_per_branch * check_cost
    return float(gen + chk)
