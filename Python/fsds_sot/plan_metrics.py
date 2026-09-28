"""Token / check / critical-path breakdown for a SoT plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np

from .budget import SoTPlan


@dataclass
class PlanTokenBreakdown:
    """Separate critical-path (span) from total work (sum of branch tokens)."""

    span_tokens: int
    total_tokens: int
    total_checks: int
    per_branch: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def breakdown_plan(plan: SoTPlan) -> PlanTokenBreakdown:
    L = [b.expansion_tokens for b in plan.branch_budgets]
    span = int(max(L)) if L else 0
    total = int(sum(L))
    checks = int(sum(b.check_budget for b in plan.branch_budgets))
    per = [
        {
            "branch_id": b.branch_id,
            "L": b.expansion_tokens,
            "checks": b.check_budget,
            "tier": b.model_tier,
            "shift": round(b.shift_score, 4),
            "quality": round(b.quality_score, 4),
        }
        for b in plan.branch_budgets
    ]
    return PlanTokenBreakdown(span_tokens=span, total_tokens=total, total_checks=checks, per_branch=per)


def uniform_baseline_breakdown(
    n_branches: int,
    *,
    L_uniform: int = 520,
    checks_per_branch: int = 2,
) -> PlanTokenBreakdown:
    per = [
        {
            "branch_id": i,
            "L": L_uniform,
            "checks": checks_per_branch,
            "tier": "medium",
        }
        for i in range(n_branches)
    ]
    return PlanTokenBreakdown(
        span_tokens=L_uniform,
        total_tokens=n_branches * L_uniform,
        total_checks=n_branches * checks_per_branch,
        per_branch=per,
    )


def compare_to_uniform(plan: SoTPlan, *, L_uniform: int = 520, checks_per_branch: int = 2) -> Dict[str, float]:
    B = len(plan.branch_budgets)
    u = uniform_baseline_breakdown(B, L_uniform=L_uniform, checks_per_branch=checks_per_branch)
    f = breakdown_plan(plan)
    return {
        "span_reduction_pct": 100.0 * (1.0 - f.span_tokens / max(u.span_tokens, 1)),
        "total_token_reduction_pct": 100.0 * (1.0 - f.total_tokens / max(u.total_tokens, 1)),
        "checks_reduction_pct": 100.0 * (1.0 - f.total_checks / max(u.total_checks, 1)),
        "uniform_span": float(u.span_tokens),
        "fsds_span": float(f.span_tokens),
        "uniform_total_tokens": float(u.total_tokens),
        "fsds_total_tokens": float(f.total_tokens),
        "uniform_checks": float(u.total_checks),
        "fsds_checks": float(f.total_checks),
    }
