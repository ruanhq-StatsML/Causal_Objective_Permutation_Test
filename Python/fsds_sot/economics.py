from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

import numpy as np

from .budget import SoTPlan


@dataclass
class SoTEconomicsReport:
    baseline_latency_tokens: int
    fsds_latency_tokens: int
    latency_reduction_pct: float
    baseline_compute_cost_usd: float
    fsds_compute_cost_usd: float
    cost_savings_pct: float
    check_cost_usd: float
    net_savings_usd: float
    fsds_overhead_usd: float
    roi_multiple: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tier_price(tier: str) -> float:
    return {"small": 0.15, "medium": 0.60, "large": 2.50}.get(tier, 0.60)


def estimate_economics(
    plan: SoTPlan,
    *,
    branches: int,
    uniform_tokens_per_branch: int = 512,
    cost_per_1k_output_tokens: float = 0.006,
    check_cost_per_call: float = 0.002,
    fsds_overhead_usd: float = 0.0005,
    parallel_slots: int = 8,
) -> SoTEconomicsReport:
    """
    Economic justification sketch:
    - Baseline SoT: equal-length parallel branches -> critical path = uniform length.
    - FSDS SoT: adaptive lengths + tier routing + targeted checks.
    """
    baseline_span = uniform_tokens_per_branch
    fsds_lengths = np.array([b.expansion_tokens for b in plan.branch_budgets], dtype=float)
    fsds_span = int(fsds_lengths.max()) if fsds_lengths.size else uniform_tokens_per_branch

    baseline_cost = branches * uniform_tokens_per_branch * (cost_per_1k_output_tokens / 1000.0)
    fsds_gen_cost = sum(
        b.expansion_tokens * _tier_price(b.model_tier) * (cost_per_1k_output_tokens / 1000.0)
        for b in plan.branch_budgets
    )
    check_cost = sum(b.check_budget for b in plan.branch_budgets) * check_cost_per_call

    latency_reduction = 100.0 * (1.0 - fsds_span / max(baseline_span, 1))
    cost_savings = 100.0 * (1.0 - (fsds_gen_cost + check_cost) / max(baseline_cost, 1e-9))
    net_savings = baseline_cost - fsds_gen_cost - check_cost - fsds_overhead_usd
    roi = net_savings / max(fsds_overhead_usd, 1e-9)

    # Parallel slots bound (Brent-style): if branches > slots, latency scales
    if branches > parallel_slots:
        baseline_span = int(np.ceil(branches / parallel_slots) * uniform_tokens_per_branch)
        fsds_sorted = np.sort(fsds_lengths)[::-1]
        padded = np.pad(
            fsds_sorted,
            (0, parallel_slots - len(fsds_sorted) % parallel_slots),
            mode="constant",
        )
        waves = padded.reshape(-1, parallel_slots).max(axis=1)
        fsds_span = int(waves.sum())
        latency_reduction = 100.0 * (1.0 - fsds_span / max(baseline_span, 1))

    return SoTEconomicsReport(
        baseline_latency_tokens=int(baseline_span),
        fsds_latency_tokens=int(fsds_span),
        latency_reduction_pct=float(latency_reduction),
        baseline_compute_cost_usd=float(baseline_cost),
        fsds_compute_cost_usd=float(fsds_gen_cost),
        cost_savings_pct=float(cost_savings),
        check_cost_usd=float(check_cost),
        net_savings_usd=float(net_savings),
        fsds_overhead_usd=float(fsds_overhead_usd),
        roi_multiple=float(roi),
    )
