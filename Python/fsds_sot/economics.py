from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np

from .budget import SoTPlan
from .pricing import (
    DEFAULT_CHECK_COST_USD,
    DEFAULT_COST_PER_1K_MEDIUM_USD,
    DEFAULT_FSDS_OVERHEAD_USD,
    DEFAULT_IMPL_COST_USD,
    DEFAULT_VALUE_PER_SUCCESS_USD,
    plan_variable_cost_from_plan,
    uniform_baseline_variable_cost_usd,
)


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
    baseline_total_cost_usd: float = 0.0
    fsds_total_cost_usd: float = 0.0
    baseline_check_cost_usd: float = 0.0
    expected_value_usd: float = 0.0
    baseline_expected_value_usd: float = 0.0
    net_economic_gain_usd: float = 0.0
    roi_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parallel_span(lengths: np.ndarray, *, parallel_slots: int) -> int:
    if lengths.size == 0:
        return 0
    if lengths.size <= parallel_slots:
        return int(lengths.max())
    fsds_sorted = np.sort(lengths)[::-1]
    pad = (-len(fsds_sorted)) % parallel_slots
    if pad:
        fsds_sorted = np.pad(fsds_sorted, (0, pad), mode="constant")
    waves = fsds_sorted.reshape(-1, parallel_slots).max(axis=1)
    return int(waves.sum())


def net_economic_gain_usd(
    *,
    success_rate: float,
    baseline_success_rate: float,
    fsds_variable_cost_usd: float,
    baseline_variable_cost_usd: float,
    fsds_overhead_usd: float = DEFAULT_FSDS_OVERHEAD_USD,
    value_per_success_usd: float = DEFAULT_VALUE_PER_SUCCESS_USD,
) -> float:
    """
    Single ROI definition used by economics, MCTS, and demos:

    net_gain = (p - p0) * v_succ + (baseline_var - fsds_var) - overhead
    """
    return (
        (success_rate - baseline_success_rate) * value_per_success_usd
        + (baseline_variable_cost_usd - fsds_variable_cost_usd)
        - fsds_overhead_usd
    )


def estimate_economics(
    plan: SoTPlan,
    *,
    branches: int,
    uniform_tokens_per_branch: int = 512,
    uniform_checks_per_branch: int = 2,
    cost_per_1k_output_tokens: float = DEFAULT_COST_PER_1K_MEDIUM_USD,
    check_cost_per_call: float = DEFAULT_CHECK_COST_USD,
    fsds_overhead_usd: float = DEFAULT_FSDS_OVERHEAD_USD,
    parallel_slots: int = 8,
    success_rate: Optional[float] = None,
    baseline_success_rate: Optional[float] = None,
    value_per_success_usd: float = DEFAULT_VALUE_PER_SUCCESS_USD,
    implementation_cost_usd: Optional[float] = None,
) -> SoTEconomicsReport:
    fsds_lengths = np.array([b.expansion_tokens for b in plan.branch_budgets], dtype=float)
    B = branches if branches > 0 else len(plan.branch_budgets)

    baseline_span = uniform_tokens_per_branch
    fsds_span = int(fsds_lengths.max()) if fsds_lengths.size else uniform_tokens_per_branch
    if B > parallel_slots:
        baseline_span = int(np.ceil(B / parallel_slots) * uniform_tokens_per_branch)
        fsds_span = _parallel_span(fsds_lengths, parallel_slots=parallel_slots)

    baseline_var = uniform_baseline_variable_cost_usd(
        B,
        tokens_per_branch=uniform_tokens_per_branch,
        checks_per_branch=uniform_checks_per_branch,
        cost_per_1k_medium=cost_per_1k_output_tokens,
        check_cost=check_cost_per_call,
    )
    baseline_gen = B * uniform_tokens_per_branch * (cost_per_1k_output_tokens / 1000.0)
    baseline_chk = B * uniform_checks_per_branch * check_cost_per_call

    fsds_gen, fsds_chk, fsds_var = plan_variable_cost_from_plan(
        plan,
        cost_per_1k_medium=cost_per_1k_output_tokens,
        check_cost=check_cost_per_call,
    )
    fsds_total = fsds_var + fsds_overhead_usd

    latency_reduction = 100.0 * (1.0 - fsds_span / max(baseline_span, 1))
    cost_savings = 100.0 * (1.0 - fsds_var / max(baseline_var, 1e-9))
    net_savings = baseline_var - fsds_total

    sr = float(success_rate if success_rate is not None else 0.0)
    sr0 = float(baseline_success_rate if baseline_success_rate is not None else sr)
    ev = sr * value_per_success_usd
    ev0 = sr0 * value_per_success_usd
    net_gain = net_economic_gain_usd(
        success_rate=sr,
        baseline_success_rate=sr0,
        fsds_variable_cost_usd=fsds_var,
        baseline_variable_cost_usd=baseline_var,
        fsds_overhead_usd=fsds_overhead_usd,
        value_per_success_usd=value_per_success_usd,
    )

    invest = implementation_cost_usd if implementation_cost_usd is not None else DEFAULT_IMPL_COST_USD
    roi = net_gain / max(invest, 1e-9)
    roi_pct = 100.0 * net_gain / max(baseline_var + ev0, 1e-9)

    return SoTEconomicsReport(
        baseline_latency_tokens=int(baseline_span),
        fsds_latency_tokens=int(fsds_span),
        latency_reduction_pct=float(latency_reduction),
        baseline_compute_cost_usd=float(baseline_gen),
        fsds_compute_cost_usd=float(fsds_gen),
        cost_savings_pct=float(cost_savings),
        check_cost_usd=float(fsds_chk),
        net_savings_usd=float(net_savings),
        fsds_overhead_usd=float(fsds_overhead_usd),
        roi_multiple=float(roi),
        baseline_total_cost_usd=float(baseline_var),
        fsds_total_cost_usd=float(fsds_total),
        baseline_check_cost_usd=float(baseline_chk),
        expected_value_usd=float(ev),
        baseline_expected_value_usd=float(ev0),
        net_economic_gain_usd=float(net_gain),
        roi_pct=float(roi_pct),
    )
