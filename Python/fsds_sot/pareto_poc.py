"""
Pareto POC: cost vs success under KPI non-inferiority guard.

Sweeps FSDS budget knobs (kappa, need floor on high-need branches) on agentic simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .incremental_value import PolicyMetrics, simulate_episode_outcome, uniform_plan
from .pipeline import FSDSSoT
from .plan_metrics import compare_to_uniform


@dataclass
class ParetoPoint:
    label: str
    kappa: float
    need_floor_high: int
    mean_success: float
    mean_cost_usd: float
    mean_total_tokens: float
    mean_span_tokens: float
    mean_checks: float
    success_delta_vs_uniform: float
    cost_reduction_pct_vs_uniform: float
    total_token_reduction_pct: float
    checks_reduction_pct: float
    kpi_pass: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _high_need_floors(need: np.ndarray, floor: int, cap: int) -> np.ndarray:
    floors = np.zeros_like(need, dtype=int)
    for i, n in enumerate(need):
        if n >= 0.95:
            floors[i] = min(floor, cap)
    return floors


def evaluate_config(
    episodes_Z: np.ndarray,
    episodes_Q: np.ndarray,
    need: np.ndarray,
    X_old: np.ndarray,
    X_new: np.ndarray,
    *,
    label: str,
    kappa: float,
    need_floor: int,
    latency_cap: int = 520,
    total_budget: int = 2800,
    seed: int = 2026,
    uniform_metrics: Optional[PolicyMetrics] = None,
    success_epsilon: float = 0.03,
) -> Tuple[ParetoPoint, PolicyMetrics]:
    rng = np.random.default_rng(seed)
    ctrl = FSDSSoT(seed=seed)
    n = episodes_Z.shape[0]
    floors = _high_need_floors(need, need_floor, latency_cap)

    succ, spans, totals, costs, checks = [], [], [], [], []
    last_plan = None
    for i in range(n):
        Z = episodes_Z[i]
        q = episodes_Q[i]
        ref = Z.mean(axis=0, keepdims=True)
        shift = np.linalg.norm(Z - ref, axis=1)
        shift = shift / (shift.max() + 1e-9)
        plan, _ = ctrl.fit_plan_branches(
            Z,
            q,
            total_token_budget=total_budget,
            latency_cap_tokens=latency_cap,
            need_weights=need,
            min_tokens_by_branch=floors,
            budget_kappa=kappa,
            entropy_lambda=0.05,
        )
        last_plan = plan
        s, sp, tot, c, ch = simulate_episode_outcome(plan, need, q, uniform_L=latency_cap, rng=rng)
        succ.append(s)
        spans.append(sp)
        totals.append(tot)
        costs.append(c)
        checks.append(ch)

    metrics = PolicyMetrics(
        name=label,
        mean_success=float(np.mean(succ)),
        mean_span_tokens=float(np.mean(spans)),
        mean_total_tokens=float(np.mean(totals)),
        mean_cost_usd=float(np.mean(costs)),
        mean_checks=float(np.mean(checks)),
    )

    if uniform_metrics is None:
        raise ValueError("uniform_metrics required")

    cmp = compare_to_uniform(last_plan, L_uniform=latency_cap) if last_plan else {}
    sd = metrics.mean_success - uniform_metrics.mean_success
    cr = 100.0 * (1.0 - metrics.mean_cost_usd / max(uniform_metrics.mean_cost_usd, 1e-9))
    kpi = sd >= -success_epsilon

    point = ParetoPoint(
        label=label,
        kappa=kappa,
        need_floor_high=need_floor,
        mean_success=metrics.mean_success,
        mean_cost_usd=metrics.mean_cost_usd,
        mean_total_tokens=metrics.mean_total_tokens,
        mean_span_tokens=metrics.mean_span_tokens,
        mean_checks=metrics.mean_checks,
        success_delta_vs_uniform=sd,
        cost_reduction_pct_vs_uniform=cr,
        total_token_reduction_pct=cmp.get("total_token_reduction_pct", 0.0),
        checks_reduction_pct=cmp.get("checks_reduction_pct", 0.0),
        kpi_pass=bool(kpi),
    )
    return point, metrics


def run_pareto_sweep(
    episodes_Z: np.ndarray,
    episodes_Q: np.ndarray,
    need: np.ndarray,
    X_old: np.ndarray,
    X_new: np.ndarray,
    *,
    seed: int = 2026,
    success_epsilon: float = 0.03,
    kappas: Optional[List[float]] = None,
    need_floors: Optional[List[int]] = None,
) -> Dict[str, Any]:
    kappas = kappas or [0.9, 1.0, 1.15]
    need_floors = need_floors or [0, 320, 400]

    rng = np.random.default_rng(seed)
    ctrl = FSDSSoT(seed=seed)
    n = episodes_Z.shape[0]

    # Uniform baseline
    succ, spans, totals, costs, checks = [], [], [], [], []
    for i in range(n):
        Z = episodes_Z[i]
        q = episodes_Q[i]
        ref = Z.mean(axis=0, keepdims=True)
        shift = np.linalg.norm(Z - ref, axis=1)
        shift = shift / (shift.max() + 1e-9)
        plan = uniform_plan(shift, q, latency_cap_tokens=520, total_token_budget=2800)
        s, sp, tot, c, ch = simulate_episode_outcome(plan, need, q, uniform_L=520, rng=rng)
        succ.append(s)
        spans.append(sp)
        totals.append(tot)
        costs.append(c)
        checks.append(ch)

    uniform = PolicyMetrics(
        name="uniform_sot",
        mean_success=float(np.mean(succ)),
        mean_span_tokens=float(np.mean(spans)),
        mean_total_tokens=float(np.mean(totals)),
        mean_cost_usd=float(np.mean(costs)),
        mean_checks=float(np.mean(checks)),
    )

    points: List[ParetoPoint] = []
    for kappa in kappas:
        for floor in need_floors:
            label = f"fsds_k{kappa:.2f}_floor{floor}"
            pt, _ = evaluate_config(
                episodes_Z,
                episodes_Q,
                need,
                X_old,
                X_new,
                label=label,
                kappa=kappa,
                need_floor=floor,
                seed=seed,
                uniform_metrics=uniform,
                success_epsilon=success_epsilon,
            )
            points.append(pt)

    # Default fsds (no need floor) for reference
    pt0, _ = evaluate_config(
        episodes_Z,
        episodes_Q,
        need,
        X_old,
        X_new,
        label="fsds_default",
        kappa=1.0,
        need_floor=0,
        seed=seed,
        uniform_metrics=uniform,
        success_epsilon=success_epsilon,
    )
    points.append(pt0)

    passing = [p for p in points if p.kpi_pass]
    passing.sort(key=lambda p: (-p.cost_reduction_pct_vs_uniform, -p.mean_success))
    recommended = passing[0] if passing else max(points, key=lambda p: p.mean_success)

    frontier = []
    for p in sorted(points, key=lambda x: x.mean_cost_usd):
        dominated = False
        for q in points:
            if q.mean_cost_usd <= p.mean_cost_usd and q.mean_success >= p.mean_success and (
                q.mean_cost_usd < p.mean_cost_usd or q.mean_success > p.mean_success
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(p.label)

    return {
        "uniform_baseline": uniform.to_dict(),
        "success_epsilon": success_epsilon,
        "points": [p.to_dict() for p in points],
        "pareto_frontier_labels": frontier,
        "kpi_pass_count": len(passing),
        "recommended": recommended.to_dict(),
        "interpretation": (
            "KPI guard: success within epsilon of uniform. "
            "Among passing configs, pick max cost reduction (Pareto POC)."
        ),
    }
