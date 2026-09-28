"""
MCTS + local search over budget hyperparameters to maximize success-rate proxy.

Search space (discrete): kappa, entropy_lambda, high-need L floor.
Uses deterministic success_probability from incremental_value simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .budget import SoTPlan, allocate_branch_budgets
from .economics import net_economic_gain_usd
from .incremental_value import success_probability, uniform_plan
from .pipeline import FSDSSoT
from .pricing import plan_variable_cost_from_plan, uniform_baseline_variable_cost_usd


@dataclass(frozen=True)
class BudgetHyperparams:
    kappa: float
    entropy_lambda: float
    need_floor: int

    def key(self) -> Tuple[float, float, int]:
        return (round(self.kappa, 3), round(self.entropy_lambda, 3), int(self.need_floor))


def _floors_from_need(need: np.ndarray, floor: int, cap: int) -> np.ndarray:
    out = np.zeros_like(need, dtype=int)
    for i, n in enumerate(need):
        if n >= 0.95:
            out[i] = min(floor, cap)
    return out


def plan_from_hyperparams(
    shift: np.ndarray,
    quality: np.ndarray,
    need: np.ndarray,
    hp: BudgetHyperparams,
    *,
    total_budget: int = 2800,
    latency_cap: int = 520,
) -> SoTPlan:
    floors = _floors_from_need(need, hp.need_floor, latency_cap)
    return allocate_branch_budgets(
        shift,
        quality,
        total_token_budget=total_budget,
        latency_cap_tokens=latency_cap,
        need_weights=need,
        min_tokens_by_branch=floors,
        kappa=hp.kappa,
        entropy_lambda=hp.entropy_lambda,
    )


def _reward(
    plan: SoTPlan,
    need: np.ndarray,
    quality: np.ndarray,
    *,
    latency_cap: int,
    uniform_plan_ref: SoTPlan | None = None,
) -> float:
    """Same net_economic_gain_usd as estimate_economics / ROI reports."""
    p = success_probability(plan, need, quality, uniform_L=latency_cap)
    B = len(plan.branch_budgets)
    if uniform_plan_ref is None:
        p0 = p
        baseline_var = uniform_baseline_variable_cost_usd(B, tokens_per_branch=latency_cap)
    else:
        p0 = success_probability(uniform_plan_ref, need, quality, uniform_L=latency_cap)
        _, _, baseline_var = plan_variable_cost_from_plan(uniform_plan_ref)
    _, _, fsds_var = plan_variable_cost_from_plan(plan)
    return net_economic_gain_usd(
        success_rate=p,
        baseline_success_rate=p0,
        fsds_variable_cost_usd=fsds_var,
        baseline_variable_cost_usd=baseline_var,
    )


def _neighbors(hp: BudgetHyperparams) -> List[BudgetHyperparams]:
    kappas = [0.85, 0.95, 1.0, 1.1, 1.2]
    lams = [0.0, 0.1, 0.15, 0.25, 0.35]
    floors = [0, 280, 360, 420, 480]
    out: List[BudgetHyperparams] = []
    for k in kappas:
        if abs(k - hp.kappa) > 1e-9:
            out.append(BudgetHyperparams(k, hp.entropy_lambda, hp.need_floor))
    for lam in lams:
        if abs(lam - hp.entropy_lambda) > 1e-9:
            out.append(BudgetHyperparams(hp.kappa, lam, hp.need_floor))
    for fl in floors:
        if fl != hp.need_floor:
            out.append(BudgetHyperparams(hp.kappa, hp.entropy_lambda, fl))
    return out[:12]


def mcts_search_success(
    shift: np.ndarray,
    quality: np.ndarray,
    need: np.ndarray,
    *,
    n_simulations: int = 120,
    c_explore: float = 1.4,
    latency_cap: int = 520,
    total_budget: int = 2800,
    seed: int = 2026,
) -> Tuple[BudgetHyperparams, float, SoTPlan]:
    rng = np.random.default_rng(seed)
    root = BudgetHyperparams(1.0, 0.15, 320)
    children: Dict[Tuple[float, float, int], List[BudgetHyperparams]] = {}
    visits: Dict[Tuple[float, float, int], int] = {}
    values: Dict[Tuple[float, float, int], float] = {}

    def ensure_children(hp: BudgetHyperparams) -> None:
        k = hp.key()
        if k not in children:
            children[k] = _neighbors(hp)

    for _ in range(n_simulations):
        path: List[BudgetHyperparams] = [root]
        hp = root
        ensure_children(hp)

        # Select
        while children.get(hp.key()):
            kids = children[hp.key()]
            untried = [c for c in kids if c.key() not in visits]
            if untried:
                hp = untried[int(rng.integers(len(untried)))]
                path.append(hp)
                break
            best = None
            best_score = -1e18
            for c in kids:
                ck = c.key()
                n = visits.get(ck, 0)
                if n == 0:
                    best = c
                    break
                q = values[ck] / n
                parent_n = max(visits.get(hp.key(), 1), 1)
                ucb = q + c_explore * np.sqrt(np.log(parent_n + 1) / (n + 1e-9))
                if ucb > best_score:
                    best_score = ucb
                    best = c
            hp = best if best is not None else hp
            path.append(hp)
            ensure_children(hp)

        plan = plan_from_hyperparams(
            shift, quality, need, hp, total_budget=total_budget, latency_cap=latency_cap
        )
        uni = uniform_plan(shift, quality, latency_cap_tokens=latency_cap, total_token_budget=total_budget)
        r = _reward(plan, need, quality, latency_cap=latency_cap, uniform_plan_ref=uni)

        for node in path:
            k = node.key()
            visits[k] = visits.get(k, 0) + 1
            values[k] = values.get(k, 0.0) + r

    # Best child of root by visit count
    ensure_children(root)
    best_hp = root
    best_q = -1e18
    for c in children[root.key()] + [root]:
        ck = c.key()
        n = visits.get(ck, 0)
        if n == 0:
            continue
        q = values[ck] / n
        if q > best_q:
            best_q = q
            best_hp = c

    plan = plan_from_hyperparams(
        shift, quality, need, best_hp, total_budget=total_budget, latency_cap=latency_cap
    )
    return best_hp, best_q, plan


def local_search_refine(
    shift: np.ndarray,
    quality: np.ndarray,
    need: np.ndarray,
    hp: BudgetHyperparams,
    *,
    max_steps: int = 24,
    latency_cap: int = 520,
    total_budget: int = 2800,
) -> Tuple[BudgetHyperparams, float, SoTPlan]:
    current = hp
    plan = plan_from_hyperparams(
        shift, quality, need, current, total_budget=total_budget, latency_cap=latency_cap
    )
    best_r = _reward(plan, need, quality, latency_cap=latency_cap)

    for _ in range(max_steps):
        improved = False
        for cand in _neighbors(current):
            p = plan_from_hyperparams(
                shift, quality, need, cand, total_budget=total_budget, latency_cap=latency_cap
            )
            uni = uniform_plan(shift, quality, latency_cap_tokens=latency_cap, total_token_budget=total_budget)
            r = _reward(p, need, quality, latency_cap=latency_cap, uniform_plan_ref=uni)
            if r > best_r + 1e-6:
                best_r = r
                current = cand
                plan = p
                improved = True
                break
        if not improved:
            break
    return current, best_r, plan


def optimize_budget_for_success(
    branch_embeddings: np.ndarray,
    branch_quality: np.ndarray,
    need: np.ndarray,
    *,
    mcts_sims: int = 100,
    seed: int = 2026,
) -> Dict[str, Any]:
    ref = branch_embeddings.mean(axis=0, keepdims=True)
    shift = np.linalg.norm(branch_embeddings - ref, axis=1)
    shift = shift / (shift.max() + 1e-9)
    q = np.asarray(branch_quality, dtype=float)
    need = np.asarray(need, dtype=float)

    hp_m, _, plan_m = mcts_search_success(
        shift, q, need, n_simulations=mcts_sims, seed=seed
    )
    hp, score, plan = local_search_refine(shift, q, need, hp_m)

    p_succ = success_probability(plan, need, q, uniform_L=520)
    p_uni = success_probability(
        uniform_plan(shift, q, latency_cap_tokens=520, total_token_budget=2800),
        need,
        q,
        uniform_L=520,
    )
    return {
        "hyperparams": {"kappa": hp.kappa, "entropy_lambda": hp.entropy_lambda, "need_floor": hp.need_floor},
        "success_rate_est": p_succ,
        "uniform_success_rate_est": p_uni,
        "success_lift": p_succ - p_uni,
        "search_score": score,
        "plan_total_tokens": sum(b.expansion_tokens for b in plan.branch_budgets),
        "plan_span_tokens": max(b.expansion_tokens for b in plan.branch_budgets),
    }
