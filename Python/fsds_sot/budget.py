from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .entropy_alloc import entropy, regularized_allocation_weights

ModelTier = Literal["small", "medium", "large"]


@dataclass
class BranchBudget:
    branch_id: int
    expansion_tokens: int
    check_budget: int
    model_tier: ModelTier
    shift_score: float
    quality_score: float


@dataclass
class SoTPlan:
    branch_budgets: list[BranchBudget]
    latency_cap_tokens: int
    total_token_budget: int
    use_parallel: bool
    clusters: list[list[int]]


def allocate_branch_budgets(
    shift_scores: np.ndarray,
    quality_scores: np.ndarray,
    *,
    total_token_budget: int = 4096,
    latency_cap_tokens: int = 512,
    min_tokens: int = 64,
    min_tokens_by_branch: np.ndarray | None = None,
    need_weights: np.ndarray | None = None,
    check_base: int = 1,
    check_scale: float = 2.0,
    kappa: float = 1.0,
    entropy_lambda: float = 0.15,
    temperature: float = 1.0,
) -> SoTPlan:
    """
    Water-filling with critical-path cap:
      L_b = clip(kappa * s_b * q_b, min_tokens, latency_cap_tokens)
    Check budget scales with concept-risk proxy (shift * (1-quality)).
    Model tier: large if high shift & low quality, small if stable.
    """
    s = np.asarray(shift_scores, dtype=float)
    q = np.asarray(quality_scores, dtype=float)
    s = s / (s.sum() + 1e-9)
    q = np.clip(q, 0.05, 1.0)

    nw = np.ones_like(s) if need_weights is None else np.asarray(need_weights, dtype=float)
    nw = nw / (nw.max() + 1e-9)
    raw = kappa * s * q * nw
    w = regularized_allocation_weights(raw, temperature=temperature, entropy_lambda=entropy_lambda)
    # Map weights to lengths: target total min(sum L, total_token_budget), cap each at C_lat
    target_sum = min(total_token_budget, len(w) * latency_cap_tokens)
    L = np.clip((w * target_sum).astype(int), min_tokens, latency_cap_tokens)
    _ = entropy(w)  # available for logging / diagnostics
    if min_tokens_by_branch is not None:
        floors = np.asarray(min_tokens_by_branch, dtype=int)
        L = np.maximum(L, np.clip(floors, min_tokens, latency_cap_tokens))

    # Respect total budget via proportional shrink if needed
    if L.sum() > total_token_budget:
        scale = total_token_budget / L.sum()
        L = np.maximum((L * scale).astype(int), min_tokens)

    risk = s * (1.0 - q)
    check = np.clip(np.round(check_base + check_scale * risk), 1, 5).astype(int)

    tiers: list[ModelTier] = []
    for r, si in zip(risk, s):
        if si > np.percentile(s, 75) and r > np.percentile(risk, 60):
            tiers.append("large")
        elif si < np.percentile(s, 40):
            tiers.append("small")
        else:
            tiers.append("medium")

    branches = [
        BranchBudget(
            branch_id=i,
            expansion_tokens=int(L[i]),
            check_budget=int(check[i]),
            model_tier=tiers[i],
            shift_score=float(shift_scores[i]),
            quality_score=float(quality_scores[i]),
        )
        for i in range(len(L))
    ]

    return SoTPlan(
        branch_budgets=branches,
        latency_cap_tokens=int(latency_cap_tokens),
        total_token_budget=int(total_token_budget),
        use_parallel=True,
        clusters=[],
    )
