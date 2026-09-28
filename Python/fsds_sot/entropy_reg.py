"""Entropy-regularized allocation weights (avoid budget collapse / starvation)."""

from __future__ import annotations

import numpy as np


def allocation_entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    p = p / (p.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def regularize_weights(
    raw: np.ndarray,
    *,
    temperature: float = 1.0,
    entropy_coef: float = 0.15,
    min_entropy_ratio: float = 0.55,
) -> tuple[np.ndarray, float]:
    """
    Map nonnegative scores -> branch weight shares with entropy floor.

    Combines tempered softmax with a convex mix toward uniform so
    H(p) >= min_entropy_ratio * log(B) (approximate).
    """
    r = np.asarray(raw, dtype=float)
    r = np.maximum(r, 1e-9)
    B = r.size
    if B <= 1:
        return np.ones_like(r), 0.0

    tau = max(temperature, 1e-6)
    logits = np.log(r) / tau
    logits -= logits.max()
    p = np.exp(logits)
    p = p / p.sum()

    h_max = float(np.log(B))
    h_target = min_entropy_ratio * h_max
    h = allocation_entropy(p)
    if h < h_target and h_max > 1e-9:
        # alpha increases until entropy reaches target (binary search)
        lo, hi = 0.0, 1.0
        uniform = np.ones(B) / B
        for _ in range(24):
            mid = (lo + hi) / 2
            mix = (1.0 - mid) * p + mid * uniform
            if allocation_entropy(mix) < h_target:
                lo = mid
            else:
                hi = mid
        p = (1.0 - hi) * p + hi * uniform
        h = allocation_entropy(p)

    # Optional entropy bonus shaping (tilt toward under-served branches)
    if entropy_coef > 0:
        bonus = entropy_coef * (1.0 / B - p)
        p = p + bonus
        p = np.maximum(p, 1e-9)
        p = p / p.sum()
        h = allocation_entropy(p)

    return p, h
