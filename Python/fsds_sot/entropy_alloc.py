"""Entropy-regularized budget weights (avoid collapse onto one branch)."""

from __future__ import annotations

import numpy as np


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def regularized_allocation_weights(
    raw_scores: np.ndarray,
    *,
    temperature: float = 1.0,
    entropy_lambda: float = 0.15,
) -> np.ndarray:
    """
    Softmax with temperature + uniform mixing (entropy regularization).

    w = (1 - λ) * softmax(log raw / τ) + λ * (1/B)
    """
    raw = np.asarray(raw_scores, dtype=float)
    raw = np.maximum(raw, 1e-12)
    B = raw.size
    logits = np.log(raw) / max(temperature, 1e-6)
    logits -= logits.max()
    z = np.exp(logits)
    p = z / (z.sum() + 1e-12)
    lam = float(np.clip(entropy_lambda, 0.0, 1.0))
    w = (1.0 - lam) * p + lam * (1.0 / B)
    return w / (w.sum() + 1e-12)
