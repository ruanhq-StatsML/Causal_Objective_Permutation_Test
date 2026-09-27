"""
Synthetic agentic SoT traces (ReAct-style skeleton branches).

Each episode = one user task decomposed into B skeleton branches:
  plan, retrieve, reason, act, verify (default labels).

Reference batch = stable agent policy; live batch = injected drift on
tool-heavy branches (retrieve/act embeddings + success rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

DEFAULT_BRANCH_NAMES = ("plan", "retrieve", "reason", "act", "verify")


@dataclass
class AgenticEpisode:
    trace_features: np.ndarray  # (p,)
    branch_embeddings: np.ndarray  # (B, d)
    branch_quality: np.ndarray  # (B,) verifier/PRM proxy in [0,1]
    success: float  # 0/1 task success
    branch_names: Tuple[str, ...]


def _branch_need_vector(names: Tuple[str, ...]) -> np.ndarray:
    """Latent 'needs budget' per branch type (agentic prior)."""
    need = []
    for n in names:
        if n in ("retrieve", "act"):
            need.append(1.0)
        elif n == "reason":
            need.append(0.7)
        elif n == "verify":
            need.append(0.5)
        else:
            need.append(0.35)
    return np.array(need, dtype=float)


def generate_agentic_batch(
    n_episodes: int,
    *,
    branches: int = 5,
    emb_dim: int = 24,
    seed: int = 0,
    live: bool = False,
    drift_strength: float = 1.2,
) -> List[AgenticEpisode]:
    rng = np.random.default_rng(seed)
    names = DEFAULT_BRANCH_NAMES[:branches]
    if len(names) < branches:
        names = names + tuple(f"step{i}" for i in range(len(names), branches))
    need = _branch_need_vector(names)

    episodes: List[AgenticEpisode] = []
    for _ in range(n_episodes):
        Z = rng.normal(size=(branches, emb_dim))
        # Trace scalars: tool calls, failures, latency proxy
        tool_calls = rng.integers(1, 8)
        tool_fail = rng.integers(0, 3 if not live else 6)
        latency = rng.normal(2.0 + 0.2 * tool_fail, 0.3)

        if live:
            # Covariate shift on retrieve/act blocks (first 8 dims of those rows)
            for bi, nm in enumerate(names):
                if nm in ("retrieve", "act"):
                    Z[bi, :8] += rng.normal(scale=drift_strength, size=8)
            tool_fail = rng.integers(2, 8)
            latency += 0.4

        q = np.clip(0.55 + 0.25 * rng.random(branches) - 0.08 * live * need, 0.05, 0.98)
        if live:
            for bi, nm in enumerate(names):
                if nm in ("retrieve", "act"):
                    q[bi] -= rng.uniform(0.15, 0.35)

        scalars = np.array([branches, tool_calls, tool_fail, latency], dtype=float)
        trace = np.concatenate([Z.mean(axis=0), scalars])

        # Success latent: depends on matching budget to need (evaluated in simulator)
        base_success = 0.7 - 0.06 * tool_fail + 0.05 * q.mean()
        success = float(rng.random() < np.clip(base_success, 0.05, 0.95))

        episodes.append(
            AgenticEpisode(
                trace_features=trace,
                branch_embeddings=Z,
                branch_quality=q,
                success=success,
                branch_names=names,
            )
        )
    return episodes


def episodes_to_arrays(episodes: List[AgenticEpisode]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([e.trace_features for e in episodes])
    Y = np.array([e.success for e in episodes], dtype=float)
    # Use last episode branch structure — for batch sim assume fixed B
    B = episodes[0].branch_embeddings.shape[0]
    Z = np.stack([e.branch_embeddings for e in episodes])  # (n, B, d)
    Q = np.stack([e.branch_quality for e in episodes])
    return X, Y, Z, Q
