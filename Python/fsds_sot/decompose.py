from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def _normalize_rows(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return Z / norms


def coupling_matrix(branch_embeddings: np.ndarray) -> np.ndarray:
    """HSIC proxy: squared cosine similarity (1 = strongly coupled)."""
    Z = _normalize_rows(branch_embeddings)
    sim = Z @ Z.T
    return np.clip(sim, 0.0, 1.0)


@dataclass
class DecomposabilityReport:
    redundancy_rate: float
    parallelizability: float
    balance: float
    n_clusters: int
    cluster_ids: np.ndarray
    decomposability: float
    allow_parallel: bool
    sequential_clusters: List[List[int]]


def spectral_clusters(H: np.ndarray, delta: float) -> Tuple[int, np.ndarray]:
    """Threshold coupling graph and count connected components."""
    B = H.shape[0]
    adj = (H >= delta).astype(int)
    np.fill_diagonal(adj, 0)
    visited = np.zeros(B, dtype=bool)
    labels = -np.ones(B, dtype=int)
    cid = 0
    for i in range(B):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        labels[i] = cid
        while stack:
            u = stack.pop()
            for v in range(B):
                if not visited[v] and adj[u, v]:
                    visited[v] = True
                    labels[v] = cid
                    stack.append(v)
        cid += 1
    return cid, labels


def evaluate_decomposability(
    branch_embeddings: np.ndarray,
    branch_shift_scores: np.ndarray,
    *,
    couple_delta: float = 0.55,
    redundancy_eps: float = 0.85,
    min_parallelizability: float = 0.35,
    max_redundancy: float = 0.25,
) -> DecomposabilityReport:
    B = branch_embeddings.shape[0]
    H = coupling_matrix(branch_embeddings)

    redundant_pairs = 0
    total_pairs = max(B * (B - 1) // 2, 1)
    for i in range(B):
        for j in range(i + 1, B):
            if H[i, j] >= redundancy_eps:
                redundant_pairs += 1
    redundancy_rate = redundant_pairs / total_pairs

    off = H[np.triu_indices(B, k=1)]
    mean_coupling = float(off.mean()) if off.size else 0.0
    parallelizability = 1.0 - mean_coupling

    s = np.asarray(branch_shift_scores, dtype=float)
    balance = 1.0 - (float(np.std(s)) / (float(np.mean(s)) + 1e-6))

    n_clusters, cluster_ids = spectral_clusters(H, couple_delta)
    decomposability = n_clusters / max(B, 1)

    allow_parallel = (
        parallelizability >= min_parallelizability and redundancy_rate <= max_redundancy
    )

    sequential_clusters: List[List[int]] = []
    for c in range(n_clusters):
        sequential_clusters.append(np.where(cluster_ids == c)[0].tolist())

    return DecomposabilityReport(
        redundancy_rate=float(redundancy_rate),
        parallelizability=float(parallelizability),
        balance=float(balance),
        n_clusters=int(n_clusters),
        cluster_ids=cluster_ids,
        decomposability=float(decomposability),
        allow_parallel=bool(allow_parallel),
        sequential_clusters=sequential_clusters,
    )
