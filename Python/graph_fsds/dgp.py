"""
Synthetic graph data-generating processes for validating graph FSDS.

Every scenario returns two snapshots of the **same node set** (a temporal graph):
an ``exist`` snapshot (control / W=0) and a ``new`` snapshot (treatment / W=1).
A snapshot is ``(graph, node_features, node_labels, community_assignment)``.

Scenarios mirror the parent project's covariate-shift / concept-drift DGPs,
lifted onto a stochastic block model so that drift can be injected at a chosen
scale:

    null              : new ~ exist (no drift anywhere)      -> FSDS should pass
    covariate_shift   : P(X) shifts on a subset of dims       -> covariate
    concept_drift     : P(Y|X) changes (label mechanism)      -> concept
    community_shift    : shift localised to specific communities-> community-scale
    structure_shift    : edge probabilities of a block change  -> structural
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import networkx as nx
import numpy as np


@dataclass
class GraphSnapshot:
    graph: nx.Graph
    features: np.ndarray          # (n_nodes, p)
    labels: np.ndarray            # (n_nodes,)
    communities: np.ndarray       # (n_nodes,) block membership
    nodelist: List


def _sbm_graph(block_sizes, p_in, p_out, features_by_block, rng) -> nx.Graph:
    K = len(block_sizes)
    probs = np.full((K, K), p_out)
    np.fill_diagonal(probs, p_in)
    seed = int(rng.integers(0, 2**31 - 1))
    G = nx.stochastic_block_model(block_sizes, probs, seed=seed)
    # ensure integer, contiguous node labels 0..n-1
    return nx.convert_node_labels_to_integers(G)


def _block_membership(block_sizes) -> np.ndarray:
    return np.concatenate([np.full(s, k) for k, s in enumerate(block_sizes)])


def _features_from_centroids(membership, centroids, rng):
    """Block-structured Gaussian features around *shared* block centroids.

    Only the per-node Gaussian noise is redrawn; the block centroids are fixed by
    the caller so that two snapshots share the same generating distribution
    unless a scenario deliberately perturbs the centroids.
    """
    n = membership.size
    p = centroids.shape[1]
    return rng.normal(size=(n, p)) + centroids[membership]


def _labels_from_features(X, beta, noise, rng, threshold):
    logits = X @ beta + rng.normal(scale=noise, size=X.shape[0])
    return (logits > threshold).astype(int)


def _make(scenario, *, n_per_block=60, n_blocks=4, p=12,
          p_in=0.18, p_out=0.02, seed=2026, strength=1.0, block_shift=1.4):
    rng = np.random.default_rng(seed)
    block_sizes = [n_per_block] * n_blocks
    membership = _block_membership(block_sizes)
    n = membership.size
    shift_dims = [0, 1, 2, 3]

    # ---- fixed, SHARED generating structure ------------------------------
    # A proper null shares these; a scenario perturbs only what it names.
    centroids = rng.normal(scale=block_shift, size=(n_blocks, p))
    beta = np.zeros(p)
    beta[:4] = np.array([2.0, 1.5, 1.0, 0.5])
    # fix the label threshold on the shared mechanism so P(Y|X) is comparable
    X_ref = _features_from_centroids(membership, centroids, np.random.default_rng(seed + 99))
    threshold = float(np.median(X_ref @ beta))

    # ---- shared topology -------------------------------------------------
    # A temporal graph with a fixed node set: both snapshots share the same
    # topology for feature/label drift (this is also what keeps the node-level
    # test calibrated -- see HierarchicalGraphEmbedding).  Only ``structure_shift``
    # rewires the graph, and that rewiring *is* the drift we want to catch.
    G_exist = _sbm_graph(block_sizes, p_in, p_out, None, rng)

    # ---- existing snapshot (control, W=0) --------------------------------
    X_exist = _features_from_centroids(membership, centroids, rng)
    y_exist = _labels_from_features(X_exist, beta, 0.6, rng, threshold)

    # ---- new snapshot (treatment, W=1): defaults == same process (null) --
    centroids_new = centroids.copy()
    beta_new = beta.copy()
    build_structure_shift = False

    if scenario == "null":
        pass
    elif scenario == "covariate_shift":
        centroids_new = centroids.copy()
        centroids_new[:, shift_dims] += strength * 1.6      # P(X) moves everywhere
    elif scenario == "concept_drift":
        beta_new = beta.copy()
        beta_new[:4] = beta_new[:4] * (-strength)           # P(Y|X) flips
    elif scenario == "community_shift":
        centroids_new = centroids.copy()
        centroids_new[[0, 1]] += strength * 2.0             # only blocks 0 & 1 move
    elif scenario == "structure_shift":
        build_structure_shift = True                        # block 0 gets denser
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    X_new = _features_from_centroids(membership, centroids_new, rng)
    y_new = _labels_from_features(X_new, beta_new, 0.6, rng, threshold)

    if build_structure_shift:
        probs = np.full((n_blocks, n_blocks), p_out)
        np.fill_diagonal(probs, p_in)
        probs[0, 0] = min(0.9, p_in * (1.0 + strength))
        s = int(rng.integers(0, 2**31 - 1))
        G_new = nx.convert_node_labels_to_integers(
            nx.stochastic_block_model(block_sizes, probs, seed=s))
    else:
        G_new = G_exist  # shared topology (fixed node set)

    nodelist = list(range(n))
    exist = GraphSnapshot(G_exist, X_exist, y_exist, membership, nodelist)
    new = GraphSnapshot(G_new, X_new, y_new, membership.copy(), nodelist)
    return exist, new


def make_scenario(scenario: str, **kwargs):
    """Public factory: return (exist_snapshot, new_snapshot) for a scenario."""
    return _make(scenario, **kwargs)


SCENARIOS = ["null", "covariate_shift", "concept_drift",
             "community_shift", "structure_shift"]
