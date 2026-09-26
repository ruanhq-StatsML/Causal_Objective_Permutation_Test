"""
Cluster stability: making "the clustering itself moved" an explicit FSDS signal.

Graph-clustering attribution is a *two-level* problem: first, did the community
structure move at all (and by how much)?  second, which features / nodes drove
that move?  This module handles the first level.  It compares the community
partitions of the existing and new snapshots (which share a node set in the
temporal-graph setting) and reports:

    * ARI / NMI                : global partition agreement
    * modularity change        : did the partition get better/worse structured
    * per-community Jaccard     : which communities are stable vs shifted

The scalar ``stability_score`` doubles as a *prior* for the hierarchical FSDS:
when it is low, jump straight to community-scale attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .graph_embedding import detect_communities


def _modularity(G: nx.Graph, labels, nodelist) -> float:
    groups: Dict[int, set] = {}
    for node, lab in zip(nodelist, labels):
        groups.setdefault(int(lab), set()).add(node)
    communities = list(groups.values())
    if not communities or G.number_of_edges() == 0:
        return 0.0
    try:
        return float(nx.algorithms.community.modularity(G, communities))
    except Exception:
        return 0.0


@dataclass
class ClusterStability:
    """Explicit representation of clustering drift between two snapshots."""
    ari: float
    nmi: float
    modularity_exist: float
    modularity_new: float
    modularity_change: float
    n_communities_exist: int
    n_communities_new: int
    n_stable_communities: int
    n_shifted_communities: int
    stable_communities: List[int] = field(default_factory=list)
    shifted_communities: List[int] = field(default_factory=list)
    community_jaccard: Dict[int, float] = field(default_factory=dict)

    def is_stable(self, ari_thresh: float = 0.8, mod_thresh: float = 0.1) -> bool:
        return self.ari > ari_thresh and abs(self.modularity_change) < mod_thresh

    def stability_score(self) -> float:
        """Composite stability in [0, 1] (1 == identical clustering)."""
        ari_score = max(0.0, self.ari)
        mod_score = 1.0 - min(abs(self.modularity_change), 1.0)
        return float(0.5 * ari_score + 0.5 * mod_score)

    def summary(self) -> str:
        return (
            f"ARI={self.ari:.3f} NMI={self.nmi:.3f} "
            f"Δmod={self.modularity_change:+.3f} "
            f"stable={self.n_stable_communities}/{self.n_communities_exist} "
            f"shifted={self.n_shifted_communities} "
            f"score={self.stability_score():.3f}"
        )


def _match_communities(labels_a, labels_b, nodelist):
    """Best-overlap match of each 'exist' community to a 'new' community.

    Returns per-community Jaccard of the best match.  Shared node set assumed.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    jaccard = {}
    for ca in np.unique(labels_a):
        set_a = set(np.asarray(nodelist)[labels_a == ca])
        best = 0.0
        for cb in np.unique(labels_b):
            set_b = set(np.asarray(nodelist)[labels_b == cb])
            union = len(set_a | set_b)
            if union:
                best = max(best, len(set_a & set_b) / union)
        jaccard[int(ca)] = float(best)
    return jaccard


def cluster_stability(
    G_exist: nx.Graph,
    G_new: nx.Graph,
    labels_exist=None,
    labels_new=None,
    nodelist=None,
    jaccard_thresh: float = 0.6,
) -> ClusterStability:
    """Compute :class:`ClusterStability` between two graph snapshots."""
    if nodelist is None:
        nodelist = sorted(set(G_exist.nodes()) & set(G_new.nodes()))
    nodelist = list(nodelist)

    if labels_exist is None:
        labels_exist = detect_communities(G_exist, nodelist)
    if labels_new is None:
        labels_new = detect_communities(G_new, nodelist)
    labels_exist = np.asarray(labels_exist)
    labels_new = np.asarray(labels_new)

    ari = float(adjusted_rand_score(labels_exist, labels_new))
    nmi = float(normalized_mutual_info_score(labels_exist, labels_new))
    mod_e = _modularity(G_exist, labels_exist, nodelist)
    mod_n = _modularity(G_new, labels_new, nodelist)

    jaccard = _match_communities(labels_exist, labels_new, nodelist)
    stable = [c for c, j in jaccard.items() if j >= jaccard_thresh]
    shifted = [c for c, j in jaccard.items() if j < jaccard_thresh]

    return ClusterStability(
        ari=ari,
        nmi=nmi,
        modularity_exist=mod_e,
        modularity_new=mod_n,
        modularity_change=mod_n - mod_e,
        n_communities_exist=int(len(np.unique(labels_exist))),
        n_communities_new=int(len(np.unique(labels_new))),
        n_stable_communities=len(stable),
        n_shifted_communities=len(shifted),
        stable_communities=sorted(stable),
        shifted_communities=sorted(shifted),
        community_jaccard=jaccard,
    )
