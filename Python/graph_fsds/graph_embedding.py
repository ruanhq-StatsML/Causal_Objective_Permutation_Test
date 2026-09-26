"""
Hierarchical graph embeddings.

We build deterministic, training-free embeddings that mix graph *structure* with
node *features* at four nested scales, so that FSDS can localise a drift to the
scale where it actually lives:

    node       : per-node embedding  (feature diffusion + local structure)
    community  : per-community pooled embedding
    global     : whole-graph summary (feature diffusion pooled + structural stats)
    structural : purely topological node embedding (degree / clustering / core / ...)

The node embedding is an SGC-style feature propagation ``(D^-1/2 A D^-1/2)^k X``
concatenated with a handful of local structural descriptors.  It needs no
training, is stable across snapshots that share a node set, and keeps the
embedding dimensions interpretable enough for LOCO / domain VIMP to be read as
"which structural or feature channel moved".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler


def _normalized_adjacency(G: nx.Graph, nodelist: List) -> np.ndarray:
    A = nx.to_numpy_array(G, nodelist=nodelist, weight="weight")
    A = A + np.eye(A.shape[0])  # self-loops (renormalisation trick)
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (A * d_inv_sqrt).T * d_inv_sqrt  # D^-1/2 A D^-1/2


def _structural_node_features(G: nx.Graph, nodelist: List) -> np.ndarray:
    """Purely topological per-node descriptors (drift here == structure drift)."""
    deg = dict(G.degree())
    clus = nx.clustering(G)
    try:
        core = nx.core_number(G)
    except nx.NetworkXError:
        core = {n: 0 for n in G.nodes()}
    pr = nx.pagerank(G, max_iter=200) if G.number_of_edges() else {n: 0.0 for n in G.nodes()}
    avg_neigh_deg = nx.average_neighbor_degree(G)
    feats = []
    for n in nodelist:
        feats.append([
            deg.get(n, 0),
            clus.get(n, 0.0),
            core.get(n, 0),
            pr.get(n, 0.0),
            avg_neigh_deg.get(n, 0.0),
        ])
    return np.asarray(feats, dtype=float)


@dataclass
class HierarchicalEmbedding:
    """Container of the four embedding scales for one graph snapshot."""
    nodelist: List
    communities: np.ndarray                # community id per node (aligned to nodelist)
    node_embeddings: np.ndarray            # (n_nodes, d_node)
    structural_embeddings: np.ndarray      # (n_nodes, d_struct)
    community_embeddings: np.ndarray       # (n_comm, d_node)
    community_ids: np.ndarray              # (n_comm,)
    global_embedding: np.ndarray           # (d_global,)
    feature_names: List[str]
    structural_feature_names: List[str]

    @property
    def n_nodes(self) -> int:
        return len(self.nodelist)


class HierarchicalGraphEmbedding:
    """Compute :class:`HierarchicalEmbedding` for graph snapshots.

    Parameters
    ----------
    n_hops:
        Number of feature-propagation hops (SGC depth).
    include_structural:
        Concatenate local structural descriptors onto the node embedding.
    standardize:
        Fit a StandardScaler on the reference snapshot and reuse it, so that the
        two snapshots live in a common, comparable embedding space (this is what
        makes column-wise VIMP meaningful across batches).
    """

    def __init__(self, n_hops: int = 2, include_structural: bool = True,
                 include_raw_features: bool = True, include_diffusion: bool = False,
                 standardize: bool = True):
        self.n_hops = int(n_hops)
        self.include_structural = include_structural
        self.include_raw_features = include_raw_features
        # NOTE on ``include_diffusion``: the k-hop feature diffusion S^k X is a
        # lovely representation, but it *couples* nodes through the (single)
        # graph.  A flexible domain classifier can then fingerprint which graph a
        # node came from via higher-order structure, which inflates the
        # node-level permutation test's type-I error (an empirical null AUC ~0.8
        # even when nothing has drifted).  It is therefore OFF by default and
        # should only be switched on for exploratory attribution, never for the
        # calibrated drift test.  The calibrated node representation is the raw
        # node features (exchangeable across nodes) plus per-snapshot-standardized
        # structural descriptors (inert under a shared topology, and the carrier
        # of genuine *structural* drift when the topology changes).
        self.include_diffusion = include_diffusion
        # ``standardize`` = per-snapshot standardization of structure-derived
        # columns, removing each graph's nuisance offset/scale.
        self.standardize = standardize

    # -- node scale -------------------------------------------------------- #
    def _raw_node_embedding(self, G: nx.Graph, nodelist, node_features):
        S = _normalized_adjacency(G, nodelist)
        H = np.asarray(node_features, dtype=float)
        for _ in range(self.n_hops):
            H = S @ H
        struct = _structural_node_features(G, nodelist)
        return H, struct

    def fit_transform(self, G, node_features, communities=None, nodelist=None):
        """Embed the reference snapshot (kept symmetric with :meth:`transform`)."""
        return self.transform(G, node_features, communities, nodelist)

    def transform(self, G, node_features, communities=None, nodelist=None):
        """Embed a snapshot; structure-derived columns are standardized in-snapshot."""
        nodelist = list(G.nodes()) if nodelist is None else list(nodelist)
        H, struct = self._raw_node_embedding(G, nodelist, node_features)
        return self._assemble(G, nodelist, H, struct, communities, node_features)

    # -- assembly across scales ------------------------------------------- #
    def _assemble(self, G, nodelist, H, struct, communities, node_features):
        n_feat = H.shape[1]

        def _snap_std(M):
            return StandardScaler().fit_transform(M) if self.standardize else M

        # structure-derived columns: standardized *within this snapshot*
        diff_s = _snap_std(H)
        struct_s = _snap_std(struct)

        raw = np.asarray(node_features, dtype=float)
        struct_names = ["degree", "clustering", "coreness", "pagerank", "avg_neigh_deg"]

        blocks, names = [], []
        if self.include_raw_features:
            blocks.append(raw)
            names += [f"feat_{i}" for i in range(raw.shape[1])]
        if self.include_diffusion:
            blocks.append(diff_s)
            names += [f"diff_feat_{i}" for i in range(n_feat)]
        if self.include_structural:
            blocks.append(struct_s)
            names += [f"struct_{s}" for s in struct_names]
        if not blocks:  # safety: never return an empty embedding
            blocks.append(raw)
            names += [f"feat_{i}" for i in range(raw.shape[1])]

        node_emb = np.hstack(blocks)
        node_names = names

        if communities is None:
            communities = _detect_communities(G, nodelist)
        communities = np.asarray(communities)

        comm_ids = np.unique(communities)
        comm_emb = np.vstack([
            node_emb[communities == c].mean(axis=0) for c in comm_ids
        ])

        # global scale: pooled node embedding (mean+std) + graph-level structure
        global_emb = np.concatenate([
            node_emb.mean(axis=0),
            node_emb.std(axis=0),
            _global_structural_summary(G),
        ])

        return HierarchicalEmbedding(
            nodelist=nodelist,
            communities=communities,
            node_embeddings=node_emb,
            structural_embeddings=struct_s,
            community_embeddings=comm_emb,
            community_ids=comm_ids,
            global_embedding=global_emb,
            feature_names=node_names,
            structural_feature_names=struct_names,
        )


def _global_structural_summary(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    return np.asarray([
        n,
        m,
        np.mean(degrees) if degrees else 0.0,
        np.std(degrees) if degrees else 0.0,
        nx.transitivity(G) if m else 0.0,
        nx.density(G),
    ], dtype=float)


def _detect_communities(G: nx.Graph, nodelist) -> np.ndarray:
    """Louvain community detection, aligned to ``nodelist``."""
    try:
        import community as community_louvain  # python-louvain
        partition = community_louvain.best_partition(G, random_state=0)
    except Exception:
        # fall back to greedy modularity communities
        comms = list(nx.algorithms.community.greedy_modularity_communities(G))
        partition = {}
        for cid, cset in enumerate(comms):
            for node in cset:
                partition[node] = cid
    return np.asarray([partition.get(n, 0) for n in nodelist])


def detect_communities(G: nx.Graph, nodelist=None) -> np.ndarray:
    """Public helper: community id per node aligned to ``nodelist``."""
    nodelist = list(G.nodes()) if nodelist is None else list(nodelist)
    return _detect_communities(G, nodelist)
