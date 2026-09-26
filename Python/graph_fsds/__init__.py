"""
graph_fsds
==========

Applying the FSDS (Feature Selection under Distribution Shift) meta-learner
objective-function framework to **graph embeddings**.

This package sits *on top of* the causal-objective permutation test that powers
this repository (``DRPerm`` / PO-risk, ``RRPerm`` / R-risk and the LOCO variable
importance). It turns the raw "is there a shift, and which feature drives it?"
signal into a *structured prior for graph learning*:

    - WHERE does the drift live?   (global / community / node scale)
    - WHAT kind of drift is it?    (covariate P(X) vs concept P(Y|X))
    - HOW MUCH has clustering moved? (ARI / NMI / modularity stability)

The public surface is intentionally small:

    >>> from graph_fsds import (
    ...     HierarchicalGraphEmbedding,
    ...     ConditionalHierarchicalFSDS,
    ...     cluster_stability,
    ... )

See ``graph_fsds.demo`` for an end-to-end runnable example.
"""

from .fsds_core import (
    FSDSResult,
    fsds_test,
    loco_vimp,
    domain_vimp,
    classify_drift_type,
    node_drift_scores,
)
from .graph_embedding import HierarchicalGraphEmbedding, HierarchicalEmbedding
from .cluster_stability import ClusterStability, cluster_stability
from .hierarchical_fsds import ConditionalHierarchicalFSDS, HierarchicalDrift
from . import dgp
from . import visualize

__all__ = [
    "FSDSResult",
    "fsds_test",
    "loco_vimp",
    "domain_vimp",
    "classify_drift_type",
    "node_drift_scores",
    "HierarchicalGraphEmbedding",
    "HierarchicalEmbedding",
    "ClusterStability",
    "cluster_stability",
    "ConditionalHierarchicalFSDS",
    "HierarchicalDrift",
    "dgp",
    "visualize",
]
