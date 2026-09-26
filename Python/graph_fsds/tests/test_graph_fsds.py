"""Fast smoke tests for the graph_fsds package.

These use tiny graphs and a small permutation budget so they run in seconds and
stay deterministic; they check plumbing and qualitative behaviour rather than
statistical power (which the demo exercises).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graph_fsds import (  # noqa: E402
    ConditionalHierarchicalFSDS,
    HierarchicalGraphEmbedding,
    cluster_stability,
    dgp,
)
from graph_fsds.fsds_core import (  # noqa: E402
    classify_drift_type,
    covariate_shift_test,
    domain_vimp,
    node_drift_scores,
)


def _embed(scenario, seed=0):
    exist, new = dgp.make_scenario(scenario, n_per_block=25, n_blocks=3, seed=seed)
    emb = HierarchicalGraphEmbedding(n_hops=2, standardize=True)
    he_e = emb.fit_transform(exist.graph, exist.features, exist.communities)
    he_n = emb.transform(new.graph, new.features, exist.communities)
    return exist, new, he_e, he_n


def test_dgp_shapes_and_scenarios():
    for sc in dgp.SCENARIOS:
        exist, new = dgp.make_scenario(sc, n_per_block=20, n_blocks=3, seed=1)
        assert exist.features.shape == new.features.shape
        assert exist.graph.number_of_nodes() == new.graph.number_of_nodes()
        assert set(np.unique(exist.labels)).issubset({0, 1})


def test_embedding_scales():
    _, _, he_e, he_n = _embed("null", seed=2)
    assert he_e.node_embeddings.shape[0] == he_e.n_nodes
    assert he_e.node_embeddings.shape[1] == len(he_e.feature_names)
    assert he_e.community_embeddings.shape[0] == len(he_e.community_ids)
    assert he_e.global_embedding.ndim == 1
    # standardized reference embedding is comparable to the new snapshot's
    assert he_e.node_embeddings.shape[1] == he_n.node_embeddings.shape[1]


def test_cluster_stability_null_is_high():
    exist, new = dgp.make_scenario("null", n_per_block=30, n_blocks=3, seed=3)
    stab = cluster_stability(exist.graph, new.graph)
    assert 0.0 <= stab.stability_score() <= 1.0
    assert stab.ari > 0.5  # same block structure -> partitions largely agree


def test_covariate_auc_separates_shift_from_null():
    # a genuine covariate shift is easier to separate than the null
    _, _, he_e_null, he_n_null = _embed("null", seed=4)
    _, _, he_e_cov, he_n_cov = _embed("covariate_shift", seed=4)
    X_null = np.vstack([he_e_null.node_embeddings, he_n_null.node_embeddings])
    W_null = np.r_[np.zeros(he_e_null.n_nodes), np.ones(he_n_null.n_nodes)]
    X_cov = np.vstack([he_e_cov.node_embeddings, he_n_cov.node_embeddings])
    W_cov = np.r_[np.zeros(he_e_cov.n_nodes), np.ones(he_n_cov.n_nodes)]
    auc_null = covariate_shift_test(X_null, W_null, n_perm=5, n_folds=3)["auc"]
    auc_cov = covariate_shift_test(X_cov, W_cov, n_perm=5, n_folds=3)["auc"]
    assert auc_cov > auc_null


def test_node_scores_and_domain_vimp_shapes():
    _, _, he_e, he_n = _embed("covariate_shift", seed=5)
    X = np.vstack([he_e.node_embeddings, he_n.node_embeddings])
    W = np.r_[np.zeros(he_e.n_nodes), np.ones(he_n.n_nodes)]
    scores = node_drift_scores(X, W, n_folds=3)
    assert scores.shape[0] == X.shape[0]
    assert np.all((scores >= 0) & (scores <= 1))
    v = domain_vimp(X, W, n_folds=3, n_repeats=3)["vimp"]
    assert v.shape[0] == X.shape[1]


def test_hierarchical_pipeline_runs():
    exist, new, he_e, he_n = _embed("community_shift", seed=6)
    stab = cluster_stability(exist.graph, new.graph)
    result = ConditionalHierarchicalFSDS(n_perm=8, n_folds=3, seed=6).run(
        he_e, he_n, exist.labels, new.labels, stability=stab)
    assert result.primary_level in {"global", "community", "node", "none"}
    assert result.overall_drift_type in {"covariate", "concept", "both", "none"}
    assert result.node_attribution is None or \
        result.node_attribution.shape[0] == he_e.n_nodes
    assert isinstance(result.report(), str)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
