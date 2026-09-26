"""
Conditional hierarchical FSDS.

The four embedding scales are *not* tested independently.  Each level is
conditioned on the level above it, both to save permutation budget and to make
the attribution coherent:

    global  : pool every node, run the causal-objective test on the full batch.
              This decides the *drift type* (covariate / concept / both / none).
              -> if nothing is significant globally, stop early.

    community : for every community, run a scale-appropriate test:
                  covariate-type drift -> RF-domain VIMP  (P(X) separation)
                  concept-type  drift  -> PO-risk LOCO     (P(Y|X) movement)
                This decides *which communities* moved.

    node    : per-node drift score (cross-fitted domain identifiability),
              refined inside the communities flagged above.
              This decides *which nodes* drove it.

The aggregation then reports the primary level, so a downstream graph learner
knows *where* to adapt, *what* to adapt to and *how much*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from .cluster_stability import ClusterStability
from .fsds_core import (
    FSDSResult,
    classify_drift_type,
    domain_vimp,
    fsds_test,
    loco_vimp,
    node_drift_scores,
)
from .graph_embedding import HierarchicalEmbedding


def _top_k(vimp: np.ndarray, k: int = 5) -> List[int]:
    if vimp is None:
        return []
    return [int(i) for i in np.argsort(-vimp)[:k]]


@dataclass
class HierarchicalDrift:
    """Aggregated multi-scale attribution result."""
    primary_level: str = "none"                 # global / community / node / none
    overall_drift_type: str = "none"            # covariate / concept / both / none
    stability_score: float = float("nan")
    global_result: Optional[FSDSResult] = None
    community_results: Dict[int, FSDSResult] = field(default_factory=dict)
    shifted_communities: List[int] = field(default_factory=list)
    node_attribution: Optional[np.ndarray] = None      # score per node (aligned)
    nodelist: List = field(default_factory=list)
    top_global_dims: List[int] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = ["=" * 68, "FSDS Hierarchical Drift Attribution", "=" * 68]
        lines.append(f"primary level      : {self.primary_level}")
        lines.append(f"overall drift type : {self.overall_drift_type}")
        if not np.isnan(self.stability_score):
            lines.append(f"cluster stability  : {self.stability_score:.3f}")
        if self.global_result is not None:
            lines.append("")
            lines.append("[global] " + self.global_result.summary())
            if self.feature_names and self.top_global_dims:
                named = ", ".join(
                    self.feature_names[i] for i in self.top_global_dims[:5]
                    if i < len(self.feature_names)
                )
                lines.append(f"         top driving dims: {named}")
        if self.community_results:
            lines.append("")
            lines.append(f"[community] {len(self.shifted_communities)} shifted "
                         f"of {len(self.community_results)} tested")
            for cid, res in sorted(self.community_results.items()):
                flag = "SHIFT" if res.significant else "  ok "
                lines.append(f"   community {cid:>3} [{flag}] " + res.summary())
        if self.node_attribution is not None:
            top_nodes = np.argsort(-self.node_attribution)[:8]
            lines.append("")
            lines.append("[node] top-8 drifting nodes (id: score):")
            lines.append("   " + ", ".join(
                f"{self.nodelist[i]}:{self.node_attribution[i]:.2f}" for i in top_nodes))
        lines.append("=" * 68)
        return "\n".join(lines)


class ConditionalHierarchicalFSDS:
    """Run conditioned global -> community -> node FSDS attribution."""

    def __init__(
        self, *,
        risk: str = "po",
        n_perm: int = 100,
        n_folds: int = 4,
        alpha: float = 0.05,
        domain_model: str = "rf",
        min_community_size: int = 12,
        seed: int = 2026,
        verbose: bool = False,
    ):
        self.risk = risk
        self.n_perm = n_perm
        self.n_folds = n_folds
        self.alpha = alpha
        self.domain_model = domain_model
        self.min_community_size = min_community_size
        self.seed = seed
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # -- global scale ------------------------------------------------------ #
    def _global(self, X, Y, W) -> FSDSResult:
        dtype, cov_p, con_p = classify_drift_type(
            X, Y, W, n_perm=self.n_perm, n_folds=self.n_folds, alpha=self.alpha,
            risk=self.risk, domain_model=self.domain_model, seed=self.seed)
        # attribution: domain VIMP (covariate) + LOCO VIMP (concept), combined
        dvimp = domain_vimp(X, W, domain_model=self.domain_model,
                            n_folds=self.n_folds, seed=self.seed)["vimp"]
        if dtype in ("concept", "both"):
            lvimp = loco_vimp(X, Y, W, risk=self.risk, n_folds=self.n_folds,
                              domain_model=self.domain_model, seed=self.seed)["vimp"]
        else:
            lvimp = np.zeros_like(dvimp)
        combined = _normalize(dvimp) + _normalize(lvimp)
        reject = dtype != "none"
        return FSDSResult(
            scale="global", risk=self.risk,
            statistic=float(np.max(combined)),
            p_value=min(cov_p, con_p),
            reject=reject, alpha=self.alpha, drift_type=dtype,
            covariate_p=cov_p, concept_p=con_p,
            vimp=combined, vimp_rank=np.argsort(-combined),
            top_features=_top_k(combined), n_samples=len(W), n_features=X.shape[1],
        )

    # -- community scale --------------------------------------------------- #
    def _community(self, X, Y, W, communities, drift_type) -> Dict[int, FSDSResult]:
        results: Dict[int, FSDSResult] = {}
        method = "covariate" if drift_type in ("covariate", "both", "none") else "concept"
        for cid in np.unique(communities):
            mask = communities == cid
            if mask.sum() < self.min_community_size:
                continue
            Wc = W[mask]
            if len(np.unique(Wc)) < 2:
                continue
            Xc, Yc = X[mask], Y[mask]
            seed_c = self.seed + 1000 + int(cid)
            if method == "concept":
                test = fsds_test(Xc, Yc, Wc, risk=self.risk, n_perm=self.n_perm,
                                 n_folds=min(self.n_folds, 3), alpha=self.alpha,
                                 domain_model=self.domain_model, seed=seed_c)
                vimp = loco_vimp(Xc, Yc, Wc, risk=self.risk,
                                 n_folds=min(self.n_folds, 3),
                                 domain_model=self.domain_model, seed=seed_c)["vimp"]
                p_value, reject, stat = test["p_value"], test["reject"], test["statistic"]
            else:
                from .fsds_core import covariate_shift_test
                test = covariate_shift_test(Xc, Wc, n_perm=self.n_perm,
                                            n_folds=min(self.n_folds, 3), alpha=self.alpha,
                                            domain_model=self.domain_model, seed=seed_c)
                vimp = domain_vimp(Xc, Wc, domain_model=self.domain_model,
                                   n_folds=min(self.n_folds, 3), seed=seed_c)["vimp"]
                p_value, reject, stat = test["p_value"], test["reject"], test["auc"]
            results[int(cid)] = FSDSResult(
                scale=f"comm[{cid}]", risk=self.risk, statistic=float(stat),
                p_value=float(p_value), reject=bool(reject), alpha=self.alpha,
                drift_type=drift_type, vimp=vimp, vimp_rank=np.argsort(-vimp),
                top_features=_top_k(vimp), n_samples=int(mask.sum()),
                n_features=X.shape[1],
            )
            self._log(f"    community {cid}: " + results[int(cid)].summary())
        return results

    # -- orchestration ----------------------------------------------------- #
    def run(
        self,
        he_exist: HierarchicalEmbedding,
        he_new: HierarchicalEmbedding,
        y_exist,
        y_new,
        stability: Optional[ClusterStability] = None,
    ) -> HierarchicalDrift:
        """Execute the conditioned global -> community -> node cascade."""
        X = np.vstack([he_exist.node_embeddings, he_new.node_embeddings])
        # Pool-standardize the stacked embeddings.  Standardizing on the two
        # batches jointly keeps the batches exchangeable under the null, which is
        # exactly what the permute-then-refit test needs to stay calibrated (a
        # per-batch / reference-only scaler would leave the new batch off-centre
        # and manufacture a spurious covariate shift).
        X = StandardScaler().fit_transform(X)
        Y = np.concatenate([np.asarray(y_exist, float), np.asarray(y_new, float)])
        W = np.concatenate([np.zeros(he_exist.n_nodes), np.ones(he_new.n_nodes)])
        # community id per stacked row: use reference-snapshot communities, shared
        # node set means he_exist.communities aligns with he_new by node.
        communities = np.concatenate([he_exist.communities, he_exist.communities]) \
            if he_exist.n_nodes == he_new.n_nodes \
            else np.concatenate([he_exist.communities, he_new.communities])

        agg = HierarchicalDrift(
            nodelist=list(he_exist.nodelist),
            feature_names=list(he_exist.feature_names),
        )
        if stability is not None:
            agg.stability_score = stability.stability_score()

        # ---- level 1: global ---------------------------------------------
        self._log("[global] running pooled causal-objective test ...")
        g = self._global(X, Y, W)
        agg.global_result = g
        agg.overall_drift_type = g.drift_type
        agg.top_global_dims = list(g.top_features)
        self._log("[global] " + g.summary())

        # a low clustering-stability prior forces community-scale attribution
        force_community = stability is not None and not stability.is_stable()

        if not g.significant and not force_community:
            agg.primary_level = "none"
            return agg

        # ---- level 2: community (conditioned on drift type) --------------
        self._log("[community] running conditioned per-community tests ...")
        comm = self._community(X, Y, W, communities, g.drift_type)
        agg.community_results = comm
        agg.shifted_communities = sorted(c for c, r in comm.items() if r.significant)

        # ---- level 3: node ------------------------------------------------
        self._log("[node] scoring per-node drift ...")
        node_scores = node_drift_scores(X, W, domain_model=self.domain_model,
                                        n_folds=self.n_folds, seed=self.seed)
        # fold back to per-node (shared node set): average exist/new halves
        if he_exist.n_nodes == he_new.n_nodes:
            n = he_exist.n_nodes
            agg.node_attribution = 0.5 * (node_scores[:n] + node_scores[n:])
        else:
            agg.node_attribution = node_scores[: he_exist.n_nodes]

        # ---- aggregate primary level -------------------------------------
        n_tested = max(1, len(comm))
        shifted_frac = len(agg.shifted_communities) / n_tested
        if shifted_frac >= 0.75:
            # pervasive move touching (almost) every community == global primary
            agg.primary_level = "global"
        elif agg.shifted_communities:
            # localised to a subset of communities == community primary
            agg.primary_level = "community"
        elif g.significant:
            agg.primary_level = "global"
        else:
            agg.primary_level = "node"
        return agg


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo, hi = np.min(v), np.max(v)
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)
