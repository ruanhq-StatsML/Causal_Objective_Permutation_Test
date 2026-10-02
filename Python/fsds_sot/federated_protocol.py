"""
Explicit three-phase federated FSDS attribution protocol (vertical feature blocks).

Phase 1: local attribution on F_k only
Phase 2: upload (plain or secure-aggregation stub)
Phase 3: server merge, drift typing, FDR + OFS closed loop
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .attribution import covariate_attribution
from .drift_ofs_fdr_loop import ClosedLoopStepResult, DriftOFSFDRClosedLoop


@dataclass
class LocalFeatureRecord:
    feature: str
    po_risk_distance: float
    distribution_shift: float
    local_rank: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalAttributionPayload:
    node_id: str
    n_ref: int
    n_live: int
    block_mmd2: float
    block_domain_auc: float
    block_ess: float
    features: List[LocalFeatureRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "n_ref": self.n_ref,
            "n_live": self.n_live,
            "block_mmd2": self.block_mmd2,
            "block_domain_auc": self.block_domain_auc,
            "block_ess": self.block_ess,
            "features": [f.to_dict() for f in self.features],
        }


@dataclass
class FederatedProtocolConfig:
    secure_aggregation: bool = False
    fdr_alpha: float = 0.05
    d_high: float = 0.02
    dp_high: float = 0.01


@dataclass
class FederatedProtocolResult:
    local_payloads: List[LocalAttributionPayload]
    global_ranking: List[Tuple[str, float]]
    closed_loop: ClosedLoopStepResult
    server_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_payloads": [p.to_dict() for p in self.local_payloads],
            "global_ranking": [{"feature": f, "score": s} for f, s in self.global_ranking],
            "closed_loop": self.closed_loop.to_dict(),
            "server_notes": self.server_notes,
        }


class FederatedNode:
    """Client: owns column indices and feature names."""

    def __init__(self, node_id: str, col_indices: Sequence[int], feature_names: Sequence[str]):
        self.node_id = node_id
        self.col_indices = list(col_indices)
        self.feature_names = list(feature_names)

    def compute_local_attribution(
        self,
        X_ref: np.ndarray,
        X_live: np.ndarray,
    ) -> LocalAttributionPayload:
        idx = self.col_indices
        Xr, Xl = X_ref[:, idx], X_live[:, idx]
        cov = covariate_attribution(Xr, Xl)
        vimp = np.asarray(cov.vimp, dtype=float)
        loco = np.asarray(cov.mmd_loco, dtype=float)
        # PO-risk proxy: combined ranking signal; shift proxy: MMD-LOCO
        scores = np.maximum(vimp, 0) + np.maximum(loco, 0)
        order = np.argsort(-scores)
        records: List[LocalFeatureRecord] = []
        for rank, j in enumerate(order):
            gidx = idx[j]
            fname = (
                self.feature_names[gidx]
                if gidx < len(self.feature_names)
                else f"f{gidx}"
            )
            records.append(
                LocalFeatureRecord(
                    feature=fname,
                    po_risk_distance=float(scores[j]),
                    distribution_shift=float(max(loco[j], 0.0)),
                    local_rank=int(rank + 1),
                )
            )
        return LocalAttributionPayload(
            node_id=self.node_id,
            n_ref=int(Xr.shape[0]),
            n_live=int(Xl.shape[0]),
            block_mmd2=float(cov.mmd2),
            block_domain_auc=float(cov.domain_auc),
            block_ess=float(cov.overlap_ess),
            features=records,
        )


class FederatedServer:
    """Coordinator: merge uploads, run closed loop."""

    def __init__(self, config: FederatedProtocolConfig):
        self.config = config
        self._uploads: List[LocalAttributionPayload] = []

    def ingest_plain(self, payload: LocalAttributionPayload) -> None:
        self._uploads.append(payload)

    def ingest_secure_stub(self, payload: LocalAttributionPayload) -> None:
        """Placeholder for masked aggregation — plain ingest in prototype."""
        self._uploads.append(payload)

    def merge_and_close_loop(self) -> Tuple[List[Tuple[str, float]], ClosedLoopStepResult, List[str]]:
        notes: List[str] = []
        po: Dict[str, float] = {}
        dp: Dict[str, float] = {}
        for upl in self._uploads:
            if upl.block_ess < 0.15:
                notes.append(f"{upl.node_id}: low ESS={upl.block_ess:.3f}; defer OFS on this block.")
            for rec in upl.features:
                po[rec.feature] = rec.po_risk_distance
                dp[rec.feature] = rec.distribution_shift

        # Global rank: z-score po distances across nodes
        vals = np.array(list(po.values())) if po else np.array([0.0])
        mu, sig = float(vals.mean()), float(vals.std()) + 1e-9
        ranking = sorted(
            ((f, (po[f] - mu) / sig) for f in po),
            key=lambda t: -t[1],
        )

        loop = DriftOFSFDRClosedLoop(alpha=self.config.fdr_alpha)
        closed = loop.step(
            po,
            dp,
            d_high=self.config.d_high,
            dp_high=self.config.dp_high,
        )
        n_sig = len(closed.significant)
        notes.append(f"BH-FDR alpha={self.config.fdr_alpha}: {n_sig} features flagged.")
        return ranking, closed, notes


class FederatedAttributionProtocol:
    def __init__(
        self,
        nodes: Sequence[FederatedNode],
        config: Optional[FederatedProtocolConfig] = None,
    ):
        self.nodes = list(nodes)
        self.config = config or FederatedProtocolConfig()
        self.server = FederatedServer(self.config)

    def run(self, X_ref: np.ndarray, X_live: np.ndarray) -> FederatedProtocolResult:
        payloads: List[LocalAttributionPayload] = []
        for node in self.nodes:
            pay = node.compute_local_attribution(X_ref, X_live)
            payloads.append(pay)
            if self.config.secure_aggregation:
                self.server.ingest_secure_stub(pay)
            else:
                self.server.ingest_plain(pay)

        ranking, closed, notes = self.server.merge_and_close_loop()
        return FederatedProtocolResult(
            local_payloads=payloads,
            global_ranking=ranking,
            closed_loop=closed,
            server_notes=notes,
        )


def partition_feature_blocks(n_features: int, n_nodes: int) -> List[List[int]]:
    """Equal-ish column splits for vertical FL simulation."""
    n_nodes = max(1, min(n_nodes, n_features))
    chunks: List[List[int]] = []
    edges = np.linspace(0, n_features, n_nodes + 1, dtype=int)
    for i in range(n_nodes):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if lo < hi:
            chunks.append(list(range(lo, hi)))
    return chunks
