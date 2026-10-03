"""
Federated FSDS prototype: each feature block = client; minimal uplink statistics.

No cryptography — simulates vertical FL communication envelope for monitoring research.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .attribution import covariate_attribution, rbf_mmd2


@dataclass
class LocalBlockReport:
    block_id: str
    n_ref: int
    n_live: int
    mmd2: float
    domain_auc: float
    overlap_ess: float
    top_features: List[Dict[str, float]] = field(default_factory=list)
    po_risk_pvalue: Optional[float] = None
    drift_type: str = "unknown"
    hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FederatedFSDSReport:
    blocks: List[LocalBlockReport]
    global_mmd2: float
    global_domain_auc: float
    merged_rank: List[Tuple[str, float]]
    server_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "global_mmd2": self.global_mmd2,
            "global_domain_auc": self.global_domain_auc,
            "merged_rank": [{"block": b, "score": s} for b, s in self.merged_rank],
            "server_actions": self.server_actions,
        }


def _ess_batch(e_hat: np.ndarray, eps: float = 0.05) -> float:
    e = np.clip(e_hat, eps, 1.0 - eps)
    w = 1.0 / (e * (1.0 - e))
    return float((w.sum() ** 2) / np.sum(w**2) / len(w))


def local_block_fsds(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    *,
    block_id: str,
    feature_names: Optional[Sequence[str]] = None,
    Y_ref: Optional[np.ndarray] = None,
    Y_live: Optional[np.ndarray] = None,
    po_pvalue: Optional[float] = None,
    mmd_high: float = 0.02,
    auc_high: float = 0.85,
    top_k: int = 5,
) -> LocalBlockReport:
    """Client-side: REF vs LIVE on block columns only."""
    cov = covariate_attribution(X_ref, X_live)
    names = list(feature_names or [f"{block_id}_x{j}" for j in range(X_ref.shape[1])])
    vimp = np.asarray(cov.vimp, dtype=float)
    order = np.argsort(-vimp)
    top = [
        {"feature": names[int(j)], "vimp": float(vimp[j])}
        for j in order[: min(top_k, len(names))]
    ]
    drift_type, hints = classify_block_drift_type(
        cov.mmd2,
        cov.domain_auc,
        po_pvalue,
        mmd_high=mmd_high,
        auc_high=auc_high,
    )
    return LocalBlockReport(
        block_id=block_id,
        n_ref=int(X_ref.shape[0]),
        n_live=int(X_live.shape[0]),
        mmd2=float(cov.mmd2),
        domain_auc=float(cov.domain_auc),
        overlap_ess=float(cov.overlap_ess),
        top_features=top,
        po_risk_pvalue=po_pvalue,
        drift_type=drift_type,
        hints=hints,
    )


def classify_block_drift_type(
    mmd2: float,
    domain_auc: float,
    po_pvalue: Optional[float],
    *,
    mmd_high: float = 0.02,
    auc_high: float = 0.85,
    po_alpha: float = 0.05,
) -> Tuple[str, List[str]]:
    """FedORA-style coarse typing from block summaries (not full decision tree)."""
    x_shift = mmd2 >= mmd_high or domain_auc >= auc_high
    po_reject = po_pvalue is not None and po_pvalue <= po_alpha
    hints: List[str] = []
    if x_shift and po_reject:
        label = "compound"
        hints.append("Covariate + outcome shift: stratify REF per block before OFS retire.")
    elif x_shift:
        label = "feature_drift"
        hints.append("Covariate shift on block: LOGO retire candidate; RefreshRef if ESS low.")
    elif po_reject:
        label = "concept"
        hints.append("Flat X on block but PO-risk reject: do not auto-drop features; retrain/Y audit.")
    else:
        label = "stable"
        hints.append("No strong block-level shift signal.")
    return label, hints


def merge_block_reports(
    reports: Sequence[LocalBlockReport],
    X_ref_full: np.ndarray,
    X_live_full: np.ndarray,
) -> FederatedFSDSReport:
    """Server: global FSDS on horizontally stacked blocks + federated ranking."""
    g = covariate_attribution(X_ref_full, X_live_full)
    # Score blocks by LOGO-style drop: here use uploaded mmd2 + domain_auc as proxy
    scored: List[Tuple[str, float]] = []
    for r in reports:
        logo_proxy = r.mmd2 + max(0.0, r.domain_auc - 0.5)
        scored.append((r.block_id, logo_proxy))
    scored.sort(key=lambda t: -t[1])

    actions: List[str] = []
    for r in reports:
        if r.overlap_ess < 0.15:
            actions.append(f"[{r.block_id}] RefreshRef: ESS={r.overlap_ess:.3f} too low for OFS.")
        elif r.drift_type == "feature_drift":
            actions.append(
                f"[{r.block_id}] OFS candidate: covariate drift — review top VIMP {r.top_features[:2]}."
            )
        elif r.drift_type == "concept":
            actions.append(f"[{r.block_id}] Monitor/retrain path; do not retire columns on PO alone.")
        elif r.drift_type == "compound":
            actions.append(f"[{r.block_id}] Stratify then partial retire + label extension.")

    if g.overlap_ess < 0.15:
        actions.insert(0, "Global RefreshRef before cross-block ranking actions.")

    return FederatedFSDSReport(
        blocks=list(reports),
        global_mmd2=float(g.mmd2),
        global_domain_auc=float(g.domain_auc),
        merged_rank=scored,
        server_actions=actions,
    )


def simulate_vertical_partitions(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    slices: Dict[str, slice],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ref_parts = {k: X_ref[:, sl] for k, sl in slices.items()}
    live_parts = {k: X_live[:, sl] for k, sl in slices.items()}
    return ref_parts, live_parts
