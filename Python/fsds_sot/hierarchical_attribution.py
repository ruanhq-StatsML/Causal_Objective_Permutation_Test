"""
Two-layer attribution (delivery contract):

  Layer 1 — leave-one-**group**-out on ref vs live:
      groups = modalities on concat stack, or sub-groups inside a modality
      (e.g. embedding branches ``emb_b*``).

  Layer 2 — leave-one-**feature**-out (MMD-LOCO + RF-VIMP) **within** each group,
      using the global concat domain model slices (efficient) or a refit on the block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .attribution import (
    covariate_attribution,
    domain_logo_group_delta,
    mmd_logo_group_delta,
    rbf_mmd2,
)


@dataclass
class GroupLogoRow:
    group: str
    n_features: int
    mmd2_block: float
    mmd_logo_delta: float
    domain_logo_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TwoLayerAttribution:
    """Layer-1 LOGO table + layer-2 per-group feature rankings."""

    mmd2_full: float
    domain_auc_full: float
    layer1_group_logo: List[GroupLogoRow]
    layer2_within_group: Dict[str, List[Dict[str, float]]]

    def to_dict(self) -> Dict[str, Any]:
        rank = sorted(
            self.layer1_group_logo,
            key=lambda r: (r.mmd_logo_delta, r.domain_logo_delta),
            reverse=True,
        )
        return {
            "mmd2_full": self.mmd2_full,
            "domain_auc_full": self.domain_auc_full,
            "layer1_group_logo": [r.to_dict() for r in rank],
            "layer1_rank": [r.group for r in rank],
            "layer2_within_group": self.layer2_within_group,
        }


def _feature_layer2(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    names: List[str],
    vimp: np.ndarray,
    loco: np.ndarray,
    *,
    top_k: int = 12,
) -> List[Dict[str, float]]:
    order = np.argsort(-np.maximum(vimp, loco))
    out: List[Dict[str, float]] = []
    for j in order[: min(top_k, len(names))]:
        out.append(
            {
                "feature": names[j],
                "vimp": float(vimp[j]),
                "mmd_loco": float(loco[j]),
            }
        )
    return out


def run_two_layer_hierarchical(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    feature_names: List[str],
    group_slices: Dict[str, slice],
    *,
    n_estimators: int = 100,
    seed: int = 2026,
    top_k_layer2: int = 12,
    refit_layer2_blocks: bool = False,
) -> TwoLayerAttribution:
    """
    Hierarchical LOGO on ``group_slices``, then feature LOCO/VIMP inside each group.

    When ``refit_layer2_blocks=False`` (default, fast path), layer-2 uses VIMP/LOCO
    from one global ``covariate_attribution`` on the full ``X_ref||X_live`` matrix.
    """
    X_ref = np.asarray(X_ref, dtype=float)
    X_live = np.asarray(X_live, dtype=float)
    mmd_full = rbf_mmd2(X_ref, X_live)
    global_cov = covariate_attribution(
        X_ref, X_live, n_estimators=n_estimators, seed=seed
    )
    auc_full = global_cov.domain_auc

    layer1: List[GroupLogoRow] = []
    layer2: Dict[str, List[Dict[str, float]]] = {}

    for gname, sl in sorted(group_slices.items()):
        block_names = feature_names[sl.start : sl.stop]
        mmd_block = rbf_mmd2(X_ref[:, sl], X_live[:, sl])
        mmd_logo = mmd_logo_group_delta(X_ref, X_live, sl)
        dom_logo = domain_logo_group_delta(
            X_ref, X_live, sl, n_estimators=n_estimators, seed=seed + hash(gname) % 500
        )
        layer1.append(
            GroupLogoRow(
                group=gname,
                n_features=sl.stop - sl.start,
                mmd2_block=mmd_block,
                mmd_logo_delta=float(mmd_logo),
                domain_logo_delta=float(dom_logo),
            )
        )

        if refit_layer2_blocks:
            block_cov = covariate_attribution(
                X_ref[:, sl],
                X_live[:, sl],
                n_estimators=n_estimators,
                seed=seed + 17 + hash(gname) % 500,
            )
            vimp = block_cov.vimp
            loco = block_cov.mmd_loco
        else:
            vimp = global_cov.vimp[sl]
            loco = global_cov.mmd_loco[sl]

        layer2[gname] = _feature_layer2(
            X_ref[:, sl],
            X_live[:, sl],
            block_names,
            vimp,
            loco,
            top_k=top_k_layer2,
        )

    return TwoLayerAttribution(
        mmd2_full=mmd_full,
        domain_auc_full=auc_full,
        layer1_group_logo=layer1,
        layer2_within_group=layer2,
    )


def embedding_branch_group_slices(feature_names: Sequence[str]) -> Dict[str, slice]:
    """
    Parse ``emb_b{b}_d{j}`` names → one slice per branch block (layer-1 sub-groups).
    """
    branches: Dict[str, List[int]] = {}
    for j, name in enumerate(feature_names):
        if not name.startswith("emb_b"):
            continue
        # emb_b3_d1 → branch key emb_b3
        parts = name.split("_")
        if len(parts) < 3:
            continue
        bkey = "_".join(parts[:2])
        branches.setdefault(bkey, []).append(j)

    slices: Dict[str, slice] = {}
    for bkey, cols in sorted(branches.items()):
        cols_sorted = sorted(cols)
        slices[bkey] = slice(cols_sorted[0], cols_sorted[-1] + 1)
    return slices


def text_half_group_slices(n_features: int) -> Dict[str, slice]:
    """Optional layer-1 sub-groups for text-only (first/second half of hashed dims)."""
    mid = n_features // 2
    return {
        "text_low": slice(0, mid),
        "text_high": slice(mid, n_features),
    }
