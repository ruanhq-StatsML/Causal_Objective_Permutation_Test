"""
Token perturbation → re-encode → **MMD** and **PO-risk** (Model Registry path).

Use cases:
  - Global perturbation (mask/replace/shuffle/drop fraction) on live batch
  - Position-wise LOCO: mask token ``i`` on live → ΔMMD, ΔPO-risk vs baseline
  - Prefix groups (``BR_retrieve``, ``TC_``) → group-level counterfactual
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .agentic_dgp import AgenticEpisode
from .attribution import rbf_mmd2
from .online_pfi_registry import prediction_deltas_on_window
from .text_tokens import (
    PerturbOp,
    episode_to_tokens,
    episodes_to_token_lists,
    perturb_episodes_tokens,
    perturb_tokens,
    perturb_tokens_matching_prefix,
    text_features_from_episodes,
    tokens_to_feature_matrix,
)


def _outcomes(episodes: Sequence[AgenticEpisode]) -> np.ndarray:
    return np.array([e.success for e in episodes], dtype=float)


@dataclass
class ShiftMetrics:
    mmd2: float
    po_risk_observed: float
    delta_tau_mean: float
    domain_separable: bool  # po_risk > 0 proxy

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def metrics_on_text_features(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    Y_ref: np.ndarray,
    Y_live: np.ndarray,
    *,
    seed: int = 2026,
) -> ShiftMetrics:
    deltas = prediction_deltas_on_window(X_ref, X_live, Y_ref, Y_live, seed=seed)
    return ShiftMetrics(
        mmd2=float(rbf_mmd2(X_ref, X_live)),
        po_risk_observed=float(deltas["po_risk_observed"]),
        delta_tau_mean=float(deltas["delta_tau_mean"]),
        domain_separable=bool(deltas["po_risk_observed"] > 1e-6),
    )


def baseline_text_metrics(
    ref_episodes: Sequence[AgenticEpisode],
    live_episodes: Sequence[AgenticEpisode],
    *,
    dim: int = 16,
    seed: int = 0,
) -> tuple[ShiftMetrics, np.ndarray, np.ndarray]:
    X_ref = text_features_from_episodes(ref_episodes, dim=dim, seed=seed)
    X_live = text_features_from_episodes(live_episodes, dim=dim, seed=seed)
    Y_ref = _outcomes(ref_episodes)
    Y_live = _outcomes(live_episodes)
    m = metrics_on_text_features(X_ref, X_live, Y_ref, Y_live, seed=seed)
    return m, X_ref, X_live


@dataclass
class PerturbationRow:
    label: str
    op: str
    mmd2: float
    po_risk_observed: float
    delta_mmd: float
    delta_po_risk: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenPerturbationReport:
    baseline: ShiftMetrics
    global_perturbations: List[PerturbationRow] = field(default_factory=list)
    token_position_loco: List[PerturbationRow] = field(default_factory=list)
    prefix_group_perturbations: List[PerturbationRow] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "global_perturbations": [r.to_dict() for r in self.global_perturbations],
            "token_position_loco": [r.to_dict() for r in self.token_position_loco],
            "prefix_group_perturbations": [r.to_dict() for r in self.prefix_group_perturbations],
        }


def _row(
    label: str,
    op: str,
    m: ShiftMetrics,
    base: ShiftMetrics,
) -> PerturbationRow:
    return PerturbationRow(
        label=label,
        op=op,
        mmd2=m.mmd2,
        po_risk_observed=m.po_risk_observed,
        delta_mmd=float(base.mmd2 - m.mmd2),
        delta_po_risk=float(base.po_risk_observed - m.po_risk_observed),
    )


def run_token_perturbation_audit(
    ref_episodes: Sequence[AgenticEpisode],
    live_episodes: Sequence[AgenticEpisode],
    *,
    dim: int = 16,
    seed: int = 2026,
    mask_frac: float = 0.25,
    max_token_positions: int = 32,
    prefix_groups: Optional[Sequence[str]] = None,
) -> TokenPerturbationReport:
    """
    Baseline text features, then perturb **live** tokens → re-encode → MMD / PO-risk.

    ``delta_mmd > 0`` after perturbation means MMD dropped (that content drove shift).
    """
    base, X_ref, _ = baseline_text_metrics(
        ref_episodes, live_episodes, dim=dim, seed=seed
    )
    Y_ref = _outcomes(ref_episodes)
    Y_live = _outcomes(live_episodes)

    report = TokenPerturbationReport(baseline=base)
    live_token_lists = episodes_to_token_lists(live_episodes)

    # --- Global perturbations on live batch ---
    for op in (PerturbOp.MASK, PerturbOp.REPLACE, PerturbOp.SHUFFLE, PerturbOp.DROP):
        pert = perturb_episodes_tokens(
            live_episodes, op, seed=seed + hash(op.value) % 999, mask_frac=mask_frac
        )
        X_live_p = tokens_to_feature_matrix(pert, dim=dim, seed=seed)
        m = metrics_on_text_features(X_ref, X_live_p, Y_ref, Y_live, seed=seed)
        report.global_perturbations.append(
            _row(f"live_{op.value}_frac{mask_frac}", op.value, m, base)
        )

    # --- Token-position LOCO (mask token i on every live episode) ---
    max_len = min(
        max_token_positions,
        max((len(t) for t in live_token_lists), default=0),
    )
    for i in range(max_len):
        pert_lists: List[List[str]] = []
        for toks in live_token_lists:
            if i < len(toks):
                pert_lists.append(perturb_tokens(toks, PerturbOp.MASK, index=i))
            else:
                pert_lists.append(list(toks))
        X_live_p = tokens_to_feature_matrix(pert_lists, dim=dim, seed=seed)
        m = metrics_on_text_features(X_ref, X_live_p, Y_ref, Y_live, seed=seed + i)
        tok_label = live_token_lists[0][i] if live_token_lists and i < len(live_token_lists[0]) else f"pos_{i}"
        report.token_position_loco.append(
            _row(f"mask_pos{i}_{tok_label}", "mask", m, base)
        )

    # --- Prefix / branch groups ---
    prefixes = list(prefix_groups or ("BR_retrieve", "BR_act", "TC_", "TF_", "Q_"))
    for pref in prefixes:
        pert_lists = [
            perturb_tokens_matching_prefix(toks, pref, PerturbOp.MASK, rng=np.random.default_rng(seed))
            for toks in live_token_lists
        ]
        X_live_p = tokens_to_feature_matrix(pert_lists, dim=dim, seed=seed)
        m = metrics_on_text_features(X_ref, X_live_p, Y_ref, Y_live, seed=seed + hash(pref) % 500)
        report.prefix_group_perturbations.append(
            _row(f"mask_prefix_{pref}", "mask_prefix", m, base)
        )

    report.token_position_loco.sort(key=lambda r: r.delta_mmd, reverse=True)
    report.prefix_group_perturbations.sort(key=lambda r: r.delta_mmd, reverse=True)
    return report
