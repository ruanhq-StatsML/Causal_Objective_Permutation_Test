"""
Two-batch (reference vs live) attribution with upstream modality dataloaders.

Pipeline (your spec):
  1. Three dataloaders → text / structured / embedding blocks per batch.
  2. Compare ref vs live at **modality** level (MMD + domain RF-VIMP per block).
  3. **Concatenate** all modalities → global covariate attribution + slice back to modalities.
  4. **Fine-grain**: rank features inside each modality (VIMP + MMD-LOCO).
  5. **Text-only** track: same steps on ``text_dataloader`` only.

Optional: Model Registry prediction deltas on stacked X with outcome Y (success).

**Two-layer delivery** (hierarchical LOGO):
  Layer 1 — leave-one-modality-out on concat (or text sub-groups on text-only).
  Layer 2 — leave-one-feature-out (VIMP + MMD-LOCO) within each group.
  See ``hierarchical_attribution.run_two_layer_hierarchical``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agentic_dgp import AgenticEpisode
from .attribution import CovariateAttribution, covariate_attribution, rbf_mmd2
from .hierarchical_attribution import (
    TwoLayerAttribution,
    embedding_branch_group_slices,
    run_two_layer_hierarchical,
    text_half_group_slices,
)
from .modality_dataloaders import (
    ModalityBatch,
    concatenate_modalities,
    load_all_modalities,
    text_dataloader,
)


def _episode_outcomes(episodes: Sequence[AgenticEpisode]) -> np.ndarray:
    return np.array([e.success for e in episodes], dtype=float)


@dataclass
class ModalitySummary:
    modality: str
    n_features: int
    mmd2: float
    domain_auc: float
    modality_vimp_mass: float
    overlap_ok: bool
    overlap_ess: float
    top_features: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class ModalityAttributionReport:
    text_only: ModalitySummary
    per_modality: Dict[str, ModalitySummary]
    global_concat: ModalitySummary
    global_cov: CovariateAttribution
    feature_names: List[str]
    modality_slices: Dict[str, slice]
    fine_grain: Dict[str, List[Dict[str, float]]]
    two_layer_concat: TwoLayerAttribution
    two_layer_text: TwoLayerAttribution
    two_layer_embedding_branches: Optional[TwoLayerAttribution] = None
    registry_deltas: Optional[Dict[str, float]] = None

    def _summary_dict(self, s: ModalitySummary) -> Dict[str, Any]:
        d = asdict(s)
        d["overlap_ok"] = bool(s.overlap_ok)
        return d

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "text_only": self._summary_dict(self.text_only),
            "per_modality": {k: self._summary_dict(v) for k, v in self.per_modality.items()},
            "global_concat": self._summary_dict(self.global_concat),
            "global_cov": {
                "domain_auc": self.global_cov.domain_auc,
                "mmd2": self.global_cov.mmd2,
                "overlap_ok": bool(self.global_cov.overlap_ok),
                "overlap_ess": self.global_cov.overlap_ess,
            },
            "feature_names": self.feature_names,
            "modality_slices": {k: [sl.start, sl.stop] for k, sl in self.modality_slices.items()},
            "fine_grain": self.fine_grain,
            "two_layer_concat": self.two_layer_concat.to_dict(),
            "two_layer_text": self.two_layer_text.to_dict(),
        }
        if self.two_layer_embedding_branches is not None:
            d["two_layer_embedding_branches"] = self.two_layer_embedding_branches.to_dict()
        if self.registry_deltas is not None:
            d["registry_deltas"] = self.registry_deltas
        return d


def _top_k_features(
    names: List[str],
    vimp: np.ndarray,
    loco: np.ndarray,
    *,
    k: int = 8,
) -> List[Dict[str, float]]:
    order = np.argsort(-vimp)
    out: List[Dict[str, float]] = []
    for j in order[: min(k, len(names))]:
        out.append(
            {
                "feature": names[j],
                "vimp": float(vimp[j]),
                "mmd_loco": float(loco[j]),
            }
        )
    return out


def _summarize_modality_block(
    modality: str,
    X_ref: np.ndarray,
    X_live: np.ndarray,
    names: List[str],
    *,
    global_vimp: Optional[np.ndarray] = None,
    global_loco: Optional[np.ndarray] = None,
    sl: Optional[slice] = None,
    n_estimators: int = 100,
    seed: int = 2026,
    top_k: int = 8,
) -> Tuple[ModalitySummary, CovariateAttribution]:
    cov = covariate_attribution(X_ref, X_live, n_estimators=n_estimators, seed=seed)
    if global_vimp is not None and sl is not None:
        vimp = global_vimp[sl]
        loco = global_loco[sl] if global_loco is not None else cov.mmd_loco
        block_names = names[sl.start : sl.stop]
    else:
        vimp = cov.vimp
        loco = cov.mmd_loco
        block_names = names

    mass = float(vimp.sum())
    tops = _top_k_features(block_names, vimp, loco, k=top_k)
    summary = ModalitySummary(
        modality=modality,
        n_features=len(block_names),
        mmd2=cov.mmd2,
        domain_auc=cov.domain_auc,
        modality_vimp_mass=mass,
        overlap_ok=cov.overlap_ok,
        overlap_ess=cov.overlap_ess,
        top_features=tops,
    )
    return summary, cov


def _load_batches(
    ref: Sequence[AgenticEpisode],
    live: Sequence[AgenticEpisode],
    *,
    text_dim: int = 16,
    seed: int = 0,
) -> Tuple[Dict[str, ModalityBatch], Dict[str, ModalityBatch]]:
    ref_map = load_all_modalities(ref, text_dim=text_dim, seed=seed)
    live_map = load_all_modalities(live, text_dim=text_dim, seed=seed)
    return ref_map, live_map


def run_modality_attribution(
    ref_episodes: Sequence[AgenticEpisode],
    live_episodes: Sequence[AgenticEpisode],
    *,
    text_dim: int = 16,
    seed: int = 2026,
    n_estimators: int = 100,
    top_k: int = 8,
    include_registry_deltas: bool = False,
) -> ModalityAttributionReport:
    ref_map, live_map = _load_batches(
        ref_episodes, live_episodes, text_dim=text_dim, seed=seed
    )

    # --- Text-only track ---
    text_ref = ref_map["text"].X
    text_live = live_map["text"].X
    text_names = ref_map["text"].feature_names
    text_summary, _ = _summarize_modality_block(
        "text",
        text_ref,
        text_live,
        text_names,
        n_estimators=n_estimators,
        seed=seed,
        top_k=top_k,
    )

    # --- Per-modality (separate two-batch attribution) ---
    per_modality: Dict[str, ModalitySummary] = {}
    for key in sorted(ref_map.keys()):
        mb_r, mb_l = ref_map[key], live_map[key]
        summ, _ = _summarize_modality_block(
            key,
            mb_r.X,
            mb_l.X,
            mb_r.feature_names,
            n_estimators=n_estimators,
            seed=seed + hash(key) % 997,
            top_k=top_k,
        )
        per_modality[key] = summ

    # --- Concatenate three loaders → global attribution → slice modalities ---
    X_ref, names, slices = concatenate_modalities(ref_map)
    X_live, _, _ = concatenate_modalities(live_map)
    global_cov = covariate_attribution(
        X_ref, X_live, n_estimators=n_estimators, seed=seed + 1
    )

    global_summ = ModalitySummary(
        modality="concat_all",
        n_features=X_ref.shape[1],
        mmd2=global_cov.mmd2,
        domain_auc=global_cov.domain_auc,
        modality_vimp_mass=float(global_cov.vimp.sum()),
        overlap_ok=global_cov.overlap_ok,
        overlap_ess=global_cov.overlap_ess,
        top_features=_top_k_features(names, global_cov.vimp, global_cov.mmd_loco, k=top_k),
    )

    fine_grain: Dict[str, List[Dict[str, float]]] = {}
    for key, sl in slices.items():
        block_names = names[sl.start : sl.stop]
        vimp = global_cov.vimp[sl]
        loco = global_cov.mmd_loco[sl]
        mmd_block = rbf_mmd2(X_ref[:, sl], X_live[:, sl])
        tops = _top_k_features(block_names, vimp, loco, k=max(top_k, 12))
        fine_grain[key] = tops

    two_layer_concat = run_two_layer_hierarchical(
        X_ref,
        X_live,
        names,
        slices,
        n_estimators=n_estimators,
        seed=seed + 2,
        top_k_layer2=max(top_k, 12),
    )
    text_slices = text_half_group_slices(text_ref.shape[1])
    two_layer_text = run_two_layer_hierarchical(
        text_ref,
        text_live,
        text_names,
        text_slices,
        n_estimators=n_estimators,
        seed=seed + 3,
        top_k_layer2=max(top_k, 12),
    )

    emb_names = ref_map["embedding"].feature_names
    branch_slices = embedding_branch_group_slices(emb_names)
    two_layer_emb: Optional[TwoLayerAttribution] = None
    if branch_slices:
        two_layer_emb = run_two_layer_hierarchical(
            ref_map["embedding"].X,
            live_map["embedding"].X,
            emb_names,
            branch_slices,
            n_estimators=n_estimators,
            seed=seed + 4,
            top_k_layer2=8,
        )

    registry_deltas: Optional[Dict[str, float]] = None
    if include_registry_deltas:
        from .online_pfi_registry import prediction_deltas_on_window

        Y_ref = _episode_outcomes(ref_episodes)
        Y_live = _episode_outcomes(live_episodes)
        registry_deltas = prediction_deltas_on_window(
            X_ref, X_live, Y_ref, Y_live, seed=seed
        )

    return ModalityAttributionReport(
        text_only=text_summary,
        per_modality=per_modality,
        global_concat=global_summ,
        global_cov=global_cov,
        feature_names=names,
        modality_slices=slices,
        fine_grain=fine_grain,
        two_layer_concat=two_layer_concat,
        two_layer_text=two_layer_text,
        two_layer_embedding_branches=two_layer_emb,
        registry_deltas=registry_deltas,
    )


def run_text_modality_attribution(
    ref_episodes: Sequence[AgenticEpisode],
    live_episodes: Sequence[AgenticEpisode],
    *,
    text_dim: int = 16,
    seed: int = 2026,
    n_estimators: int = 100,
    top_k: int = 12,
) -> Tuple[ModalitySummary, CovariateAttribution]:
    """Standalone text dataloader → ref vs live attribution."""
    text_ref = text_dataloader(ref_episodes, dim=text_dim, seed=seed)
    text_live = text_dataloader(live_episodes, dim=text_dim, seed=seed)
    return _summarize_modality_block(
        "text",
        text_ref.X,
        text_live.X,
        text_ref.feature_names,
        n_estimators=n_estimators,
        seed=seed,
        top_k=top_k,
    )
