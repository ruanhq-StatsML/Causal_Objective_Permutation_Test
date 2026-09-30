"""
Upstream dataloaders → modality blocks for two-batch FSDS attribution.

Three loaders (your spec):
  1. text      — episode-level text / token statistics
  2. structured — trace scalars (tool_calls, failures, latency, …)
  3. embedding  — branch embedding summaries (mean / std per branch block)

Then: concatenate for global batch test OR run per-modality attribution separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from .agentic_dgp import AgenticEpisode
except ImportError:
    from fsds_sot.agentic_dgp import AgenticEpisode


@dataclass
class ModalityBatch:
    modality: str
    X: np.ndarray  # (n, p_m)
    feature_names: List[str]

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _episodes_to_list(episodes: Sequence[AgenticEpisode]) -> List[AgenticEpisode]:
    return list(episodes)


def text_dataloader(episodes: Sequence[AgenticEpisode], *, dim: int = 16, seed: int = 0) -> ModalityBatch:
    """
    Text modality: tokens → ``qwen_episode_encoder.encode_episode_texts`` (see ``text_encoder``).
    """
    try:
        from .text_tokens import text_features_from_episodes
    except ImportError:
        from fsds_sot.text_tokens import text_features_from_episodes

    X = text_features_from_episodes(episodes, dim=dim, seed=seed)
    names = [f"text_h{i}" for i in range(dim)]
    return ModalityBatch("text", X, names)


def structured_dataloader(episodes: Sequence[AgenticEpisode]) -> ModalityBatch:
    """Structured / tabular trace scalars (last 4 dims of trace_features in agentic DGP)."""
    X = np.vstack([ep.trace_features[-4:] for ep in episodes])
    names = ["n_branches", "tool_calls", "tool_fail", "latency_proxy"]
    return ModalityBatch("structured", X, names)


def embedding_dataloader(episodes: Sequence[AgenticEpisode]) -> ModalityBatch:
    """Embedding modality: per-branch mean embedding concatenated."""
    B = episodes[0].branch_embeddings.shape[0]
    d = episodes[0].branch_embeddings.shape[1]
    rows = []
    for ep in episodes:
        rows.append(ep.branch_embeddings.reshape(-1))
    X = np.vstack(rows)
    names = []
    for b in range(B):
        for j in range(d):
            names.append(f"emb_b{b}_d{j}")
    return ModalityBatch("embedding", X, names)


def load_all_modalities(
    episodes: Sequence[AgenticEpisode],
    *,
    text_dim: int = 16,
    seed: int = 0,
) -> Dict[str, ModalityBatch]:
    eps = _episodes_to_list(episodes)
    return {
        "text": text_dataloader(eps, dim=text_dim, seed=seed),
        "structured": structured_dataloader(eps),
        "embedding": embedding_dataloader(eps),
    }


def concatenate_modalities(modality_map: Dict[str, ModalityBatch]) -> Tuple[np.ndarray, List[str], Dict[str, slice]]:
    """Stack modalities horizontally; return slices for downstream fine attribution."""
    order = sorted(modality_map.keys())
    parts, names, slices = [], [], {}
    col = 0
    for key in order:
        mb = modality_map[key]
        parts.append(mb.X)
        n = mb.n_features
        slices[key] = slice(col, col + n)
        names.extend([f"{key}::{n}" for n in mb.feature_names])
        col += n
    return np.hstack(parts), names, slices
