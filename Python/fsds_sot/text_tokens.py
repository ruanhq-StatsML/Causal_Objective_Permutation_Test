"""
Token-level text representation for agent episodes + perturbations.

Token perturbation stays here; **embeddings** go through ``text_encoder.encode_token_lists``
(your ``qwen_episode_encoder.encode_episode_texts`` — not bag-of-hash in production).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from .agentic_dgp import AgenticEpisode
except ImportError:
    from fsds_sot.agentic_dgp import AgenticEpisode

MASK_TOKEN = "[MASK]"
UNK_TOKEN = "UNK"


class PerturbOp(str, Enum):
    MASK = "mask"
    REPLACE = "replace"
    SHUFFLE = "shuffle"
    DROP = "drop"


def episode_to_tokens(ep: AgenticEpisode) -> List[str]:
    """Skeleton trace as a token sequence (branch tags + scalars)."""
    toks: List[str] = ["EP_START"]
    for i, name in enumerate(ep.branch_names):
        toks.append(f"BR_{name}")
        toks.append(f"Q_{ep.branch_quality[i]:.2f}")
    # trace_features tail: branches, tool_calls, tool_fail, latency
    tail = ep.trace_features[-4:]
    toks.extend(
        [
            f"NBR_{int(tail[0])}",
            f"TC_{int(tail[1])}",
            f"TF_{int(tail[2])}",
            f"LAT_{tail[3]:.2f}",
        ]
    )
    toks.append("EP_END")
    return toks


def episodes_to_token_lists(episodes: Sequence[AgenticEpisode]) -> List[List[str]]:
    return [episode_to_tokens(ep) for ep in episodes]


def tokens_to_feature_matrix(
    token_lists: Sequence[Sequence[str]],
    *,
    dim: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Join tokens → strings → Qwen (or user encoder); hash only if encoder missing."""
    try:
        from .text_encoder import encode_token_lists
    except ImportError:
        from fsds_sot.text_encoder import encode_token_lists

    return encode_token_lists(token_lists, dim=dim, seed=seed)


def text_features_from_episodes(
    episodes: Sequence[AgenticEpisode],
    *,
    dim: int = 16,
    seed: int = 0,
) -> np.ndarray:
    return tokens_to_feature_matrix(episodes_to_token_lists(episodes), dim=dim, seed=seed)


def _copy_tokens(toks: Sequence[str]) -> List[str]:
    return list(toks)


def perturb_tokens(
    toks: Sequence[str],
    op: PerturbOp,
    *,
    index: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    mask_frac: float = 0.25,
    replace_with: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    vocab: Optional[Sequence[str]] = None,
) -> List[str]:
    """Apply one perturbation; returns a new token list."""
    out = _copy_tokens(toks)
    rng = rng or np.random.default_rng(0)
    vocab = list(vocab) if vocab else [UNK_TOKEN]

    if op == PerturbOp.MASK:
        if indices is not None:
            for i in indices:
                if 0 <= i < len(out):
                    out[i] = MASK_TOKEN
        elif index is not None and 0 <= index < len(out):
            out[index] = MASK_TOKEN
        else:
            n_mask = max(1, int(round(mask_frac * len(out))))
            pick = rng.choice(len(out), size=min(n_mask, len(out)), replace=False)
            for i in pick:
                out[int(i)] = MASK_TOKEN
        return out

    if op == PerturbOp.REPLACE:
        rep = replace_with if replace_with is not None else str(rng.choice(vocab))
        if indices is not None:
            for i in indices:
                if 0 <= i < len(out):
                    out[i] = rep
        elif index is not None and 0 <= index < len(out):
            out[index] = rep
        else:
            n_rep = max(1, int(round(mask_frac * len(out))))
            pick = rng.choice(len(out), size=min(n_rep, len(out)), replace=False)
            for i in pick:
                out[int(i)] = rep
        return out

    if op == PerturbOp.DROP:
        if index is not None and 0 <= index < len(out):
            del out[index]
        elif indices is not None:
            drop = sorted(set(int(i) for i in indices if 0 <= int(i) < len(out)), reverse=True)
            for i in drop:
                del out[i]
        else:
            n_drop = max(1, int(round(mask_frac * len(out))))
            pick = sorted(rng.choice(len(out), size=min(n_drop, len(out)), replace=False), reverse=True)
            for i in pick:
                del out[int(i)]
        return out

    if op == PerturbOp.SHUFFLE:
        if len(out) < 3:
            rng.shuffle(out)
            return out
        # shuffle interior (keep EP_START / EP_END fixed if present)
        start = 1 if out and out[0] == "EP_START" else 0
        end = len(out) - 1 if out and out[-1] == "EP_END" else len(out)
        mid = out[start:end]
        rng.shuffle(mid)
        out[start:end] = mid
        return out

    raise ValueError(f"Unknown op {op}")


def perturb_episodes_tokens(
    episodes: Sequence[AgenticEpisode],
    op: PerturbOp,
    *,
    per_episode_fn: Optional[Callable[[List[str], int], List[str]]] = None,
    seed: int = 0,
    **perturb_kw,
) -> List[List[str]]:
    """Perturb token lists for each episode (same op or custom fn)."""
    rng = np.random.default_rng(seed)
    out_lists: List[List[str]] = []
    for ei, ep in enumerate(episodes):
        toks = episode_to_tokens(ep)
        if per_episode_fn is not None:
            out_lists.append(per_episode_fn(toks, ei))
        else:
            out_lists.append(perturb_tokens(toks, op, rng=rng, **perturb_kw))
    return out_lists


def perturb_tokens_matching_prefix(
    toks: Sequence[str],
    prefix: str,
    op: PerturbOp,
    *,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """Mask/replace/drop all tokens whose string starts with ``prefix`` (e.g. ``BR_retrieve``)."""
    idx = [i for i, t in enumerate(toks) if str(t).startswith(prefix)]
    if not idx:
        return _copy_tokens(toks)
    if op == PerturbOp.DROP:
        return perturb_tokens(toks, op, indices=idx, rng=rng)
    return perturb_tokens(toks, op, indices=idx, rng=rng)
