"""
Text feature encoder for modality / token-perturbation pipelines.

**Default:** import your Qwen (or other) script if present:
  ``Python/fsds_sot/qwen_episode_encoder.py`` must define::

      def encode_episode_texts(texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray

  ``texts[i]`` = space-joined token string for episode ``i`` (after perturbation).

**Fallback:** ``FSDS_TEXT_ENCODER=hash`` (CI / no GPU only — not for production).

**Override module path:** ``FSDS_TEXT_ENCODER_MODULE=my_package.my_qwen_embed``
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import List, Sequence

import numpy as np


def token_lists_to_texts(token_lists: Sequence[Sequence[str]]) -> List[str]:
    return [" ".join(str(t) for t in toks) for toks in token_lists]


def _load_user_encoder():
    mod_path = os.environ.get("FSDS_TEXT_ENCODER_MODULE", "fsds_sot.qwen_episode_encoder")
    try:
        mod = importlib.import_module(mod_path)
    except ImportError:
        return None
    fn = getattr(mod, "encode_episode_texts", None)
    if fn is None:
        warnings.warn(f"{mod_path} has no encode_episode_texts; using hash fallback.")
        return None
    return fn


def _hash_encode(texts: Sequence[str], *, dim: int, seed: int) -> np.ndarray:
    rows = []
    for text in texts:
        vec = np.zeros(dim, dtype=float)
        for t in text.split():
            idx = abs(hash((seed, t))) % dim
            vec[idx] += 1.0
        n = max(len(text.split()), 1)
        rows.append(vec / n)
    return np.vstack(rows)


def encode_token_lists(
    token_lists: Sequence[Sequence[str]],
    *,
    dim: int = 16,
    seed: int = 0,
) -> np.ndarray:
    texts = token_lists_to_texts(token_lists)
    mode = os.environ.get("FSDS_TEXT_ENCODER", "auto").lower()

    if mode == "hash":
        return _hash_encode(texts, dim=dim, seed=seed)

    user_fn = _load_user_encoder() if mode in ("auto", "qwen", "user") else None
    if user_fn is not None:
        X = np.asarray(user_fn(texts), dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    if mode not in ("auto", "hash"):
        raise ValueError(f"Unknown FSDS_TEXT_ENCODER={mode!r}")

    warnings.warn(
        "No qwen_episode_encoder.encode_episode_texts found; "
        "using hash fallback. Add fsds_sot/qwen_episode_encoder.py or set FSDS_TEXT_ENCODER_MODULE.",
        stacklevel=2,
    )
    return _hash_encode(texts, dim=dim, seed=seed)
