"""Upper-level wrapper for multimodal distribution-shift attribution.

The procedure follows the image-shift localization logic:

1. Turn each modality into a feature matrix. Tabular columns and token
   embeddings are used as-is. Images are encoded as non-overlapping patch
   means, unless the caller supplies an encode/decode pair (for example a VAE).
2. Fit a domain random forest on reference versus query samples and read
   feature importance.
3. For images, add the query-minus-reference mean shift on the top-k
   coordinates, decode, and summarize the spatial difference with Gini,
   top-10% mass, and active area.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


Array = np.ndarray
EncodeFn = Callable[[Array], Array]
DecodeFn = Callable[[Array], Array]


def gini_coefficient(values: Array) -> float:
    """Gini of a nonnegative map. Larger means the mass is more concentrated."""
    h = np.asarray(values, dtype=float).ravel()
    h = np.clip(h, 0.0, None)
    total = float(h.sum())
    if total <= 1e-12:
        return 0.0
    h_sorted = np.sort(h)
    n = h_sorted.size
    gini = (2.0 * np.sum(np.arange(1, n + 1) * h_sorted) / (n * total)) - (n + 1) / n
    return float(np.clip(gini, 0.0, 1.0))


def spatial_summary(heatmap: Array) -> Dict[str, float]:
    """Concentration summaries of one spatial difference map."""
    h = np.asarray(heatmap, dtype=float).ravel()
    h = np.clip(h, 0.0, None)
    total = float(h.sum()) + 1e-8
    h_sorted = np.sort(h)
    k = max(1, int(0.1 * h_sorted.size))
    active = float(np.mean(h > (h.mean() + h.std())))
    return {
        "gini": gini_coefficient(h),
        "top10_mass": float(h_sorted[-k:].sum() / total),
        "active_area": active,
    }


def domain_importance(
    X_ref: Array,
    X_query: Array,
    *,
    n_estimators: int = 80,
    seed: int = 0,
) -> Tuple[Array, float]:
    """Random-forest importance for separating reference (0) from query (1)."""
    X_ref = np.asarray(X_ref, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    if X_ref.ndim != 2 or X_query.ndim != 2:
        raise ValueError("Feature modalities must be 2-D arrays (n, d).")
    if X_ref.shape[1] != X_query.shape[1]:
        raise ValueError("Reference and query must have the same number of features.")
    X = np.vstack([X_ref, X_query])
    y = np.concatenate([
        np.zeros(X_ref.shape[0], dtype=int),
        np.ones(X_query.shape[0], dtype=int),
    ])
    n, p = X.shape
    rf = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_features="sqrt",
        min_samples_leaf=max(1, int(round(np.sqrt(n) / 2.0))),
        random_state=int(seed),
        oob_score=True,
        bootstrap=True,
        n_jobs=1,
    )
    rf.fit(X, y)
    return rf.feature_importances_.astype(float), float(rf.oob_score_)


def recall_at_k(importance: Array, truth_idx: Sequence[int], k: int) -> float:
    if not truth_idx:
        return float("nan")
    k = min(int(k), importance.size)
    top = set(np.argsort(-importance)[:k].tolist())
    truth = set(int(i) for i in truth_idx)
    return float(len(top & truth) / len(truth))


def importance_mass(importance: Array, truth_idx: Sequence[int]) -> float:
    if not truth_idx:
        return float("nan")
    total = float(np.sum(importance)) + 1e-12
    return float(np.sum(importance[list(truth_idx)]) / total)


def importance_entropy(importance: Array) -> float:
    p = np.clip(np.asarray(importance, dtype=float), 0.0, None)
    p = p / (p.sum() + 1e-12)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


@dataclass
class PatchCodec:
    """Encode an image as per-patch channel means and decode by tiling."""

    patch: int
    channels: int
    grid_h: int
    grid_w: int

    @classmethod
    def from_images(cls, images: Array, patch: int) -> "PatchCodec":
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError("Images must have shape (n, c, h, w).")
        _, channels, height, width = images.shape
        if height % patch or width % patch:
            raise ValueError(f"Image size {(height, width)} is not divisible by patch {patch}.")
        return cls(patch=patch, channels=channels, grid_h=height // patch, grid_w=width // patch)

    @property
    def n_features(self) -> int:
        return self.grid_h * self.grid_w

    def encode(self, images: Array) -> Array:
        images = np.asarray(images, dtype=float)
        n, channels, height, width = images.shape
        ph = self.grid_h * self.patch
        pw = self.grid_w * self.patch
        cropped = images[:, :, :ph, :pw]
        blocks = cropped.reshape(n, channels, self.grid_h, self.patch, self.grid_w, self.patch)
        means = blocks.mean(axis=(1, 3, 5))
        return means.reshape(n, -1)

    def decode(self, features: Array) -> Array:
        features = np.asarray(features, dtype=float)
        means = features.reshape(-1, 1, self.grid_h, self.grid_w)
        tiled = np.repeat(np.repeat(means, self.patch, axis=2), self.patch, axis=3)
        return np.repeat(tiled, self.channels, axis=1)


def localize_latents(
    z_ref: Array,
    z_query: Array,
    decode: DecodeFn,
    images_query: Array,
    importance: Array,
    top_k: Iterable[int],
    *,
    max_images: int = 16,
) -> Dict[int, Dict[str, float]]:
    """Perturb the top-k query latents and summarize the decoded difference."""
    z_delta = z_query.mean(axis=0) - z_ref.mean(axis=0)
    order = np.argsort(-importance)
    n_use = min(int(max_images), z_query.shape[0], images_query.shape[0])
    z_base = z_query[:n_use]
    x_base = np.asarray(images_query[:n_use], dtype=float)
    x_hat = np.asarray(decode(z_base), dtype=float)
    summaries: Dict[int, Dict[str, float]] = {}
    for k in top_k:
        k_use = min(int(k), importance.size)
        if k_use <= 0:
            continue
        top_idx = order[:k_use]
        z_pert = z_base.copy()
        z_pert[:, top_idx] = z_pert[:, top_idx] + z_delta[top_idx]
        x_pert = np.asarray(decode(z_pert), dtype=float)
        heat_input = np.mean(np.abs(x_base - x_pert), axis=1)
        heat_code = np.mean(np.abs(x_hat - x_pert), axis=1)
        per_image = [spatial_summary(heat_input[i]) for i in range(n_use)]
        per_code = [spatial_summary(heat_code[i]) for i in range(n_use)]
        summaries[k_use] = {
            "gini": float(np.mean([row["gini"] for row in per_image])),
            "top10_mass": float(np.mean([row["top10_mass"] for row in per_image])),
            "active_area": float(np.mean([row["active_area"] for row in per_image])),
            "gini_perturbation": float(np.mean([row["gini"] for row in per_code])),
        }
    return summaries


@dataclass
class ModalitySpec:
    name: str
    kind: str
    reference: Array
    query: Array
    truth_idx: Tuple[int, ...] = ()
    patch: int = 8
    encode: Optional[EncodeFn] = None
    decode: Optional[DecodeFn] = None


@dataclass
class AttributionResult:
    summary: pd.DataFrame
    importance: Dict[str, Array] = field(default_factory=dict)
    localization: Dict[str, Mapping[int, Mapping[str, float]]] = field(default_factory=dict)
    example_heatmap: Dict[str, Array] = field(default_factory=dict)


class MultimodalAttribution:
    """Run the same shift-attribution procedure on every registered modality."""

    def __init__(self, *, n_estimators: int = 80, seed: int = 2026, top_k: Sequence[int] = (1, 2, 4, 8)):
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.top_k = tuple(int(k) for k in top_k)
        self._modalities: List[ModalitySpec] = []

    def add_features(
        self,
        name: str,
        X_ref: Array,
        X_query: Array,
        truth_idx: Optional[Sequence[int]] = None,
    ) -> None:
        self._modalities.append(ModalitySpec(
            name=name,
            kind="features",
            reference=np.asarray(X_ref, dtype=float),
            query=np.asarray(X_query, dtype=float),
            truth_idx=tuple(int(i) for i in (truth_idx or ())),
        ))

    def add_images(
        self,
        name: str,
        images_ref: Array,
        images_query: Array,
        *,
        patch: int = 8,
        truth_idx: Optional[Sequence[int]] = None,
        encode: Optional[EncodeFn] = None,
        decode: Optional[DecodeFn] = None,
    ) -> None:
        if (encode is None) ^ (decode is None):
            raise ValueError("Provide both encode and decode, or neither.")
        self._modalities.append(ModalitySpec(
            name=name,
            kind="image",
            reference=np.asarray(images_ref, dtype=float),
            query=np.asarray(images_query, dtype=float),
            truth_idx=tuple(int(i) for i in (truth_idx or ())),
            patch=int(patch),
            encode=encode,
            decode=decode,
        ))

    def run(self) -> AttributionResult:
        rows = []
        importance: Dict[str, Array] = {}
        localization: Dict[str, Mapping[int, Mapping[str, float]]] = {}
        example_heatmap: Dict[str, Array] = {}
        for spec in self._modalities:
            if spec.kind == "features":
                scores, oob = domain_importance(
                    spec.reference, spec.query,
                    n_estimators=self.n_estimators, seed=self.seed,
                )
                importance[spec.name] = scores
                k = min(max(len(spec.truth_idx), 1), scores.size)
                rows.append({
                    "modality": spec.name,
                    "kind": spec.kind,
                    "n_features": int(scores.size),
                    "oob_score": oob,
                    "recall_at_truth": recall_at_k(scores, spec.truth_idx, k),
                    "mass_on_truth": importance_mass(scores, spec.truth_idx),
                    "importance_entropy": importance_entropy(scores),
                    "gini": np.nan,
                    "top10_mass": np.nan,
                    "active_area": np.nan,
                })
                continue

            codec = None
            if spec.encode is None:
                codec = PatchCodec.from_images(spec.reference, spec.patch)
                z_ref = codec.encode(spec.reference)
                z_query = codec.encode(spec.query)
                decode = codec.decode
            else:
                z_ref = np.asarray(spec.encode(spec.reference), dtype=float)
                z_query = np.asarray(spec.encode(spec.query), dtype=float)
                decode = spec.decode
            scores, oob = domain_importance(
                z_ref, z_query, n_estimators=self.n_estimators, seed=self.seed,
            )
            importance[spec.name] = scores
            loc = localize_latents(
                z_ref, z_query, decode, spec.query, scores, self.top_k,
            )
            localization[spec.name] = loc
            k_show = min(max(len(spec.truth_idx), 1), scores.size)
            chosen = loc.get(k_show) or loc[min(loc)]
            k_truth = min(max(len(spec.truth_idx), 1), scores.size)
            rows.append({
                "modality": spec.name,
                "kind": spec.kind,
                "n_features": int(scores.size),
                "oob_score": oob,
                "recall_at_truth": recall_at_k(scores, spec.truth_idx, k_truth),
                "mass_on_truth": importance_mass(scores, spec.truth_idx),
                "importance_entropy": importance_entropy(scores),
                "gini": chosen["gini_perturbation"],
                "top10_mass": chosen["top10_mass"],
                "active_area": chosen["active_area"],
            })
            top_idx = np.argsort(-scores)[:k_show]
            z_delta = z_query.mean(axis=0) - z_ref.mean(axis=0)
            z_one = z_query[:1].copy()
            z_one[:, top_idx] = z_one[:, top_idx] + z_delta[top_idx]
            x_hat = np.asarray(decode(z_query[:1]), dtype=float)
            x_pert = np.asarray(decode(z_one), dtype=float)
            example_heatmap[spec.name] = np.mean(np.abs(x_hat - x_pert), axis=1)[0]

        summary = pd.DataFrame(rows)
        return AttributionResult(
            summary=summary,
            importance=importance,
            localization=localization,
            example_heatmap=example_heatmap,
        )
