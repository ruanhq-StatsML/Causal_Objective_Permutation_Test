"""Visualization on skeleton branch / trace embeddings for FSDS-SoT."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .decompose import coupling_matrix, evaluate_decomposability


def _pca2(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    Z = Z - Z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:2].T


def plot_sot_dashboard(
    branch_embeddings: np.ndarray,
    branch_shift_scores: Optional[np.ndarray] = None,
    branch_quality: Optional[np.ndarray] = None,
    feature_vimp: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    *,
    out_path: str | Path = "fsds_sot_dashboard.png",
    title: str = "FSDS-SoT embedding dashboard",
) -> Path:
    """
    Four panels:
      1) Branch coupling heatmap (grouping granularity)
      2) Branch PCA scatter colored by cluster
      3) Per-branch shift / quality (budget inputs)
      4) Trace-level covariate VIMP (top features)
    """
    import matplotlib.pyplot as plt

    B = branch_embeddings.shape[0]
    if branch_shift_scores is None:
        ref = branch_embeddings.mean(axis=0, keepdims=True)
        branch_shift_scores = np.linalg.norm(branch_embeddings - ref, axis=1)
    if branch_quality is None:
        branch_quality = np.ones(B) * 0.7

    decomp = evaluate_decomposability(branch_embeddings, branch_shift_scores)
    H = coupling_matrix(branch_embeddings)
    xy = _pca2(branch_embeddings)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(title, fontsize=12)

    im = axes[0, 0].imshow(H, vmin=0, vmax=1, cmap="viridis")
    axes[0, 0].set_title(
        f"Coupling (K={decomp.n_clusters}, parallel={decomp.allow_parallel})"
    )
    axes[0, 0].set_xlabel("branch")
    axes[0, 0].set_ylabel("branch")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    colors = decomp.cluster_ids
    sc = axes[0, 1].scatter(xy[:, 0], xy[:, 1], c=colors, cmap="tab10", s=120, edgecolors="k")
    for i in range(B):
        axes[0, 1].annotate(str(i), (xy[i, 0], xy[i, 1]), fontsize=9, ha="center", va="center")
    axes[0, 1].set_title("Branch embeddings (PCA-2, color=cluster)")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")

    x = np.arange(B)
    w = 0.35
    axes[1, 0].bar(x - w / 2, branch_shift_scores, width=w, label="shift")
    axes[1, 0].bar(x + w / 2, branch_quality, width=w, label="quality")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"b{i}" for i in range(B)])
    axes[1, 0].set_title("Budget drivers (→ tokens / tier / checks)")
    axes[1, 0].legend(loc="upper right", fontsize=8)
    axes[1, 0].set_ylim(0, max(1.05, float(branch_shift_scores.max()) * 1.1))

    ax = axes[1, 1]
    if feature_vimp is not None and feature_vimp.size:
        k = min(12, feature_vimp.size)
        idx = np.argsort(-feature_vimp)[:k]
        labels = (
            [feature_names[i] for i in idx]
            if feature_names and len(feature_names) >= feature_vimp.size
            else [f"f{i}" for i in idx]
        )
        ax.barh(range(k), feature_vimp[idx][::-1], color="steelblue")
        ax.set_yticks(range(k))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_title("Batch covariate VIMP (top features)")
    else:
        ax.text(0.5, 0.5, "No trace VIMP provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
