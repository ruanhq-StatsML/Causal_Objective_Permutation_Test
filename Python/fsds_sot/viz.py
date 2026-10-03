"""Visualization on skeleton branch / trace embeddings for FSDS-SoT."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .budget import SoTPlan
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
    plan: Optional[SoTPlan] = None,
    *,
    out_path: str | Path = "fsds_sot_dashboard.png",
    title: str = "FSDS-SoT embedding dashboard",
) -> Path:
    """
    Four panels:
      1) Branch coupling heatmap (grouping granularity) — actionable: K, merge, sequential
      2) Branch PCA scatter colored by cluster — actionable: cluster assignment
      3) Budget executor: shift, quality, risk, and emitted L/tier/checks — actionable: deploy
      4) Trace-level covariate VIMP (top features) — actionable: re-plan / monitor
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

    ax3 = axes[1, 0]
    x = np.arange(B)
    w = 0.35
    s_norm = branch_shift_scores / (branch_shift_scores.max() + 1e-9)
    q = np.clip(branch_quality, 0, 1)
    risk = s_norm * (1.0 - q)

    ax3.bar(x - w / 2, s_norm, width=w, label="shift (norm)", color="#4C72B0")
    ax3.bar(x + w / 2, q, width=w, label="quality", color="#55A868")
    ax3.plot(x, risk, "o-", color="#C44E52", label="risk ≈ shift×(1−q)", markersize=6)

    if plan is not None and len(plan.branch_budgets) == B:
        ax3b = ax3.twinx()
        tokens = [plan.branch_budgets[i].expansion_tokens for i in range(B)]
        ax3b.bar(
            x,
            tokens,
            width=0.15,
            alpha=0.35,
            color="#8172B2",
            label="L tokens (action)",
        )
        ax3b.set_ylabel("expansion tokens", fontsize=8)
        for i, bb in enumerate(plan.branch_budgets):
            ax3.text(
                i,
                max(s_norm[i], q[i], risk[i]) + 0.06,
                f"{bb.model_tier[0].upper()}·{bb.check_budget}c",
                ha="center",
                fontsize=7,
                color="#333",
            )
        lines1, lab1 = ax3.get_legend_handles_labels()
        lines2, lab2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=7)
    else:
        ax3.legend(loc="upper right", fontsize=8)

    def _quadrant(si: float, qi: float) -> str:
        sh, qh = si >= 0.5, qi >= 0.5
        if sh and qh:
            return "HiS·HiQ"
        if sh and not qh:
            return "HiS·LoQ"
        if not sh and qh:
            return "LoS·HiQ"
        return "LoS·LoQ"

    quad_labels = [_quadrant(float(s_norm[i]), float(q[i])) for i in range(B)]
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"b{i}\n{quad_labels[i]}" for i in range(B)], fontsize=7)
    ax3.set_title("Panel 3: budget executor (drivers → L / tier / checks)")
    ax3.set_ylim(0, 1.25)
    ax3.set_ylabel("normalized driver")

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
