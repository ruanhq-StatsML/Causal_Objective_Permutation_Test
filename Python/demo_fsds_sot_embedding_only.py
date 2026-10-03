#!/usr/bin/env python3
"""Embedding-only figures for FSDS-SoT (topology panels)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from demo_fsds_sot import synthetic_sot_batch
from fsds_sot.decompose import coupling_matrix, evaluate_decomposability


def _pca2(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:2].T


def main() -> int:
    out_dir = Path("/opt/cursor/artifacts/fsds_sot_embedding")
    out_dir.mkdir(parents=True, exist_ok=True)

    branches, p = 6, 32
    _, B1, _ = synthetic_sot_batch(120, branches, p, seed=2, drift_strength=0.0)
    B1[:, 2, :] += 0.8
    B1[:, 4, :] += 0.5
    Z = B1.mean(axis=0)

    ref = Z.mean(axis=0, keepdims=True)
    shift = np.linalg.norm(Z - ref, axis=1)
    decomp = evaluate_decomposability(Z, shift)
    H = coupling_matrix(Z)
    xy = _pca2(Z)

    # Fig 1: coupling heatmap
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(H, vmin=0, vmax=1, cmap="magma")
    ax.set_title(
        f"Embedding coupling (branch×branch)\n"
        f"K={decomp.n_clusters}, parallel={decomp.allow_parallel}",
        fontsize=11,
    )
    ax.set_xlabel("branch index")
    ax.set_ylabel("branch index")
    for i in range(branches):
        ax.text(i, i, str(i), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, label="HSIC proxy (1 = tightly coupled)")
    fig.tight_layout()
    p1 = out_dir / "01_coupling_heatmap.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: PCA scatter
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cid = decomp.cluster_ids
    for c in range(decomp.n_clusters):
        mask = cid == c
        ax.scatter(xy[mask, 0], xy[mask, 1], s=200, label=f"cluster {c}", edgecolors="k", linewidths=0.8)
    for i in range(branches):
        ax.annotate(f"b{i}", (xy[i, 0], xy[i, 1]), ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_title("Branch embeddings (PCA-2, color = cluster)", fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p2 = out_dir / "02_pca_clusters.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: embedding shift magnitude (bar, still embedding-derived)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    s_norm = shift / (shift.max() + 1e-9)
    colors = plt.cm.viridis(s_norm)
    ax.bar(range(branches), s_norm, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_xticks(range(branches))
    ax.set_xticklabels([f"b{i}" for i in range(branches)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("normalized ‖z_b − z̄‖")
    ax.set_title("Per-branch shift from embedding (input to budget executor)", fontsize=11)
    fig.tight_layout()
    p3 = out_dir / "03_embedding_shift.png"
    fig.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 4: combined 1+2 embedding topology strip
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    im = axes[0].imshow(H, vmin=0, vmax=1, cmap="magma")
    axes[0].set_title(f"Coupling  (K={decomp.n_clusters})")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    for c in range(decomp.n_clusters):
        mask = cid == c
        axes[1].scatter(xy[mask, 0], xy[mask, 1], s=160, edgecolors="k")
    for i in range(branches):
        axes[1].annotate(str(i), (xy[i, 0], xy[i, 1]), ha="center", va="center", fontsize=9)
    axes[1].set_title("PCA-2 clusters")
    axes[1].grid(True, alpha=0.2)
    fig.suptitle("FSDS-SoT — embedding-only topology (procedure 1)", fontsize=12, y=1.02)
    fig.tight_layout()
    p4 = out_dir / "04_topology_strip.png"
    fig.savefig(p4, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:")
    for p in (p1, p2, p3, p4):
        print(" ", p)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
