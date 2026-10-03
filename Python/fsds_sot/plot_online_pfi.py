"""Visualize online PFI stream (Panel 4 over time + drift triggers)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .online_pfi import OnlinePFIReport


def plot_online_pfi_dashboard(
    report: OnlinePFIReport,
    out_path: str | Path,
    *,
    top_k: int = 8,
    title: str = "Online PFI — FSDS covariate stream",
) -> Path:
    T = len(report.steps)
    if T == 0:
        raise ValueError("empty OnlinePFIReport")

    steps = report.steps
    t_axis = np.arange(T)
    auc = [s.domain_auc for s in steps]
    mmd = [s.mmd2 for s in steps]
    pval = [s.c2st_pvalue for s in steps]

    mat = report.vimp_matrix
    if mat is None:
        mat = np.vstack([s.vimp for s in steps])
    # Global top features across stream
    mean_v = mat.mean(axis=0)
    idx = np.argsort(-mean_v)[:top_k]
    names = [report.feature_names[i] if i < len(report.feature_names) else f"f{i}" for i in idx]
    sub = mat[:, idx]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title, fontsize=12)

    ax0 = axes[0, 0]
    ax0.plot(t_axis, auc, "o-", color="crimson", label="domain AUC")
    ax0.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax0.set_ylabel("AUC")
    ax0.set_xlabel("window step")
    ax0.set_title("Batch separability (reference vs live)")
    ax0.legend(loc="lower right", fontsize=8)

    ax1 = axes[0, 1]
    ax1.plot(t_axis, mmd, "s-", color="darkgreen", label="MMD²")
    ax1.set_ylabel("MMD²")
    ax1.set_xlabel("window step")
    ax1.set_title("Covariate distance")
    ax1.legend(fontsize=8)

    ax2 = axes[1, 0]
    ax2.plot(t_axis, pval, "^-", color="navy", label="onlineRFPerm p")
    ax2.axhline(0.05, color="orange", ls="--", lw=1, label="α=0.05")
    ax2.set_ylabel("p-value")
    ax2.set_xlabel("window step")
    ax2.set_title("C2ST permutation trigger")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    ax3 = axes[1, 1]
    im = ax3.imshow(sub.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xticks(t_axis)
    ax3.set_xlabel("window step")
    ax3.set_title(f"Online PFI heatmap (top {top_k} features)")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
