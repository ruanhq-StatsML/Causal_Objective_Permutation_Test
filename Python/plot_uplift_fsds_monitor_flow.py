#!/usr/bin/env python3
"""English flowchart: two-batch REF vs LIVE, FSDS gates, two-level post-hoc + PO-risk."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "uplift_fsds_monitor_flow_en.png"
OUT_LEGACY = ROOT / "artifacts" / "uplift_fsds_monitor_flow.png"


def _box(ax, xy, w, h, text, fc="#eef4ff", ec="#334155", fontsize=8):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#475569",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 17.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis("off")
    ax.set_title(
        "Two-batch uplift monitor (REF vs LIVE): FSDS gates + two-level post-hoc + PO-risk",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.text(
        5,
        21.35,
        "W = batch (REF/LIVE) · T = uplift treatment · Meta-learner retrains on REF only (LOCO ablations)",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )

    _box(
        ax,
        (1.8, 19.85),
        6.4,
        1.05,
        "Two batches: REF (calibrate τ̂) + LIVE (score frozen τ̂)\nX, T, Y per row · stack for W-tests and domain RF",
        fc="#dbeafe",
    )
    _arrow(ax, (5, 19.85), (5, 19.2))

    _box(
        ax,
        (0.8, 17.35),
        8.4,
        1.65,
        "Step 0 — FSDS validity (batch geometry)\n"
        "SRM on T within each batch · MMD²(X_ref, X_live)\n"
        "Domain RF AUC(X; W) — primary covariate-shift monitor · ESS_overlap on ê(W|X)",
        fc="#ecfdf5",
        fontsize=7.8,
    )
    _arrow(ax, (5, 17.35), (5, 16.7))

    _box(
        ax,
        (0.8, 15.55),
        8.4,
        1.05,
        "Support gate: if ESS_overlap low → Refresh REF (defer post-hoc rules)",
        fc="#e0f2fe",
        fontsize=8,
    )
    _arrow(ax, (5, 15.55), (5, 14.9))

    _box(
        ax,
        (0.8, 13.15),
        8.4,
        1.55,
        "Post-hoc Level 1 — AUUC / ranking (per-batch eval + two-batch gap)\n"
        "AUUC_ref · AUUC_live · ΔAUUC · LOCO–AUUC (drop_ref, drop_live)\n"
        "LIVE slices: domain-quintile AUUC · pairwise AUUC · overlap-band AUUC",
        fc="#fef3c7",
        fontsize=7.6,
    )
    _arrow(ax, (5, 13.15), (5, 12.5))

    _box(
        ax,
        (0.8, 10.75),
        8.4,
        1.45,
        "Post-hoc Level 2 — estimated / observed uplift on LIVE slices\n"
        "Empirical ATE_k · mean τ̂_k per quintile · pairwise ATE (budget map)",
        fc="#fde68a",
        fontsize=7.6,
    )
    _arrow(ax, (5, 10.75), (5, 10.1))

    _box(
        ax,
        (0.8, 8.55),
        8.4,
        1.35,
        "Pseudo-outcome risk (stacked REF∪LIVE)\n"
        "PO-risk statistic · DRPerm p-value · optional PO-LOCO on X",
        fc="#fae8ff",
        fontsize=7.8,
    )
    _arrow(ax, (5, 8.55), (5, 7.9))

    _box(
        ax,
        (1.5, 7.0),
        7.0,
        0.75,
        "Compound insights → business rules: REALLOCATE · HOLD_FEATURE · REFRESH_REF · RELEARN",
        fc="#f1f5f9",
        fontsize=8,
    )

    y0 = 6.15
    _arrow(ax, (5, 7.0), (1.8, y0))
    _arrow(ax, (5, 7.0), (5.0, y0))
    _arrow(ax, (5, 7.0), (8.2, y0))

    _box(
        ax,
        (0.15, 4.5),
        3.25,
        1.45,
        "Mix / incomparable\nESS low · domain AUC → 1",
        fc="#fee2e2",
        fontsize=7,
    )
    _box(
        ax,
        (3.35, 4.5),
        3.25,
        1.45,
        "Covariate shift\nMMD / domain RF AUC ↑\nL1 quintile + overlap",
        fc="#ffedd5",
        fontsize=7,
    )
    _box(
        ax,
        (6.55, 4.5),
        3.25,
        1.45,
        "Concept / Y|X shift\nMMD flat · PO-risk reject\nL1 + L2 ATE REALLOCATE",
        fc="#fce7f3",
        fontsize=7,
    )

    _arrow(ax, (1.8, 4.5), (1.8, 3.55))
    _arrow(ax, (5.0, 4.5), (5.0, 3.55))
    _arrow(ax, (8.2, 4.5), (8.2, 3.55))

    _box(
        ax,
        (0.05, 1.4),
        3.45,
        1.95,
        "REFRESH_REF\n"
        "Match / extend REF\n"
        "Pause LOCO-driven cuts",
        fc="#ffffff",
        fontsize=6.8,
    )
    _box(
        ax,
        (3.25, 1.4),
        3.45,
        1.95,
        "REALLOCATE (L1)\n"
        "Cap low-AUUC quintile\n"
        "Overlap-only targeting\n"
        "LOGO-MMD feature blocks",
        fc="#ffffff",
        fontsize=6.5,
    )
    _box(
        ax,
        (6.45, 1.4),
        3.45,
        1.95,
        "REALLOCATE (L2) + RELEARN\n"
        "If PO reject: shift to high ATE slice\n"
        "If τ̂ ≠ ATE: relearn prod τ",
        fc="#ffffff",
        fontsize=6.5,
    )

    ax.text(
        5,
        0.55,
        "LaTeX: docs/latex/uplift_fsds_two_batch_formulation.tex · Run: run_uplift_subset_benchmark.py",
        ha="center",
        fontsize=7.5,
        color="#475569",
    )

    for path in (OUT, OUT_LEGACY):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT, "and", OUT_LEGACY)


if __name__ == "__main__":
    main()
