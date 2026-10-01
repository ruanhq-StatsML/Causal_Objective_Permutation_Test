#!/usr/bin/env python3
"""English flowchart: FSDS + LOCO-AUUC uplift monitoring and response actions."""

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
    fig, ax = plt.subplots(figsize=(11.5, 15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 19)
    ax.axis("off")
    ax.set_title(
        "How to monitor an uplift model with FSDS + LOCO–AUUC",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        5,
        18.35,
        "Meta-learner probe (retrain on REF) · SRM on T · Support gate: ESS overlap or domain RF AUC(X)",
        ha="center",
        fontsize=9,
        color="#475569",
    )

    _box(
        ax,
        (2.2, 16.8),
        5.6,
        1.05,
        "Inputs: REF window + LIVE window\nFeatures X · uplift treatment T · outcome Y · batch label W",
        fc="#dbeafe",
    )
    _arrow(ax, (5, 16.8), (5, 16.15))

    _box(
        ax,
        (1.0, 14.5),
        8.0,
        1.45,
        "Step 0 — Validity checks\n"
        "SRM on T (REF train & LIVE): treatment balance for AUUC\n"
        "Layer 1 — Batch FSDS on X: MMD²(ref,live) · domain RF AUC(X) · Online PFI (optional)",
        fc="#ecfdf5",
    )
    _arrow(ax, (5, 14.5), (5, 13.85))

    _box(
        ax,
        (1.0, 12.55),
        8.0,
        1.15,
        "Support / comparability gate (use either or both)\n"
        r"ESS$_{\mathrm{overlap}}$ on $\hat e(W{=}1\mid X)$  OR  high domain RF AUC(X) $\Rightarrow$ review before LOCO",
        fc="#e0f2fe",
        fontsize=7.5,
    )
    _arrow(ax, (5, 12.55), (5, 11.9))

    _box(
        ax,
        (1.0, 10.35),
        8.0,
        1.35,
        "Layer 2 — LOCO–AUUC (ranking)\n"
        "Fit meta-learner on REF train only → AUUC_ref, AUUC_live, gap\n"
        "Group LOCO: drop block g, retrain on REF → drop_live; Spearman(drop_ref, drop_live)",
        fc="#fef3c7",
    )
    _arrow(ax, (5, 10.35), (5, 9.7))

    _box(
        ax,
        (1.0, 8.35),
        8.0,
        1.15,
        "Layer 3 — PO-risk on Y|X (same window, optional)\n"
        "Pseudo-outcome learner · PO-risk · DRPerm on W · PO-LOCO features",
        fc="#fae8ff",
    )
    _arrow(ax, (5, 8.35), (5, 7.7))

    _box(
        ax,
        (2.0, 6.75),
        6.0,
        0.85,
        "Output: diagnose_shift label · bootstrap CI on AUUC_live",
        fc="#f1f5f9",
    )

    y0 = 6.0
    _arrow(ax, (5, 6.75), (1.8, y0))
    _arrow(ax, (5, 6.75), (5.0, y0))
    _arrow(ax, (5, 6.75), (8.2, y0))

    _box(
        ax,
        (0.2, 4.35),
        3.2,
        1.45,
        "Mix / new population\nMMD or domain AUC up\nESS overlap low",
        fc="#fee2e2",
        fontsize=7,
    )
    _box(
        ax,
        (3.4, 4.35),
        3.2,
        1.45,
        "Covariate shift\nMMD / domain AUC up\nAUUC gap up · ESS OK",
        fc="#ffedd5",
        fontsize=7,
    )
    _box(
        ax,
        (6.6, 4.35),
        3.2,
        1.45,
        "Concept / tau drift\nMMD flat · AUUC down\nPO-risk sig · Spearman down",
        fc="#fce7f3",
        fontsize=7,
    )

    _arrow(ax, (1.8, 4.35), (1.8, 3.45))
    _arrow(ax, (5.0, 4.35), (5.0, 3.45))
    _arrow(ax, (8.2, 4.35), (8.2, 3.45))

    _box(
        ax,
        (0.05, 1.35),
        3.5,
        1.95,
        "Actions\n"
        "• Refresh or match REF cohort\n"
        "• Stratify LIVE; pause LOCO-driven cuts\n"
        "• Re-check support gate; then re-run monitor",
        fc="#ffffff",
        fontsize=6.8,
    )
    _box(
        ax,
        (3.25, 1.35),
        3.5,
        1.95,
        "Actions (ranking first)\n"
        "• LOCO blocks: cap / trim high-shift segments\n"
        "• Recalibrate propensity e(T|X)\n"
        "• Re-score AUUC on overlap support only\n"
        "• Defer full prod tau relearn",
        fc="#ffffff",
        fontsize=6.5,
    )
    _box(
        ax,
        (6.45, 1.35),
        3.5,
        1.95,
        "Actions (outcome / tau)\n"
        "• PO-LOCO + LOCO–AUUC feature lists\n"
        "• Extend label window; retrain tau (prod)\n"
        "• REALLOCATE targeting rules, then RELEARN",
        fc="#ffffff",
        fontsize=6.5,
    )

    ax.text(
        5,
        0.45,
        "Hourly: Registry X-learner monitor · On alert: neural / prod LOCO confirm",
        ha="center",
        fontsize=8,
        color="#475569",
    )

    for path in (OUT, OUT_LEGACY):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT, "and", OUT_LEGACY)


if __name__ == "__main__":
    main()
