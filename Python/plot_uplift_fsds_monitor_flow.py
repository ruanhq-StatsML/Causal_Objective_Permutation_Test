#!/usr/bin/env python3
"""Simple flowchart: FSDS + LOCO-AUUC uplift monitoring and response actions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "uplift_fsds_monitor_flow.png"


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
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, start, end, text=None):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.1,
        color="#475569",
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.15, text, ha="center", fontsize=7, color="#64748b")


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")
    ax.set_title(
        "Monitor uplift with FSDS + LOCO–AUUC\n(meta-learner probe on REF; ESS overlap gates attribution)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    _box(ax, (2.5, 16.2), 5, 1.0, "REF window + LIVE window\n(features X, treatment T, outcome Y)", fc="#dbeafe")
    _arrow(ax, (5, 16.2), (5, 15.5))

    _box(
        ax,
        (1.2, 13.8),
        7.6,
        1.5,
        "Layer 1 — Batch FSDS on X\n"
        r"MMD$^2$(ref,live) · domain AUC(X) · ESS$_{\mathrm{ovlp}}$( $\hat e(W|X)$ ) · SRM on T",
        fc="#ecfdf5",
    )
    _arrow(ax, (5, 13.8), (5, 13.0))

    _box(
        ax,
        (1.2, 11.3),
        7.6,
        1.4,
        "Layer 2 — LOCO–AUUC (meta-learner, REF train only)\n"
        "AUUC_ref · AUUC_live · gap · Group LOCO retrain → drop_live · Spearman(ref,live)",
        fc="#fef3c7",
    )
    _arrow(ax, (5, 11.3), (5, 10.5))

    _box(
        ax,
        (1.2, 9.0),
        7.6,
        1.2,
        "Layer 3 (same window) — FSDS PO-risk on Y|X\n"
        r"PO-risk · DRPerm(W) · PO-LOCO → which features drive concept shift",
        fc="#fae8ff",
    )
    _arrow(ax, (5, 9.0), (5, 8.2))

    _box(ax, (2.0, 7.0), 6, 0.9, "diagnose_shift label + bootstrap AUUC_live CI", fc="#f1f5f9")

    # Decision branches
    y0 = 6.2
    _arrow(ax, (5, 7.0), (2.0, y0))
    _arrow(ax, (5, 7.0), (5.0, y0))
    _arrow(ax, (5, 7.0), (8.0, y0))

    _box(
        ax,
        (0.3, 4.5),
        3.4,
        1.4,
        "ESS low or\nmix_shift\n(MMD↑)",
        fc="#fee2e2",
        fontsize=7,
    )
    _box(
        ax,
        (3.3, 4.5),
        3.4,
        1.4,
        "covariate_shift\nMMD↑ gap↑\nESS OK",
        fc="#ffedd5",
        fontsize=7,
    )
    _box(
        ax,
        (6.3, 4.5),
        3.4,
        1.4,
        "concept / τ change\nMMD flat AUUC↓\n(or Spearman↓)",
        fc="#fce7f3",
        fontsize=7,
    )

    _arrow(ax, (2.0, 4.5), (2.0, 3.6))
    _arrow(ax, (5.0, 4.5), (5.0, 3.6))
    _arrow(ax, (8.0, 4.5), (8.0, 3.6))

    _box(
        ax,
        (0.15, 1.5),
        3.7,
        1.9,
        "Actions\n• Refresh / match REF\n• Stratify; pause LOCO trust\n• Fix overlap then re-monitor",
        fc="#ffffff",
        fontsize=7,
    )
    _box(
        ax,
        (3.15, 1.5),
        3.7,
        1.9,
        "Actions (adapt ranking, not full relearn first)\n• LOCO blocks → trim / cap segments\n• Recalibrate ê(T|X)\n• Target only overlap support",
        fc="#ffffff",
        fontsize=6.5,
    )
    _box(
        ax,
        (6.15, 1.5),
        3.7,
        1.9,
        "Actions\n• PO-LOCO + LOCO–AUUC features\n• Extend labels; retrain τ (prod)\n• REALLOCATE rules → then RELEARN",
        fc="#ffffff",
        fontsize=6.5,
    )

    ax.text(
        5,
        0.4,
        "Cheap monitor: Registry X-learner hourly · Deep confirm (neural LOCO) only on alert",
        ha="center",
        fontsize=8,
        color="#475569",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
