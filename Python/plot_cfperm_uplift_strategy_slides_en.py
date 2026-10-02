#!/usr/bin/env python3
"""
Slide-style concise diagram (cfperm_diagram.pptx lineage):
  Concept vs Mixture · DRPerm · Meta-learner monitor · Adapt ladder
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "cfperm_uplift_strategy_slides_en.png"

# Slide-like palette
BLUE = "#2563eb"
LIGHT_BLUE = "#dbeafe"
ORANGE = "#ea580c"
LIGHT_ORANGE = "#ffedd5"
GREEN = "#059669"
LIGHT_GREEN = "#ecfdf5"
GRAY = "#64748b"
DARK = "#0f172a"


def slide_box(ax, xy, w, h, title, body, *, fc=LIGHT_BLUE, title_color=DARK, edge=BLUE):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=edge, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=9, fontweight="bold", color=title_color)
    ax.text(x + w / 2, y + h / 2 - 0.15, body, ha="center", va="center", fontsize=7.2, color=DARK, linespacing=1.25)


def arrow_h(ax, x1, x2, y, color=GRAY):
    ax.add_patch(
        FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=color)
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.33, 7.5))  # 16:9 slide
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(6.65, 7.05, "Uplift under shift: DRPerm / FSDS · three-layer adapt", ha="center", fontsize=14, fontweight="bold")
    ax.text(
        6.65, 6.65,
        "Reference batch (REF) vs LIVE · W=batch · T=treatment · Meta-learner monitor (no online cross-fit)",
        ha="center", fontsize=8, color=GRAY,
    )

    # Row 1: Two axes (slide 28 style)
    ax.text(0.4, 6.15, "Detect", fontsize=9, fontweight="bold", color=BLUE)
    slide_box(
        ax, (0.35, 4.85), 3.0, 1.15,
        "Covariate / user mixture",
        "MMD² · RF domain AUC(W|X)\nESS overlap · LOGO-MMD",
        fc=LIGHT_ORANGE, title_color=ORANGE, edge=ORANGE,
    )
    slide_box(
        ax, (3.55, 4.85), 3.0, 1.15,
        "Concept / outcome",
        "DRPerm · PO-risk\n(batch W, clever η)",
        fc=LIGHT_GREEN, title_color=GREEN, edge=GREEN,
    )
    arrow_h(ax, 3.35, 3.55, 5.42)
    slide_box(
        ax, (6.75, 4.85), 2.5, 1.15,
        "Ranking (L1)",
        "AUUC_live · LOCO\nquintile / overlap AUUC",
        fc="#fef3c7", title_color="#a16207", edge="#ca8a04",
    )
    arrow_h(ax, 6.55, 6.75, 5.42)
    slide_box(
        ax, (9.45, 4.85), 3.5, 1.15,
        "Monitor model",
        "Meta-learner (X/R/T)\nnot full DML cross-fit online",
        fc=LIGHT_BLUE,
    )

    # Row 2: Three strategies (user summary)
    ax.text(0.4, 4.45, "Adapt", fontsize=9, fontweight="bold", color=BLUE)

    slide_box(
        ax, (0.35, 2.55), 4.0, 1.65,
        "① Outcome shift (PO-risk)",
        "Large PO-risk / DRPerm reject\n→ (re)train τ & nuisances\nMMD may be flat; AUUC ranking likely breaks on overlap core",
        fc="#fce7f3", title_color="#9d174d", edge="#db2777",
    )
    slide_box(
        ax, (4.55, 2.55), 4.0, 1.65,
        "② User-mixture shift",
        "MMD / domain AUC ↑\n→ check AUUC consistency:\nglobal vs overlap band\n→ cap / narrow support / REF (not τ first)",
        fc=LIGHT_ORANGE, title_color=ORANGE, edge=ORANGE,
    )
    slide_box(
        ax, (8.75, 2.55), 4.25, 1.65,
        "③ Feedback + overlap decay",
        "Policy polluted e(T|X)\n→ refit propensity e first\n+ matching / balance; optional KMM\npause caps until ESS OK",
        fc=LIGHT_GREEN, title_color=GREEN, edge=GREEN,
    )

    # Row 3: DRPerm mini-flow (slide 17)
    ax.text(0.4, 2.15, "DRPerm (concept test)", fontsize=9, fontweight="bold", color=BLUE)
    y_dr = 1.05
    boxes = [
        ("Stack\nREF∪LIVE", 0.35, 0.9),
        ("Cross-fit\nμ̂, ê(W|X)", 1.45, 0.9),
        ("η̂=(Y−μ̂)(W−ê)", 2.55, 0.9),
        ("Permute W\nrefit → PO-risk R", 3.65, 0.9),
        ("p-value", 4.75, 0.9),
        ("Gate L2 &\nRELEARN", 5.85, 0.9),
    ]
    for label, x, w in boxes:
        patch = FancyBboxPatch(
            (x, y_dr), w, 0.85, boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2, edgecolor=GREEN, facecolor=LIGHT_GREEN,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y_dr + 0.42, label, ha="center", va="center", fontsize=6.8, color=DARK)
    for i in range(len(boxes) - 1):
        x_end = boxes[i][1] + boxes[i][2]
        x_start = boxes[i + 1][1]
        arrow_h(ax, x_end + 0.05, x_start - 0.05, y_dr + 0.42, color=GREEN)

    ax.text(
        7.2, 1.45,
        "Joint refit (offline / snapshot X):\ntemporal split e,μ on past → pseudo on current → τ\nHoldout DR-AUUC before deploy",
        fontsize=7, color=DARK, va="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=GRAY, linewidth=1),
    )

    ax.text(
        6.65, 0.35,
        "Statistical core: batch FSDS separates P(X) vs P(Y|X); meta-learner + permute-refit replaces infeasible online cross-fit",
        ha="center", fontsize=7, color=GRAY, style="italic",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
