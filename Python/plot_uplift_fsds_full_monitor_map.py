#!/usr/bin/env python3
"""Full uplift two-batch monitoring logic + adaptation map (REF vs LIVE)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "uplift_fsds_full_monitor_logic_en.png"


def box(ax, xy, w, h, text, *, fc="#eef4ff", ec="#334155", fs=7.5, lw=1.1):
    x, y = xy
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, start, end, *, color="#475569", style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=1.0,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 20))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 26)
    ax.axis("off")

    ax.text(
        7,
        25.45,
        "Uplift two-batch FSDS — full monitoring logic & adaptation map",
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        7,
        24.85,
        "W = batch (REF/LIVE) · T = treatment · τ̂ frozen on LIVE · trial loop re-scores same τ̂",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )

    # --- Column headers ---
    ax.text(3.5, 24.35, "MONITOR", ha="center", fontsize=9, fontweight="bold", color="#1e40af")
    ax.text(10.5, 24.35, "ADAPT", ha="center", fontsize=9, fontweight="bold", color="#9a3412")

    # Contract
    box(
        ax,
        (1.2, 22.6),
        11.6,
        1.35,
        "Data contract\n"
        "REF: train τ̂ on D_ref^tr · eval AUUC_ref on D_ref^ev\n"
        "LIVE: score τ̂ only · AUUC_live · LOCO retrains on REF only",
        fc="#dbeafe",
        fs=8,
    )
    arrow(ax, (7, 22.6), (7, 22.15))

    # Step 0
    box(
        ax,
        (1.2, 20.55),
        11.6,
        1.45,
        "Step 0 — Validity gates (batch geometry)\n"
        "SRM on T (each batch) · MMD²(X) · Domain RF AUC(X;W) · ESS_overlap on ê(W|X)",
        fc="#ecfdf5",
        fs=7.8,
    )
    arrow(ax, (7, 20.55), (7, 20.05))

    # ESS branch
    box(ax, (0.4, 18.85), 5.2, 1.0, "ESS low / mix\n→ REFRESH_REF only\n(defer LOCO / L2 / caps)", fc="#fee2e2", fs=7.5)
    box(ax, (8.4, 18.85), 5.2, 1.0, "ESS OK\n→ post-hoc localization", fc="#dcfce7", fs=7.5)
    arrow(ax, (5, 20.05), (2.8, 19.85))
    arrow(ax, (9, 20.05), (11, 19.85))

    # L1
    box(
        ax,
        (1.2, 16.75),
        11.6,
        1.75,
        "Step 1 — Level 1 · τ ranking (frozen τ̂ on LIVE)\n"
        "Global: AUUC_ref · AUUC_live · ΔAUUC\n"
        "Local: LOCO–AUUC (drop_ref, drop_live) · domain-quintile AUUC_k · pairwise AUUC · overlap-band AUUC",
        fc="#fef3c7",
        fs=7.4,
    )
    arrow(ax, (11, 18.85), (7, 18.5))
    arrow(ax, (7, 18.5), (7, 18.35))
    arrow(ax, (7, 16.75), (7, 16.25))

    # L2
    box(
        ax,
        (1.2, 14.65),
        11.6,
        1.45,
        "Step 2 — Level 2 · uplift level on LIVE slices (not ranking)\n"
        "Empirical ATE_k · mean τ̂_k · pairwise ΔATE (budget map)",
        fc="#fde68a",
        fs=7.5,
    )
    arrow(ax, (7, 14.65), (7, 14.15))

    # PO
    box(
        ax,
        (1.2, 12.75),
        11.6,
        1.25,
        "Step 3 — Concept axis (clever covariates, stacked REF∪LIVE, label W)\n"
        "DRPerm / PO-risk · optional PO-LOCO (outcome vs ranking drivers)",
        fc="#fae8ff",
        fs=7.5,
    )
    arrow(ax, (7, 12.75), (7, 12.25))

    # Regime router
    box(
        ax,
        (1.2, 10.85),
        11.6,
        1.25,
        "Regime router (FSDS + PO)\n"
        "X stable + PO accept · X shift (MMD/AUC↑) · X stable + PO reject (Y|X / τ surface)",
        fc="#f1f5f9",
        fs=7.5,
    )
    arrow(ax, (7, 10.85), (7, 10.35))

    # Three regime columns
    y_reg = 8.35
    box(
        ax,
        (0.35, y_reg),
        4.1,
        1.55,
        "Support / mix\nESS was low or\nAUC → 1 tail",
        fc="#fecaca",
        fs=7,
    )
    box(
        ax,
        (4.95, y_reg),
        4.1,
        1.55,
        "Covariate shift\nRanking on new X mix\nPO often accept",
        fc="#fed7aa",
        fs=7,
    )
    box(
        ax,
        (9.55, y_reg),
        4.1,
        1.55,
        "Concept shift\nMMD flat · PO reject\nL2 spend gated",
        fc="#fbcfe8",
        fs=7,
    )
    arrow(ax, (3.5, 10.85), (2.4, 9.9))
    arrow(ax, (7, 10.85), (7, 9.9))
    arrow(ax, (10.5, 10.85), (11.6, 9.9))

    # Adapt column (right side detailed)
    adapt_y = 6.85
    actions = [
        ("REFRESH_REF", "Extend/match REF · roll stable LIVE\nPause LOCO-driven cuts", "#fee2e2"),
        ("REALLOCATE L1", "Global pause · cap worst quintile\npairwise continue · narrow support [0.15,0.85]", "#ffedd5"),
        ("HOLD_FEATURE", "LOGO-MMD block trim · LOCO segment rules\nDo NOT drop prod features blindly", "#fef9c3"),
        ("Recalibrate", "ê(T|X) or score thresholds\nwhen τ̂ ≈ ATE but AUUC weak", "#e0e7ff"),
        ("REALLOCATE L2", "Shift budget to high ATE slice\nonly if PO-risk reject", "#fce7f3"),
        ("RELEARN τ̂", "After 1–2 trial windows:\nPO reject + τ̂ ≠ ATE + AUUC_ovlp red", "#ddd6fe"),
    ]
    for i, (title, body, color) in enumerate(actions):
        row, col = divmod(i, 2)
        x = 0.35 + col * 6.85
        y = adapt_y - row * 2.05
        box(ax, (x, y), 6.45, 1.75, f"{title}\n{body}", fc=color, fs=6.8)

    # Arrows from regime to adapt groups
    arrow(ax, (2.4, 8.35), (3.2, 7.5), color="#b91c1c")
    arrow(ax, (7, 8.35), (7, 7.5), color="#c2410c")
    arrow(ax, (11.6, 8.35), (10.5, 5.0), color="#be185d")

    # Trial loop
    box(
        ax,
        (1.2, 1.55),
        11.6,
        1.35,
        "Trial loop (next LIVE window)\n"
        "Apply adapt · keep REF τ̂ frozen · re-run Steps 0–3 · measure policy-conditional AUUC\n"
        "Improve but < SLA → tighten cap / band · ovlp green / global red → stay narrow + REF · else escalate RELEARN",
        fc="#ffffff",
        ec="#0f172a",
        fs=7.5,
    )
    arrow(ax, (7, 3.2), (7, 2.9), color="#0f172a")
    # feedback
    ax.add_patch(
        FancyArrowPatch(
            (0.8, 2.2),
            (0.8, 22.0),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#64748b",
            linestyle="--",
            connectionstyle="arc3,rad=-0.15",
        )
    )
    ax.text(0.15, 12.5, "daily\nre-monitor", fontsize=7, color="#64748b", rotation=90, va="center")

    # Outputs strip
    box(
        ax,
        (1.2, 0.25),
        11.6,
        1.05,
        "Outputs: gates JSON · L1/L2 tables · business_rules[] "
        "(REALLOCATE · HOLD_FEATURE · REFRESH_REF · RELEARN · MONITOR)\n"
        "Code: run_uplift_subset_localization · plot: uplift_fsds_full_monitor_logic_en.png",
        fc="#f8fafc",
        fs=6.8,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
