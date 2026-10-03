#!/usr/bin/env python3
"""English poster: ADAPT-first methodology + economic benefit framing (two-batch uplift FSDS)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "uplift_fsds_adapt_economics_methodology_en.png"


def box(ax, xy, w, h, text, *, fc="#f8fafc", ec="#334155", fs=7, lw=1.0, title=None, title_fs=8):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    if title:
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=title_fs, fontweight="bold")
        ax.text(x + w / 2, y + (h - 0.35) / 2, text, ha="center", va="center", fontsize=fs)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, *, color="#475569", dashed=False, rad=0.0):
    style = f"arc3,rad={rad}"
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=11, linewidth=1.05, color=color,
            linestyle="--" if dashed else "-", connectionstyle=style,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(17, 20))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 20)
    ax.axis("off")

    ax.text(
        8.5, 19.45,
        "Two-Batch Uplift FSDS — Adaptation Methodology & Economic Rationale",
        ha="center", fontsize=14.5, fontweight="bold",
    )
    ax.text(
        8.5, 18.95,
        "Focus: what to change on LIVE when AUUC drops · priority P1→P5 · trial before RELEARN · $ impact order-of-magnitude",
        ha="center", fontsize=8.2, color="#475569",
    )

    # ---- Trigger strip ----
    box(
        ax, (0.4, 17.85), 16.2, 0.85,
        "Trigger: AUUC_live or ΔAUUC below SLA  →  gates OK (SRM, ESS)  →  localize (L1 ranking + L2 level + PO-risk)  →  adapt",
        fc="#dbeafe", fs=7.5, title="Entry condition", title_fs=8.5,
    )

    # ---- ADAPT FLOW (center, large) ----
    ax.text(8.5, 17.35, "How to adapt (execution ladder — same frozen τ̂ during trial)", ha="center", fontsize=11, fontweight="bold", color="#9a3412")

    ladder = [
        ("1 · REFRESH_REF  [P1]", "ESS low / new mix · extend or match REF cohort\n$ → avoid wrong LOCO cuts & false relearn tickets", "#fee2e2"),
        ("2 · REALLOCATE L1  [P1]", "Global pause · cap worst domain quintile · pairwise continue\n$ → stop treat spend where ranking is broken (immediate OPEX save)", "#ffedd5"),
        ("3 · Narrow support  [P3]", "Deploy uplift only on ê(W|X) ∈ [0.15, 0.85]\n$ → recover SLA without GPU relearn (covariate shift path)", "#fed7aa"),
        ("4 · HOLD + segment  [P2]", "LOGO-MMD block trim · LOCO segment rules (never blind feature drop)\n$ → cheaper than full τ̂ retrain; keeps prod pipeline stable", "#fef9c3"),
        ("5 · Recalibrate ê(T|X)  [P2]", "When τ̂ ≈ empirical ATE but AUUC weak\n$ → ranking fix via nuisance/thresholds, not new neural train", "#e0e7ff"),
        ("6 · REALLOCATE L2 budget  [P1 if PO reject]", "Shift same-day treat intensity to high-empirical-ATE slice (uncapped only)\n$ → redeploy budget to slices with realized lift under concept shift", "#fce7f3"),
        ("7 · RELEARN prod τ̂  [P2* deferred]", "Only after 1–2 trial LIVE windows fail: PO reject + τ̂≠ATE + AUUC_ovlp red\n$ → defer expensive train/compute until cheaper adapt exhausted", "#ddd6fe"),
    ]
    y = 16.55
    for title, body, color in ladder:
        box(ax, (0.4, y - 1.35), 10.2, 1.35, body, fc=color, fs=6.7, title=title, title_fs=7.5)
        y -= 1.48

    arrow(ax, (5.5, 16.55 - 7 * 1.48 + 0.2), (5.5, 5.95))

    # Trial loop
    box(
        ax, (0.4, 4.75), 10.2, 1.15,
        "Trial loop: apply steps 1–6 as sorted rules → next LIVE window → re-measure policy-conditional AUUC\n"
        "Escalate one rung only if SLA still fails · P2* RELEARN is last-resort CAPEX",
        fc="#ffffff", ec="#0f172a", lw=1.3, fs=7, title="Daily re-monitor", title_fs=8,
    )
    arrow(ax, (0.5, 5.35), (0.5, 16.0), dashed=True, color="#64748b", rad=-0.12)
    ax.text(0.15, 10.5, "re-monitor", fontsize=7, color="#64748b", rotation=90, va="center")

    # ---- RIGHT: Priority & conflicts (compact) ----
    ax.text(13.2, 16.9, "Rule priority & conflicts", ha="center", fontsize=9.5, fontweight="bold")
    box(
        ax, (10.85, 12.35), 5.75, 4.35,
        "Sort by P1 → P5 (code: business_rules.priority)\n\n"
        "Collisions:\n"
        "• REFRESH_REF blocks LOCO/L2 interpret\n"
        "• L1 cap on slice S blocks L2 spend into S\n"
        "• PO accept → no L2 budget move (P4 MONITOR)\n"
        "• AUUC_ovlp green → narrow + REF, not RELEARN\n"
        "• Multiple P1: apply all compatible caps\n\n"
        "Monitor steps (gates, L1, L2, PO) feed this panel only.",
        fc="#fffbeb", fs=6.5, ec="#b45309",
    )

    # ---- RIGHT: Economic benefit estimate ----
    ax.text(13.2, 11.85, "Economic benefit (order-of-magnitude)", ha="center", fontsize=9.5, fontweight="bold")
    box(
        ax, (10.85, 6.15), 5.75, 5.05,
        "Operational (uplift adapt)\n"
        "• REALLOCATE L1: if cap frees fraction f of treat\n"
        "  impressions at cost c per treat → save ≈ f × volume × c\n"
        "  per window (benchmark: single quintile ~20% LIVE n).\n"
        "• Narrow support: avoid off-support spend;\n"
        "  demo gap AUUC_ovlp − global up to ~0.01–0.02+\n"
        "  → often defers RELEARN (weeks eng + GPU).\n"
        "• RELEARN deferred: trial ladder targets\n"
        "  “policy fix before model fix” — typical CAPEX\n"
        "  only when PO + τ̂ mismatch persist.\n\n"
        "Federated FSDS (same contract, vertical X)\n"
        "• Uplink O(p) vs centralize O(np):\n"
        "  measured Eff ≈ 48×–1410× on 7 tabular demos\n"
        "  → recurring bandwidth + governance latency ↓\n"
        "  when raw X cannot leave site.\n\n"
        "Not counted: uplift lift recovered on overlap core.",
        fc="#ecfdf5", fs=6.35, ec="#047857", lw=1.2,
    )

    arrow(ax, (10.2, 8.5), (10.85, 8.5), color="#047857")

    # Footer
    ax.text(
        8.5, 0.45,
        "PNG: uplift_fsds_adapt_economics_methodology_en.png  ·  "
        "plot_uplift_fsds_adapt_economics_en.py  ·  "
        "rules: loco_auuc/subset_localization.py",
        ha="center", fontsize=6.5, color="#64748b",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
