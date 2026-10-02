#!/usr/bin/env python3
"""English methodology poster: monitor pipeline + rule priority & conflict resolution."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "uplift_fsds_methodology_priority_en.png"


def box(ax, xy, w, h, text, *, fc="#f8fafc", ec="#334155", fs=7, lw=1.0, bold_title=None):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.05",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    if bold_title:
        ax.text(x + w / 2, y + h - 0.22, bold_title, ha="center", va="top", fontsize=fs + 0.5, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.12, text, ha="center", va="center", fontsize=fs)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, *, color="#475569", dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=10, linewidth=0.95, color=color,
            linestyle="--" if dashed else "-",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 22))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 22)
    ax.axis("off")

    ax.text(8, 21.55, "Two-Batch Uplift FSDS — Monitoring Methodology & Rule Priority", ha="center", fontsize=15, fontweight="bold")
    ax.text(
        8, 21.05,
        "W = batch (REF/LIVE) · T = treatment · τ̂ trained on REF only · LIVE scores frozen τ̂ · outputs sorted by priority (P1 first)",
        ha="center", fontsize=8, color="#475569",
    )

    # ========== LEFT: MONITOR PIPELINE ==========
    ax.text(4, 20.55, "A. Monitor pipeline (single LIVE window)", ha="center", fontsize=10, fontweight="bold", color="#1e3a8a")

    steps = [
        ("Contract", "REF: fit τ̂ on D_ref^tr · AUUC_ref on D_ref^ev\nLIVE: score τ̂ · AUUC_live · LOCO retrains on REF only", "#dbeafe"),
        ("Step 0 · Gates", "SRM(T) per batch · MMD²(X) · Domain RF AUC(X;W)\nESS_overlap on ê(W|X) — if ESS low → stop post-hoc", "#ecfdf5"),
        ("Step 1 · L1 ranking", "ΔAUUC · LOCO–AUUC (drop_ref, drop_live)\nQuintile AUUC_k · pairwise AUUC · overlap-band AUUC", "#fef3c7"),
        ("Step 2 · L2 level", "Empirical ATE_k · mean τ̂_k on LIVE slices\nPairwise ΔATE (budget map — PO-gated)", "#fde68a"),
        ("Step 3 · Concept", "Stack REF∪LIVE · DRPerm / PO-risk on W\n(Optional PO-LOCO: outcome vs ranking drivers)", "#fae8ff"),
        ("Regime", "Support/mix · Covariate shift (X) · Concept (Y|X, PO reject)", "#f1f5f9"),
    ]
    y = 19.5
    for title, body, color in steps:
        box(ax, (0.35, y - 1.05), 7.3, 1.05, body, fc=color, fs=6.6, bold_title=title)
        y -= 1.15
    arrow(ax, (3.5, 19.5 - 6 * 1.15), (3.5, 12.85))

    box(
        ax, (0.35, 11.75), 7.3, 0.95,
        "Emit business_rules[] + gates JSON + L1/L2 tables\nSort rules by priority field (ascending): P1 → P5",
        fc="#e2e8f0", fs=6.8, bold_title="Outputs",
    )

    # ========== RIGHT: ADAPT ACTIONS ==========
    ax.text(12, 20.55, "B. Adaptation actions (by type)", ha="center", fontsize=10, fontweight="bold", color="#9a3412")

    acts = [
        ("REFRESH_REF", "Match/extend REF · pause LOCO-driven cuts\nTrigger: ESS low · mix_shift diagnosis", "#fee2e2", "P1"),
        ("REALLOCATE · global", "Pause/cap all uplift targeting on LIVE\nTrigger: AUUC_live < SLA floor", "#ffedd5", "P1"),
        ("REALLOCATE · quintile", "Cap worst domain quintile Q* · continue best pair\nTrigger: AUUC_k* < floor & pairwise spread", "#ffedd5", "P1"),
        ("REALLOCATE · L2 budget", "Shift spend to high empirical ATE slice\nTrigger: PO-risk REJECT only", "#ffedd5", "P1"),
        ("HOLD_FEATURE", "LOGO-MMD block down-weight · segment trim\nDo not blind-drop prod features", "#fef9c3", "P2"),
        ("REALLOCATE · LOCO segment", "Segment rules on LOCO-top group\nRetrain stays on REF in monitor", "#fef9c3", "P2"),
        ("RELEARN τ̂", "Schedule prod retrain AFTER REALLOCATE trial\nTrigger: concept label · persistent gap", "#ddd6fe", "P2*"),
        ("REALLOCATE · narrow support", "Treat only ê(W|X) ∈ [0.15, 0.85] on LIVE\nTrigger: AUUC_ovlp ≫ AUUC_global (+0.01)", "#fed7aa", "P3"),
        ("MONITOR · L2 only", "Large ΔATE but PO accept — no budget move\nUse L1 AUUC rules first", "#f8fafc", "P4"),
        ("MONITOR · default", "No rule fired — continue scheduled monitor", "#f8fafc", "P5"),
    ]
    y0 = 19.35
    for i, (title, body, color, pri) in enumerate(acts):
        row, col = divmod(i, 2)
        x = 8.05 + col * 3.95
        y = y0 - row * 1.55
        box(ax, (x, y - 1.35), 3.75, 1.35, body, fc=color, fs=5.8, bold_title=f"{title}  [{pri}]")

    # ========== BOTTOM: PRIORITY & CONFLICTS ==========
    ax.text(8, 10.35, "C. Rule priority & conflict resolution (policy engine)", ha="center", fontsize=11, fontweight="bold")

    # Priority table as text block
    pri_text = (
        "Execution order (code: rules.sort by priority ascending):\n"
        "  P1 — Safety & SLA: REFRESH_REF (mix) · global REALLOCATE · quintile REALLOCATE · L2 budget (PO reject)\n"
        "  P2 — Localization: HOLD_FEATURE (LOGO) · LOCO segment REALLOCATE · RELEARN ticket (deferred until trial)\n"
        "  P3 — Support geometry: narrow overlap-band REALLOCATE (runs WITH P1 caps, not instead of)\n"
        "  P4 — Diagnostics only: MONITOR on pairwise ATE when PO accept\n"
        "  P5 — Default: continue monitor\n"
        "* RELEARN is P2 in the queue but MUST NOT run until 1–2 LIVE trial windows fail under REALLOCATE/narrow support."
    )
    box(ax, (0.35, 7.85), 15.3, 2.35, pri_text, fc="#fffbeb", fs=7.2, ec="#b45309", lw=1.3)

    conflicts = (
        "When rules collide — apply ALL compatible P1 actions, then resolve overlaps:\n"
        "  1. ESS low / REFRESH_REF  →  suppress interpretation of LOCO & L2; no LOCO-driven feature retirement.\n"
        "  2. L1 quintile CAP on slice S  →  block L2 budget REALLOCATE into S (cannot spend into a capped slice).\n"
        "  3. PO-risk ACCEPT  →  veto L2 budget REALLOCATE; keep P4 MONITOR on ΔATE; ranking adapt = L1 only.\n"
        "  4. AUUC_ovlp green & AUUC_global red  →  prefer P3 narrow support + REFRESH_REF; veto immediate RELEARN.\n"
        "  5. LOCO vs HOLD_FEATURE  →  complementary: LOGO = which block shifted; LOCO = ranking depends on whom.\n"
        "  6. Multiple P1 REALLOCATE  →  apply global pause AND quintile cap AND (if PO reject) L2 shift to uncapped high-ATE slice.\n"
        "  7. τ̂ ≈ empirical ATE on capped slices but AUUC weak  →  recalibrate ê(T|X) / score thresholds before RELEARN.\n"
        "  8. Trial loop: next LIVE window uses same frozen τ̂; re-run Steps 0–3; escalate only if policy-conditional AUUC still fails."
    )
    box(ax, (0.35, 4.55), 15.3, 3.05, conflicts, fc="#eff6ff", fs=7.0, ec="#1d4ed8", lw=1.3)

    # Trial loop
    box(
        ax, (0.35, 2.35), 15.3, 1.85,
        "D. Trial loop (daily re-monitor)\n"
        "Apply sorted rules → measure policy-conditional AUUC & AUUC_ovlp → tighten caps / widen REF / recalibrate e(T|X) → "
        "schedule RELEARN only if PO reject + τ̂≠ATE + ranking fails on deployed support.",
        fc="#ffffff", fs=7.2, ec="#0f172a", lw=1.2,
    )

    ax.text(
        8, 0.55,
        "Generate: Python/plot_uplift_fsds_methodology_en.py  ·  "
        "Implement: loco_auuc/subset_localization.py → business_rules_from_localization  ·  "
        "LaTeX: docs/latex/uplift_fsds_two_batch_formulation.tex",
        ha="center", fontsize=6.5, color="#64748b",
    )

    # Decorative divider
    ax.add_patch(Rectangle((7.85, 2.2), 0.06, 18.5, facecolor="#cbd5e1", edgecolor="none"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
