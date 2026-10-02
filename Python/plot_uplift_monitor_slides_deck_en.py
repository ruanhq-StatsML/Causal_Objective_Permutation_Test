#!/usr/bin/env python3
"""Three slide-style pages: uplift two-batch monitoring (cfperm_diagram.pptx style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "uplift_monitor_slides"

BLUE, LBLUE = "#2563eb", "#dbeafe"
ORANGE, LORANGE = "#ea580c", "#ffedd5"
GREEN, LGREEN = "#059669", "#ecfdf5"
GRAY, DARK = "#64748b", "#0f172a"
W, H = 13.333, 7.5


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", p)


def box(ax, x, y, w, h, title, body, *, fc=LBLUE, ec=BLUE, tfs=9, bfs=7.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4, edgecolor=ec, facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=tfs, fontweight="bold", color=DARK)
    ax.text(x + w / 2, y + h / 2 - 0.12, body, ha="center", va="center", fontsize=bfs, color=DARK, linespacing=1.2)


def arr(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=GRAY))


def slide_frame(ax, title: str, subtitle: str = ""):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(W / 2, H - 0.45, title, ha="center", fontsize=14, fontweight="bold", color=DARK)
    if subtitle:
        ax.text(W / 2, H - 0.85, subtitle, ha="center", fontsize=8, color=GRAY)


def page1():
    fig, ax = plt.subplots(figsize=(W, H))
    slide_frame(
        ax,
        "Uplift model monitoring — two continuous batches (REF vs LIVE)",
        "W = batch label · T = treatment · τ̂ fit on REF only · score LIVE with frozen τ̂",
    )
    # REF / LIVE (slide 2 style)
    box(ax, 0.5, 5.5, 2.8, 1.35, "Reference batch REF", "Train τ̂\nEval AUUC_ref\nLOCO retrains here", fc=LBLUE)
    box(ax, 3.6, 5.5, 2.8, 1.35, "Live batch LIVE", "Score τ̂ (frozen)\nAUUC_live · ΔAUUC\nEmpirical ATE slices", fc=LORANGE, ec=ORANGE)
    arr(ax, 3.3, 3.6, 6.15)
    ax.text(3.45, 6.95, "Monitor ranking degradation", fontsize=8, color=ORANGE, fontweight="bold")

    # Gates row
    box(
        ax, 0.5, 3.85, 12.3, 1.35,
        "Step 0 — FSDS validity gates",
        "SRM(T)  ·  MMD²(X_ref, X_live)  ·  Domain RF AUC(X;W)  ·  ESS_overlap on ê(W|X)\n"
        "If ESS low → Refresh REF only (defer post-hoc rules)",
        fc=LGREEN, ec=GREEN, bfs=7,
    )

    # L1 L2 PO row
    box(ax, 0.5, 2.2, 3.9, 1.4, "L1 · τ ranking", "AUUC · LOCO–AUUC\nQuintile & pairwise AUUC\nOverlap-band AUUC", fc="#fef3c7", ec="#ca8a04")
    arr(ax, 4.4, 4.55, 2.9)
    box(ax, 4.55, 2.2, 3.9, 1.4, "L2 · uplift level", "Empirical ATE_k\nMean τ̂_k · pairwise ΔATE\n(PO-gated for budget)", fc="#fde68a", ec="#ca8a04")
    arr(ax, 8.45, 8.6, 2.9)
    box(ax, 8.6, 2.2, 4.2, 1.4, "Concept axis", "Stack REF∪LIVE\nDRPerm / PO-risk\nPermute batch W → refit", fc=LGREEN, ec=GREEN)

    box(
        ax, 0.5, 0.55, 12.3, 1.35,
        "Post-hoc subset localization → business_rules",
        "Named subsets: LOCO group · domain quintile Q_k · overlap band · LOGO-MMD block · L2 pair  →  REALLOCATE · HOLD · REFRESH_REF · RELEARN · MONITOR",
        fc="#f1f5f9", ec=GRAY, bfs=6.8,
    )
    save(fig, "slide1_two_batch_monitor_pipeline.png")


def page2():
    fig, ax = plt.subplots(figsize=(W, H))
    slide_frame(
        ax,
        "Detect: covariate shift vs concept drift (FSDS + DRPerm)",
        "Same split as CFPerm roadmap — MMD/RF for P(X) · PO-risk for P(Y|X)",
    )
    box(ax, 0.45, 5.35, 5.8, 1.55, "Covariate / user-mixture shift", "MMD² ↑ · Domain RF AUC ↑\nLOGO-MMD · ESS on ê(W|X)\nAUUC may drop on new support", fc=LORANGE, ec=ORANGE)
    box(ax, 6.55, 5.35, 6.35, 1.55, "Concept / outcome shift", "MMD may be flat\nDRPerm reject · PO-risk ↑\nRanking breaks on overlap core", fc="#fce7f3", ec="#db2777")

    # DRPerm strip
    ax.text(0.45, 4.85, "DRPerm (batch W, clever covariates η̂)", fontsize=9, fontweight="bold", color=GREEN)
    steps = ["REF∪LIVE", "Cross-fit μ̂, ê", "η̂=(Y−μ̂)(W−ê)", "Permute W\nrefit", "PO-risk R", "p-value"]
    x = 0.45
    for i, s in enumerate(steps):
        w = 1.95 if i < 5 else 1.5
        box(ax, x, 3.55, w, 1.05, "", s, fc=LGREEN, ec=GREEN, tfs=1, bfs=7)
        if i < len(steps) - 1:
            arr(ax, x + w + 0.02, x + w + 0.28, 4.05)
        x += w + 0.3

    box(
        ax, 0.45, 1.85, 6.2, 1.45,
        "Mixture → read AUUC consistency",
        "AUUC_global  vs  AUUC_overlap (ê∈[0.15,0.85])\n"
        "ovlp ≫ global → narrow support + cap (not τ relearn first)",
        fc=LORANGE, ec=ORANGE, bfs=7,
    )
    box(
        ax, 6.85, 1.85, 6.05, 1.45,
        "Concept → gate heavy adapt",
        "PO reject → L2 budget REALLOCATE allowed\n"
        "Persistent gap on overlap → schedule τ / joint refit",
        fc="#fce7f3", ec="#db2777", bfs=7,
    )

    box(ax, 0.45, 0.45, 12.45, 1.15, "Meta-learner monitor", "X-learner / R-learner / registry for LOCO–AUUC & daily SLA — prod DragonNet optional (shadow)", fc=LBLUE, bfs=7)
    save(fig, "slide2_detect_drperm_fsds.png")


def page3():
    fig, ax = plt.subplots(figsize=(W, H))
    slide_frame(
        ax,
        "Adapt: three layers · rule priority · online vs refit",
        "Policy first (frozen τ̂ trial) · joint refit only with snapshot X + holdout",
    )
    # Three pillars
    box(
        ax, 0.4, 4.95, 4.0, 2.05,
        "① Outcome shift",
        "DRPerm / PO-risk reject\n→ train or joint-refit τ, μ, e\n(even if MMD not significant)\nAUUC ranking likely unstable on core",
        fc="#fce7f3", ec="#db2777", bfs=6.8,
    )
    box(
        ax, 4.55, 4.95, 4.0, 2.05,
        "② User-mixture",
        "MMD / domain AUC high\n→ overlap AUUC vs global\nCap quintile · narrow support · REF\nOther knobs (LOCO, thresholds) easy",
        fc=LORANGE, ec=ORANGE, bfs=6.8,
    )
    box(
        ax, 8.7, 4.95, 4.25, 2.05,
        "③ Feedback + overlap",
        "Policy → e(T|X) biased\nRefit propensity e first\nMatching / balance · pause caps\nRestore ESS then re-monitor",
        fc=LGREEN, ec=GREEN, bfs=6.8,
    )

    box(
        ax, 0.4, 2.85, 12.55, 1.75,
        "Rule priority (after localization)",
        "P1: REFRESH_REF (ESS) · global/quintile REALLOCATE · L2 budget if PO reject   |   "
        "P2: HOLD/LOCO segment · RELEARN ticket (deferred)   |   P3: narrow overlap band   |   "
        "Collisions: cap blocks L2 into same slice · PO accept → no ATE-only spend",
        fc="#fffbeb", ec="#b45309", bfs=6.5,
    )

    box(
        ax, 0.4, 0.45, 12.55, 2.15,
        "Online streaming · no cross-fit",
        "Monitor: meta-learner + batch DRPerm (not full online DML K-fold)\n"
        "Joint refit: temporal split past→e,μ · current→pseudo→τ · independent holdout DR-AUUC\n"
        "Requires feature snapshot / warehouse — else summary-only alarms",
        fc=LBLUE, bfs=7,
    )
    save(fig, "slide3_adapt_priority_online.png")


def main():
    page1()
    page2()
    page3()
    # Combined contact sheet for one-file preview
    fig, axes = plt.subplots(3, 1, figsize=(W, H * 3))
    for ax, name in zip(
        axes,
        ["slide1_two_batch_monitor_pipeline.png", "slide2_detect_drperm_fsds.png", "slide3_adapt_priority_online.png"],
    ):
        img = plt.imread(OUT_DIR / name)
        ax.imshow(img)
        ax.axis("off")
    fig.savefig(OUT_DIR / "uplift_monitor_3slides_combined.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT_DIR / "uplift_monitor_3slides_combined.png")


if __name__ == "__main__":
    main()
