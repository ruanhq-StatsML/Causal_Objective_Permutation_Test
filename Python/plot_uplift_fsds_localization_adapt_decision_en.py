#!/usr/bin/env python3
"""
English decision poster:
  localization outputs → strategy sort order → MMD × overlap AUUC × PO → adapt vs RELEARN
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "uplift_fsds_localization_adapt_decision_en.png"


def box(ax, xy, w, h, text, *, fc="#f8fafc", ec="#334155", fs=6.5, lw=1.0, title=None):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    if title:
        ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=7.2, fontweight="bold")
        ax.text(x + w / 2, y + (h - 0.22) / 2, text, ha="center", va="center", fontsize=fs)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 24))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 24)
    ax.axis("off")

    ax.text(
        9, 23.35,
        "After Localization: Strategy Sort Order · MMD vs Overlap AUUC · Adapt vs RELEARN",
        ha="center", fontsize=13.5, fontweight="bold",
    )
    ax.text(
        9, 22.85,
        "Two-batch uplift FSDS · thresholds: mmd_high≈0.02 · ess_low≈0.15 · δ_ovlp≈0.01 · PO α=0.05 · code: diagnose_shift + business_rules",
        ha="center", fontsize=7.5, color="#475569",
    )

    # --- Section 1: Localization outputs ---
    ax.text(9, 22.35, "1. Localization outputs (named subsets — do not conflate)", ha="center", fontsize=10, fontweight="bold")
    locs = [
        ("LOCO group g", "drop_ref, drop_live, drop_gap\nRanking dependence on feature block"),
        ("Domain quintile Q_k", "AUUC_k on LIVE · ŝ(x)=P(W=1|X)\nUser slice (LIVE-like vs REF-like)"),
        ("Pairwise AUUC", "Cap worse slice · continue better\nDirect L1 REALLOCATE pair"),
        ("Overlap band", "AUUC on ê(W|X)∈[0.15,0.85]\nComparable batch support core"),
        ("LOGO-MMD group", "Which X block carries MMD\nCovariate driver"),
        ("L2 pair + PO", "ΔATE · mean τ̂_k\nBudget move if PO reject"),
    ]
    for i, (t, b) in enumerate(locs):
        col = i % 3
        row = i // 3
        box(ax, (0.35 + col * 5.9, 20.35 - row * 1.55), 5.6, 1.4, b, fc="#fef3c7", title=t)

    # --- Section 2: Strategy sort (3 phases) ---
    ax.text(9, 17.55, "2. How to sort adjustments AFTER localization (execution phases)", ha="center", fontsize=10, fontweight="bold")
    box(
        ax, (0.35, 14.85), 17.3, 2.45,
        "Phase A — Regime gate (before any cap)\n"
        "  ESS_overlap < ess_low  →  ONLY REFRESH_REF / match REF  (ignore LOCO, quintile caps, L2)\n"
        "  SRM(T) fail  →  fix experiment; no uplift adapt\n\n"
        "Phase B — Which localization table to trust first (subset_insight_regime + diagnose_shift.label)\n"
        "  mix_shift_or_new_population  →  REFRESH_REF then re-localize\n"
        "  covariate_shift_ranking_degradation (MMD↑, ESS OK)  →  LOGO-MMD → quintile AUUC → overlap AUUC\n"
        "  concept_drift_or_tau_change (MMD flat, gap↑)  →  LOCO → PO-risk → quintile + L2\n"
        "  concept_drift_feature_reallocation (LOCO ρ low)  →  LOCO segment + PO before relearn\n\n"
        "Phase C — Emit & sort business_rules by priority P1→P5 (compatible P1 run together; see conflicts panel)",
        fc="#eff6ff", ec="#1d4ed8", fs=6.8, title="Three-phase strategy sort",
    )

    # --- Section 3: MMD × overlap decision matrix ---
    ax.text(9, 14.35, "3. MMD² vs AUUC_overlap — RELEARN or other? (primary path)", ha="center", fontsize=10, fontweight="bold")

    headers = ["MMD²(X)", "ESS", "AUUC_ovlp vs global", "PO-risk", "Primary adapt (no RELEARN first)", "RELEARN when"]
    col_w = [1.35, 1.05, 2.0, 1.05, 4.5, 3.2]
    x0 = 0.35
    y_table = 13.55
    row_h = 0.72
    xs = [x0]
    for w in col_w[:-1]:
        xs.append(xs[-1] + w)

    for j, (h, w) in enumerate(zip(headers, col_w)):
        box(ax, (xs[j], y_table), w, row_h, h, fc="#e2e8f0", fs=6.2, title=None)

    rows = [
        ("high", "OK", "ovlp ≫ global (+δ)", "accept", "Narrow support + quintile cap high-ŝ + REFRESH_REF parallel + LOGO HOLD", "Only if ovlp red after 2 trials AND PO reject + τ̂≠ATE"),
        ("high", "OK", "both weak", "accept", "Quintile cap + recalibrate ê(T|X) + segment LOCO", "If τ̂≈ATE but AUUC still red → then relearn rare; try e(T|X) first"),
        ("high", "low", "any", "any", "REFRESH_REF / stratify ONLY — no LOCO-driven cuts", "After ESS recovered; then reassess MMD/overlap"),
        ("low", "OK", "ovlp ≫ global", "accept", "Narrow support (policy gate) — τ̂ likely OK on core", "Do NOT relearn; widen REF when MMD cools"),
        ("low", "OK", "both weak", "reject", "L1 cap + L2 budget to high-ATE slice + RELEARN ticket (deferred trial)", "After trial: ovlp & capped Q still red + τ̂≠ATE"),
        ("low", "OK", "both weak", "accept", "LOCO segment + recalibrate ê(T|X); L2 MONITOR only", "RELEARN if LOCO ρ collapse persists + ranking fails on ovlp"),
        ("low", "OK", "global red, ovlp green", "accept", "Narrow support + REF extend (classic covariate tail)", "Explicit NO immediate RELEARN (formulation §narrow support)"),
    ]

    y = y_table - row_h
    for row in rows:
        for j, (cell, w) in enumerate(zip(row, col_w)):
            fc = "#ffffff" if j < 4 else ("#ffedd5" if "Narrow" in cell or "REFRESH" in cell else "#fce7f3" if "RELEARN" in cell else "#f0fdf4")
            if j == 5 and "Do NOT" in cell:
                fc = "#dcfce7"
            box(ax, (xs[j], y), w, row_h, cell, fc=fc, fs=5.5)
        y -= row_h

    # --- Section 4: What MMD vs overlap MEASURE ---
    ax.text(9, 7.85, "4. Do not confuse signals", ha="center", fontsize=10, fontweight="bold")
    box(
        ax, (0.35, 5.95), 8.4, 1.75,
        "MMD²(ref, live) on X\n"
        "• Detects covariate / mix shift magnitude\n"
        "• High MMD → trust quintile + LOGO; suspect off-support tails\n"
        "• Does NOT tell you if τ̂ ranks well on overlap core",
        fc="#ffedd5", fs=6.8, title="MMD²",
    )
    box(
        ax, (9.25, 5.95), 8.4, 1.75,
        "AUUC_overlap vs AUUC_global\n"
        "• Same frozen τ̂; different LIVE row sets\n"
        "• ovlp ≫ global → ranking fix = support/policy, not new τ̂\n"
        "• Both bad → ranking broken even on comparable support → escalate trial → RELEARN path",
        fc="#fed7aa", fs=6.8, title="Overlap AUUC",
    )

    # --- Section 5: Economic one-liner ---
    box(
        ax, (0.35, 4.35), 17.3, 1.35,
        "Economic rationale: narrow/cap/REFRESH = OPEX & latency (avoid treat on bad-rank slices; defer GPU relearn). "
        "RELEARN = CAPEX only when overlap-core AUUC fails under PO+τ̂ mismatch. "
        "Federated same contract: O(p) uplink vs O(np) when X cannot centralize (Eff 48×–1410× in demos).",
        fc="#ecfdf5", ec="#047857", fs=6.8,
    )

    ax.text(
        9, 0.45,
        "Generate: plot_uplift_fsds_localization_adapt_decision_en.py · "
        "Companion: uplift_fsds_adapt_economics_methodology_en.png",
        ha="center", fontsize=6.5, color="#64748b",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
