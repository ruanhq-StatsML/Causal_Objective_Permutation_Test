#!/usr/bin/env python3
"""
Run two-layer uplift subset localization on all benchmark datasets.
Outputs JSON + markdown with direct business rules and L1/L2 insights.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from DRPerm import DRPerm  # noqa: E402
from loco_auuc.benchmark import _feature_groups  # noqa: E402
from loco_auuc.benchmark_datasets import UpliftBatch, all_benchmarks  # noqa: E402
from loco_auuc.data import split_ref_live  # noqa: E402
from loco_auuc.learners import make_learner  # noqa: E402
from loco_auuc.monitor import propensity_overlap_ess  # noqa: E402
from loco_auuc.subset_localization import run_uplift_subset_localization  # noqa: E402

ART = ROOT / "artifacts"
DOCS = ROOT / "docs"


def _batch_e(X_ref: np.ndarray, X_live: np.ndarray) -> np.ndarray:
    X = np.vstack([X_ref, X_live])
    w = np.array([0] * len(X_ref) + [1] * len(X_live), dtype=int)
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(X, w)
    return clf.predict_proba(X_live)[:, 1]


def _po_risk_pvalue(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    X_live: np.ndarray,
    y_live: np.ndarray,
    *,
    n_perm: int = 48,
    seed: int = 2026,
) -> Dict[str, Any]:
    X = np.vstack([X_ref, X_live])
    Y = np.concatenate([y_ref.ravel(), y_live.ravel()])
    W = np.array([0] * X_ref.shape[0] + [1] * X_live.shape[0], dtype=int)
    out = DRPerm(
        X,
        Y,
        W,
        n_perm=n_perm,
        seed=seed,
        model_m="rf_regressor",
        model_e="rf_classifier",
        return_detail=True,
    )
    return {
        "po_risk_statistic": float(out.get("statistic", 0)),
        "po_risk_pvalue": float(out.get("p_value", 1.0)),
        "po_risk_reject": bool(out.get("p_value", 1.0) < 0.05),
    }


def _layer1_summary(report: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    lines.append(
        f"Global ranking: AUUC_ref={report['auuc_ref_global']:.4f}, "
        f"AUUC_live={report['auuc_live_global']:.4f}, gap={report['auuc_gap']:.4f}."
    )
    loco = report.get("loco_high_drop_live") or []
    if loco:
        top = loco[0]
        lines.append(
            f"Top LOCO (L1 feature): '{top.get('feature')}' drop_live={top.get('drop_live', 0):.4f} "
            f"— ranking on LIVE leans on this input."
        )
    pw = report.get("pairwise_auuc_quintiles") or []
    if pw:
        p0 = pw[0]
        lines.append(
            f"Top pairwise AUUC (L1 users): {p0['subset_a']} vs {p0['subset_b']} "
            f"Δ={p0.get('delta', p0.get('delta_a_minus_b', 0)):.4f}; "
            f"cap worse-ranking slice '{p0.get('worse', '?')}'."
        )
    ins = report.get("insights") or {}
    w, b = ins.get("worst_quintile"), ins.get("best_quintile")
    if w and b and np.isfinite(w.get("auuc_live", np.nan)):
        lines.append(
            f"Quintile spread: worst={w['subset']} AUUC={w['auuc_live']:.4f}; "
            f"best={b['subset']} AUUC={b.get('auuc_live', float('nan')):.4f}."
        )
    return lines


def _layer2_summary(report: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if not report.get("po_risk_concept_mode"):
        lines.append(
            "Layer 2 (effect): PO-risk not flagged — use empirical ATE / mean τ̂ as diagnostic only, "
            "not primary REALLOCATE on ATE pairs."
        )
    else:
        lines.append("Layer 2 (effect): PO-risk reject — prioritize pairwise empirical ATE for budget moves.")
    slices = report.get("concept_slice_uplift_live") or []
    top_ate = report.get("insights", {}).get("top_empirical_ate_quintile")
    if top_ate:
        lines.append(
            f"Highest observed uplift slice: {top_ate['subset']} "
            f"empirical_ATE={top_ate.get('empirical_ate', float('nan')):.4f}, "
            f"mean_tau_hat={top_ate.get('mean_tau_hat', float('nan')):.4f}."
        )
    pw_ate = report.get("pairwise_empirical_ate") or []
    if pw_ate:
        p0 = pw_ate[0]
        lines.append(
            f"Top pairwise ATE (L2): {p0['subset_a']} vs {p0['subset_b']} "
            f"ΔATE={p0['delta_a_minus_b']:.4f}; shift spend toward '{p0['higher_uplift_subset']}'."
        )
    pw_tau = report.get("pairwise_mean_tau_by_slice") or []
    if pw_tau and slices:
        p0 = pw_tau[0]
        lines.append(
            f"Top pairwise mean τ̂ (model level): {p0['subset_a']} vs {p0['subset_b']} "
            f"Δ={p0['delta_a_minus_b']:.4f}."
        )
    return lines


def _strategy_read(report: Dict[str, Any], po: Dict[str, Any]) -> str:
    regime = report.get("subset_insight_regime", "monitor")
    label = report.get("diagnosis", {}).get("label", "stable")
    parts = [f"Regime={regime}; diagnose={label}; PO-risk p={po['po_risk_pvalue']:.3f}."]
    if regime == "mix_shift_refresh_ref_first":
        parts.append("Strategy: REFRESH_REF before LOCO/ATE rules.")
    elif regime == "x_shift_check_logo_mmd_and_quintiles":
        parts.append("Strategy: L1 first — LOGO-MMD groups + cap low-AUUC quintiles; defer RELEARN.")
    elif regime == "x_stable_gap_large_check_loco_and_po_risk":
        parts.append("Strategy: LOCO–AUUC → L1 pairwise AUUC; if PO reject add L2 ATE REALLOCATE.")
    else:
        parts.append("Strategy: MONITOR; rules below are weak signals only.")
    return " ".join(parts)


def run_one_batch(batch: UpliftBatch, *, n_perm: int = 48, seed: int = 42) -> Dict[str, Any]:
    (X_tr, t_tr, y_tr), (X_ev, t_ev, y_ev), (X_lv, t_lv, y_lv) = split_ref_live(
        batch.X_ref,
        batch.t_ref,
        batch.y_ref,
        batch.X_live,
        batch.t_live,
        batch.y_live,
        eval_frac=0.2,
        seed=seed,
    )
    p = X_tr.shape[1]
    groups = _feature_groups(p)
    X_pool = np.vstack([X_ev, X_tr[: min(500, len(X_tr))]])
    e_live = _batch_e(X_pool, X_lv)
    e_stack = np.concatenate(
        [
            np.full(len(X_pool), 0.5),
            e_live,
        ]
    )
    overlap_ess = propensity_overlap_ess(
        np.array([0] * len(X_pool) + [1] * len(X_lv)), e_stack
    )

    po = _po_risk_pvalue(X_ev, y_ev, X_lv, y_lv, n_perm=n_perm, seed=seed + 17)

    report = run_uplift_subset_localization(
        X_tr,
        t_tr,
        y_tr,
        X_ev,
        t_ev,
        y_ev,
        X_lv,
        t_lv,
        y_lv,
        lambda: make_learner("xlearner", random_state=seed),
        groups,
        feature_names=batch.feature_names[:p],
        e_batch_live=e_live,
        overlap_ess=overlap_ess,
        po_risk_reject=po["po_risk_reject"],
        seed=seed,
    )

    compact = {
        "dataset": batch.name,
        "gates": {
            "overlap_ess_batch": overlap_ess,
            "mmd2_x": report["mmd2_x_global"],
            **po,
        },
        "layer1_auuc_ranking": {
            "auuc_ref": report["auuc_ref_global"],
            "auuc_live": report["auuc_live_global"],
            "auuc_gap": report["auuc_gap"],
            "loco_top3": report["loco_high_drop_live"],
            "pairwise_auuc_top3": (report.get("pairwise_auuc_quintiles") or [])[:3],
            "logo_mmd_top2": (report.get("logo_mmd") or [])[:2],
        },
        "layer2_uplift_effect": {
            "po_risk_mode": report["po_risk_concept_mode"],
            "slices": report.get("concept_slice_uplift_live"),
            "pairwise_empirical_ate_top3": (report.get("pairwise_empirical_ate") or [])[:3],
            "pairwise_mean_tau_top3": (report.get("pairwise_mean_tau_by_slice") or [])[:3],
        },
        "diagnosis": report["diagnosis"],
        "subset_insight_regime": report["subset_insight_regime"],
        "strategy": _strategy_read(report, po),
        "layer1_narrative": _layer1_summary(report),
        "layer2_narrative": _layer2_summary(report),
        "business_rules": report["business_rules"],
    }
    return compact


def write_markdown(results: List[Dict[str, Any]], path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Two-layer uplift localization — benchmark business insights",
        "",
        f"Generated: {ts}. Script: `Python/run_uplift_subset_benchmark.py`.",
        "",
        "Layers: **L1** = AUUC / LOCO / pairwise AUUC (ranking). **L2** = empirical ATE + mean τ̂ (effect).",
        "",
    ]
    for r in results:
        lines.append(f"## {r['dataset']}")
        lines.append("")
        g = r["gates"]
        lines.append(
            f"- **Gates:** MMD²={g['mmd2_x']:.4f}, ESS_overlap={g['overlap_ess_batch']:.3f}, "
            f"PO-risk p={g['po_risk_pvalue']:.4f} ({'reject' if g['po_risk_reject'] else 'accept'})."
        )
        lines.append(f"- **Strategy:** {r['strategy']}")
        lines.append("")
        lines.append("### Layer 1 (ranking)")
        for ln in r["layer1_narrative"]:
            lines.append(f"- {ln}")
        lines.append("")
        lines.append("### Layer 2 (uplift / effect)")
        for ln in r["layer2_narrative"]:
            lines.append(f"- {ln}")
        lines.append("")
        lines.append("### Business rules (ops)")
        for br in r["business_rules"]:
            lines.append(f"- **[{br['action']}]** (P{br['priority']}): {br['rule']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=48, help="DRPerm permutations for PO-risk")
    ap.add_argument("--quick", action="store_true", help="First dataset only")
    args = ap.parse_args()

    batches = all_benchmarks()
    if args.quick:
        batches = batches[:1]

    results = [run_one_batch(b, n_perm=args.n_perm) for b in batches]

    ART.mkdir(exist_ok=True)
    json_path = ART / "uplift_two_layer_benchmark.json"
    md_path = ART / "uplift_two_layer_business_insights.md"
    with open(json_path, "w") as f:
        json.dump({"generated_utc": datetime.now(timezone.utc).isoformat(), "runs": results}, f, indent=2)

    write_markdown(results, md_path)
    print("Wrote", json_path)
    print("Wrote", md_path)
    for r in results:
        print("\n===", r["dataset"], "===")
        print(r["strategy"])
        for br in r["business_rules"][:4]:
            print(f"  [{br['action']}] {br['rule'][:100]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
