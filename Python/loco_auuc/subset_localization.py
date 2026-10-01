"""
Uplift subset localization — same logic as FSDS on predictive models under shift,
with MSE replaced by AUUC on the ranking layer.

Predictive FSDS pattern          Uplift analogue
-----------------------          ---------------
Global MMD on $X$                Global MMD + domain AUC (unchanged)
Global loss / calibration        Global AUUC$_{live}$, gap
LOGO-MMD (drop group, MMD fall)   LOGO-MMD on feature groups
Per-group MMD p-value            Per-group MMD$^2$ on $(X_g)$
LOGO on outcome model            LOCO--AUUC (drop group, retrain, AUUC fall)
Post-hoc population subset       Domain-quintile AUUC on LIVE
Output                           **Business rules** (targeting caps, feature holds)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .loco import loco_auuc_monitor
from .metrics import auuc
from .monitor import _rbf_mmd2, diagnose_shift
from .posthoc_localization import (
    auuc_live_by_domain_quintile,
    auuc_live_on_overlap_support,
    mmd_subset_by_groups,
)


def logo_mmd_groups(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    groups: Dict[str, Sequence[int]],
) -> tuple[float, List[Dict[str, Any]]]:
    """
    Leave-one-group-out MMD (mirrors ``feature_store.logo_mmd.logo_mmd``).
    Large ``logo_drop`` => that group carried covariate shift on $X$.
    """
    mmd_full = _rbf_mmd2(X_ref, X_live)
    rows: List[Dict[str, Any]] = []
    p = X_ref.shape[1]
    for gname, gidx in groups.items():
        drop = set(int(i) for i in gidx)
        keep = [j for j in range(p) if j not in drop]
        if not keep:
            continue
        mmd_sub = _rbf_mmd2(X_ref[:, keep], X_live[:, keep])
        rows.append(
            {
                "group": gname,
                "logo_mmd_drop": float(mmd_full - mmd_sub),
                "mmd_without_group": float(mmd_sub),
                "mmd_full": float(mmd_full),
            }
        )
    rows.sort(key=lambda r: -r["logo_mmd_drop"])
    return mmd_full, rows


@dataclass
class BusinessRule:
    priority: int  # 1 = do first
    action: str  # REALLOCATE | HOLD_FEATURE | REFRESH_REF | RELEARN | MONITOR
    rule: str  # plain English for ops / policy engine
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def business_rules_from_localization(
    loc: Dict[str, Any],
    *,
    auuc_floor: float = 0.0,
    mmd_logo_min: float = 0.005,
    loco_drop_min: float = 0.003,
) -> List[BusinessRule]:
    """Turn localization tables into direct business rules."""
    rules: List[BusinessRule] = []
    diag = loc.get("diagnosis", {})
    label = diag.get("label", "stable")
    global_auuc = loc.get("auuc_live_global")
    if global_auuc is not None and global_auuc < auuc_floor:
        rules.append(
            BusinessRule(
                1,
                "REALLOCATE",
                f"Pause or cap uplift targeting on LIVE until AUUC_live ({global_auuc:.4f}) recovers above {auuc_floor:.3f}.",
                {"auuc_live_global": global_auuc, "diagnosis": label},
            )
        )

    for row in loc.get("logo_mmd", [])[:2]:
        if row["logo_mmd_drop"] >= mmd_logo_min:
            rules.append(
                BusinessRule(
                    2,
                    "HOLD_FEATURE",
                    f"Covariate shift localized to feature group '{row['group']}': "
                    f"exclude or down-weight this block in targeting rules until REF is refreshed "
                    f"(LOGO-MMD drop={row['logo_mmd_drop']:.4f}).",
                    row,
                )
            )

    loco_rows = loc.get("loco_rows", [])
    for row in sorted(loco_rows, key=lambda r: -abs(r.get("drop_live", 0)))[:2]:
        if abs(row.get("drop_live", 0)) >= loco_drop_min:
            rules.append(
                BusinessRule(
                    2,
                    "REALLOCATE",
                    f"Uplift ranking on LIVE depends on group '{row['feature']}': "
                    f"do not drop this block from the model without replacement; "
                    f"if trimming segments, keep users where this group's LOCO signal is stable "
                    f"(LOCO AUUC drop_live={row['drop_live']:.4f}).",
                    row,
                )
            )

    worst = loc.get("insights", {}).get("worst_quintile")
    best = loc.get("insights", {}).get("best_quintile")
    if worst and best and np.isfinite(worst.get("auuc_live", np.nan)):
        if worst["auuc_live"] < auuc_floor and best.get("auuc_live", -1) > worst["auuc_live"] + 0.01:
            rules.append(
                BusinessRule(
                    1,
                    "REALLOCATE",
                    f"Cap uplift in domain quintile '{worst['subset']}' "
                    f"(AUUC_live={worst['auuc_live']:.4f}, n={worst['n_live']}); "
                    f"continue in '{best['subset']}' where ranking remains stronger.",
                    {"worst": worst, "best": best},
                )
            )

    overlap = loc.get("overlap_support_auuc")
    if overlap and np.isfinite(overlap.get("auuc_live", np.nan)):
        g = loc.get("auuc_live_global")
        if g is not None and overlap["auuc_live"] > g + 0.01:
            rules.append(
                BusinessRule(
                    3,
                    "REALLOCATE",
                    "Restrict targeting to overlap-support users on LIVE "
                    f"(AUUC_live={overlap['auuc_live']:.4f} on support band vs global {g:.4f}).",
                    overlap,
                )
            )

    if label == "mix_shift_or_new_population":
        rules.append(
            BusinessRule(
                1,
                "REFRESH_REF",
                "Low overlap / new population: refresh or match REF cohort before applying LOCO-driven feature cuts.",
                diag,
            )
        )
    elif label == "concept_drift_or_tau_change":
        rules.append(
            BusinessRule(
                2,
                "RELEARN",
                "X stable but AUUC gap large: extend labels and schedule prod uplift model relearn after REALLOCATE trial.",
                diag,
            )
        )

    if not rules:
        rules.append(
            BusinessRule(
                5,
                "MONITOR",
                "No subset rule triggered; continue scheduled FSDS + LOCO--AUUC monitoring.",
                {"diagnosis": label},
            )
        )
    rules.sort(key=lambda r: r.priority)
    return rules


def run_uplift_subset_localization(
    X_ref_tr: np.ndarray,
    t_ref_tr: np.ndarray,
    y_ref_tr: np.ndarray,
    X_ref_eval: np.ndarray,
    t_ref_eval: np.ndarray,
    y_ref_eval: np.ndarray,
    X_live: np.ndarray,
    t_live: np.ndarray,
    y_live: np.ndarray,
    learner_factory: Callable[[], Any],
    groups: Dict[str, List[int]],
    feature_names: Optional[List[str]] = None,
    *,
    e_batch_live: Optional[np.ndarray] = None,
    mmd2_x: Optional[float] = None,
    overlap_ess: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Full uplift subset localization (FSDS shift playbook + AUUC).
    """
    X_ref_pool = np.vstack([X_ref_eval, X_ref_tr[: min(500, len(X_ref_tr))]])

    loco_out = loco_auuc_monitor(
        X_ref_tr,
        t_ref_tr,
        y_ref_tr,
        X_ref_eval,
        t_ref_eval,
        y_ref_eval,
        X_live,
        t_live,
        y_live,
        learner_factory,
        feature_names=feature_names,
        groups=groups,
    )

    mmd_full, logo_mmd = logo_mmd_groups(X_ref_eval, X_live, groups)
    if mmd2_x is None:
        mmd2_x = _rbf_mmd2(X_ref_pool, X_live)

    model = learner_factory()
    model.fit(X_ref_tr, t_ref_tr, y_ref_tr)
    tau_live = model.predict_tau(X_live)

    quintiles = auuc_live_by_domain_quintile(
        X_ref_pool, X_live, t_live, y_live, tau_live, seed=seed
    )
    overlap_row = None
    if e_batch_live is not None and len(e_batch_live) == len(X_live):
        overlap_row = auuc_live_on_overlap_support(t_live, y_live, tau_live, e_batch_live)

    drops_ref = [r["drop_ref"] for r in loco_out["loco_rows"]]
    drops_live = [r["drop_live"] for r in loco_out["loco_rows"]]
    if len(drops_ref) >= 2:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(drops_ref, drops_live)
        loco_spearman = float(rho) if np.isfinite(rho) else 0.0
    else:
        loco_spearman = 1.0

    diag = diagnose_shift(
        mmd2_x,
        loco_out["auuc_gap"],
        overlap_ess,
        loco_spearman,
    ).to_dict()

    finite = [r for r in quintiles if np.isfinite(r.get("auuc_live", np.nan))]
    worst_q = min(finite, key=lambda r: r["auuc_live"], default=None)
    best_q = max(finite, key=lambda r: r["auuc_live"], default=None)

    loc = {
        "auuc_live_global": loco_out["auuc_live_full"],
        "auuc_ref_global": loco_out["auuc_ref_full"],
        "auuc_gap": loco_out["auuc_gap"],
        "mmd2_x_global": mmd2_x,
        "mmd2_x_logo_full": mmd_full,
        "logo_mmd": logo_mmd,
        "group_mmd2_standalone": mmd_subset_by_groups(X_ref_eval, X_live, groups),
        "loco_rows": loco_out["loco_rows"],
        "domain_quintile_auuc": quintiles,
        "overlap_support_auuc": overlap_row,
        "diagnosis": diag,
        "insights": {
            "worst_quintile": worst_q,
            "best_quintile": best_q,
            "top_logo_mmd_group": logo_mmd[0]["group"] if logo_mmd else None,
            "top_loco_live_group": max(
                loco_out["loco_rows"], key=lambda r: r["drop_live"]
            )["feature"]
            if loco_out["loco_rows"]
            else None,
        },
    }
    loc["business_rules"] = [r.to_dict() for r in business_rules_from_localization(loc)]
    return loc
