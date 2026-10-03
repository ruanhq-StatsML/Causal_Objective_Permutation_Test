"""Drift-type diagnosis + FDR + OFS action closed loop (monitoring)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class ClosedLoopStepResult:
    drift_types: Dict[str, str]
    p_values: Dict[str, float]
    significant: List[str]
    actions: Dict[str, str]
    fdr_estimate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_types": self.drift_types,
            "p_values": self.p_values,
            "significant": self.significant,
            "actions": self.actions,
            "fdr_estimate": self.fdr_estimate,
        }


def distance_to_pvalue(d: float, scale: float = 0.15) -> float:
    """Map non-negative score d to a one-sided p-value proxy (calibrated for demo)."""
    d = max(float(d), 0.0)
    z = d / max(scale, 1e-9)
    return float(1.0 - norm.cdf(z))


def benjamini_hochberg(p_values: Dict[str, float], alpha: float = 0.05) -> List[str]:
    """Offline BH-FDR; returns rejected feature names."""
    if not p_values:
        return []
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    rejected: List[str] = []
    for i, (f, p) in enumerate(items, start=1):
        if p <= (i / m) * alpha:
            rejected.append(f)
    # standard BH: take largest k such that p_(k) <= k/m alpha
    significant: List[str] = []
    for i in range(m, 0, -1):
        f, p = items[i - 1]
        if p <= (i / m) * alpha:
            significant = [items[j - 1][0] for j in range(1, i + 1)]
            break
    return significant


def diagnose_feature_drift_type(
    d: float,
    delta_p: float,
    *,
    d_high: float = 0.02,
    dp_high: float = 0.01,
) -> str:
    """Feature-level typing from PO-risk proxy d and distribution-shift proxy delta_p."""
    d_ok = d >= d_high
    dp_ok = delta_p >= dp_high
    if d_ok and dp_ok:
        return "compound"
    if d_ok and not dp_ok:
        return "concept"
    if not d_ok and dp_ok:
        return "covariate"
    return "stable"


def ofs_action(
    drift_type: str,
    fdr_significant: bool,
) -> str:
    """OFS policy (federated feature monitoring)."""
    if not fdr_significant:
        return "keep"
    if drift_type == "covariate":
        return "recalibrate"
    if drift_type == "concept":
        return "candidate_retire"
    if drift_type == "compound":
        return "alert_recalibrate_and_review"
    return "keep"


class DriftOFSFDRClosedLoop:
    """One-window closed loop: types → p-values → BH-FDR → OFS actions."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def step(
        self,
        po_risk_distances: Dict[str, float],
        distribution_shifts: Dict[str, float],
        *,
        d_high: float = 0.02,
        dp_high: float = 0.01,
    ) -> ClosedLoopStepResult:
        drift_types = {
            f: diagnose_feature_drift_type(
                po_risk_distances.get(f, 0.0),
                distribution_shifts.get(f, 0.0),
                d_high=d_high,
                dp_high=dp_high,
            )
            for f in po_risk_distances
        }
        p_values = {f: distance_to_pvalue(d) for f, d in po_risk_distances.items()}
        significant = benjamini_hochberg(p_values, alpha=self.alpha)
        actions = {
            f: ofs_action(drift_types[f], f in significant) for f in po_risk_distances
        }
        sig_set = set(significant)
        fdr_est = (
            len([f for f in significant if drift_types.get(f) != "stable"]) / max(len(sig_set), 1)
            if sig_set
            else 0.0
        )
        return ClosedLoopStepResult(
            drift_types=drift_types,
            p_values=p_values,
            significant=significant,
            actions=actions,
            fdr_estimate=fdr_est,
        )
