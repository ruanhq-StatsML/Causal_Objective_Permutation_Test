"""
Uplift model monitoring under batch shift (FSDS-style windows).

Pairs:
  - Covariate shift diagnostics on X (MMD proxy, propensity overlap)
  - Ranking stability via AUUC ref vs live + LOCO recomputation after each leave-out
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.stats import chisquare, spearmanr

from .loco import loco_auuc_monitor
from .metrics import bootstrap_auuc


def _rbf_mmd2(X0: np.ndarray, X1: np.ndarray, sigma: float = 1.0) -> float:
    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    if X0.shape[0] < 2 or X1.shape[0] < 2:
        return 0.0

    def k(a, b):
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-np.sum(diff * diff, axis=2) / (2.0 * sigma * sigma))

    n0, n1 = X0.shape[0], X1.shape[0]
    k_xx = k(X0, X0)
    k_yy = k(X1, X1)
    k_xy = k(X0, X1)
    mmd2 = k_xx.sum() / (n0 * n0) + k_yy.sum() / (n1 * n1) - 2.0 * k_xy.sum() / (n0 * n1)
    return float(max(mmd2, 0.0))


def propensity_overlap_ess(t: np.ndarray, e_hat: np.ndarray, *, eps: float = 0.05) -> float:
    """ESS fraction for overlap on stacked batch (t = live indicator)."""
    e_hat = np.clip(e_hat, eps, 1.0 - eps)
    w = 1.0 / (e_hat * (1.0 - e_hat))
    ess = (w.sum() ** 2) / np.sum(w**2)
    return float(ess / len(w))


def check_srm(t: np.ndarray, expected_ratio: float = 0.5, alpha: float = 0.01) -> dict:
    n = len(t)
    observed = [int(np.sum(t == 1)), int(np.sum(t == 0))]
    expected = [n * expected_ratio, n * (1.0 - expected_ratio)]
    stat, p = chisquare(observed, expected)
    return {
        "chi2": float(stat),
        "p_value": float(p),
        "srm_flag": bool(p < alpha),
        "observed": observed,
        "expected": expected,
    }


@dataclass
class ShiftDiagnosis:
    mmd2_x: float
    auuc_gap: float
    overlap_ess: float
    loco_spearman: float
    label: str
    hints: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def diagnose_shift(
    mmd2_x: float,
    auuc_gap: float,
    overlap_ess: float,
    loco_spearman: float,
    *,
    mmd_high: float = 0.02,
    gap_high: float = 0.005,
    ess_low: float = 0.15,
) -> ShiftDiagnosis:
    hints: List[str] = []
    if mmd2_x >= mmd_high and abs(auuc_gap) >= gap_high and overlap_ess >= ess_low:
        label = "covariate_shift_ranking_degradation"
        hints.append("X shifted but overlap OK: AUUC gap likely from propensity / support mismatch on LIVE.")
        hints.append("Prefer Group LOCO on X blocks; retrain e(X) or calibrate; avoid full tau relearn first.")
    elif mmd2_x >= mmd_high and overlap_ess < ess_low:
        label = "mix_shift_or_new_population"
        hints.append("Low overlap: split strata or refresh reference before trusting LOCO drops.")
    elif mmd2_x < mmd_high and abs(auuc_gap) >= gap_high:
        label = "concept_drift_or_tau_change"
        hints.append("X stable but AUUC gap: P(Y|T,X) or true CATE may have changed.")
        hints.append("Compare LOCO profiles; if spearman low, retrain tau / refresh labels.")
    elif loco_spearman < 0.5 and mmd2_x < mmd_high:
        label = "concept_drift_feature_reallocation"
        hints.append("Which features drive ranking changed with modest X shift.")
    else:
        label = "stable"
        hints.append("AUUC and LOCO profile consistent across windows.")

    return ShiftDiagnosis(mmd2_x, auuc_gap, overlap_ess, loco_spearman, label, hints)


def run_uplift_monitor(
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
    feature_names: Optional[List[str]] = None,
    groups: Optional[Dict[str, List[int]]] = None,
    *,
    fit_propensity_for_overlap: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> dict:
    """
    End-to-end: SRM on live, MMD on X, LOCO-AUUC with retrain-after-each-drop, diagnosis.
    """
    srm_ref = check_srm(t_ref_tr)
    srm_live = check_srm(t_live)

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

    mmd2_x = _rbf_mmd2(
        np.vstack([X_ref_eval, X_ref_tr[: min(200, len(X_ref_tr))]]),
        X_live,
    )

    overlap_ess = float("nan")
    if fit_propensity_for_overlap is not None:
        X_stack = np.vstack([X_ref_eval, X_live])
        w = np.array([0] * len(X_ref_eval) + [1] * len(X_live), dtype=int)
        e_hat = fit_propensity_for_overlap(X_stack, w)
        overlap_ess = propensity_overlap_ess(w, e_hat)

    drops_ref = [r["drop_ref"] for r in loco_out["loco_rows"]]
    drops_live = [r["drop_live"] for r in loco_out["loco_rows"]]
    if len(drops_ref) >= 2:
        rho, _ = spearmanr(drops_ref, drops_live)
        loco_spearman = float(rho) if np.isfinite(rho) else 0.0
    else:
        loco_spearman = 1.0

    diag = diagnose_shift(
        mmd2_x,
        loco_out["auuc_gap"],
        overlap_ess if np.isfinite(overlap_ess) else 0.2,
        loco_spearman,
    )

    tau_live_full = learner_factory()
    tau_live_full.fit(X_ref_tr, t_ref_tr, y_ref_tr)
    uplift_live = tau_live_full.predict_tau(X_live)
    ci = bootstrap_auuc(y_live, t_live, uplift_live, n_boot=200)

    return {
        "srm_ref": srm_ref,
        "srm_live": srm_live,
        "mmd2_x_ref_vs_live": mmd2_x,
        "overlap_ess_batch": overlap_ess,
        "loco_auuc": loco_out,
        "diagnosis": diag.to_dict(),
        "auuc_live_bootstrap": ci,
    }
