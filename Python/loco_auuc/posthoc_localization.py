"""
Post-hoc subset localization after a global uplift monitor alert.

Global AUUC/MMD answer *whether* ranking or $X$ broke.
Subsets answer *where* on LIVE (which region of $X$-space or overlap support).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .metrics import auuc


def _batch_domain_scores(X_ref: np.ndarray, X_live: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """P(W=1|X) on LIVE rows (W=1), trained on stacked REF+ LIVE."""
    X = np.vstack([X_ref, X_live])
    w = np.array([0] * len(X_ref) + [1] * len(X_live), dtype=int)
    clf = RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=1)
    clf.fit(X, w)
    return clf.predict_proba(X_live)[:, 1]


def auuc_live_by_domain_quintile(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    t_live: np.ndarray,
    y_live: np.ndarray,
    tau_live: np.ndarray,
    *,
    n_bins: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Split LIVE by domain score quintiles (how ``live-like'' each row looks vs REF).
    High quintile = rows most separable as LIVE; often where AUUC drops first under covariate shift.
    """
    scores = _batch_domain_scores(X_ref, X_live, seed=seed)
    bins = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-9
    rows: List[Dict[str, Any]] = []
    for q in range(n_bins):
        lo, hi = bins[q], bins[q + 1]
        mask = (scores >= lo) & (scores < hi)
        n = int(mask.sum())
        if n < 30 or len(np.unique(t_live[mask])) < 2:
            rows.append(
                {
                    "subset": f"domain_quintile_{q + 1}",
                    "n_live": n,
                    "auuc_live": float("nan"),
                    "mean_domain_score": float(scores[mask].mean()) if n else float("nan"),
                }
            )
            continue
        rows.append(
            {
                "subset": f"domain_quintile_{q + 1}",
                "n_live": n,
                "auuc_live": float(auuc(y_live[mask], t_live[mask], tau_live[mask])),
                "mean_domain_score": float(scores[mask].mean()),
            }
        )
    return rows


def auuc_live_on_overlap_support(
    t_live: np.ndarray,
    y_live: np.ndarray,
    tau_live: np.ndarray,
    e_batch_live: np.ndarray,
    *,
    eps: float = 0.05,
    lo: float = 0.15,
    hi: float = 0.85,
) -> Dict[str, Any]:
    """AUUC on LIVE rows with batch propensity in [lo, hi] (post-hoc comparable support)."""
    e = np.clip(e_batch_live, eps, 1.0 - eps)
    mask = (e >= lo) & (e <= hi)
    n = int(mask.sum())
    out = {"subset": "overlap_support_band", "n_live": n, "lo": lo, "hi": hi}
    if n < 30 or len(np.unique(t_live[mask])) < 2:
        out["auuc_live"] = float("nan")
        return out
    out["auuc_live"] = float(auuc(y_live[mask], t_live[mask], tau_live[mask]))
    return out


def mmd_subset_by_groups(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    groups: Dict[str, List[int]],
    *,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Which feature-group subsets drive covariate shift (post-hoc on $X$)."""
    from .monitor import _rbf_mmd2

    rows = []
    for gname, cols in groups.items():
        mmd2 = _rbf_mmd2(X_ref[:, cols], X_live[:, cols])
        rows.append({"group": gname, "mmd2": mmd2, "n_features": len(cols)})
    rows.sort(key=lambda r: -r["mmd2"])
    return rows


def posthoc_from_monitor(
    monitor_report: Dict[str, Any],
    X_ref_tr: np.ndarray,
    t_ref_tr: np.ndarray,
    y_ref_tr: np.ndarray,
    X_ref_eval: np.ndarray,
    X_live: np.ndarray,
    t_live: np.ndarray,
    y_live: np.ndarray,
    learner_factory: Callable[[], Any],
    groups: Optional[Dict[str, List[int]]] = None,
    *,
    e_batch_stacked: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    model = learner_factory()
    model.fit(X_ref_tr, t_ref_tr, y_ref_tr)
    tau_live = model.predict_tau(X_live)

    global_auuc = monitor_report.get("loco_auuc", {}).get("auuc_live_full")
    quintiles = auuc_live_by_domain_quintile(
        np.vstack([X_ref_eval, X_ref_tr[: min(500, len(X_ref_tr))]]),
        X_live,
        t_live,
        y_live,
        tau_live,
        seed=seed,
    )

    overlap_row = None
    if e_batch_stacked is not None and len(e_batch_stacked) == len(X_live):
        overlap_row = auuc_live_on_overlap_support(
            t_live, y_live, tau_live, e_batch_stacked
        )

    group_mmd = mmd_subset_by_groups(X_ref_eval, X_live, groups) if groups else []

    finite = [r for r in quintiles if np.isfinite(r.get("auuc_live", np.nan))]
    worst_q = min(finite, key=lambda r: r["auuc_live"], default=None)
    best_q = max(finite, key=lambda r: r["auuc_live"], default=None)

    return {
        "auuc_live_global": global_auuc,
        "domain_quintile_auuc": quintiles,
        "overlap_support_auuc": overlap_row,
        "group_mmd2": group_mmd,
        "insights": {
            "worst_quintile": worst_q,
            "best_quintile": best_q,
            "top_mmd_group": group_mmd[0]["group"] if group_mmd else None,
        },
    }
