"""
LOCO for uplift models: drop feature (or group), **retrain on REF**, re-score, **recompute AUUC**
on REF holdout and LIVE window.

Importance = AUUC drop on a chosen eval window (default: LIVE for deployment risk).
Stability = compare LOCO drops REF vs LIVE (concept vs covariate patterns).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .metrics import auuc, evaluate_window


@dataclass
class LocoRow:
    feature: str
    auuc_ref: float
    auuc_live: float
    drop_ref: float
    drop_live: float
    drop_gap: float  # drop_live - drop_ref

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _column_indices(n_features: int, drop: Sequence[int]) -> np.ndarray:
    drop_set = set(int(i) for i in drop)
    return np.array([j for j in range(n_features) if j not in drop_set], dtype=int)


def _fit_predict_tau(
    learner_factory: Callable[[], Any],
    X_tr: np.ndarray,
    t_tr: np.ndarray,
    y_tr: np.ndarray,
    X_eval: np.ndarray,
    col_idx: Optional[np.ndarray],
) -> np.ndarray:
    if col_idx is not None:
        X_tr = X_tr[:, col_idx]
        X_eval = X_eval[:, col_idx]
    model = learner_factory()
    model.fit(X_tr, t_tr, y_tr)
    return model.predict_tau(X_eval)


def loco_auuc_monitor(
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
    *,
    groups: Optional[Dict[str, List[int]]] = None,
) -> dict:
    """
    Full-feature baseline AUUC on REF eval + LIVE, then LOCO each feature (or group):
    retrain on REF train, predict on both eval windows, recompute AUUC each time.
    """
    n_features = X_ref_tr.shape[1]
    if feature_names is None:
        feature_names = [f"x{j}" for j in range(n_features)]

    tau_ref = _fit_predict_tau(learner_factory, X_ref_tr, t_ref_tr, y_ref_tr, X_ref_eval, None)
    tau_live = _fit_predict_tau(learner_factory, X_ref_tr, t_ref_tr, y_ref_tr, X_live, None)

    auuc_ref_full = auuc(y_ref_eval, t_ref_eval, tau_ref)
    auuc_live_full = auuc(y_live, t_live, tau_live)

    rows: List[LocoRow] = []

    def _run_one(label: str, drop_idx: Sequence[int]) -> LocoRow:
        keep = _column_indices(n_features, drop_idx)
        tau_r = _fit_predict_tau(
            learner_factory, X_ref_tr, t_ref_tr, y_ref_tr, X_ref_eval, keep
        )
        tau_l = _fit_predict_tau(
            learner_factory, X_ref_tr, t_ref_tr, y_ref_tr, X_live, keep
        )
        ar = auuc(y_ref_eval, t_ref_eval, tau_r)
        al = auuc(y_live, t_live, tau_l)
        dr = auuc_ref_full - ar
        dl = auuc_live_full - al
        return LocoRow(label, ar, al, dr, dl, dl - dr)

    if groups:
        for gname, gidx in groups.items():
            rows.append(_run_one(gname, gidx))
    else:
        for j in range(n_features):
            rows.append(_run_one(feature_names[j], [j]))

    return {
        "auuc_ref_full": auuc_ref_full,
        "auuc_live_full": auuc_live_full,
        "auuc_gap": auuc_ref_full - auuc_live_full,
        "metrics_ref": evaluate_window(y_ref_eval, t_ref_eval, tau_ref),
        "metrics_live": evaluate_window(y_live, t_live, tau_live),
        "loco_rows": [r.to_dict() for r in rows],
    }
