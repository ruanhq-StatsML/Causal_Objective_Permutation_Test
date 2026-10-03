"""
Online PFI (permutation feature importance) for FSDS covariate monitoring.

Streaming recipe (matches manuscript rolling-window + warm-start narrative):
  1) Fix or decay a reference window D_old
  2) Each step: live batch D_t vs D_old → RF domain classifier + sklearn PFI
  3) Optional: onlineRFPerm-style label-permutation p-value on the same batch

Batch analogue: ``attribution.covariate_attribution`` (single window).
Panel 4 in ``viz.plot_sot_dashboard`` shows one snapshot; this module produces
the *time series* behind that panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .attribution import covariate_attribution, rbf_mmd2
from .closed_loop import decay_reference


def _rf_perm_pvalue(X: np.ndarray, w: np.ndarray, *, n_perm: int = 40, seed: int = 2026) -> float:
    """Classifier two-sample permutation p-value (onlineRFPerm recipe)."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    clf = RandomForestClassifier(n_estimators=60, oob_score=True, random_state=seed, n_jobs=1)
    clf.fit(X, w)
    obs = float(clf.oob_score_ if clf.oob_score_ is not None else (clf.predict(X) == w).mean())
    null = []
    for b in range(n_perm):
        wp = rng.permutation(w)
        clf_b = RandomForestClassifier(n_estimators=60, oob_score=True, random_state=seed + b + 1, n_jobs=1)
        clf_b.fit(X, wp)
        sc = clf_b.oob_score_ if clf_b.oob_score_ is not None else (clf_b.predict(X) == wp).mean()
        null.append(float(sc))
    null = np.asarray(null)
    return float((1.0 + np.sum(null >= obs)) / (1.0 + n_perm))


@dataclass
class OnlinePFIStep:
    step: int
    n_ref: int
    n_live: int
    domain_auc: float
    mmd2: float
    overlap_ess: float
    c2st_pvalue: float
    vimp: np.ndarray
    top_features: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "n_ref": self.n_ref,
            "n_live": self.n_live,
            "domain_auc": self.domain_auc,
            "mmd2": self.mmd2,
            "overlap_ess": self.overlap_ess,
            "c2st_pvalue": self.c2st_pvalue,
            "top_features": self.top_features,
            "vimp": self.vimp.tolist(),
        }


@dataclass
class OnlinePFIReport:
    feature_names: List[str]
    steps: List[OnlinePFIStep] = field(default_factory=list)
    vimp_matrix: Optional[np.ndarray] = None  # (T, p)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "steps": [s.to_dict() for s in self.steps],
        }


def default_agentic_feature_names(p: int) -> List[str]:
    d_emb = max(p - 4, 0)
    names = [f"emb_{i}" for i in range(d_emb)]
    names.extend(["n_branches", "tool_calls", "tool_fail", "latency"][: max(p - d_emb, 0)])
    while len(names) < p:
        names.append(f"f{len(names)}")
    return names[:p]


def run_online_pfi_stream(
    X_reference: np.ndarray,
    live_windows: Sequence[np.ndarray],
    *,
    feature_names: Optional[Sequence[str]] = None,
    decay_alpha: float = 0.85,
    update_reference: bool = True,
    seed: int = 2026,
    n_perm: int = 40,
) -> OnlinePFIReport:
    """
    Compute online PFI for each live window vs (decaying) reference.

    Parameters
    ----------
    X_reference : initial D_old traces (n_ref, p)
    live_windows : list of live batches, e.g. rolling production windows
    """
    X_ref = np.asarray(X_reference, dtype=float)
    p = X_ref.shape[1]
    names = list(feature_names) if feature_names else default_agentic_feature_names(p)
    steps: List[OnlinePFIStep] = []
    vimp_rows: List[np.ndarray] = []

    for t, X_live in enumerate(live_windows):
        X_live = np.asarray(X_live, dtype=float)
        cov = covariate_attribution(X_ref, X_live, seed=seed + t)
        X_stack = np.vstack([X_ref, X_live])
        w = np.array([0] * X_ref.shape[0] + [1] * X_live.shape[0], dtype=int)
        pval = _rf_perm_pvalue(X_stack, w, n_perm=n_perm, seed=seed + 1000 + t)
        top_k = min(8, cov.vimp.size)
        top = np.argsort(-cov.vimp)[:top_k].tolist()
        step = OnlinePFIStep(
            step=t,
            n_ref=X_ref.shape[0],
            n_live=X_live.shape[0],
            domain_auc=cov.domain_auc,
            mmd2=cov.mmd2,
            overlap_ess=cov.overlap_ess,
            c2st_pvalue=pval,
            vimp=cov.vimp,
            top_features=top,
        )
        steps.append(step)
        vimp_rows.append(cov.vimp)
        if update_reference:
            X_ref = decay_reference(X_ref, X_live, alpha=decay_alpha)

    mat = np.vstack(vimp_rows) if vimp_rows else None
    return OnlinePFIReport(feature_names=names, steps=steps, vimp_matrix=mat)
