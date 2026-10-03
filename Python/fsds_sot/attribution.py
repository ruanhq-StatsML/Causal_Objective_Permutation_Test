from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X


def rbf_mmd2(X0: np.ndarray, X1: np.ndarray, sigma: float = 1.0) -> float:
    """Unbiased-style MMD^2 with RBF kernel (sigma fixed as in FSDS paper)."""
    X0, X1 = _as_2d(X0), _as_2d(X1)
    if X0.shape[0] < 2 or X1.shape[0] < 2:
        return 0.0

    def k(a, b):
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-np.sum(diff * diff, axis=2) / (2.0 * sigma * sigma))

    k_xx = k(X0, X0)
    k_yy = k(X1, X1)
    k_xy = k(X0, X1)
    n0, n1 = X0.shape[0], X1.shape[0]
    mmd2 = k_xx.sum() / (n0 * n0) + k_yy.sum() / (n1 * n1) - 2.0 * k_xy.sum() / (n0 * n1)
    return float(max(mmd2, 0.0))


def mmd_loco_delta(X0: np.ndarray, X1: np.ndarray, j: int, sigma: float = 1.0) -> float:
    """Positive delta => feature j drives covariate shift (drop in MMD when removed)."""
    full = rbf_mmd2(X0, X1, sigma=sigma)
    X0_m = np.delete(X0, j, axis=1)
    X1_m = np.delete(X1, j, axis=1)
    reduced = rbf_mmd2(X0_m, X1_m, sigma=sigma)
    return full - reduced


def _keep_columns(p: int, drop: slice) -> np.ndarray:
    return np.array([j for j in range(p) if not (drop.start <= j < drop.stop)], dtype=int)


def mmd_logo_group_delta(
    X0: np.ndarray,
    X1: np.ndarray,
    group: slice,
    *,
    sigma: float = 1.0,
) -> float:
    """Leave-one-group-out MMD: full MMD² − MMD² with group columns removed."""
    X0, X1 = _as_2d(X0), _as_2d(X1)
    full = rbf_mmd2(X0, X1, sigma=sigma)
    keep = _keep_columns(X0.shape[1], group)
    if keep.size == 0:
        return 0.0
    reduced = rbf_mmd2(X0[:, keep], X1[:, keep], sigma=sigma)
    return full - reduced


def domain_batch_auc(
    X_old: np.ndarray,
    X_new: np.ndarray,
    col_idx: np.ndarray,
    *,
    n_estimators: int = 100,
    seed: int = 2026,
) -> float:
    """RF domain classifier AUC on selected columns (ref=0, live=1)."""
    X_old, X_new = _as_2d(X_old), _as_2d(X_new)
    if col_idx.size == 0:
        return 0.5
    X = np.vstack([X_old[:, col_idx], X_new[:, col_idx]])
    W = np.array([0] * X_old.shape[0] + [1] * X_new.shape[0], dtype=int)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    clf.fit(X, W)
    try:
        return float(roc_auc_score(W, clf.predict_proba(X)[:, 1]))
    except ValueError:
        return 0.5


def domain_logo_group_delta(
    X_old: np.ndarray,
    X_new: np.ndarray,
    group: slice,
    *,
    n_estimators: int = 100,
    seed: int = 2026,
) -> float:
    """Leave-one-group-out domain AUC drop (full AUC − AUC without group)."""
    p = _as_2d(X_old).shape[1]
    all_cols = np.arange(p, dtype=int)
    keep = _keep_columns(p, group)
    auc_full = domain_batch_auc(X_old, X_new, all_cols, n_estimators=n_estimators, seed=seed)
    auc_out = domain_batch_auc(X_old, X_new, keep, n_estimators=n_estimators, seed=seed + 1)
    return auc_full - auc_out


@dataclass
class CovariateAttribution:
    vimp: np.ndarray
    domain_auc: float
    mmd2: float
    mmd_loco: np.ndarray
    overlap_ok: bool
    overlap_ess: float


def covariate_attribution(
    X_old: np.ndarray,
    X_new: np.ndarray,
    *,
    n_estimators: int = 100,
    seed: int = 2026,
    overlap_eps: float = 0.05,
    min_ess_frac: float = 0.15,
) -> CovariateAttribution:
    X_old, X_new = _as_2d(X_old), _as_2d(X_new)
    X = np.vstack([X_old, X_new])
    W = np.array([0] * X_old.shape[0] + [1] * X_new.shape[0], dtype=int)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    clf.fit(X, W)
    try:
        auc = float(roc_auc_score(W, clf.predict_proba(X)[:, 1]))
    except ValueError:
        auc = 0.5

    perm = permutation_importance(
        clf, X, W, n_repeats=8, random_state=seed, n_jobs=-1
    )
    vimp = perm.importances_mean.copy()
    vimp = np.maximum(vimp, 0.0)

    e_hat = np.clip(clf.predict_proba(X)[:, 1], overlap_eps, 1.0 - overlap_eps)
    w_ipw = 1.0 / (e_hat * (1.0 - e_hat))
    ess = (w_ipw.sum() ** 2) / np.sum(w_ipw ** 2)
    ess_frac = ess / len(w_ipw)
    overlap_ok = ess_frac >= min_ess_frac and overlap_eps <= e_hat.mean() <= 1.0 - overlap_eps

    mmd2 = rbf_mmd2(X_old, X_new)
    loco = np.array([mmd_loco_delta(X_old, X_new, j) for j in range(X.shape[1])])

    return CovariateAttribution(
        vimp=vimp,
        domain_auc=auc,
        mmd2=mmd2,
        mmd_loco=loco,
        overlap_ok=overlap_ok,
        overlap_ess=float(ess_frac),
    )
