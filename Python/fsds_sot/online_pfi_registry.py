"""
Online monitoring via **Model Registry** → cross-fit refit → predict → **value delta**.

This is the canonical repo logic (DRPerm / PO-risk), not the lightweight sklearn
RF-PFI shortcut in ``online_pfi.py``.

Flow per window t:
  1. Stack reference (W=0) and live (W=1) traces; optional outcome Y.
  2. ``model_registry[model_e/m]``: refit nuisances → ``mu_hat``, ``e_hat`` (OOF).
  3. Pseudo-outcome → refit ``tau`` model → ``tau_hat`` on all rows.
  4. **Prediction delta**: mean(tau_hat | live) − mean(tau_hat | ref), same for mu_hat.
  5. Optional: ``DRPerm`` permute-refit p-value on the same window.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Repo root DRPerm + ModelRegistry live under Python/
_PY_ROOT = Path(__file__).resolve().parents[1]
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from DRPerm import DRPerm, MODEL_REGISTRY  # noqa: E402
from model_registry_class import default_model_registry  # noqa: E402


def _as_1d(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=float).ravel()


def cross_fit_mu_ehat(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    *,
    model_registry: dict,
    model_m: str = "rf_regressor",
    model_e: str = "rf_classifier",
    n_folds: int = 5,
    clip_e: float = 0.01,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Model Registry cross-fit (same contract as ``DRPerm`` nuisances)."""
    from DRPerm import make_folds

    X = np.asarray(X, dtype=float)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    mu_hat = np.zeros(n)
    e_hat = np.zeros(n)
    folds = make_folds(n, n_folds=n_folds, seed=seed)
    model_propensity_score = model_registry[model_e]
    model_outcome = model_registry[model_m]
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        fit_mu = model_outcome["fit"](X_train, Y[train_idx], seed=seed + k)
        mu_hat[test_idx] = model_outcome["predict"](fit_mu, X_test)
        fit_e = model_propensity_score["fit"](X_train, W[train_idx], seed=seed + 100 + k)
        e_hat[test_idx] = model_propensity_score["predict"](fit_e, X_test)
    e_hat = np.clip(e_hat, clip_e, 1.0 - clip_e)
    return mu_hat, e_hat


def prediction_deltas_on_window(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    Y_ref: np.ndarray,
    Y_live: np.ndarray,
    *,
    model_registry: Optional[dict] = None,
    model_m: str = "rf_regressor",
    model_e: str = "rf_classifier",
    n_folds: int = 5,
    seed: int = 2026,
) -> Dict[str, float]:
    """
    Refit on stacked batch; return mean prediction **delta** (live − ref) for mu and tau.
    """
    reg = model_registry or MODEL_REGISTRY
    X = np.vstack([X_ref, X_live])
    Y = np.concatenate([_as_1d(Y_ref), _as_1d(Y_live)])
    W = np.array([0] * X_ref.shape[0] + [1] * X_live.shape[0], dtype=int)

    mu_hat, e_hat = cross_fit_mu_ehat(
        X, Y, W, model_registry=reg, model_m=model_m, model_e=model_e, n_folds=n_folds, seed=seed
    )
    residual_y = Y - mu_hat
    residual_t = W - e_hat
    pseudo = residual_y * residual_t

    model_outcome = reg[model_m]
    fit_tau = model_outcome["fit"](X, pseudo, seed=seed + 200)
    tau_hat = model_outcome["predict"](fit_tau, X)

    ref_sl = slice(0, X_ref.shape[0])
    live_sl = slice(X_ref.shape[0], X.shape[0])

    return {
        "delta_mu_mean": float(mu_hat[live_sl].mean() - mu_hat[ref_sl].mean()),
        "delta_tau_mean": float(tau_hat[live_sl].mean() - tau_hat[ref_sl].mean()),
        "delta_pseudo_mean": float(pseudo[live_sl].mean() - pseudo[ref_sl].mean()),
        "po_risk_observed": float(np.mean(tau_hat**2)),
        "mean_mu_ref": float(mu_hat[ref_sl].mean()),
        "mean_mu_live": float(mu_hat[live_sl].mean()),
        "mean_tau_ref": float(tau_hat[ref_sl].mean()),
        "mean_tau_live": float(tau_hat[live_sl].mean()),
    }


@dataclass
class RegistryOnlineStep:
    step: int
    prediction_delta: Dict[str, float]
    drperm: Dict[str, Any]
    top_feature_idx: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "prediction_delta": self.prediction_delta,
            "drperm": self.drperm,
            "top_feature_idx": self.top_feature_idx,
        }


@dataclass
class RegistryOnlineReport:
    steps: List[RegistryOnlineStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps]}


def run_online_registry_stream(
    X_ref: np.ndarray,
    Y_ref: np.ndarray,
    live_windows: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    model_registry: Optional[dict] = None,
    n_perm: int = 40,
    seed: int = 2026,
) -> RegistryOnlineReport:
    """
    For each (X_live, Y_live) window: Model Registry refit → predict → deltas + DRPerm p.
    """
    reg = model_registry or MODEL_REGISTRY
    report = RegistryOnlineReport()
    for t, (X_live, Y_live) in enumerate(live_windows):
        deltas = prediction_deltas_on_window(
            X_ref, X_live, Y_ref, Y_live, model_registry=reg, seed=seed + t
        )
        X = np.vstack([X_ref, X_live])
        Y = np.concatenate([_as_1d(Y_ref), _as_1d(Y_live)])
        W = np.array([0] * X_ref.shape[0] + [1] * X_live.shape[0], dtype=int)
        dr = DRPerm(
            X, Y, W,
            n_perm=n_perm,
            seed=seed + 1000 + t,
            model_registry=reg,
            return_detail=True,
        )
        report.steps.append(
            RegistryOnlineStep(step=t, prediction_delta=deltas, drperm=dr)
        )
    return report
