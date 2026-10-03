"""T-/X-learners backed by ``default_model_registry`` outcome & propensity components."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

_PY = Path(__file__).resolve().parents[1]
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from model_registry_class import default_model_registry


def _proba(predict_fn, fit_obj, X: np.ndarray) -> np.ndarray:
    out = np.asarray(predict_fn(fit_obj, X), dtype=float).ravel()
    if out.ndim == 0:
        return out.reshape(1)
    if np.all((out >= 0) & (out <= 1)) and out.size == X.shape[0]:
        return out
    # multiclass proba matrix
    if out.ndim == 2 and out.shape[1] >= 2:
        return out[:, 1]
    return out


class RegistryTLearner:
    """Two outcome models (T=1 / T=0) from registry keys ``model_mu1``, ``model_mu0``."""

    def __init__(
        self,
        *,
        model_mu: str = "rf_classifier",
        registry: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.model_mu = model_mu
        self.registry = registry or default_model_registry(ntree=40, nthread=1)
        self.seed = seed
        self._mu1 = None
        self._mu0 = None

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> RegistryTLearner:
        reg = self.registry[self.model_mu]
        t = np.asarray(t, dtype=int).ravel()
        y = np.asarray(y, dtype=float).ravel()
        self._mu1 = reg["fit"](X[t == 1], y[t == 1], seed=self.seed)
        self._mu0 = reg["fit"](X[t == 0], y[t == 0], seed=self.seed + 1)
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        reg = self.registry[self.model_mu]
        p1 = _proba(reg["predict"], self._mu1, X)
        p0 = _proba(reg["predict"], self._mu0, X)
        return p1 - p0


class RegistryXLearner:
    """X-learner with registry outcome ``model_mu`` and propensity ``model_e``."""

    def __init__(
        self,
        *,
        model_mu: str = "rf_classifier",
        model_tau: str = "rf_regressor",
        model_e: str = "rf_classifier",
        registry: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.model_mu = model_mu
        self.model_tau = model_tau
        self.model_e = model_e
        self.registry = registry or default_model_registry(ntree=40, nthread=1)
        self.seed = seed

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> RegistryXLearner:
        t = np.asarray(t, dtype=int).ravel()
        y = np.asarray(y, dtype=float).ravel()
        mu_reg = self.registry[self.model_mu]
        tau_reg = self.registry[self.model_tau]
        e_reg = self.registry[self.model_e]

        mu1 = mu_reg["fit"](X[t == 1], y[t == 1], seed=self.seed)
        mu0 = mu_reg["fit"](X[t == 0], y[t == 0], seed=self.seed + 1)

        tau_hat = np.zeros(len(y), dtype=float)
        tau_hat[t == 1] = y[t == 1] - _proba(mu_reg["predict"], mu0, X[t == 1])
        tau_hat[t == 0] = _proba(mu_reg["predict"], mu1, X[t == 0]) - y[t == 0]

        self._tau1 = tau_reg["fit"](X[t == 1], tau_hat[t == 1], seed=self.seed + 2)
        self._tau0 = tau_reg["fit"](X[t == 0], tau_hat[t == 0], seed=self.seed + 3)
        self._e = e_reg["fit"](X, t, seed=self.seed + 4)
        self._mu_reg = mu_reg
        self._tau_reg = tau_reg
        self._e_reg = e_reg
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        e = _proba(self._e_reg["predict"], self._e, X)
        t0 = self._tau_reg["predict"](self._tau0, X)
        t1 = self._tau_reg["predict"](self._tau1, X)
        return e * t0 + (1.0 - e) * t1


def make_registry_learner(
    kind: str = "tlearner",
    *,
    model_mu: str = "rf_classifier",
    model_tau: str = "rf_regressor",
    model_e: str = "rf_classifier",
    seed: int = 42,
):
    if kind == "tlearner":
        return RegistryTLearner(model_mu=model_mu, seed=seed)
    if kind == "xlearner":
        return RegistryXLearner(
            model_mu=model_mu, model_tau=model_tau, model_e=model_e, seed=seed
        )
    raise ValueError(kind)
