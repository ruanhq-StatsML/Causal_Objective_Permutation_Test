"""Minimal uplift learners (train on REF only for monitoring)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class TLearner:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> TLearner:
        kw = {"random_state": self.random_state}
        self.mu1 = GradientBoostingClassifier(**kw).fit(X[t == 1], y[t == 1])
        self.mu0 = GradientBoostingClassifier(**kw).fit(X[t == 0], y[t == 0])
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        return self.mu1.predict_proba(X)[:, 1] - self.mu0.predict_proba(X)[:, 1]


class XLearner:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> XLearner:
        kw = {"random_state": self.random_state}
        self.mu1 = GradientBoostingClassifier(**kw).fit(X[t == 1], y[t == 1])
        self.mu0 = GradientBoostingClassifier(**kw).fit(X[t == 0], y[t == 0])

        tau_hat = np.zeros(len(y), dtype=float)
        tau_hat[t == 1] = y[t == 1] - self.mu0.predict_proba(X[t == 1])[:, 1]
        tau_hat[t == 0] = self.mu1.predict_proba(X[t == 0])[:, 1] - y[t == 0]

        self.tau1 = GradientBoostingRegressor(**kw).fit(X[t == 1], tau_hat[t == 1])
        self.tau0 = GradientBoostingRegressor(**kw).fit(X[t == 0], tau_hat[t == 0])
        self.e_model = GradientBoostingClassifier(**kw).fit(X, t)
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        e = self.e_model.predict_proba(X)[:, 1]
        return e * self.tau0.predict(X) + (1.0 - e) * self.tau1.predict(X)


class RLearner:
    def __init__(self, random_state: int = 42, clip_t: float = 1e-3):
        self.random_state = random_state
        self.clip_t = clip_t

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> RLearner:
        kw = {"random_state": self.random_state}
        self.m = GradientBoostingRegressor(**kw).fit(X, y)
        self.e = GradientBoostingClassifier(**kw).fit(X, t)
        y_res = y - self.m.predict(X)
        t_res = t - self.e.predict_proba(X)[:, 1]
        w = np.clip(np.abs(t_res), self.clip_t, None) ** 2
        z = y_res / np.clip(t_res, self.clip_t, None)
        self.tau_model = GradientBoostingRegressor(**kw).fit(X, z, sample_weight=w)
        return self

    def predict_tau(self, X: np.ndarray) -> np.ndarray:
        return self.tau_model.predict(X)


LEARNER_REGISTRY = {
    "tlearner": TLearner,
    "xlearner": XLearner,
    "rlearner": RLearner,
}


def make_learner(name: str, **kwargs):
    cls = LEARNER_REGISTRY[name.lower()]
    return cls(**kwargs)
