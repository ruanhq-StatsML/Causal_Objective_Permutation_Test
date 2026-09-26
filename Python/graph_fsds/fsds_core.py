"""
FSDS core: causal-objective permutation tests + variable importance.

This is a clean, self-contained restatement of the repository's meta-learner
distribution-shift machinery (``DRPerm`` / ``RRPerm`` / ``rrisk_LOCO``), packaged
so the graph layer can call it without the notebook-era import glue.

The framing is unchanged from the parent project:

    existing batch  -> control group   (W = 0)
    new batch       -> treatment group (W = 1)

We regard the *batch label* ``W`` as a treatment, fit a meta-learner that tries
to explain the outcome ``Y`` from the covariates ``X`` and the batch, and read
off a causal objective function (PO-risk or R-risk).  A permute-then-refit
procedure turns that objective into a valid permutation test; a LOCO pass turns
it into per-feature variable importance ("OOD-VIMP").

Two complementary signals are exposed:

    * ``po`` / ``r`` risk  -> sensitive to **concept drift** P(Y|X)
    * domain classifier    -> sensitive to **covariate shift** P(X)

Only numpy / scipy / scikit-learn are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RiskName = Literal["po", "r"]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _as_1d(y) -> np.ndarray:
    return np.asarray(y, dtype=float).ravel()


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def make_folds(n: int, n_folds: int = 5, seed: int = 2026):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, n_folds)


# --------------------------------------------------------------------------- #
# nuisance models (a deliberately small, robust registry)
# --------------------------------------------------------------------------- #
def _rf_regressor(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=120, min_samples_leaf=5, max_features="sqrt",
        random_state=seed, n_jobs=-1,
    )


def _rf_domain(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=120, min_samples_leaf=5, max_features="sqrt",
        random_state=seed, n_jobs=-1,
    )


def _logistic_domain(seed: int) -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()),
         ("clf", LogisticRegression(max_iter=500, random_state=seed))]
    )


def _domain_model(kind: str, seed: int):
    return _rf_domain(seed) if kind == "rf" else _logistic_domain(seed)


# --------------------------------------------------------------------------- #
# cross-fitted nuisances: outcome mu(x) and propensity/domain e(x) = P(W=1|x)
# --------------------------------------------------------------------------- #
def cross_nuisance_fit(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray, *,
    n_folds: int = 5, domain_model: str = "rf", clip_e: float = 1e-2, seed: int = 2026,
):
    X, Y, W = _as_2d(X), _as_1d(Y), _as_1d(W).astype(int)
    n = X.shape[0]
    mu_hat = np.zeros(n)
    e_hat = np.zeros(n)
    for k, test_idx in enumerate(make_folds(n, n_folds, seed)):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        mu = _rf_regressor(seed + k).fit(X[train_idx], Y[train_idx])
        mu_hat[test_idx] = mu.predict(X[test_idx])
        e = _domain_model(domain_model, seed + 100 + k).fit(X[train_idx], W[train_idx])
        e_hat[test_idx] = e.predict_proba(X[test_idx])[:, 1]
    return mu_hat, np.clip(e_hat, clip_e, 1 - clip_e)


def _fit_tau_rlearner_weighted(X, Y_tilde, W_tilde, seed=0, clip=1e-3):
    """R-learner: regress (Y_tilde / W_tilde) on X with weights W_tilde^2."""
    X = _as_2d(X)
    Y_tilde, W_tilde = _as_1d(Y_tilde), _as_1d(W_tilde)
    mask = np.abs(W_tilde) > clip
    if mask.sum() < 5:
        mask = np.ones_like(W_tilde, dtype=bool)
    z = Y_tilde[mask] / np.where(np.abs(W_tilde[mask]) < clip, clip, W_tilde[mask])
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=seed))])
    model.fit(X[mask], z, ridge__sample_weight=W_tilde[mask] ** 2)
    return model


# --------------------------------------------------------------------------- #
# risk objectives
# --------------------------------------------------------------------------- #
def _po_risk(X, Y, W, *, n_folds, domain_model, clip_e, seed) -> float:
    """Doubly-robust pseudo-outcome risk (DRPerm)."""
    mu_hat, e_hat = cross_nuisance_fit(
        X, Y, W, n_folds=n_folds, domain_model=domain_model, clip_e=clip_e, seed=seed)
    pseudo = (_as_1d(Y) - mu_hat) * (_as_1d(W) - e_hat)
    tau = _rf_regressor(seed + 7).fit(_as_2d(X), pseudo)
    return float(np.mean(tau.predict(_as_2d(X)) ** 2))


def _r_risk(X, Y, W, *, n_folds, domain_model, clip_e, seed) -> float:
    """R-learner residual risk (RRPerm)."""
    mu_hat, e_hat = cross_nuisance_fit(
        X, Y, W, n_folds=n_folds, domain_model=domain_model, clip_e=clip_e, seed=seed)
    Y_tilde, W_tilde = _as_1d(Y) - mu_hat, _as_1d(W) - e_hat
    tau_model = _fit_tau_rlearner_weighted(X, Y_tilde, W_tilde, seed=seed)
    tau = tau_model.predict(_as_2d(X))
    return float(np.mean((Y_tilde - tau * W_tilde) ** 2))


_RISK_FN = {"po": _po_risk, "r": _r_risk}
# For R-risk the null (no concept drift) drives the risk *up* toward Var(Y_tilde);
# for PO-risk the effect signal drives the statistic up.  We therefore compare in
# the direction that makes "observed more extreme than permuted" mean "shift".
_RISK_TAIL = {"po": "upper", "r": "lower"}


# --------------------------------------------------------------------------- #
# public results container
# --------------------------------------------------------------------------- #
@dataclass
class FSDSResult:
    scale: str = "node"
    risk: str = "po"
    statistic: float = float("nan")
    p_value: float = float("nan")
    reject: bool = False
    alpha: float = 0.05
    drift_type: str = "unknown"          # covariate / concept / both / none
    covariate_p: float = float("nan")
    concept_p: float = float("nan")
    vimp: Optional[np.ndarray] = None    # per-feature (embedding-dim) importance
    vimp_rank: Optional[np.ndarray] = None
    top_features: Sequence[int] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0
    extra: Dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return bool(self.reject)

    def summary(self) -> str:
        top = ", ".join(str(i) for i in self.top_features[:5])
        return (
            f"[{self.scale:<9}] drift={self.drift_type:<9} "
            f"p={self.p_value:.3f} stat={self.statistic:.4f} "
            f"n={self.n_samples} top_dims=[{top}]"
        )


# --------------------------------------------------------------------------- #
# permutation test (permute-then-refit)
# --------------------------------------------------------------------------- #
def fsds_test(
    X, Y, W, *,
    risk: RiskName = "po",
    n_perm: int = 100, n_folds: int = 5, alpha: float = 0.05,
    domain_model: str = "rf", clip_e: float = 1e-2, seed: int = 2026,
) -> Dict:
    """Permute-then-refit test for distribution shift via a causal objective."""
    X, Y, W = _as_2d(X), _as_1d(Y), _as_1d(W).astype(int)
    risk_fn = _RISK_FN[risk]
    observed = risk_fn(X, Y, W, n_folds=n_folds, domain_model=domain_model, clip_e=clip_e, seed=seed)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for b in range(n_perm):
        W_perm = rng.permutation(W)
        perm[b] = risk_fn(X, Y, W_perm, n_folds=n_folds, domain_model=domain_model,
                          clip_e=clip_e, seed=seed + 1 + b)
    if _RISK_TAIL[risk] == "upper":
        p = (1.0 + np.sum(perm >= observed)) / (1.0 + n_perm)
    else:
        p = (1.0 + np.sum(perm <= observed)) / (1.0 + n_perm)
    return {
        "statistic": float(observed),
        "p_value": float(p),
        "reject": bool(p < alpha),
        "alpha": float(alpha),
        "perm_mean": float(np.mean(perm)),
        "perm_std": float(np.std(perm)),
    }


# --------------------------------------------------------------------------- #
# covariate-shift test: cross-fitted domain-classifier AUC with permutation null
# --------------------------------------------------------------------------- #
def covariate_shift_test(
    X, W, *, n_perm: int = 100, n_folds: int = 5, alpha: float = 0.05,
    domain_model: str = "rf", seed: int = 2026,
) -> Dict:
    """Is P(X) different across batches?  AUC of a W~X classifier vs a permuted null."""
    X, W = _as_2d(X), _as_1d(W).astype(int)
    n = X.shape[0]

    def _auc(Wc, s):
        oof = np.zeros(n)
        for k, test_idx in enumerate(make_folds(n, n_folds, s)):
            tr = np.setdiff1d(np.arange(n), test_idx)
            if len(np.unique(Wc[tr])) < 2:
                oof[test_idx] = 0.5
                continue
            clf = _domain_model(domain_model, s + k).fit(X[tr], Wc[tr])
            oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        try:
            return roc_auc_score(Wc, oof)
        except ValueError:
            return 0.5

    observed = _auc(W, seed)
    rng = np.random.default_rng(seed)
    perm = np.array([_auc(rng.permutation(W), seed + 1 + b) for b in range(n_perm)])
    p = (1.0 + np.sum(perm >= observed)) / (1.0 + n_perm)
    return {"auc": float(observed), "p_value": float(p), "reject": bool(p < alpha),
            "perm_mean": float(np.mean(perm))}


def classify_drift_type(
    X, Y, W, *,
    n_perm: int = 100, n_folds: int = 5, alpha: float = 0.05,
    risk: RiskName = "po", domain_model: str = "rf", seed: int = 2026,
):
    """Return ('covariate'|'concept'|'both'|'none', covariate_p, concept_p)."""
    cov = covariate_shift_test(X, W, n_perm=n_perm, n_folds=n_folds, alpha=alpha,
                               domain_model=domain_model, seed=seed)
    con = fsds_test(X, Y, W, risk=risk, n_perm=n_perm, n_folds=n_folds, alpha=alpha,
                    domain_model=domain_model, seed=seed)
    cov_sig, con_sig = cov["reject"], con["reject"]
    if cov_sig and con_sig:
        dtype = "both"
    elif cov_sig:
        dtype = "covariate"
    elif con_sig:
        dtype = "concept"
    else:
        dtype = "none"
    return dtype, cov["p_value"], con["p_value"]


# --------------------------------------------------------------------------- #
# variable importance
# --------------------------------------------------------------------------- #
def loco_vimp(
    X, Y, W, *,
    risk: RiskName = "po", n_folds: int = 5, domain_model: str = "rf",
    clip_e: float = 1e-2, seed: int = 2026,
) -> Dict:
    """Leave-One-Covariate-Out importance of the causal objective (concept-side)."""
    X, Y, W = _as_2d(X), _as_1d(Y), _as_1d(W).astype(int)
    p = X.shape[1]
    risk_fn = _RISK_FN[risk]
    full = risk_fn(X, Y, W, n_folds=n_folds, domain_model=domain_model, clip_e=clip_e, seed=seed)
    imp = np.zeros(p)
    for j in range(p):
        Xj = np.delete(X, j, axis=1)
        if Xj.shape[1] == 0:
            Xj = np.zeros((X.shape[0], 1))
        rj = risk_fn(Xj, Y, W, n_folds=n_folds, domain_model=domain_model,
                     clip_e=clip_e, seed=seed + 11 + j)
        # importance = how much the objective moves toward "shift" when j is removed
        imp[j] = (full - rj) if _RISK_TAIL[risk] == "upper" else (rj - full)
    rank = np.argsort(-imp)
    return {"vimp": imp, "vimp_rank": rank, "full_risk": float(full)}


def domain_vimp(
    X, W, *, domain_model: str = "rf", n_repeats: int = 10, n_folds: int = 5, seed: int = 2026,
) -> Dict:
    """Covariate-side importance: permutation importance of a W~X domain classifier.

    This is the "RF-domain VIMP" prior: which embedding dimensions separate the
    existing batch from the new one, i.e. which dims carry the P(X) shift.
    """
    from sklearn.inspection import permutation_importance

    X, W = _as_2d(X), _as_1d(W).astype(int)
    n, p = X.shape
    imp = np.zeros(p)
    counts = np.zeros(p)
    for k, test_idx in enumerate(make_folds(n, n_folds, seed)):
        tr = np.setdiff1d(np.arange(n), test_idx)
        if len(np.unique(W[tr])) < 2 or len(test_idx) < 3:
            continue
        clf = _domain_model(domain_model, seed + k).fit(X[tr], W[tr])
        r = permutation_importance(clf, X[test_idx], W[test_idx],
                                   n_repeats=n_repeats, random_state=seed + k, scoring="roc_auc")
        imp += r.importances_mean
        counts += 1
    if counts.max() > 0:
        imp = imp / max(1, counts.max())
    rank = np.argsort(-imp)
    return {"vimp": imp, "vimp_rank": rank}


# --------------------------------------------------------------------------- #
# node-level attribution (turns a global test into per-observation scores)
# --------------------------------------------------------------------------- #
def node_drift_scores(
    X, W, *, domain_model: str = "rf", n_folds: int = 5, clip_e: float = 1e-3, seed: int = 2026,
) -> np.ndarray:
    """Per-observation drift score in [0, 1].

    Cross-fitted P(belongs to new batch | embedding).  A node that the domain
    model can confidently place in its own batch is a node whose local
    neighbourhood in embedding space has moved -- exactly the nodes that drive
    the aggregate shift.  Scores are folded so 1 == strongly batch-identifiable.
    """
    X, W = _as_2d(X), _as_1d(W).astype(int)
    n = X.shape[0]
    e_hat = np.full(n, 0.5)
    for k, test_idx in enumerate(make_folds(n, n_folds, seed)):
        tr = np.setdiff1d(np.arange(n), test_idx)
        if len(np.unique(W[tr])) < 2:
            continue
        clf = _domain_model(domain_model, seed + k).fit(X[tr], W[tr])
        e_hat[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    # identifiability = |P(new|x) - P(new)| scaled to [0, 1]
    base = float(np.mean(W))
    score = np.abs(e_hat - base) / max(base, 1 - base, 1e-6)
    return np.clip(score, 0.0, 1.0)
