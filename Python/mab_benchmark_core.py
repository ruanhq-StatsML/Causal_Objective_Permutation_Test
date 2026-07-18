"""Core MAB benchmark: DGP, model pool, policies, experiment runner."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

try:
    from config_MAB import COMPLEXITY_SCORES
except ImportError:
    COMPLEXITY_SCORES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]


# ---------------------------------------------------------------------------
# Permutation p-value (local, feature-wise)
# ---------------------------------------------------------------------------


def empirical_pval_local(
    ref: np.ndarray,
    cur: np.ndarray,
    n_perm: int = 199,
    seed: Optional[int] = None,
) -> float:
    """Two-sample permutation p-value using mean L2 distance between batch means."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 1.0
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)
    rng = np.random.default_rng(seed)
    obs = float(np.linalg.norm(ref.mean(axis=0) - cur.mean(axis=0)))
    pooled = np.vstack([ref, cur])
    n_ref = len(ref)
    stats = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        idx = rng.permutation(len(pooled))
        r = pooled[idx[:n_ref]]
        c = pooled[idx[n_ref:]]
        stats[i] = float(np.linalg.norm(r.mean(axis=0) - c.mean(axis=0)))
    return float((1 + np.sum(stats >= obs)) / (1 + n_perm))


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------


def _fit_predict_ridge(X_tr, y_tr, X_te, alpha: float, seed: int) -> np.ndarray:
    m = Ridge(alpha=alpha, random_state=seed)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def _fit_predict_tree(X_tr, y_tr, X_te, max_depth: int, seed: int) -> np.ndarray:
    m = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def _fit_predict_rf(X_tr, y_tr, X_te, n_estimators: int, max_depth: int, seed: int) -> np.ndarray:
    m = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=max(2, len(X_tr) // 20),
        random_state=seed,
        n_jobs=1,
    )
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def _fit_predict_gbr(X_tr, y_tr, X_te, n_estimators: int, max_depth: int, seed: int) -> np.ndarray:
    m = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def _fit_predict_mlp(X_tr, y_tr, X_te, hidden: int, seed: int) -> np.ndarray:
    m = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        max_iter=400,
        random_state=seed,
        early_stopping=True,
    )
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def build_model_pool(n_arms: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Return arm specs: name, complexity, and a fit_predict callable."""
    specs: List[Dict[str, Any]] = []
    complexities = list(COMPLEXITY_SCORES)
    if len(complexities) < n_arms:
        complexities = complexities + [1.0] * (n_arms - len(complexities))
    complexities = complexities[:n_arms]

    builders: List[Tuple[str, Callable[..., np.ndarray], Dict[str, Any]]] = [
        ("ridge_shallow", _fit_predict_ridge, {"alpha": 1.0}),
        ("ridge_mid", _fit_predict_ridge, {"alpha": 0.1}),
        ("tree_depth3", _fit_predict_tree, {"max_depth": 3}),
        ("tree_depth6", _fit_predict_tree, {"max_depth": 6}),
        ("tree_depth10", _fit_predict_tree, {"max_depth": 10}),
        ("rf_small", _fit_predict_rf, {"n_estimators": 25, "max_depth": 4}),
        ("rf_mid", _fit_predict_rf, {"n_estimators": 75, "max_depth": 6}),
        ("rf_large", _fit_predict_rf, {"n_estimators": 150, "max_depth": 10}),
        ("gbr_mid", _fit_predict_gbr, {"n_estimators": 50, "max_depth": 3}),
        ("gbr_deep", _fit_predict_gbr, {"n_estimators": 100, "max_depth": 5}),
        ("mlp_small", _fit_predict_mlp, {"hidden": 16}),
        ("mlp_large", _fit_predict_mlp, {"hidden": 64}),
    ]
    while len(builders) < n_arms:
        builders.append(builders[-1])

    for arm_id in range(n_arms):
        name, fn, kwargs = builders[arm_id]

        def _make_predictor(f=fn, kw=kwargs, arm=arm_id):
            def predict(X_tr, y_tr, X_te, seed=seed, **extra):
                del extra
                return f(X_tr, y_tr, X_te, seed=int(seed) + arm * 17, **kw)

            return predict

        specs.append(
            {
                "arm_id": arm_id,
                "name": f"{name}_{arm_id}",
                "complexity": float(complexities[arm_id]),
                "predict": _make_predictor(),
            }
        )
    return specs


# ---------------------------------------------------------------------------
# DGP generators
# ---------------------------------------------------------------------------


def _make_batches(total_samples: int, ref_samples: int, batch_size: int) -> Tuple[List[Tuple[int, int]], int]:
    stream_start = int(ref_samples)
    n_stream = max(0, int(total_samples) - stream_start)
    n_batches = max(1, int(np.ceil(n_stream / max(1, batch_size))))
    batches: List[Tuple[int, int]] = []
    for b in range(n_batches):
        lo = stream_start + b * batch_size
        hi = min(stream_start + (b + 1) * batch_size, total_samples)
        if lo < hi:
            batches.append((lo, hi))
    return batches, n_batches


def _concept_linear(X: np.ndarray, w: np.ndarray, bias: float = 0.0) -> np.ndarray:
    return X @ w + bias


def _concept_nonlinear_messy(X: np.ndarray, w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = X.shape[1]
    w2 = rng.normal(0, 0.5, size=d)
    inter = (X[:, : min(5, d)] ** 2).sum(axis=1) * 0.15
    sin_term = np.sin(X[:, 0] * 1.7) * 0.35
    thresh = (X[:, min(1, d - 1)] > 0).astype(float) * 0.25
    return X @ w + (X**2) @ w2 * 0.08 + inter + sin_term + thresh


def _concept_complex_periodic(X: np.ndarray, w: np.ndarray, t_index: np.ndarray, period: float) -> np.ndarray:
    phase = 2 * np.pi * t_index / max(period, 1.0)
    seasonal = 0.45 * np.sin(phase) + 0.25 * np.cos(2 * phase)
    return X @ w + seasonal


def _concept_hyper_nonlinear(
    X: np.ndarray,
    regimes: List[np.ndarray],
    regime_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    w = regimes[regime_id % len(regimes)]
    cubic = 0.05 * (X[:, : min(4, X.shape[1])] ** 3).sum(axis=1)
    noise_inter = rng.normal(0, 0.02, size=len(X))
    return X @ w + cubic + noise_inter


def _apply_covariate_drift(X: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return X
    shift = rng.normal(0, strength, size=X.shape[1])
    scale = 1.0 + rng.normal(0, strength * 0.25, size=X.shape[1])
    return (X + shift) * scale


def _oracle_neg_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_true - y_pred
    return float(-np.mean(err**2))


def _pack_dgp(
    X: np.ndarray,
    Y: np.ndarray,
    ref_samples: int,
    batch_size: int,
    shift_batch_index: int,
    model_pool: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    batches, n_batches = _make_batches(len(X), ref_samples, batch_size)
    ref_X = X[:ref_samples]
    ref_Y = Y[:ref_samples]
    oracle = np.zeros((len(batches), len(model_pool)), dtype=float)
    for b_idx, (lo, hi) in enumerate(batches):
        Xb = X[lo:hi]
        yb = Y[lo:hi]
        if len(Xb) < 4:
            oracle[b_idx, :] = 0.0
            continue
        split = max(2, len(Xb) // 2)
        X_tr, X_te = Xb[:split], Xb[split:]
        y_tr, y_te = yb[:split], yb[split:]
        for a, arm in enumerate(model_pool):
            try:
                pred = arm["predict"](X_tr, y_tr, X_te, seed=int(rng.integers(0, 10_000)))
                oracle[b_idx, a] = _oracle_neg_mse(y_te, pred)
            except Exception:
                oracle[b_idx, a] = -1e6
    return {
        "X": X,
        "Y": Y,
        "ref_samples": int(ref_samples),
        "batch_size": int(batch_size),
        "batches": batches,
        "n_batches": len(batches),
        "ref_X": ref_X,
        "ref_Y": ref_Y,
        "shift_batch_index": int(shift_batch_index),
        "model_pool": model_pool,
        "oracle_rewards": oracle,
        "rng": rng,
    }


def generate_dgp(
    *,
    random_seed: int = 0,
    total_samples: int = 20000,
    ref_samples: int = 5000,
    batch_size: int = 25,
    feature_dim: int = 100,
    noise_scale: float = 2.5,
    shift_magnitude: float = 0.5,
    shift_point_index: int = 10000,
    n_arms: int = 12,
) -> Dict[str, Any]:
    """Linear covariate shift DGP with mild concept drift at shift point."""
    rng = np.random.default_rng(random_seed)
    X = rng.normal(0, 1, size=(total_samples, feature_dim))
    w = rng.normal(0, 1, size=feature_dim)
    w = w / (np.linalg.norm(w) + 1e-8)
    Y = np.empty(total_samples, dtype=float)
    shift_idx = int(min(max(0, shift_point_index), total_samples))
    for i in range(total_samples):
        wi = w.copy()
        if i >= shift_idx:
            wi[: min(10, feature_dim)] += shift_magnitude
        Y[i] = _concept_linear(X[i : i + 1], wi)[0] + rng.normal(0, noise_scale)
    batches, _ = _make_batches(total_samples, ref_samples, batch_size)
    shift_batch = 0
    for b_idx, (lo, _) in enumerate(batches):
        if lo >= shift_idx:
            shift_batch = b_idx
            break
    pool = build_model_pool(n_arms, seed=random_seed)
    return _pack_dgp(X, Y, ref_samples, batch_size, shift_batch, pool, rng)


def generate_complex_dgp(
    *,
    random_seed: int = 0,
    total_samples: int = 20000,
    ref_samples: int = 5000,
    batch_size: int = 25,
    feature_dim: int = 100,
    noise_scale: float = 2.5,
    shift_interval: int = 4000,
    n_arms: int = 12,
) -> Dict[str, Any]:
    """Periodic concept + mild covariate drift."""
    rng = np.random.default_rng(random_seed)
    X = rng.normal(0, 1, size=(total_samples, feature_dim))
    w = rng.normal(0, 1, size=feature_dim)
    period = max(500, int(shift_interval))
    Y = np.empty(total_samples, dtype=float)
    for i in range(total_samples):
        x_i = X[i]
        if (i // period) % 2 == 1:
            x_i = x_i + rng.normal(0, 0.15, size=feature_dim)
        Y[i] = _concept_complex_periodic(x_i.reshape(1, -1), w, np.array([i]), period)[0]
        Y[i] += rng.normal(0, noise_scale)
        X[i] = x_i
    batches, _ = _make_batches(total_samples, ref_samples, batch_size)
    shift_batch = min(len(batches) - 1, max(0, ref_samples // max(1, batch_size)))
    pool = build_model_pool(n_arms, seed=random_seed)
    return _pack_dgp(X, Y, ref_samples, batch_size, shift_batch, pool, rng)


def generate_hyper_nonlinear_shift_dgp(
    *,
    random_seed: int = 0,
    total_samples: int = 20000,
    ref_samples: int = 5000,
    batch_size: int = 25,
    feature_dim: int = 100,
    noise_scale: float = 2.5,
    shift_interval: int = 4000,
    mini_shift_interval: int = 500,
    covariate_drift_strength: float = 0.85,
    shift_magnitude: float = 0.8,
    n_regimes: int = 6,
    n_arms: int = 12,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    X = rng.normal(0, 1, size=(total_samples, feature_dim))
    regimes = [rng.normal(0, 1, size=feature_dim) for _ in range(max(2, n_regimes))]
    regimes = [r / (np.linalg.norm(r) + 1e-8) for r in regimes]
    Y = np.empty(total_samples, dtype=float)
    mini = max(50, int(mini_shift_interval))
    major = max(mini, int(shift_interval))
    for i in range(total_samples):
        regime_id = (i // major) + (i // mini)
        x_i = X[i]
        if (i // mini) % 2 == 1:
            x_i = _apply_covariate_drift(x_i.reshape(1, -1), covariate_drift_strength * 0.15, rng)[0]
        w = regimes[regime_id % len(regimes)]
        if (i // major) % 2 == 1:
            w = w + shift_magnitude * rng.normal(0, 0.2, size=feature_dim)
        Y[i] = _concept_hyper_nonlinear(x_i.reshape(1, -1), regimes, regime_id, rng)[0]
        Y[i] += rng.normal(0, noise_scale)
        X[i] = x_i
    batches, _ = _make_batches(total_samples, ref_samples, batch_size)
    shift_batch = min(len(batches) - 1, max(1, major // max(1, batch_size)))
    pool = build_model_pool(n_arms, seed=random_seed)
    return _pack_dgp(X, Y, ref_samples, batch_size, shift_batch, pool, rng)


def generate_ultra_messy_dgp(
    *,
    random_seed: int = 0,
    total_samples: int = 20000,
    ref_samples: int = 5000,
    batch_size: int = 25,
    feature_dim: int = 100,
    noise_scale: float = 2.5,
    shift_magnitude: float = 0.5,
    shift_point_index: int = 10000,
    shift_interval: int = 2000,
    mini_shift_interval: int = 400,
    covariate_drift_strength: float = 0.45,
    n_arms: int = 12,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    X = rng.normal(0, 1, size=(total_samples, feature_dim))
    w = rng.normal(0, 1, size=feature_dim)
    w2 = rng.normal(0, 0.7, size=feature_dim)
    Y = np.empty(total_samples, dtype=float)
    shift_idx = int(min(max(0, shift_point_index), total_samples))
    major = max(100, int(shift_interval))
    mini = max(50, int(mini_shift_interval))
    for i in range(total_samples):
        x_i = X[i].copy()
        if (i // mini) % 3 != 0:
            x_i = _apply_covariate_drift(x_i.reshape(1, -1), covariate_drift_strength * 0.1, rng)[0]
        if i >= shift_idx and (i // major) % 2 == 1:
            w_eff = w + shift_magnitude * w2
        else:
            w_eff = w
        y_base = _concept_nonlinear_messy(x_i.reshape(1, -1), w_eff, rng)[0]
        spike = 0.0
        if i % 97 == 0:
            spike = rng.normal(0, 3.0)
        Y[i] = y_base + rng.normal(0, noise_scale) + spike
        X[i] = x_i
    batches, _ = _make_batches(total_samples, ref_samples, batch_size)
    shift_batch = 0
    for b_idx, (lo, _) in enumerate(batches):
        if lo >= shift_idx:
            shift_batch = b_idx
            break
    pool = build_model_pool(n_arms, seed=random_seed)
    return _pack_dgp(X, Y, ref_samples, batch_size, shift_batch, pool, rng)


def generate_dgp_from_config(cfg: Dict[str, Any], n_arms: int = 12) -> Dict[str, Any]:
    dgp = str(cfg.get("dgp", "nonlinear_messy"))
    common = dict(
        random_seed=int(cfg.get("random_seed", 0)),
        total_samples=int(cfg.get("total_samples", 20000)),
        ref_samples=int(cfg.get("ref_samples", 5000)),
        batch_size=int(cfg.get("batch_size", 25)),
        feature_dim=int(cfg.get("feature_dim", 100)),
        noise_scale=float(cfg.get("noise_scale", 2.5)),
        n_arms=n_arms,
    )
    if dgp == "linear_shift":
        return generate_dgp(
            shift_magnitude=float(cfg.get("shift_magnitude", 0.5)),
            shift_point_index=int(cfg.get("shift_point_index", 10000)),
            **common,
        )
    if dgp in ("nonlinear_messy", "messy"):
        rng = np.random.default_rng(common["random_seed"])
        X = rng.normal(0, 1, size=(common["total_samples"], common["feature_dim"]))
        w = rng.normal(0, 1, size=common["feature_dim"])
        shift_idx = int(cfg.get("shift_point_index", 10000))
        sm = float(cfg.get("shift_magnitude", 0.5))
        Y = np.empty(common["total_samples"], dtype=float)
        for i in range(common["total_samples"]):
            wi = w.copy()
            if i >= shift_idx:
                wi += sm * rng.normal(0, 0.5, size=common["feature_dim"])
            Y[i] = _concept_nonlinear_messy(X[i : i + 1], wi, rng)[0] + rng.normal(0, common["noise_scale"])
        batches, _ = _make_batches(common["total_samples"], common["ref_samples"], common["batch_size"])
        shift_batch = 0
        for b_idx, (lo, _) in enumerate(batches):
            if lo >= shift_idx:
                shift_batch = b_idx
                break
        pool = build_model_pool(n_arms, seed=common["random_seed"])
        return _pack_dgp(
            X,
            Y,
            common["ref_samples"],
            common["batch_size"],
            shift_batch,
            pool,
            rng,
        )
    if dgp in ("complex_periodic_shift", "complex_periodic"):
        return generate_complex_dgp(
            shift_interval=int(cfg.get("shift_interval", 4000)),
            noise_scale=common["noise_scale"],
            **{k: v for k, v in common.items() if k != "noise_scale"},
        )
    if dgp in ("hyper_nonlinear_shift", "hyper_nonlinear"):
        return generate_hyper_nonlinear_shift_dgp(
            shift_interval=int(cfg.get("shift_interval", 4000)),
            mini_shift_interval=int(cfg.get("mini_shift_interval", 500)),
            covariate_drift_strength=float(cfg.get("covariate_drift_strength", 0.85)),
            shift_magnitude=float(cfg.get("shift_magnitude", 0.8)),
            n_regimes=int(cfg.get("n_regimes", 6)),
            noise_scale=common["noise_scale"],
            **{k: v for k, v in common.items() if k != "noise_scale"},
        )
    if dgp in ("nonlinear_messy_ultra", "ultra"):
        return generate_ultra_messy_dgp(
            shift_magnitude=float(cfg.get("shift_magnitude", 0.5)),
            shift_point_index=int(cfg.get("shift_point_index", 10000)),
            shift_interval=int(cfg.get("shift_interval", 2000)),
            mini_shift_interval=int(cfg.get("mini_shift_interval", 400)),
            covariate_drift_strength=float(cfg.get("covariate_drift_strength", 0.45)),
            noise_scale=common["noise_scale"],
            **{k: v for k, v in common.items() if k != "noise_scale"},
        )
    raise ValueError(f"unknown dgp: {dgp}")


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def context_dim(n_base_features: int, aggregations: Optional[Sequence[str]] = None) -> int:
    aggs = aggregations or ("mean", "std", "quantile")
    return int(n_base_features) * (1 + len(aggs)) + 2


def build_context(
    batch_X: np.ndarray,
    ref_X: np.ndarray,
    prev_batch_X: Optional[np.ndarray],
    cfg: Dict[str, Any],
    batch_idx: int,
    pval: float,
    cov_drift: float,
) -> np.ndarray:
    n_base = int(cfg.get("n_base_features", 10))
    aggs = list(cfg.get("aggregations", ["mean", "std", "quantile"]))
    Xb = np.asarray(batch_X, dtype=float)
    if Xb.ndim == 1:
        Xb = Xb.reshape(-1, 1)
    d = Xb.shape[1]
    take = min(n_base, d)
    base = Xb[:, :take].mean(axis=0)
    if take < n_base:
        base = np.pad(base, (0, n_base - take))
    pieces = [base]
    for agg in aggs:
        if agg == "mean":
            val = Xb.mean(axis=0)[:take]
        elif agg == "std":
            val = Xb.std(axis=0, ddof=0)[:take]
        elif agg == "quantile":
            val = np.quantile(Xb, 0.5, axis=0)[:take]
        else:
            val = Xb.mean(axis=0)[:take]
        if len(val) < n_base:
            val = np.pad(val, (0, n_base - len(val)))
        pieces.append(val)
    burn = int(cfg.get("pval_burnin", 5))
    anomaly = 0.0 if batch_idx < burn else float(1.0 - pval)
    if prev_batch_X is not None and len(prev_batch_X):
        prev = np.asarray(prev_batch_X, dtype=float)
        if prev.ndim == 1:
            prev = prev.reshape(-1, 1)
        cov_drift = float(np.linalg.norm(Xb.mean(axis=0) - prev.mean(axis=0)))
    ctx = np.concatenate([np.concatenate(pieces), np.array([anomaly, cov_drift], dtype=float)])
    return ctx.astype(float)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class Policy(ABC):
    def __init__(self, n_arms: int, seed: int = 0):
        self.n_arms = int(n_arms)
        self.rng = np.random.default_rng(seed)
        self.t = 0

    @abstractmethod
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        self.t = 0


class Random(Policy):
    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        del context
        return int(self.rng.integers(0, self.n_arms))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        del arm, reward, context
        self.t += 1


class EpsilonGreedy(Policy):
    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 0):
        super().__init__(n_arms, seed=seed)
        self.epsilon = float(epsilon)
        self.counts = np.zeros(n_arms, dtype=float)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        del context
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        del context
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.t += 1


class EpsilonMomentum(Policy):
    """ε-greedy with exponential moving-average value estimates (momentum γ)."""

    def __init__(self, n_arms: int, epsilon: float = 0.05, gamma: float = 0.8, seed: int = 0):
        super().__init__(n_arms, seed=seed)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.counts = np.zeros(n_arms, dtype=float)
        self.values = np.zeros(n_arms, dtype=float)
        self.initialized = np.zeros(n_arms, dtype=bool)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        del context
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        del context
        self.counts[arm] += 1
        if not self.initialized[arm]:
            self.values[arm] = float(reward)
            self.initialized[arm] = True
        else:
            self.values[arm] = self.gamma * self.values[arm] + (1.0 - self.gamma) * float(reward)
        self.t += 1


class AdaptiveEpsilonGreedy(Policy):
    def __init__(
        self,
        n_arms: int,
        base_epsilon: float = 0.05,
        max_epsilon: float = 0.5,
        gamma: float = 0.9,
        anomaly_sensitivity: float = 0.5,
        seed: int = 0,
    ):
        super().__init__(n_arms, seed=seed)
        self.base_epsilon = float(base_epsilon)
        self.max_epsilon = float(max_epsilon)
        self.gamma = float(gamma)
        self.anomaly_sensitivity = float(anomaly_sensitivity)
        self.counts = np.zeros(n_arms, dtype=float)
        self.values = np.zeros(n_arms, dtype=float)
        self.anomaly_ema = 0.0

    def _epsilon(self) -> float:
        boost = self.anomaly_sensitivity * self.anomaly_ema
        return float(np.clip(self.base_epsilon + boost, self.base_epsilon, self.max_epsilon))

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        if context is not None and len(context) >= 2:
            self.anomaly_ema = self.gamma * self.anomaly_ema + (1 - self.gamma) * float(context[-2])
        if self.rng.random() < self._epsilon():
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        del context
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.t += 1


class AdaptiveEpsilonGreedyEWMA(Policy):
    """EWMA mean + drift-adaptive epsilon (cov shift + MSE anomaly via drift_delta)."""

    uses_drift_delta = True

    def __init__(
        self,
        n_arms: int,
        base_epsilon: float = 0.025,
        max_epsilon: float = 0.5,
        gamma: float = 0.9,
        anomaly_sensitivity: float = 0.5,
        seed: int = 0,
    ):
        super().__init__(n_arms, seed=seed)
        self.base_epsilon = float(base_epsilon)
        self.max_epsilon = float(max_epsilon)
        self.gamma = float(gamma)
        self.anomaly_sensitivity = float(anomaly_sensitivity)
        self.mu = np.zeros(n_arms, dtype=float)
        self.counts = np.zeros(n_arms, dtype=float)

    def select_arm(
        self,
        context: Optional[np.ndarray] = None,
        drift_delta: float = 0.0,
    ) -> int:
        del context
        epsilon = min(
            self.max_epsilon,
            self.base_epsilon + float(drift_delta) * self.anomaly_sensitivity,
        )
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_arms))
        return int(np.argmax(self.mu))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        del context
        self.mu[arm] = self.gamma * self.mu[arm] + (1.0 - self.gamma) * float(reward)
        self.counts[arm] += 1
        self.t += 1

    def reset(self) -> None:
        super().reset()
        self.mu.fill(0.0)
        self.counts.fill(0.0)


class UCB(Policy):
    def __init__(self, n_arms: int, c: float = 1.0, seed: int = 0):
        super().__init__(n_arms, seed=seed)
        self.c = float(c)
        self.counts = np.zeros(n_arms, dtype=float)
        self.values = np.zeros(n_arms, dtype=float)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        del context
        for a in range(self.n_arms):
            if self.counts[a] <= 0:
                return a
        bonus = self.c * np.sqrt(np.log(max(1, self.t)) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        del context
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.t += 1


class Thompson(Policy):
    def __init__(self, n_arms: int, prior_lambda: float = 0.1, seed: int = 0):
        super().__init__(n_arms, seed=seed)
        self.prior_lambda = float(prior_lambda)
        self.alpha = np.ones(n_arms, dtype=float)
        self.beta = np.ones(n_arms, dtype=float)
        self.mu = np.zeros(n_arms, dtype=float)
        self.prec = np.ones(n_arms, dtype=float) * self.prior_lambda
        self.n = np.zeros(n_arms, dtype=float)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        del context
        samples = self.rng.normal(self.mu, 1.0 / np.sqrt(self.prec + 1e-8))
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        del context
        self.n[arm] += 1
        self.prec[arm] += self.prior_lambda
        self.mu[arm] = (self.prec[arm] * self.mu[arm] + reward) / (self.prec[arm] + 1)
        self.alpha[arm] += max(0.0, reward)
        self.beta[arm] += max(0.0, -reward)
        self.t += 1


class Gaussian(Policy):
    def __init__(self, n_arms: int, learning_rate: float = 0.1, init_sigma: float = 5.0, seed: int = 0):
        super().__init__(n_arms, seed=seed)
        self.lr = float(learning_rate)
        self.sigma = np.ones(n_arms, dtype=float) * float(init_sigma)
        self.mu = np.zeros(n_arms, dtype=float)
        self.counts = np.zeros(n_arms, dtype=float)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        del context
        samples = self.rng.normal(self.mu, self.sigma)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        del context
        self.counts[arm] += 1
        self.mu[arm] += self.lr * (reward - self.mu[arm])
        resid = reward - self.mu[arm]
        self.sigma[arm] = np.sqrt(
            max(1e-6, (1 - self.lr) * self.sigma[arm] ** 2 + self.lr * resid**2)
        )
        self.t += 1


class LinUCB(Policy):
    def __init__(
        self,
        n_arms: int,
        d: int,
        alpha: float = 0.1,
        lambda_reg: float = 1.0,
        use_momentum: bool = False,
        base_gamma: float = 0.9,
        window: int = 30,
        threshold: float = 1.5,
        gamma_decay: float = 0.9,
        alpha_growth: float = 1.02,
        seed: int = 0,
    ):
        super().__init__(n_arms, seed=seed)
        self.d = int(d)
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)
        self.use_momentum = bool(use_momentum)
        self.base_gamma = float(base_gamma)
        self.window = int(window)
        self.threshold = float(threshold)
        self.gamma_decay = float(gamma_decay)
        self.alpha_growth = float(alpha_growth)
        self.A = [np.eye(self.d) * self.lambda_reg for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]
        self.reward_hist: List[float] = []
        self.gamma = self.base_gamma

    def _theta(self, a: int) -> np.ndarray:
        A_inv = np.linalg.inv(self.A[a])
        return A_inv @ self.b[a]

    def _adapt(self) -> None:
        if not self.use_momentum or len(self.reward_hist) < 2:
            return
        recent = self.reward_hist[-self.window :]
        if len(recent) < 2:
            return
        vol = float(np.std(recent))
        if vol > self.threshold:
            self.gamma = max(self.gamma_decay, self.gamma * self.gamma_decay)
            self.alpha *= self.alpha_growth
        else:
            self.gamma = min(self.base_gamma, self.gamma + 0.01)

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        if context is None:
            return int(self.rng.integers(0, self.n_arms))
        x = np.asarray(context, dtype=float).reshape(-1)
        if len(x) < self.d:
            x = np.pad(x, (0, self.d - len(x)))
        x = x[: self.d]
        scores = np.empty(self.n_arms, dtype=float)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            scores[a] = float(theta @ x + self.alpha * np.sqrt(max(0.0, x @ A_inv @ x)))
        return int(np.argmax(scores))

    def update(self, arm: int, reward: float, context: Optional[np.ndarray] = None) -> None:
        if context is None:
            self.t += 1
            return
        x = np.asarray(context, dtype=float).reshape(-1)
        if len(x) < self.d:
            x = np.pad(x, (0, self.d - len(x)))
        x = x[: self.d]
        if self.use_momentum and self.reward_hist:
            reward = self.gamma * reward + (1 - self.gamma) * self.reward_hist[-1]
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.reward_hist.append(float(reward))
        self._adapt()
        self.t += 1


def make_policy(name: str, n_arms: int, d_context: int, cfg: Dict[str, Any]) -> Policy:
    seed = int(cfg.get("seed", 0))
    if name == "Random":
        return Random(n_arms, seed=seed)
    if name == "Epsilon_Greedy":
        return EpsilonGreedy(n_arms, epsilon=float(cfg.get("epsilon", 0.1)), seed=seed)
    if name == "Epsilon_Momentum":
        return EpsilonMomentum(
            n_arms,
            epsilon=float(cfg.get("epsilon", 0.05)),
            gamma=float(cfg.get("gamma", 0.8)),
            seed=seed,
        )
    if name == "AdaptiveEpsilonGreedy":
        return AdaptiveEpsilonGreedy(
            n_arms,
            base_epsilon=float(cfg.get("base_epsilon", 0.05)),
            max_epsilon=float(cfg.get("max_epsilon", 0.5)),
            gamma=float(cfg.get("gamma", 0.9)),
            anomaly_sensitivity=float(cfg.get("anomaly_sensitivity", 0.5)),
            seed=seed,
        )
    if name == "AdaptiveEpsilonGreedyEWMA":
        return AdaptiveEpsilonGreedyEWMA(
            n_arms,
            base_epsilon=float(cfg.get("base_epsilon", 0.05)),
            max_epsilon=float(cfg.get("max_epsilon", 0.5)),
            gamma=float(cfg.get("gamma", 0.9)),
            anomaly_sensitivity=float(cfg.get("anomaly_sensitivity", 0.5)),
            seed=seed,
        )
    if name == "UCB":
        return UCB(n_arms, c=float(cfg.get("c", 1.0)), seed=seed)
    if name == "Thompson_Sampling":
        return Thompson(n_arms, prior_lambda=float(cfg.get("prior_lambda", 0.1)), seed=seed)
    if name == "Gaussian_Sampling":
        return Gaussian(
            n_arms,
            learning_rate=float(cfg.get("learning_rate", 0.1)),
            init_sigma=float(cfg.get("init_sigma", 5.0)),
            seed=seed,
        )
    if name in ("LinUCB_Vanilla", "LinUCB_Momentum"):
        return LinUCB(
            n_arms,
            d=d_context,
            alpha=float(cfg.get("alpha", 0.1)),
            lambda_reg=float(cfg.get("lambda_reg", 1.0)),
            use_momentum=name == "LinUCB_Momentum" or bool(cfg.get("use_momentum", False)),
            base_gamma=float(cfg.get("base_gamma", 0.9)),
            window=int(cfg.get("window", 30)),
            threshold=float(cfg.get("threshold", 1.5)),
            gamma_decay=float(cfg.get("gamma_decay", cfg.get("base_gamma", 0.9))),
            alpha_growth=float(cfg.get("alpha_growth", 1.02)),
            seed=seed,
        )
    raise ValueError(f"unknown policy: {name}")


# ---------------------------------------------------------------------------
# Batch cache + experiment runner
# ---------------------------------------------------------------------------


def precompute_batch_cache(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    ctx_cfg = config.get("context", {})
    ref_X = data["ref_X"]
    batches = data["batches"]
    seed = int(config.get("data", {}).get("random_seed", 0))
    cache = {
        "contexts": [],
        "pvals": [],
        "cov_drifts": [],
        "batch_X": [],
        "batch_Y": [],
    }
    prev_X: Optional[np.ndarray] = None
    for b_idx, (lo, hi) in enumerate(batches):
        Xb = data["X"][lo:hi]
        yb = data["Y"][lo:hi]
        pval = empirical_pval_local(ref_X, Xb, seed=seed + b_idx)
        cov = 0.0
        if prev_X is not None and len(prev_X):
            cov = float(np.linalg.norm(Xb.mean(axis=0) - prev_X.mean(axis=0)))
        ctx = build_context(Xb, ref_X, prev_X, ctx_cfg, b_idx, pval, cov)
        cache["contexts"].append(ctx)
        cache["pvals"].append(pval)
        cache["cov_drifts"].append(cov)
        cache["batch_X"].append(Xb)
        cache["batch_Y"].append(yb)
        prev_X = Xb
    return cache


def _fuse_drift_signal(
    pval: float,
    cov_drift: float,
    anomaly_weight: float,
    batch_idx: int,
    burnin: int,
) -> float:
    if batch_idx < burnin:
        return 0.0
    anomaly = 1.0 - float(pval)
    cov_term = float(np.tanh(cov_drift))
    return float(np.clip(anomaly_weight * anomaly + (1.0 - anomaly_weight) * cov_term, 0.0, 1.0))


def run_experiment(
    policy: Policy,
    data: Dict[str, Any],
    config: Dict[str, Any],
    use_context: bool = False,
    batch_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ctx_cfg = config.get("context", {})
    safety = config.get("safety", {})
    risky = set(int(x) for x in safety.get("risky_arms", []))
    fallback = int(safety.get("fallback_arm", 0))
    drift_guard = float(safety.get("drift_guard_threshold", 0.85))
    reward_shaping = float(safety.get("reward_shaping_coef", 0.01))
    complexity_coef = float(safety.get("complexity_risk_coef", 0.2))

    cache = batch_cache or precompute_batch_cache(data, config)
    model_pool = data["model_pool"]
    oracle = data["oracle_rewards"]
    n_batches = data["n_batches"]

    anomaly_weight = float(ctx_cfg.get("anomaly_weight", 0.6))
    min_anomaly_weight = float(ctx_cfg.get("min_anomaly_weight", 0.15))
    decay_window = int(ctx_cfg.get("decay_window", 20))
    decay_rate = float(ctx_cfg.get("decay_rate", 0.95))
    burnin = int(ctx_cfg.get("pval_burnin", 5))
    reset_anom = float(ctx_cfg.get("reset_threshold_anomaly", 0.4))
    reset_cov = float(ctx_cfg.get("reset_threshold_cov", 0.3))

    policy.reset()
    rewards: List[float] = []
    cum_regret: List[float] = []
    total_regret = 0.0
    calm_streak = 0
    prev_X: Optional[np.ndarray] = None

    for b_idx in range(n_batches):
        Xb = cache["batch_X"][b_idx]
        yb = cache["batch_Y"][b_idx]
        pval = float(cache["pvals"][b_idx])
        cov_drift = float(cache["cov_drifts"][b_idx])
        context = cache["contexts"][b_idx] if use_context else None

        fused = _fuse_drift_signal(pval, cov_drift, anomaly_weight, b_idx, burnin)
        if fused < reset_anom and cov_drift < reset_cov:
            calm_streak += 1
        else:
            calm_streak = 0
        if calm_streak >= decay_window:
            anomaly_weight = max(min_anomaly_weight, anomaly_weight * decay_rate)

        if getattr(policy, "uses_drift_delta", False):
            arm = policy.select_arm(context, drift_delta=fused)
        elif use_context and hasattr(policy, "select_arm"):
            arm = policy.select_arm(context)
        else:
            arm = policy.select_arm(None)

        if fused >= drift_guard and arm in risky:
            arm = fallback

        split = max(2, len(Xb) // 2)
        X_tr, X_te = Xb[:split], Xb[split:]
        y_tr, y_te = yb[:split], yb[split:]
        try:
            pred = model_pool[arm]["predict"](X_tr, y_tr, X_te, seed=1000 + b_idx)
            reward = _oracle_neg_mse(y_te, pred)
        except Exception:
            reward = -1e6

        complexity = float(model_pool[arm].get("complexity", 0.5))
        reward = reward - reward_shaping * fused - complexity_coef * complexity * fused

        policy.update(arm, reward, context if use_context else None)

        best = float(np.max(oracle[b_idx]))
        inst_regret = best - float(oracle[b_idx, arm])
        total_regret += inst_regret
        rewards.append(reward)
        cum_regret.append(total_regret)
        prev_X = Xb

    return {
        "rewards": rewards,
        "cum_regret": cum_regret,
        "final_regret": float(cum_regret[-1] if cum_regret else 0.0),
        "mean_reward": float(np.mean(rewards) if rewards else 0.0),
        "n_batches": int(n_batches),
    }
