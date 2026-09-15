"""Core MAB benchmark: DGP, model pool, policies, experiment runner."""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from config_MAB import COMPLEXITY_SCORES


def empirical_pval_local(seq: List[float], burnin: int = 5) -> List[float]:
    pvals = [0.5] * len(seq)
    for i in range(burnin, len(seq)):
        hist = seq[:i]
        cnt = sum(1 for v in hist if v >= seq[i])
        pvals[i] = (cnt + 1) / (len(hist) + 1)
    return pvals


try:
    from benchmark_methods import empirical_pval
except ImportError:
    empirical_pval = empirical_pval_local


def build_model_pool(n_arms: int = 12) -> List[Tuple[str, Any]]:
    pool = [
        ("RF1", RandomForestRegressor(n_estimators=50, max_features=0.25, max_depth=3, min_samples_leaf=50, random_state=2020)),
        ("RF2", RandomForestRegressor(n_estimators=70, max_features=0.75, max_depth=4, min_samples_leaf=40, random_state=2021)),
        ("RF3", RandomForestRegressor(n_estimators=90, max_features=0.25, max_depth=5, min_samples_leaf=30, random_state=2022)),
        ("RF4", RandomForestRegressor(n_estimators=110, max_features=0.75, max_depth=6, min_samples_leaf=20, random_state=2023)),
        ("RF5", RandomForestRegressor(n_estimators=130, max_features=0.25, max_depth=7, min_samples_leaf=10, random_state=2024)),
        ("RF6", RandomForestRegressor(n_estimators=500, max_features=0.75, max_depth=8, min_samples_leaf=5, random_state=2025)),
        ("XGB1", XGBRegressor(n_estimators=50, max_depth=3, colsample_bytree=0.33, subsample=0.2, random_state=2012, verbosity=0)),
        ("XGB2", XGBRegressor(n_estimators=100, max_depth=5, colsample_bytree=0.4, subsample=0.3, random_state=2014, verbosity=0)),
        ("XGB3", XGBRegressor(n_estimators=150, max_depth=7, colsample_bytree=0.5, subsample=0.4, random_state=2016, verbosity=0)),
        ("XGB4", XGBRegressor(n_estimators=200, max_depth=9, colsample_bytree=0.6, subsample=0.5, random_state=2018, verbosity=0)),
        ("XGB5", XGBRegressor(n_estimators=250, max_depth=11, colsample_bytree=0.7, subsample=0.6, random_state=2020, verbosity=0)),
        ("XGB6", XGBRegressor(n_estimators=300, max_depth=13, colsample_bytree=0.8, subsample=0.7, random_state=2022, verbosity=0)),
        ("Ridge_a1", Ridge(alpha=1.0, random_state=2026)),
        ("Ridge_a10", Ridge(alpha=10.0, random_state=2026)),
        ("LM", LinearRegression()),
        ("MLP_32_16", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=2022)),
        ("MLP_64_32", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=2022)),
        ("MLP_128_64", MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=2022)),
        ("KRR", KernelRidge(kernel="rbf", gamma=0.1, alpha=0.15)),
    ]
    return pool[:n_arms]


def generate_dgp(
    total_samples: int,
    feature_dim: int,
    ref_samples: int,
    batch_size: int,
    noise_scale: float,
    shift_magnitude: float,
    random_seed: int,
    n_arms: int,
    shift_point_index: int | None = None,
    dgp: str = "nonlinear_messy",
) -> Dict[str, Any]:
    """Synthetic DGP with optional concept drift at ``shift_point_index``."""
    rng = np.random.default_rng(random_seed)
    ref_samples = min(ref_samples, total_samples - batch_size * 5)
    ref_samples = max(ref_samples, batch_size * 5)
    shift_point = shift_point_index if shift_point_index is not None else total_samples // 2
    shift_point = min(max(shift_point, ref_samples + batch_size), total_samples - batch_size)

    X = rng.normal(0, 1, size=(total_samples, feature_dim))
    beta = np.zeros(feature_dim)
    n_sig = min(15, feature_dim)
    beta[:n_sig] = np.arange(n_sig) / 10.0

    def _xi(col: int) -> np.ndarray:
        return X[:, col % feature_dim]

    if dgp in ("stationary", "nonlinear_stationary"):
        # Fully stationary: same conditional law for all t (no CD / covariate shift).
        Y = (
            X @ beta
            + (_xi(1) ** 2) * 0.015
            + np.sin(_xi(2) ** 2) * 0.05
            + (_xi(7) ** 2) * 0.01
            + (5.0 / (_xi(11) ** 2 + 0.15))
            + rng.normal(0, noise_scale * 2, size=total_samples)
        ) / 25.0
        shift_point = total_samples  # no post-shift regime
    elif dgp == "linear_shift":
        Y = (X @ beta + rng.normal(0, noise_scale * 2, size=total_samples)) / 25.0
    elif dgp == "nonlinear_messy_rich":
        Y = (
            X @ beta
            + (_xi(1) ** 2) * 0.015
            + np.sin(_xi(2) ** 2) * 0.05
            + (_xi(7) ** 2) * 0.25
            + (_xi(27) ** 2) * 0.2
            + (5.0 / (_xi(11) ** 2 + 0.15))
            + (_xi(3) * _xi(5)) * 0.1
            + np.cos(_xi(8) * _xi(9)) * 0.2
            + np.exp(_xi(4) * 0.01) * 0.5
            + rng.normal(0, noise_scale * 2, size=total_samples)
        ) / 5.0
    else:
        Y = (
            X @ beta
            + (_xi(1) ** 2) * 0.015
            + np.sin(_xi(2) ** 2) * 0.05
            + (_xi(7) ** 2) * 0.01
            + (5.0 / (_xi(11) ** 2 + 0.15))
            + rng.normal(0, noise_scale * 2, size=total_samples)
        ) / 25.0

    if dgp not in ("stationary", "nonlinear_stationary"):
        beta_new = np.zeros(feature_dim)
        n_jump = min(15, max(0, feature_dim - n_sig))
        if n_jump > 0:
            beta_new[n_sig : n_sig + n_jump] = np.arange(n_jump) / 50.0
        post_len = total_samples - shift_point
        Y[shift_point:] = (
            Y[shift_point:]
            + (X[shift_point:] @ beta_new) * shift_magnitude
            + rng.normal(0, noise_scale * 0.8, size=post_len)
        )
    scaler = StandardScaler()
    X_ref = X[:ref_samples]
    scaler.fit(X_ref)
    X_scaled = scaler.transform(X)
    Y_ref = Y[:ref_samples]
    fitted_models = []
    for name, model in build_model_pool(n_arms):
        m = copy.deepcopy(model)
        m.fit(X_ref, Y_ref)
        fitted_models.append((name, m))
    n_batches = (total_samples - ref_samples) // batch_size
    return {
        "X_scaled": X_scaled,
        "Y": Y,
        "X_ref": X_ref,
        "fitted_models": fitted_models,
        "ref_samples": ref_samples,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "shift_point": shift_point,
        "shift_batch_index": max(0, (shift_point - ref_samples) // batch_size),
        "dgp": dgp,
    }


def generate_complex_dgp(
    total_samples: int,
    feature_dim: int,
    ref_samples: int,
    batch_size: int,
    noise_scale: float,
    random_seed: int,
    n_arms: int,
    shift_interval: int = 4000,
) -> Dict[str, Any]:
    """Periodic concept drift + covariate shift + heteroscedastic noise."""
    rng = np.random.default_rng(random_seed)
    ref_samples = min(ref_samples, total_samples - batch_size * 5)
    ref_samples = max(ref_samples, batch_size * 5)

    X = rng.normal(0, 1, size=(total_samples, feature_dim))
    for t in range(ref_samples, total_samples):
        phase = (t - ref_samples) % shift_interval / shift_interval
        mean_shift = 0.5 * np.sin(2 * np.pi * phase * 2) + 0.3 * np.cos(2 * np.pi * phase * 3)
        var_scale = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * phase * 1.5))
        for j in range(min(10, feature_dim)):
            X[t, j] = X[t, j] * var_scale + mean_shift * 0.5

    beta_A = np.zeros(feature_dim)
    beta_A[:15] = np.arange(15) / 10.0
    beta_B = np.zeros(feature_dim)
    beta_B[20:35] = np.arange(15) / 8.0
    beta_B[50:55] = np.array([0.5, -0.3, 0.8, -0.6, 0.4])
    beta_C = np.zeros(feature_dim)
    beta_C[10:25] = np.arange(15) / 6.0
    beta_C[60:70] = np.linspace(-0.5, 0.5, 10)
    beta_D = -beta_A.copy()
    beta_D[15:20] = np.array([0.2, -0.5, 0.3, -0.7, 0.1])
    betas = [beta_A, beta_B, beta_C, beta_D]

    Y = np.zeros(total_samples)
    for t in range(total_samples):
        x_t = X[t]
        if t < ref_samples:
            state = 0
        else:
            cycle_idx = (t - ref_samples) // shift_interval
            state = cycle_idx % len(betas)
        beta_t = betas[state]
        linear_part = x_t @ beta_t
        if state == 0:
            nonlinear = (
                (x_t[1] ** 2) * 0.015
                + np.sin(x_t[2] ** 2) * 0.05
                + (5.0 / (x_t[11] ** 2 + 0.15))
            )
        elif state == 1:
            nonlinear = (
                (x_t[3] * x_t[5]) * 0.3
                + np.cos(x_t[7] * x_t[8]) * 0.2
                + np.exp(x_t[4] * 0.01) * 0.5
            )
        elif state == 2:
            nonlinear = (
                (x_t[1] ** 3) * 0.01
                + np.sin(x_t[2] * x_t[3]) * 0.15
                + (x_t[9] ** 2) * 0.1
            )
        else:
            nonlinear = (
                (x_t[0] * x_t[1] * x_t[2]) * 0.05
                + np.cos(x_t[5] ** 2) * 0.1
                + (x_t[6] ** 2) * 0.08
            )
        if t < ref_samples:
            noise_scale_t = noise_scale * 0.5
        else:
            phase = (t - ref_samples) % shift_interval / shift_interval
            noise_mod = 0.5 + 0.5 * np.exp(-((phase - 0.5) ** 2) * 20)
            noise_scale_t = noise_scale * (1.0 + noise_mod)
        Y[t] = (linear_part + nonlinear + rng.normal(0, noise_scale_t)) / 5.0

    scaler = StandardScaler()
    X_ref = X[:ref_samples]
    scaler.fit(X_ref)
    X_scaled = scaler.transform(X)
    Y_ref = Y[:ref_samples]
    fitted_models = []
    for name, model in build_model_pool(n_arms):
        m = copy.deepcopy(model)
        m.fit(X_ref, Y_ref)
        fitted_models.append((name, m))

    n_batches = (total_samples - ref_samples) // batch_size
    shift_batch_indices = []
    for sp in range(ref_samples + shift_interval, total_samples, shift_interval):
        shift_batch_indices.append(max(0, (sp - ref_samples) // batch_size))
    first_shift = ref_samples + shift_interval
    return {
        "X_scaled": X_scaled,
        "Y": Y,
        "X_ref": X_ref,
        "fitted_models": fitted_models,
        "ref_samples": ref_samples,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "shift_point": first_shift,
        "shift_batch_index": shift_batch_indices[0] if shift_batch_indices else 0,
        "shift_batch_indices": shift_batch_indices,
        "dgp": "complex_periodic_shift",
        "shift_interval": shift_interval,
    }


def _hyper_nonlinear_map(
    x: np.ndarray,
    regime: int,
    mini_flip: int,
    shift_magnitude: float,
) -> float:
    """Regime-specific highly nonlinear response; mini_flip toggles interaction signs."""
    sgn = -1.0 if mini_flip % 2 else 1.0
    x = np.clip(x, -4.5, 4.5)
    if regime == 0:
        return sgn * (
            (x[1] ** 2) * 0.022
            + np.sin(x[2] ** 2 + x[3]) * 0.08
            + (6.0 / (x[11] ** 2 + 0.08))
            + np.tanh(x[4] * x[5]) * 0.35
            + (x[7] ** 3) * 0.006
        )
    if regime == 1:
        return sgn * (
            np.exp(np.clip(x[3], -2.5, 2.5) * 0.04) * 0.55
            + np.cos(x[7] * x[8] + x[1]) * 0.28
            + (x[12] ** 3) * 0.012
            + np.sin(x[1] * x[9] * x[2]) * 0.14
            + (x[4] * x[6] - x[5] ** 2) * 0.11
        )
    if regime == 2:
        return sgn * (
            (x[0] * x[1] * x[2]) * 0.07
            + np.sin(x[2] * x[3] + x[4] ** 2) * 0.18
            + (x[9] ** 2) * 0.16
            + np.tanh(x[14] * x[15] + x[6]) * 0.32
            + (3.5 / (x[13] ** 2 + 0.15))
        )
    if regime == 3:
        return sgn * (
            np.sin(x[0] * x[5]) * np.cos(x[8] * x[10]) * 0.25
            + (x[16] ** 2 - x[17] ** 2) * 0.09
            + np.log1p(np.abs(x[18] * x[19]) + 0.05) * 0.4
            + (x[2] * x[11] * x[14]) * 0.05
            + shift_magnitude * np.sin(x[20] + x[21] ** 2) * 0.12
        )
    if regime == 4:
        soft = np.tanh(x[3:8])
        return sgn * (
            float(np.dot(soft, np.array([0.3, -0.2, 0.45, 0.15, -0.35])))
            + (x[22] * x[23] - x[24] ** 3) * 0.08
            + np.sin(x[6] ** 2 + x[25]) * 0.15
            + (x[1] / (1.0 + x[7] ** 2)) * 0.5
        )
    return sgn * (
        (x[0] ** 2 * x[1] - x[2] ** 3) * 0.04
        + np.cos(x[5] ** 2 + x[9]) * 0.14
        + (x[8] * x[10] * x[11]) * 0.06
        + np.sinh(np.clip(x[12], -2, 2)) * 0.12
        + shift_magnitude * (x[26] * x[27] - x[28] * x[29]) * 0.09
    )


def _hyper_covariate_warp(
    x: np.ndarray,
    elapsed: int,
    shift_interval: int,
    mini_shift_interval: int,
    strength: float,
    feature_dim: int,
    rot_cache: Dict[int, Tuple[np.ndarray, np.ndarray, float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Strong covariate shift: rotation, mixture morphing, skew, heavy tails."""
    major_phase = (elapsed % shift_interval) / shift_interval
    mini_phase = (elapsed % mini_shift_interval) / mini_shift_interval
    major_cycle = elapsed // shift_interval
    mini_cycle = elapsed // mini_shift_interval

    if major_cycle not in rot_cache:
        rot_rng = np.random.default_rng(10007 * major_cycle + 17)
        block = min(25, feature_dim)
        theta = rot_rng.uniform(-np.pi, np.pi, size=block // 5 + 1)
        alt_center = rot_rng.normal(0, 1.2 + 0.4 * strength, size=block)
        rot_cache[major_cycle] = (theta, alt_center, block, float(major_cycle))

    theta, alt_center, block, _ = rot_cache[major_cycle]
    out = x.copy()
    for b, th in enumerate(theta):
        i0 = b * 5
        if i0 + 1 >= block:
            continue
        c, s = np.cos(th), np.sin(th)
        pair = out[i0 : i0 + 2].copy()
        out[i0] = c * pair[0] - s * pair[1]
        out[i0 + 1] = s * pair[0] + c * pair[1]

    mean_wave = strength * (
        0.55 * np.sin(2 * np.pi * major_phase * 2.1 + major_cycle * 0.7)
        + 0.35 * np.cos(2 * np.pi * major_phase * 3.3 - mini_cycle * 0.4)
        + 0.25 * np.sin(2 * np.pi * mini_phase * 5.0)
    )
    var_scale = 0.45 + 0.55 * np.abs(np.sin(2 * np.pi * major_phase * 1.4 + 0.5))
    out[:block] = out[:block] * var_scale + mean_wave * (
        0.25 + 0.08 * np.arange(block) % 7
    )

    skew_dims = min(12, feature_dim - 30)
    if skew_dims > 0 and feature_dim > 30:
        idx = np.arange(30, 30 + skew_dims)
        out[idx] = np.sinh(np.clip(out[idx], -2.5, 2.5) * (0.35 + 0.15 * strength))

    if mini_cycle != rot_cache.get("_last_tail_cycle"):
        rot_cache["_last_tail_cycle"] = mini_cycle
        rot_cache["_tail_mask"] = rng.random(size=feature_dim) < (
            0.10 + 0.05 * np.abs(np.sin(mini_phase * np.pi))
        )
    tail_mask = rot_cache.get("_tail_mask")
    if tail_mask is not None and tail_mask.any():
        n_tail = int(tail_mask.sum())
        out[tail_mask] = rng.standard_t(df=3, size=n_tail) * (0.8 + 0.4 * strength)

    mix_w = 0.5 + 0.5 * np.sin(2 * np.pi * major_phase)
    out[:block] = mix_w * out[:block] + (1.0 - mix_w) * alt_center
    return out


def generate_hyper_nonlinear_shift_dgp(
    total_samples: int,
    feature_dim: int,
    ref_samples: int,
    batch_size: int,
    noise_scale: float,
    random_seed: int,
    n_arms: int,
    shift_interval: int = 4000,
    mini_shift_interval: int = 500,
    covariate_drift_strength: float = 0.85,
    shift_magnitude: float = 0.8,
    n_regimes: int = 6,
) -> Dict[str, Any]:
    """Extreme nonstationary DGP: deep nonlinear CD + strong covariate shift + multi-scale drift."""
    rng = np.random.default_rng(random_seed)
    ref_samples = min(ref_samples, total_samples - batch_size * 5)
    ref_samples = max(ref_samples, batch_size * 5)
    shift_interval = max(batch_size * 4, int(shift_interval))
    mini_shift_interval = max(batch_size * 2, int(mini_shift_interval))
    n_regimes = max(2, int(n_regimes))

    X = rng.normal(0, 1, size=(total_samples, feature_dim))

    betas: List[np.ndarray] = []
    for k in range(n_regimes):
        b = np.zeros(feature_dim)
        start = (k * 13) % max(1, feature_dim - 22)
        b[start : start + 18] = np.linspace(0.55, -0.35, 18) * (0.9 + 0.12 * k)
        flip_idx = np.arange(start + 35, start + 42) % feature_dim
        b[flip_idx] = rng.normal(0, 0.22, size=len(flip_idx)) * (1.0 if k % 2 == 0 else -1.0)
        betas.append(b)

    beta_anti = np.zeros(feature_dim)
    beta_anti[40:60] = np.linspace(-0.6, 0.45, 20)

    Y = np.zeros(total_samples)
    shift_batch_indices: List[int] = []
    mini_shift_batches: List[int] = []
    rot_cache: Dict[int, Any] = {}

    for t in range(total_samples):
        if t >= ref_samples:
            elapsed = t - ref_samples
            X[t] = _hyper_covariate_warp(
                X[t],
                elapsed,
                shift_interval,
                mini_shift_interval,
                covariate_drift_strength,
                feature_dim,
                rot_cache,
                rng,
            )
        x_t = X[t]

        if t < ref_samples:
            regime = 0
            mini_flip = 0
        else:
            elapsed = t - ref_samples
            major_cycle = elapsed // shift_interval
            mini_cycle = elapsed // mini_shift_interval
            regime = int(major_cycle % n_regimes)
            mini_flip = mini_cycle
            if elapsed > 0 and elapsed % shift_interval == 0:
                shift_batch_indices.append(max(0, (t - ref_samples) // batch_size))
            if elapsed > 0 and elapsed % mini_shift_interval == 0:
                mini_shift_batches.append(max(0, (t - ref_samples) // batch_size))

        beta_t = betas[regime].copy()
        if t >= ref_samples:
            jitter_rng = np.random.default_rng(random_seed + mini_flip * 7919 + regime * 104729)
            beta_t += jitter_rng.normal(0, 0.06, size=feature_dim)

        linear = float(x_t @ beta_t)
        if t >= ref_samples + shift_interval:
            linear += float(x_t @ beta_anti) * shift_magnitude * (
                0.6 + 0.4 * np.sin(2 * np.pi * ((t - ref_samples) % shift_interval) / shift_interval)
            )

        nonlinear = _hyper_nonlinear_map(x_t, regime, mini_flip, shift_magnitude)

        if t < ref_samples:
            noise_sd = noise_scale * 0.5
        else:
            phase = ((t - ref_samples) % shift_interval) / shift_interval
            mini_p = ((t - ref_samples) % mini_shift_interval) / mini_shift_interval
            noise_sd = noise_scale * (
                0.9
                + 0.6 * np.exp(-((phase - 0.4) ** 2) * 12)
                + 0.25 * np.abs(np.sin(2 * np.pi * mini_p * 3))
                + 0.15 * np.linalg.norm(x_t[:10]) / np.sqrt(10)
            )

        if rng.random() < 0.10:
            eps = rng.standard_t(df=3) * noise_sd * 1.6
        elif rng.random() < 0.03:
            eps = rng.choice([-1.0, 1.0]) * noise_sd * rng.uniform(5.0, 10.0)
        else:
            eps = rng.normal(0, noise_sd)

        Y[t] = (linear + nonlinear + eps) / 4.0

    scaler = StandardScaler()
    X_ref = X[:ref_samples]
    scaler.fit(X_ref)
    X_scaled = scaler.transform(X)
    Y_ref = Y[:ref_samples]
    fitted_models = []
    for name, model in build_model_pool(n_arms):
        m = copy.deepcopy(model)
        m.fit(X_ref, Y_ref)
        fitted_models.append((name, m))

    n_batches = (total_samples - ref_samples) // batch_size
    first_shift = ref_samples + shift_interval
    if not shift_batch_indices and first_shift < total_samples:
        shift_batch_indices.append(max(0, (first_shift - ref_samples) // batch_size))
    shift_batch_indices = sorted(set(shift_batch_indices))

    return {
        "X_scaled": X_scaled,
        "Y": Y,
        "X_ref": X_ref,
        "fitted_models": fitted_models,
        "ref_samples": ref_samples,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "shift_point": first_shift,
        "shift_batch_index": shift_batch_indices[0] if shift_batch_indices else 0,
        "shift_batch_indices": shift_batch_indices,
        "mini_shift_batches": mini_shift_batches,
        "dgp": "hyper_nonlinear_shift",
        "shift_interval": shift_interval,
        "mini_shift_interval": mini_shift_interval,
        "shift_magnitude": shift_magnitude,
        "covariate_drift_strength": covariate_drift_strength,
        "n_regimes": n_regimes,
    }


def _ultra_nonlinear(x: np.ndarray, regime: int) -> float:
    """Regime-specific messy nonlinear terms (dimension-safe via wrap)."""
    d = len(x)

    def xi(i: int) -> float:
        return float(x[i % d])

    if regime == 0:
        return (
            (xi(1) ** 2) * 0.018
            + np.sin(xi(2) ** 2) * 0.06
            + (xi(7) ** 2) * 0.22
            + (5.0 / (xi(11) ** 2 + 0.12))
            + (xi(3) * xi(5)) * 0.08
        )
    if regime == 1:
        return (
            (xi(3) * xi(5)) * 0.28
            + np.cos(xi(7) * xi(8)) * 0.22
            + np.exp(np.clip(xi(4), -3, 3) * 0.02) * 0.45
            + (xi(12) ** 3) * 0.008
            + np.sin(xi(1) * xi(9)) * 0.12
        )
    if regime == 2:
        return (
            (xi(1) ** 3) * 0.012
            + np.sin(xi(2) * xi(3)) * 0.16
            + (xi(9) ** 2) * 0.14
            + (xi(0) * xi(6) - xi(4) ** 2) * 0.09
            + (3.0 / (xi(13) ** 2 + 0.2))
        )
    return (
        (xi(0) * xi(1) * xi(2)) * 0.055
        + np.cos(xi(5) ** 2) * 0.12
        + (xi(6) ** 2) * 0.1
        + np.tanh(xi(14) * xi(15)) * 0.25
        + (xi(8) * xi(10) * xi(11)) * 0.04
    )


def generate_ultra_messy_dgp(
    total_samples: int,
    feature_dim: int,
    ref_samples: int,
    batch_size: int,
    noise_scale: float,
    shift_magnitude: float,
    random_seed: int,
    n_arms: int,
    shift_point_index: int | None = None,
    shift_interval: int = 2000,
    mini_shift_interval: int = 400,
    covariate_drift_strength: float = 0.45,
) -> Dict[str, Any]:
    """Highly nonstationary DGP: covariate drift + multi-regime CD + abrupt shift + heavy tails."""
    rng = np.random.default_rng(random_seed)
    ref_samples = min(ref_samples, total_samples - batch_size * 5)
    ref_samples = max(ref_samples, batch_size * 5)
    shift_point = shift_point_index if shift_point_index is not None else total_samples // 2
    shift_point = min(max(shift_point, ref_samples + batch_size), total_samples - batch_size)
    shift_interval = max(batch_size * 4, int(shift_interval))
    mini_shift_interval = max(batch_size * 2, int(mini_shift_interval))

    X = rng.normal(0, 1, size=(total_samples, feature_dim))

    # Four rotating concept regimes + micro-perturbation coefficients.
    # Slice widths scale with d so low-dim (e.g. d=10) still runs.
    block = max(3, min(15, feature_dim))
    jump_w = max(2, min(20, feature_dim // 2 if feature_dim >= 4 else feature_dim))
    betas = []
    for k in range(4):
        b = np.zeros(feature_dim)
        start = (k * max(1, feature_dim // 4)) % feature_dim
        idx = (np.arange(block) + start) % feature_dim
        b[idx] = np.linspace(0.4, -0.2, block) * (0.8 + 0.1 * k)
        extra_n = max(1, min(5, feature_dim))
        extra_idx = (np.arange(extra_n) + start + block) % feature_dim
        b[extra_idx] = rng.normal(0, 0.15, size=extra_n)
        betas.append(b)
    beta_jump = np.zeros(feature_dim)
    jump_start = min(feature_dim // 2, max(0, feature_dim - jump_w))
    jump_idx = (np.arange(jump_w) + jump_start) % feature_dim
    beta_jump[jump_idx] = np.linspace(0.5, -0.35, jump_w)

    Y = np.zeros(total_samples)
    shift_batch_indices: List[int] = []
    mini_shift_batches: List[int] = []

    for t in range(total_samples):
        # Covariate drift (post-ref): mean + variance warping on leading features.
        if t >= ref_samples:
            phase = (t - ref_samples) % shift_interval / shift_interval
            mean_shift = covariate_drift_strength * (
                0.6 * np.sin(2 * np.pi * phase * 2.3)
                + 0.4 * np.cos(2 * np.pi * phase * 3.7)
            )
            var_scale = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * phase * 1.6))
            for j in range(min(18, feature_dim)):
                X[t, j] = X[t, j] * var_scale + mean_shift * (0.35 + 0.05 * (j % 5))

        x_t = X[t]
        if t < ref_samples:
            regime = 0
        else:
            cycle = (t - ref_samples) // shift_interval
            regime = int(cycle % len(betas))
            if t == ref_samples + cycle * shift_interval and cycle > 0:
                shift_batch_indices.append(max(0, (t - ref_samples) // batch_size))

        beta_t = betas[regime].copy()
        if t >= ref_samples:
            mini_cycle = (t - ref_samples) // mini_shift_interval
            rng_mini = np.random.default_rng(random_seed + mini_cycle * 9973)
            beta_t += rng_mini.normal(0, 0.04, size=feature_dim)
            if t == ref_samples + mini_cycle * mini_shift_interval and mini_cycle > 0:
                mini_shift_batches.append(max(0, (t - ref_samples) // batch_size))

        nonlinear = _ultra_nonlinear(x_t, regime)
        linear = float(x_t @ beta_t)

        if t >= shift_point:
            linear += float(x_t @ beta_jump) * shift_magnitude

        if t < ref_samples:
            noise_sd = noise_scale * 0.55
        else:
            phase = (t - ref_samples) % shift_interval / shift_interval
            noise_mod = 0.45 + 0.55 * np.exp(-((phase - 0.35) ** 2) * 18)
            noise_sd = noise_scale * (0.85 + noise_mod + 0.15 * np.abs(x_t[0]))

        if rng.random() < 0.08:
            eps = rng.standard_t(df=3) * noise_sd * 1.4
        else:
            eps = rng.normal(0, noise_sd)
        if rng.random() < 0.02:
            eps += rng.choice([-1, 1]) * noise_sd * rng.uniform(4.0, 8.0)

        Y[t] = (linear + nonlinear + eps) / 4.5

    scaler = StandardScaler()
    X_ref = X[:ref_samples]
    scaler.fit(X_ref)
    X_scaled = scaler.transform(X)
    Y_ref = Y[:ref_samples]
    fitted_models = []
    for name, model in build_model_pool(n_arms):
        m = copy.deepcopy(model)
        m.fit(X_ref, Y_ref)
        fitted_models.append((name, m))

    n_batches = (total_samples - ref_samples) // batch_size
    major_shift_batch = max(0, (shift_point - ref_samples) // batch_size)
    if major_shift_batch not in shift_batch_indices:
        shift_batch_indices.append(major_shift_batch)
    shift_batch_indices = sorted(set(shift_batch_indices))

    return {
        "X_scaled": X_scaled,
        "Y": Y,
        "X_ref": X_ref,
        "fitted_models": fitted_models,
        "ref_samples": ref_samples,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "shift_point": shift_point,
        "shift_batch_index": major_shift_batch,
        "shift_batch_indices": shift_batch_indices,
        "mini_shift_batches": mini_shift_batches,
        "dgp": "nonlinear_messy_ultra",
        "shift_interval": shift_interval,
        "mini_shift_interval": mini_shift_interval,
        "shift_magnitude": shift_magnitude,
    }


def generate_dgp_from_config(data_cfg: Dict[str, Any], n_arms: int) -> Dict[str, Any]:
    dgp = data_cfg.get("dgp", "nonlinear_messy")
    common = dict(
        total_samples=data_cfg["total_samples"],
        feature_dim=data_cfg["feature_dim"],
        ref_samples=data_cfg["ref_samples"],
        batch_size=data_cfg["batch_size"],
        noise_scale=data_cfg["noise_scale"],
        random_seed=data_cfg["random_seed"],
        n_arms=n_arms,
    )
    if dgp == "complex_periodic_shift":
        return generate_complex_dgp(
            **common,
            shift_interval=int(data_cfg.get("shift_interval", 4000)),
        )
    if dgp == "hyper_nonlinear_shift":
        return generate_hyper_nonlinear_shift_dgp(
            **common,
            shift_interval=int(data_cfg.get("shift_interval", 4000)),
            mini_shift_interval=int(data_cfg.get("mini_shift_interval", 500)),
            covariate_drift_strength=float(data_cfg.get("covariate_drift_strength", 0.85)),
            shift_magnitude=float(data_cfg.get("shift_magnitude", 0.8)),
            n_regimes=int(data_cfg.get("n_regimes", 6)),
        )
    if dgp == "nonlinear_messy_ultra":
        return generate_ultra_messy_dgp(
            **common,
            shift_magnitude=float(data_cfg.get("shift_magnitude", 0.5)),
            shift_point_index=data_cfg.get("shift_point_index"),
            shift_interval=int(data_cfg.get("shift_interval", 2000)),
            mini_shift_interval=int(data_cfg.get("mini_shift_interval", 400)),
            covariate_drift_strength=float(data_cfg.get("covariate_drift_strength", 0.45)),
        )
    return generate_dgp(
        **common,
        shift_magnitude=float(data_cfg.get("shift_magnitude", 0.5)),
        shift_point_index=data_cfg.get("shift_point_index"),
        dgp=dgp,
    )


def context_dim(n_base_features: int) -> int:
    return n_base_features * 5 + 3


def build_context(
    X_batch: np.ndarray,
    X_ref: np.ndarray,
    batch_idx: int,
    n_base_features: int,
) -> Tuple[np.ndarray, float]:
    feats = X_batch[:, :n_base_features]
    ref_feats = X_ref[:, :n_base_features]
    mean_vec = np.mean(feats, axis=0)
    std_vec = np.std(feats, axis=0)
    quant_vec = np.quantile(feats, [0.25, 0.5, 0.75], axis=0).flatten()
    ref_mean = np.mean(ref_feats, axis=0)
    drift_delta = min(1.0, np.linalg.norm(mean_vec - ref_mean) / 2.0)
    time_sin = np.sin(2 * np.pi * batch_idx / 100)
    time_cos = np.cos(2 * np.pi * batch_idx / 100)
    x_vec = np.concatenate([mean_vec, std_vec, quant_vec, [drift_delta, time_sin, time_cos]])
    return x_vec.astype(float), float(drift_delta)


class LinUCBDisjointArmVanilla:
    def __init__(
        self,
        arm_idx: int,
        d: int,
        alpha: float = 0.1,
        lam: float = 0.05,
        complexity_risk_coef: float = 0.2,
    ):
        self.arm_idx = arm_idx
        self.alpha = alpha
        self.lam = lam
        self.complexity_risk_coef = complexity_risk_coef
        self.d = d
        self.A = np.identity(d)
        self.b = np.zeros((d, 1))
    def calc_ucb(self, x: np.ndarray, drift_delta: float = 0.0, complexity: float = 0.0) -> float:
        x = x.reshape(-1, 1)
        a_inv = np.linalg.inv(self.A + self.lam * np.identity(self.d))
        theta = a_inv @ self.b
        mean = (theta.T @ x).item()
        uncertainty = self.alpha * np.sqrt((x.T @ a_inv @ x).item() + 1e-8)
        risk_penalty = drift_delta * complexity * self.complexity_risk_coef
        return mean + uncertainty - risk_penalty
    def update(self, reward: float, x: np.ndarray) -> None:
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x


class LinUCBDisjointArmMomentum(LinUCBDisjointArmVanilla):
    def __init__(
        self,
        arm_idx: int,
        d: int,
        alpha: float = 0.1,
        lam: float = 0.05,
        gamma: float = 0.95,
        window: int = 30,
        threshold: float = 1.5,
        gamma_decay: float = 0.98,
        alpha_growth: float = 1.02,
        complexity_risk_coef: float = 0.2,
    ):
        super().__init__(arm_idx, d, alpha, lam, complexity_risk_coef)
        self.base_gamma = gamma
        self.gamma = gamma
        self.window = window
        self.threshold = threshold
        self.gamma_decay = gamma_decay
        self.alpha_growth = alpha_growth
        self.recent_rewards: List[float] = []
        self.recent_var = 1.0
    def calc_ucb(self, x: np.ndarray, drift_delta: float = 0.0, complexity: float = 0.0) -> float:
        x = x.reshape(-1, 1)
        a_inv = np.linalg.inv(self.A + self.lam * np.identity(self.d))
        theta = a_inv @ self.b
        mean = (theta.T @ x).item()
        noise_aware = self.alpha * np.sqrt(self.recent_var) * np.sqrt((x.T @ a_inv @ x).item() + 1e-8)
        risk_penalty = drift_delta * complexity * self.complexity_risk_coef
        return mean + noise_aware - risk_penalty
    def update(self, reward: float, x: np.ndarray) -> None:
        x = x.reshape(-1, 1)
        self.A = self.gamma * self.A + x @ x.T
        self.b = self.gamma * self.b + reward * x
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window:
            self.recent_rewards.pop(0)
        if len(self.recent_rewards) > 5:
            trimmed = np.sort(self.recent_rewards)[1:-1]
            self.recent_var = max(np.var(trimmed) if len(trimmed) > 1 else 1.0, 0.01)
        if len(self.recent_rewards) >= 10:
            median = np.median(self.recent_rewards)
            mad = np.median(np.abs(self.recent_rewards - median))
            if mad > 1e-6:
                z = 0.6745 * (self.recent_rewards[-1] - median) / mad
                if abs(z) > self.threshold:
                    self.gamma = max(0.7, self.gamma * self.gamma_decay)
                    self.alpha = min(1.0, self.alpha * self.alpha_growth)
                else:
                    self.gamma = self.base_gamma
                    self.alpha = max(0.05, self.alpha * 0.99)


class LinUCBPolicy:
    def __init__(self, k: int, d: int, policy_cfg: Dict[str, Any]):
        self.k = k
        complexity_risk_coef = policy_cfg.get("complexity_risk_coef", 0.2)
        if policy_cfg.get("use_momentum", False):
            self.arms = [
                LinUCBDisjointArmMomentum(
                    i,
                    d,
                    alpha=policy_cfg.get("alpha", 0.1),
                    lam=policy_cfg["lambda_reg"],
                    gamma=policy_cfg.get("base_gamma", 0.95),
                    window=policy_cfg.get("window", 30),
                    threshold=policy_cfg.get("threshold", 1.5),
                    gamma_decay=policy_cfg.get("gamma_decay", 0.98),
                    alpha_growth=policy_cfg.get("alpha_growth", 1.02),
                    complexity_risk_coef=complexity_risk_coef,
                )
                for i in range(k)
            ]
        else:
            self.arms = [
                LinUCBDisjointArmVanilla(
                    i,
                    d,
                    alpha=policy_cfg.get("alpha", 0.1),
                    lam=policy_cfg["lambda_reg"],
                    complexity_risk_coef=complexity_risk_coef,
                )
                for i in range(k)
            ]
    def select_arm(self, x: np.ndarray, drift_delta: float = 0.0) -> int:
        vals = [
            arm.calc_ucb(x, drift_delta, COMPLEXITY_SCORES[i])
            for i, arm in enumerate(self.arms)
        ]
        return int(np.argmax(vals))
    def update(self, arm: int, reward: float, x: np.ndarray) -> None:
        self.arms[arm].update(reward, x)




class AdaptiveEpsilonGreedyEWMA:
    """EWMA mean + drift-adaptive epsilon (cov shift + MSE anomaly via drift_delta)."""

    def __init__(
        self,
        n_arms: int,
        base_epsilon: float = 0.025,
        max_epsilon: float = 0.5,
        gamma: float = 0.9,
        anomaly_sensitivity: float = 0.5,
    ):
        self.n_arms = n_arms
        self.base_epsilon = base_epsilon
        self.max_epsilon = max_epsilon
        self.gamma = gamma
        self.anomaly_sensitivity = anomaly_sensitivity
        self.mu = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)

    def select_arm(self, x_array=None, drift_delta: float = 0.0) -> int:
        epsilon = min(
            self.max_epsilon,
            self.base_epsilon + drift_delta * self.anomaly_sensitivity,
        )
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_arms))
        return int(np.argmax(self.mu))

    def update(self, arm: int, reward: float, x=None) -> None:
        self.mu[arm] = self.gamma * self.mu[arm] + (1.0 - self.gamma) * reward
        self.counts[arm] += 1


class ThompsonSamplingPolicy:
    def __init__(self, k: int, prior_lambda: float = 0.1):
        self.k = k
        self.mu = np.zeros(k)
        self.lambda_ = np.ones(k) * prior_lambda
        self.alpha = np.ones(k) * 1.0
        self.beta = np.ones(k) * 1.0

    def select_arm(self, x=None, drift_delta: float = 0.0) -> int:
        samples = []
        for i in range(self.k):
            tau = np.random.gamma(self.alpha[i], 1.0 / self.beta[i])
            var_mu = 1.0 / (self.lambda_[i] * tau + 1e-8)
            samples.append(np.random.normal(self.mu[i], np.sqrt(var_mu)))
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, x=None) -> None:
        n = 1
        r = reward
        mu_old = self.mu[arm]
        lam_old = self.lambda_[arm]
        self.mu[arm] = (lam_old * mu_old + n * r) / (lam_old + n)
        self.lambda_[arm] = lam_old + n
        alpha_old = self.alpha[arm]
        beta_old = self.beta[arm]
        self.alpha[arm] = alpha_old + 0.5
        delta = r - mu_old
        self.beta[arm] = beta_old + (n * lam_old * delta**2) / (2.0 * (lam_old + n))


class EpsilonGreedyPolicy:
    def __init__(self, k: int, eps: float = 0.1):
        self.k = k
        self.eps = eps
        self.history: List[List[float]] = [[] for _ in range(k)]
    def select_arm(self, x=None, drift_delta: float = 0.0) -> int:
        if np.random.rand() < self.eps:
            return int(np.random.randint(self.k))
        means = [np.mean(h) if h else -1e10 for h in self.history]
        return int(np.argmax(means))
    def update(self, arm: int, reward: float, x=None) -> None:
        self.history[arm].append(reward)


class GaussianSamplingPolicy:
    def __init__(self, k: int, learning_rate: float = 0.1, init_sigma: float = 5.0):
        self.k = k
        self.lr = learning_rate
        self.mu = np.zeros(k)
        self.sigma = np.ones(k) * init_sigma
        self.history: List[List[float]] = [[] for _ in range(k)]
    def select_arm(self, x=None, drift_delta: float = 0.0) -> int:
        samples = [norm.rvs(self.mu[i], self.sigma[i]) for i in range(self.k)]
        return int(np.argmax(samples))
    def update(self, arm: int, reward: float, x=None) -> None:
        self.history[arm].append(reward)
        self.mu[arm] = (1 - self.lr) * self.mu[arm] + self.lr * reward
        delta = reward - self.mu[arm]
        self.sigma[arm] = np.sqrt(
            (1 - self.lr) * (self.sigma[arm] ** 2) + self.lr * (delta**2)
        )
        self.sigma[arm] = max(self.sigma[arm], 0.05)


class UCBPolicy:
    """Classic UCB1 (Auer et al.): value + sqrt(2 log t / n_a)."""
    def __init__(self, k: int):
        self.counts = [0 for _ in range(k)]
        self.values = [0.0 for _ in range(k)]
    def select_arm(self, x=None, drift_delta: float = 0.0) -> int:
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
        total_counts = sum(self.counts)
        ucb_values = []
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values.append(self.values[arm] + bonus)
        return int(np.argmax(ucb_values))
    def update(self, arm: int, reward: float, x=None) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / float(n)) * value + (1.0 / float(n)) * reward


class RandomPolicy:
    def __init__(self, k: int):
        self.k = k
    def select_arm(self, x=None, drift_delta: float = 0.0) -> int:
        return int(np.random.randint(self.k))
    def update(self, arm: int, reward: float, x=None) -> None:
        pass


def make_policy(policy_name: str, k: int, d: int, policy_cfg: Dict[str, Any]):
    if policy_name == "AdaptiveEpsilonGreedy":
        return AdaptiveEpsilonGreedyEWMA(
            n_arms=k,
            base_epsilon=policy_cfg.get("base_epsilon", 0.025),
            max_epsilon=policy_cfg.get("max_epsilon", 0.5),
            gamma=policy_cfg.get("gamma", 0.9),
            anomaly_sensitivity=policy_cfg.get("anomaly_sensitivity", 0.5),
        )
    if policy_name.startswith("LinUCB"):
        return LinUCBPolicy(k, d, policy_cfg)
    if policy_name == "Thompson_Sampling":
        return ThompsonSamplingPolicy(k, prior_lambda=policy_cfg["prior_lambda"])
    if policy_name == "Epsilon_Greedy":
        return EpsilonGreedyPolicy(k, eps=policy_cfg["epsilon"])
    if policy_name == "Gaussian_Sampling":
        return GaussianSamplingPolicy(
            k,
            learning_rate=policy_cfg["learning_rate"],
            init_sigma=policy_cfg["init_sigma"],
        )
    if policy_name == "UCB":
        return UCBPolicy(k)
    if policy_name == "Random":
        return RandomPolicy(k)
    raise ValueError(f"unknown policy: {policy_name}")


def precompute_batch_cache(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Predict once per batch; reuse across all policy configs."""
    ctx_cfg = config["context"]
    n_base = ctx_cfg["n_base_features"]
    X_scaled = data["X_scaled"]
    Y = data["Y"]
    X_ref = data["X_ref"]
    fitted_models = data["fitted_models"]
    ref_samples = data["ref_samples"]
    batch_size = data["batch_size"]
    n_batches = data["n_batches"]
    n_arms = len(fitted_models)
    rewards_all = np.zeros((n_batches, n_arms))
    opt_arms = np.zeros(n_batches, dtype=int)
    contexts = np.zeros((n_batches, context_dim(n_base)))
    drift_deltas = np.zeros(n_batches)
    opt_mse_list = []
    for i in range(n_batches):
        start = ref_samples + i * batch_size
        end = min(start + batch_size, len(Y))
        X_batch = X_scaled[start:end]
        Y_batch = Y[start:end]
        mse_vals = []
        for _, model in fitted_models:
            pred = model.predict(X_batch)
            mse_vals.append(mean_squared_error(Y_batch, pred))
        batch_rewards = -np.array(mse_vals)
        rewards_all[i] = batch_rewards
        opt_arms[i] = int(np.argmax(batch_rewards))
        opt_mse_list.append(-batch_rewards[int(opt_arms[i])])
        x_vec, drift_delta = build_context(X_batch, X_ref, i, n_base)
        contexts[i] = x_vec
        drift_deltas[i] = drift_delta
    pvals = empirical_pval(opt_mse_list, burnin=ctx_cfg.get("pval_burnin", 5))
    anomaly_scores = np.array([1.0 - p for p in pvals])
    return {
        "rewards_all": rewards_all,
        "opt_arms": opt_arms,
        "contexts": contexts,
        "n_batches": n_batches,
        "opt_mse_list": opt_mse_list,
        "pvals": pvals,
        "anomaly_scores": anomaly_scores,
        "cov_shift_only": drift_deltas.copy(),
    }


def run_experiment(
    policy,
    data: Dict[str, Any],
    config: Dict[str, Any],
    use_context: bool = True,
    batch_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safety = config["safety"]
    ctx_cfg = config["context"]
    complexity_scores = config.get("complexity_scores", COMPLEXITY_SCORES)
    if batch_cache is None:
        batch_cache = precompute_batch_cache(data, config)
    n_batches = batch_cache["n_batches"]
    rewards_all_mat = batch_cache["rewards_all"]
    opt_arms = batch_cache["opt_arms"]
    contexts = batch_cache["contexts"]
    cov_shift_only = batch_cache["cov_shift_only"]
    anomaly_scores = batch_cache["anomaly_scores"]

    current_anomaly_weight = float(ctx_cfg.get("anomaly_weight", 0.6))
    min_weight = float(ctx_cfg.get("min_anomaly_weight", 0.15))
    decay_window = int(ctx_cfg.get("decay_window", 20))
    decay_rate = float(ctx_cfg.get("decay_rate", 0.95))
    reset_threshold_anomaly = float(ctx_cfg.get("reset_threshold_anomaly", 0.4))
    reset_threshold_cov = float(ctx_cfg.get("reset_threshold_cov", 0.3))
    normal_count = 0

    rewards = np.zeros(n_batches)
    regrets = np.zeros(n_batches)
    chosen = np.zeros(n_batches, dtype=int)
    drift_deltas_record = np.zeros(n_batches)

    for i in range(n_batches):
        rewards_all = rewards_all_mat[i]
        opt_arm = int(opt_arms[i])
        cov_shift = float(cov_shift_only[i])
        anomaly_score = float(anomaly_scores[i])
        is_normal = (anomaly_score < reset_threshold_anomaly) and (cov_shift < reset_threshold_cov)
        if is_normal:
            normal_count += 1
            if normal_count > decay_window:
                current_anomaly_weight = max(min_weight, current_anomaly_weight * decay_rate)
        else:
            normal_count = 0
            current_anomaly_weight = float(ctx_cfg.get("anomaly_weight", 0.6))

        drift_delta = max(cov_shift, anomaly_score * current_anomaly_weight)
        drift_delta = float(np.clip(drift_delta, 0.0, 1.0))
        drift_deltas_record[i] = drift_delta

        if use_context:
            x_vec = contexts[i]
            arm = policy.select_arm(x_vec, drift_delta)
        else:
            x_vec = None
            arm = policy.select_arm(drift_delta=drift_delta)
        if drift_delta > safety["drift_guard_threshold"] and arm in safety["risky_arms"]:
            arm = safety["fallback_arm"]
        raw_reward = rewards_all[arm]
        shaped_reward = raw_reward - safety["reward_shaping_coef"] * drift_delta * complexity_scores[arm]
        if use_context:
            policy.update(arm, shaped_reward, x_vec)
        else:
            policy.update(arm, shaped_reward, None)
        chosen[i] = arm
        rewards[i] = raw_reward
        regrets[i] = rewards_all[opt_arm] - raw_reward
    return {
        "cum_regret": np.cumsum(regrets),
        "final_regret": float(np.sum(regrets)),
        "mean_reward": float(np.mean(rewards)),
        "rewards": rewards,
        "chosen": chosen,
        "n_batches": n_batches,
        "drift_deltas": drift_deltas_record,
    }
