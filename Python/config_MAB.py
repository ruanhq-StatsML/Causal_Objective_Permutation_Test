"""MAB benchmark configuration."""
import numpy as np

GAMMA_GRID = np.linspace(0.5, 0.95, 10).round(2).tolist()

# Run every policy grid on each DGP scenario.
DGP_GRID = [
    {
        "name": "nonlinear_messy_sm025",
        "dgp": "nonlinear_messy",
        "shift_magnitude": 0.25,
        "shift_point_index": 10000,
    },
    {
        "name": "nonlinear_messy_sm05",
        "dgp": "nonlinear_messy",
        "shift_magnitude": 0.5,
        "shift_point_index": 10000,
    },
    {
        "name": "nonlinear_messy_sm10",
        "dgp": "nonlinear_messy",
        "shift_magnitude": 1.0,
        "shift_point_index": 10000,
    },
    {
        "name": "linear_shift_sm05",
        "dgp": "linear_shift",
        "shift_magnitude": 0.5,
        "shift_point_index": 10000,
    },
    {
        "name": "nonlinear_messy_ultra_sm05",
        "dgp": "nonlinear_messy_ultra",
        "shift_magnitude": 0.5,
        "shift_point_index": 10000,
        "shift_interval": 2000,
        "mini_shift_interval": 400,
        "covariate_drift_strength": 0.45,
    },
    {
        "name": "nonlinear_messy_ultra_sm10",
        "dgp": "nonlinear_messy_ultra",
        "shift_magnitude": 1.0,
        "shift_point_index": 10000,
        "shift_interval": 2000,
        "mini_shift_interval": 400,
        "covariate_drift_strength": 0.45,
    },
    {
        "name": "nonlinear_messy_ultra_sm025",
        "dgp": "nonlinear_messy_ultra",
        "shift_magnitude": 0.25,
        "shift_point_index": 10000,
        "shift_interval": 2000,
        "mini_shift_interval": 400,
        "covariate_drift_strength": 0.45,
    },
]

# Prototype: hyper-nonlinear multi-regime CD + strong covariate shift (replaces simpler periodic DGP).
PROTOTYPE_DGP_GRID = [
    {
        "name": "hyper_nonlinear_si4000",
        "dgp": "hyper_nonlinear_shift",
        "shift_interval": 4000,
        "mini_shift_interval": 500,
        "covariate_drift_strength": 0.85,
        "shift_magnitude": 0.8,
        "n_regimes": 6,
    },
    {
        "name": "complex_periodic_si4000",
        "dgp": "complex_periodic_shift",
        "shift_interval": 4000,
    },
    # Large-scale messy DGP for epsilon calibration (override dims/n via CLI if needed).
    {
        "name": "messy_d500_n100k",
        "dgp": "nonlinear_messy",
        "feature_dim": 500,
        "total_samples": 100000,
        "ref_samples": 10000,
        "batch_size": 500,
        "shift_point_index": 50000,
        "shift_magnitude": 0.5,
        "noise_scale": 2.5,
    },
]

# Dense epsilon grid for large-scale calibration.
PROTOTYPE_EPSILON_CALIB_POLICIES = {
    "Epsilon_Greedy": {
        "epsilon": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
    },
}

# Dual-drift / adaptive-cooling presets (applied to run_experiment drift fusion).
PROTOTYPE_COOLING_PRESETS = {
    "cool_default": {},
    "cool_slow": {
        "decay_rate": 0.98,
        "decay_window": 30,
        "min_anomaly_weight": 0.10,
    },
    "cool_fast": {
        "decay_rate": 0.90,
        "decay_window": 10,
        "min_anomaly_weight": 0.20,
    },
}

PROTOTYPE_POLICIES = {
    "Epsilon_Greedy": {
        "epsilon": [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
    },
    "AdaptiveEpsilonGreedy": {
        "base_epsilon": [0.025, 0.05, 0.10],
        "max_epsilon": [0.5],
        "gamma": [0.9],
        "anomaly_sensitivity": [0.3, 0.5, 0.7],
    },
    "UCB": {},
    "Random": {},
}

# Dedicated epsilon sweep (1/2/5/8/10/15/20/30%) for high-repeat runs.
PROTOTYPE_EPSILON_SWEEP_POLICIES = {
    "Epsilon_Greedy": {
        "epsilon": [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
    },
}

# Extra adaptive / baseline settings for 5-repeat comparison runs.
PROTOTYPE_EXTENDED_POLICIES = {
    "AdaptiveEpsilonGreedy": {
        "base_epsilon": [0.01, 0.02, 0.05, 0.08, 0.10],
        "max_epsilon": [0.3, 0.5],
        "gamma": [0.9],
        "anomaly_sensitivity": [0.2, 0.3, 0.5, 0.7],
    },
    "UCB": {},
    "Random": {},
}

# Full large grid on hyper_nonlinear DGP: 11 eps + 144 adaptive + baselines.
PROTOTYPE_LARGE_POLICIES = {
    "Epsilon_Greedy": {
        "epsilon": [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
    },
    "AdaptiveEpsilonGreedy": {
        "base_epsilon": [0.01, 0.02, 0.05, 0.08, 0.10, 0.15],
        "max_epsilon": [0.3, 0.5, 0.7],
        "gamma": [0.9, 0.95],
        "anomaly_sensitivity": [0.2, 0.3, 0.5, 0.7],
    },
    "UCB": {},
    "Random": {},
}

CONFIG = {
    "data": {
        "random_seed": 2026,
        "total_samples": 20000,
        "ref_samples": 5000,
        "batch_size": 25,
        "feature_dim": 100,
        "noise_scale": 2.5,
        "shift_magnitude": 0.5,
        "shift_point_index": 10000,
        "dgp": "nonlinear_messy",
    },
    "context": {
        "n_base_features": 10,
        "aggregations": ["mean", "std", "quantile"],
        "pval_burnin": 5,
        "anomaly_weight": 0.6,
        "min_anomaly_weight": 0.15,
        "decay_window": 20,
        "decay_rate": 0.95,
        "reset_threshold_anomaly": 0.4,
        "reset_threshold_cov": 0.3,
    },
    "safety": {
        "risky_arms": [8, 9, 10, 11],
        "fallback_arm": 0,
        "drift_guard_threshold": 0.85,
        "reward_shaping_coef": 0.01,
        "complexity_risk_coef": 0.2,
    },
    "model_pool": {
        "n_arms": 12,
    },
    "policies": {
        "LinUCB_Vanilla": {
            "alpha": 0.1,
            "lambda_reg": [10, 50, 100, 500, 1000],
            "use_momentum": False,
        },
        "LinUCB_Momentum": {
            "alpha": 0.1,
            "lambda_reg": [10, 50, 100, 500, 1000],
            "base_gamma": GAMMA_GRID,
            "window": 30,
            "threshold": 1.5,
            "use_momentum": True,
            "gamma_decay": GAMMA_GRID,
            "alpha_growth": 1.02,
        },
        "Thompson_Sampling": {
            "prior_lambda": [0.05, 0.1, 0.15, 0.2],
        },
        "Epsilon_Greedy": {
            "epsilon": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        },
        "AdaptiveEpsilonGreedy": {
            "base_epsilon": [0.025, 0.05, 0.1],
            "max_epsilon": [0.5],
            "gamma": [0.9],
            "anomaly_sensitivity": [0.3, 0.5, 0.7],
        },
        "Gaussian_Sampling": {
            "learning_rate": [0.1],
            "init_sigma": [5.0],
        },
        "UCB": {},
        "Random": {},
    },
}

COMPLEXITY_SCORES = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0
]
