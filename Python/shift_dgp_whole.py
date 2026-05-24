#!/usr/bin/env python3
"""
Baseline + shift DGP grid: ref vs new batches, covariate / concept / mixed.
Writes:
  results/DGP_whole_outputs.npy
  results/DGP_whole_meta.csv
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

ShiftType = ["no_shift", "covariate_only", "concept_only", "mixed"]
XShiftType = ["none", "mean", "covariance", "mixture", "support"]
ConceptDriftType = ["none", "linear", "nonlinear", "cubic", "interaction", "threshold", "none"]
YType = ["Discrete", "Continuous"]


@dataclass
class DGPConfig:
    dgp_id: int = 1
    n_ref: int = 1000
    n_new: int = 1000
    d: int = 10
    x_shift_type: str = "none"
    concept_drift_type: str = "none"
    delta_x: float = 0.0
    delta_c: float = 0.0
    y_type: str = "regression"
    noise_sd: float = 1.0
    seed: int = 123
    mixture_degree: float = 2.0
    # Used when ``x_shift_type == "eigenshift"`` (comprehensive grid).
    mean_shift_count: int = 0
    eigen_shift_count: int = 4
    eigen_shift_mag: float = 1.5
    @property
    def S_X(self) -> int:
        return int(self.x_shift_type != "none" and self.delta_x > 0)
    @property
    def S_C(self) -> int:
        return int(self.concept_drift_type != "none" and self.delta_c > 0)
    @property
    def label(self) -> str:
        if self.S_X == 0 and self.S_C == 0:
            return "no_shift"
        if self.S_X == 1 and self.S_C == 0:
            return "covariate_only"
        if self.S_X == 0 and self.S_C == 1:
            return "concept_only"
        return "mixed"


@dataclass
class DGPOutput:
    X_ref: np.ndarray
    Y_ref: np.ndarray
    X_new: np.ndarray
    Y_new: np.ndarray
    config: DGPConfig
    metadata: Dict[str, Any]
    def to_metadata_row(self) -> dict[str, Any]:
        row = asdict(self.config)
        row.update(self.metadata)
        return row


class ShiftDGP:
    def __init__(self, config: DGPConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
    def generate_X_ref(self) -> np.ndarray:
        cfg = self.config
        return self.rng.normal(loc=0.0, scale=1.0, size=(cfg.n_ref, cfg.d))
    def generate_X_new(self) -> np.ndarray:
        cfg = self.config
        if cfg.x_shift_type == "none" or cfg.delta_x == 0:
            return self.rng.normal(loc=0.0, scale=1.0, size=(cfg.n_new, cfg.d))
        if cfg.x_shift_type == "mean":
            mean = np.zeros(cfg.d)
            mean[:3] = cfg.delta_x
            return self.rng.normal(loc=mean, scale=1.0, size=(cfg.n_new, cfg.d))
        if cfg.x_shift_type == "covariance":
            cov = np.eye(cfg.d)
            cov[0, 1] = cov[1, 0] = min(0.8, cfg.delta_x)
            cov[2, 3] = cov[3, 2] = min(0.8, cfg.delta_x / 2)
            return self.rng.multivariate_normal(mean=np.zeros(cfg.d), cov=cov, size=cfg.n_new)
        if cfg.x_shift_type == "mixture":
            z = self.rng.binomial(1, p=min(0.9, cfg.delta_x), size=cfg.n_new)
            X = self.rng.normal(0, 1, size=(cfg.n_new, cfg.d))
            X[z == 1, :3] += cfg.mixture_degree
            return X
        if cfg.x_shift_type == "support":
            half = 1.0 + float(cfg.delta_x)
            return self.rng.uniform(-half, half, size=(cfg.n_new, cfg.d))
        if cfg.x_shift_type == "eigenshift":
            k_e = max(1, min(int(cfg.eigen_shift_count), int(cfg.d)))
            mag = float(cfg.eigen_shift_mag)
            dx = float(cfg.delta_x)
            A = self.rng.standard_normal((cfg.d, cfg.d))
            Q, _ = np.linalg.qr(A)
            evals = np.ones(cfg.d, dtype=float)
            mult = 1.0 + mag * dx
            evals[:k_e] *= mult
            sigma = Q @ np.diag(evals) @ Q.T
            sigma = 0.5 * (sigma + sigma.T)
            mu = np.zeros(cfg.d, dtype=float)
            msc = int(cfg.mean_shift_count)
            if msc > 0:
                mu[: min(msc, cfg.d)] = dx
            return self.rng.multivariate_normal(mu, sigma, size=cfg.n_new)
        raise ValueError(f"Unknown x_shift_type: {cfg.x_shift_type}")
    def m0(self, X: np.ndarray) -> np.ndarray:
        return (
            0.5 * X[:, 0]
            + 0.75 * X[:, 1]
            - 0.25 * X[:, 2]
            + 0.25 * np.sin(X[:, 3])
            + 0.75 * X[:, 4] * X[:, 5]
        )
    def concept_drift(self, X: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.concept_drift_type == "none" or cfg.delta_c == 0:
            return np.zeros(X.shape[0], dtype=float)
        if cfg.concept_drift_type == "linear":
            return cfg.delta_c * (X[:, 0] + X[:, 1])
        if cfg.concept_drift_type == "nonlinear":
            return cfg.delta_c * np.sin(X[:, 0] * X[:, 1])
        if cfg.concept_drift_type == "cubic":
            return cfg.delta_c * ((X[:, 0] ** 2) + X[:, 2] ** 3)
        if cfg.concept_drift_type == "interaction":
            return cfg.delta_c * X[:, 0] * X[:, 2] * X[:, 3]
        if cfg.concept_drift_type == "threshold":
            return cfg.delta_c * (X[:, 1] > 0).astype(float)
        raise ValueError(f"Unknown concept drift type: {cfg.concept_drift_type}")
    def m1(self, X: np.ndarray) -> np.ndarray:
        return self.m0(X) + self.concept_drift(X)
    def generate_y(self, X: np.ndarray, batch: str) -> np.ndarray:
        cfg = self.config
        mu = self.m0(X) if batch == "ref" else self.m1(X)
        if cfg.y_type == "regression":
            return mu + self.rng.normal(0, cfg.noise_sd, size=X.shape[0])
        if cfg.y_type == "classification":
            probs = 1.0 / (1.0 + np.exp(-np.clip(mu, -50, 50)))
            return self.rng.binomial(1, probs, size=X.shape[0])
        raise ValueError(f"Unknown y_type: {cfg.y_type}")
    def generate(self) -> DGPOutput:
        X_ref = self.generate_X_ref()
        X_new = self.generate_X_new()
        Y_ref = self.generate_y(X_ref, batch="ref")
        Y_new = self.generate_y(X_new, batch="new")
        cd_new = self.concept_drift(X_new)
        metadata = {
            "oracle_mean_shift": float(np.linalg.norm(X_new.mean(axis=0) - X_ref.mean(axis=0))),
            "oracle_m0_mean_ref": float(np.mean(self.m0(X_ref))),
            "oracle_m1_mean_new": float(np.mean(self.m1(X_new))),
            "oracle_concept_signal_abs_mean_new": float(np.mean(np.abs(cd_new))),
        }
        return DGPOutput(
            X_ref=X_ref,
            Y_ref=Y_ref,
            X_new=X_new,
            Y_new=Y_new,
            config=self.config,
            metadata=metadata,
        )


def make_whole_DGP(
    n_ref: int = 1000,
    n_new: int = 1000,
    d: int = 10,
    y_type: str = "regression",
    base_seed: int = 2026,
) -> list[DGPConfig]:
    x_shift_type = ["none", "mean", "covariance", "mixture", "support"]
    concept_drift_type = ["none", "linear", "nonlinear", "cubic", "interaction", "threshold"]
    delta_x_values: dict[str, list[float]] = {
        "none": [0.0],
        "mean": [0.25, 0.5, 0.75, 1.0],
        "covariance": [0.25, 0.5, 0.75, 1.0],
        "mixture": [0.25, 0.5, 0.75, 1.0],
        "support": [0.25, 0.5, 0.75, 1.0],
    }
    delta_c_values: dict[str, list[float]] = {
        "none": [0.0],
        "linear": [0.25, 0.5, 0.75, 1.0],
        "nonlinear": [0.25, 0.5, 0.75, 1.0],
        "cubic": [0.25, 0.5, 0.75, 1.0],
        "interaction": [0.25, 0.5, 0.75, 1.0],
        "threshold": [0.25, 0.5, 0.75],
    }
    configs: list[DGPConfig] = []
    dgp_id = 0
    for xs in x_shift_type:
        for cs in concept_drift_type:
            for dx in delta_x_values[xs]:
                for dc in delta_c_values[cs]:
                    cfg = DGPConfig(
                        dgp_id=dgp_id,
                        n_ref=n_ref,
                        n_new=n_new,
                        d=d,
                        x_shift_type=xs,
                        concept_drift_type=cs,
                        delta_x=float(dx),
                        delta_c=float(dc),
                        y_type=y_type,
                        seed=base_seed + dgp_id * 2,
                    )
                    configs.append(cfg)
                    dgp_id += 1
    for xs in x_shift_type:
        for dx in delta_x_values[xs]:
            cfg = DGPConfig(
                dgp_id = dgp_id,
                n_ref = n_ref,
                n_new = n_new,
                d = d,
                x_shift_type = xs,
                concept_drift_type = 'none',
                delta_x = float(dx),
                delta_c = 0,
                y_type = y_type,
                seed = base_seed + dgp_id ** 2
            )
            configs.append(cfg)
            dgp_id += 1
    for cs in concept_drift_type:
        for dc in delta_c_values[cs]:
            cfg = DGPConfig(
                dgp_id = dgp_id,
                n_ref = n_ref,
                n_new = n_new,
                d = d,
                x_shift_type = 'none',
                concept_drift_type = cs,
                delta_x = 0,
                delta_c = float(dc),
                y_type = y_type,
                seed = base_seed + dgp_id ** 3
            )
            configs.append(cfg)
            dgp_id += 1
    return configs


def generate_all_dgp(configs: list[DGPConfig]) -> tuple[list[DGPOutput], pd.DataFrame]:
    outputs: list[DGPOutput] = []
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        dgp = ShiftDGP(cfg)
        out = dgp.generate()
        outputs.append(out)
        rows.append(out.to_metadata_row())
    metadata_df = pd.DataFrame(rows)
    return outputs, metadata_df


configs = make_whole_DGP()
outputs, metadata_df = generate_all_dgp(configs)
np.save(
    "DGP_whole_outputs_onesided.npy",
    np.array(outputs, dtype=object),
    allow_pickle=True,
)
metadata_df.to_csv("DGP_whole_meta_onesided.csv", index=False)
print(f"wrote results/DGP_whole_outputs.npy ({len(outputs)} configs)")
print(f"wrote results/DGP_whole_meta.csv shape={metadata_df.shape}")


