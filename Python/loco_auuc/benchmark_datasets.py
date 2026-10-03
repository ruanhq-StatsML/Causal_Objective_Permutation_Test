"""Open uplift benchmarks + synthetic shift scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class UpliftBatch:
    name: str
    X_ref: np.ndarray
    t_ref: np.ndarray
    y_ref: np.ndarray
    X_live: np.ndarray
    t_live: np.ndarray
    y_live: np.ndarray
    feature_names: list


def load_hillstrom_ref_live(*, live_frac: float = 0.35, seed: int = 42) -> UpliftBatch:
    from sklift.datasets import fetch_hillstrom
    from sklearn.model_selection import train_test_split

    raw = fetch_hillstrom()
    df = raw["data"].copy()
    seg = np.asarray(raw["treatment"]).astype(str).ravel()
    y_all = np.asarray(raw["target"]).astype(int).ravel()
    mask = (df["mens"].values == 1) & np.isin(seg, ["Mens E-Mail", "No E-Mail"])
    df = df.loc[mask].reset_index(drop=True)
    seg = seg[mask]
    y_all = y_all[mask]
    t_all = (seg == "Mens E-Mail").astype(int)

    num = [c for c in ["recency", "history", "mens", "womens", "newbie"] if c in df.columns]
    X = df[num].astype(float).values
    t = t_all
    y = y_all

    X_ref, X_live, t_ref, t_live, y_ref, y_live = train_test_split(
        X, t, y, test_size=live_frac, stratify=t, random_state=seed
    )
    # Cap size for LOCO retrain budget (monitoring benchmark, not SOTA leaderboard)
    cap_ref, cap_live = 3500, 1200
    if X_ref.shape[0] > cap_ref:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X_ref.shape[0], cap_ref, replace=False)
        X_ref, t_ref, y_ref = X_ref[idx], t_ref[idx], y_ref[idx]
    if X_live.shape[0] > cap_live:
        rng = np.random.default_rng(seed + 1)
        idx = rng.choice(X_live.shape[0], cap_live, replace=False)
        X_live, t_live, y_live = X_live[idx], t_live[idx], y_live[idx]
    names = [f"x{j}" for j in range(X.shape[1])]
    return UpliftBatch("hillstrom", X_ref, t_ref, y_ref, X_live, t_live, y_live, names)


def load_synthetic_covariate_shift(n_ref: int = 2500, n_live: int = 1000, seed: int = 11) -> UpliftBatch:
    rng = np.random.default_rng(seed)

    def gen(n, live):
        X = rng.normal(size=(n, 10))
        if live:
            X[:, 0] += 1.0
            X[:, 1] += 0.5
        t = rng.binomial(1, 0.5, size=n)
        logit = 0.5 * X[:, 2] + 0.3 * X[:, 3]
        y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
        return X, t, y

    X_ref, t_ref, y_ref = gen(n_ref, False)
    X_live, t_live, y_live = gen(n_live, True)
    return UpliftBatch(
        "synthetic_covariate",
        X_ref,
        t_ref,
        y_ref,
        X_live,
        t_live,
        y_live,
        [f"f{j}" for j in range(10)],
    )


def load_synthetic_concept_shift(n_ref: int = 2500, n_live: int = 1000, seed: int = 22) -> UpliftBatch:
    rng = np.random.default_rng(seed)

    def gen(n, live):
        X = rng.normal(size=(n, 10))
        t = rng.binomial(1, 0.5, size=n)
        logit = 0.5 * X[:, 2] + 0.3 * X[:, 3]
        if live:
            logit += 0.8 * X[:, 4]
        y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
        return X, t, y

    X_ref, t_ref, y_ref = gen(n_ref, False)
    X_live, t_live, y_live = gen(n_live, True)
    return UpliftBatch(
        "synthetic_concept",
        X_ref,
        t_ref,
        y_ref,
        X_live,
        t_live,
        y_live,
        [f"f{j}" for j in range(10)],
    )


def load_hillstrom_covariate_drift(seed: int = 7) -> UpliftBatch:
    base = load_hillstrom_ref_live(live_frac=0.3, seed=seed)
    X_live = base.X_live.copy()
    X_live[:, : min(3, X_live.shape[1])] += 0.75
    return UpliftBatch(
        "hillstrom_covariate_drift",
        base.X_ref,
        base.t_ref,
        base.y_ref,
        X_live,
        base.t_live,
        base.y_live,
        base.feature_names,
    )


def all_benchmarks():
    return [
        load_hillstrom_ref_live(),
        load_hillstrom_covariate_drift(),
        load_synthetic_covariate_shift(),
        load_synthetic_concept_shift(),
    ]
