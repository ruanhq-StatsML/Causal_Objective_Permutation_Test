"""Online PO-risk prototype.

The original batch is scored with the DRPerm PO-risk: cross-fit ``model_m`` and
``model_e`` from ``MODEL_REGISTRY``, pseudo-outcome ``(Y - mu) * (T - e)``,
then ``mean(tau^2)`` with tau fit by ``model_m``.

Those two fits, trained on the original batch, predict each later sample.
Every later sample has ``T = 1``. The online bootstrap is the stratified
with-replacement resample and the 2.5 / 97.5 percentiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model_registry_class import ModelRegistry


def make_folds(n: int, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return np.array_split(indices, n_folds)


def build_registry(ntree: int = 40):
    factory = ModelRegistry(
        ntree=ntree,
        ridge_alpha=0.25,
        nthread=1,
        maxit=200,
        max_depth=4,
        mlp_hidden_size=4,
        positive_class=1,
    )
    return factory.as_r_style_dict()


def cross_fit_nuisance(X, Y, T, outcome, propensity, seed: int, n_folds: int, clip_e: float):
    """DRPerm cross-fit: model_m and model_e predict the held-out fold."""
    n = len(Y)
    mu = np.zeros(n, dtype=float)
    e = np.zeros(n, dtype=float)
    for k, test_idx in enumerate(make_folds(n, n_folds, seed)):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        fit_mu = outcome["fit"](X[train_idx], Y[train_idx], seed=seed + k)
        mu[test_idx] = outcome["predict"](fit_mu, X[test_idx])
        fit_e = propensity["fit"](X[train_idx], T[train_idx], seed=seed + 100 + k)
        e[test_idx] = propensity["predict"](fit_e, X[test_idx])
    return mu, np.clip(e, clip_e, 1.0 - clip_e)


def po_risk_from_pseudo(X, pseudo, outcome, seed: int) -> float:
    fit_tau = outcome["fit"](X, pseudo, seed=seed)
    tau = outcome["predict"](fit_tau, X)
    return float(np.mean(tau ** 2))


def fit_original_and_score(X0, Y0, T0, X1, Y1, registry, model_m: str, model_e: str,
                           seed: int, n_folds: int = 2, clip_e: float = 0.01) -> float:
    """Original-batch PO-risk, then next-step prediction for each T=1 sample."""
    outcome = registry[model_m]
    propensity = registry[model_e]
    mu0, e0 = cross_fit_nuisance(X0, Y0, T0, outcome, propensity, seed, n_folds, clip_e)
    pseudo0 = (Y0 - mu0) * (T0.astype(float) - e0)
    if len(Y1) == 0:
        return po_risk_from_pseudo(X0, pseudo0, outcome, seed + 200)

    fit_mu = outcome["fit"](X0, Y0, seed=seed + 7)
    fit_e = propensity["fit"](X0, T0, seed=seed + 11)
    mu1 = outcome["predict"](fit_mu, X1)
    e1 = np.clip(propensity["predict"](fit_e, X1), clip_e, 1.0 - clip_e)
    pseudo1 = (Y1 - mu1) * (1.0 - e1)
    X = np.vstack([X0, X1])
    pseudo = np.concatenate([pseudo0, pseudo1])
    return po_risk_from_pseudo(X, pseudo, outcome, seed + 200)


def _stratified_bootstrap(groups, rng) -> np.ndarray:
    parts = []
    for idx in groups:
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(parts)


def online_series(X0, Y0, T0, X1, Y1, registry, model_m: str, model_e: str,
                  step: int, n_boot: int, seed: int) -> list:
    """Grow the T=1 tail. Bootstrap resamples the original batch and the tail."""
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(T0 == 0)
    idx1 = np.flatnonzero(T0 == 1)
    rows = []
    n1 = len(Y1)
    for m in range(step, n1 + 1, step):
        point = fit_original_and_score(
            X0, Y0, T0, X1[:m], Y1[:m], registry, model_m, model_e, seed + m,
        )
        draws = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            take0 = _stratified_bootstrap((idx0, idx1), rng)
            take1 = rng.choice(m, size=m, replace=True)
            draws[b] = fit_original_and_score(
                X0[take0], Y0[take0], T0[take0],
                X1[:m][take1], Y1[:m][take1],
                registry, model_m, model_e, seed + 1000 + m + b,
            )
        rows.append({
            "m": int(m),
            "porisk": float(point),
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)),
        })
    return rows


def make_stream(delta: float, seed: int, n0: int = 36, n1: int = 16, p: int = 4):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(size=(n0, p))
    T0 = np.zeros(n0, dtype=int)
    T0[n0 // 2:] = 1
    rng.shuffle(T0)
    beta = np.ones(p)
    Y0 = X0 @ beta + rng.normal(size=n0)
    X1 = rng.normal(size=(n1, p))
    Y1 = X1 @ beta + float(delta) * X1[:, 0] + rng.normal(size=n1)
    return X0, Y0, T0, X1, Y1


def plot_series(original_risk: float, series, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(series), figsize=(6.2 * len(series), 5.0), sharey=False)
    if len(series) == 1:
        axes = [axes]
    for ax, (title, rows) in zip(axes, series):
        m = np.array([r["m"] for r in rows])
        y = np.array([r["porisk"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        ax.fill_between(m, lo, hi, color="#1f77b4", alpha=0.18, label="95% online bootstrap CI")
        ax.plot(m, y, color="#1f77b4", lw=1.8, marker="o", label="online PO-risk, T=1")
        ax.axhline(original_risk, color="#d62728", ls="--", lw=1.2, label="original-batch PO-risk")
        ax.set_title(title)
        ax.set_xlabel("samples after the original batch (each T=1)")
        ax.set_ylabel("PO-risk  mean(tau^2)")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    registry = build_registry(ntree=30)
    model_m, model_e = "rf_regressor", "logistic_classifier"
    X0, Y0, T0, _, _ = make_stream(0.0, seed=11)
    base = fit_original_and_score(X0, Y0, T0, X0[:0], Y0[:0], registry, model_m, model_e, seed=11)
    print(f"original-batch PO-risk {base:.4f}  model_m={model_m}  model_e={model_e}")

    shifted = make_stream(2.5, seed=11)
    quiet = make_stream(0.0, seed=11)
    # same original batch; only the T=1 tail changes
    shift_rows = online_series(*shifted, registry, model_m, model_e, step=4, n_boot=12, seed=21)
    null_rows = online_series(*quiet, registry, model_m, model_e, step=4, n_boot=12, seed=22)
    for name, rows in (("concept", shift_rows), ("null", null_rows)):
        for row in rows:
            print(
                f"{name} T=1 count={row['m']} "
                f"PO-risk={row['porisk']:.4f} "
                f"CI=[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
            )
    out = Path("/opt/cursor/artifacts/online_porisk_registry_prototype.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_series(
        base,
        [("concept drift on T=1", shift_rows), ("no drift on T=1", null_rows)],
        str(out),
    )
    payload = {"original_porisk": base, "concept": shift_rows, "null": null_rows,
               "model_m": model_m, "model_e": model_e}
    Path("/opt/cursor/artifacts/online_porisk_registry_prototype.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    repo = Path("Python/results")
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "online_porisk_registry_prototype.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    print("PROTOTYPE_OK", out)


if __name__ == "__main__":
    main()
