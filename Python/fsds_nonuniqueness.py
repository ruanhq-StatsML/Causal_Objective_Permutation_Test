#!/usr/bin/env python3
"""Path-order non-uniqueness of distribution-shift attribution (MMD / PO-risk).

Following the ``fsds_nonuniqueness`` methodology: build random feature-addition
paths ``pi = np.random.permutation(p)``; along each path add one feature at a
time and record the metric on the growing feature block. Two metrics (concise,
trustworthy -- preferred over Shapley, which is only the path average as a
proxy):

    * MMD^2  for covariate shift  P(X);
    * PO-risk / R-risk for concept drift P(Y|X)  (the emphasis, per experiment).

``summarize_path_nonuniqueness`` then shows the key fact:

    all paths END at the same full-set value (the TOTAL shift is order-invariant
    and identifiable), but the incremental curves DIVERGE across orders (the
    per-feature accrual is path-dependent) -> the per-feature contribution is
    NOT uniquely attributable; only subset localization is well-posed.

No cross-validation: linear nuisances (well-specified for these DGPs) keep it
concise. DGPs are the repo's banded-Toeplitz covariate-shift / concept-drift
generators.
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.linalg import toeplitz
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge


# --------------------------------------------------------------------------- #
# DGPs (repo's banded-Toeplitz covariate-shift / concept-drift)
# --------------------------------------------------------------------------- #
def banded_toeplitz_corr(rho, p=20, bandwidth=6):
    corr = toeplitz(rho ** np.arange(p))
    return np.triu(np.tril(corr, bandwidth), -bandwidth)


def generate_covariate_shift_dgp(n=600, p=20, gamma=0.6, rho=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(p), banded_toeplitz_corr(rho, p), n)
    beta = np.zeros(p)
    beta[:4] = gamma
    shift = X[: n // 2, :4] @ beta[:4]
    X[: n // 2, :4] = X[: n // 2, :4] + np.tile(shift.reshape(-1, 1), (1, 4))
    Y = X @ beta + rng.normal(0, 1, n)
    W = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(int)
    return X, Y, W


def generate_concept_drift_dgp(n=600, p=20, delta_beta=0.6, rho=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(p), banded_toeplitz_corr(rho, p), n)
    Y = X @ np.ones(p) + rng.normal(0, 1, n)
    beta = np.zeros(p)
    beta[:4] = np.arange(4) * delta_beta
    Y[n // 2:] = Y[n // 2:] + X[n // 2:, :] @ beta          # mapping changes on new batch
    W = np.concatenate([np.zeros(n // 2), np.ones(n - n // 2)]).astype(int)
    return X, Y, W


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _gamma(Z):
    d = pdist(Z, "sqeuclidean")
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return 1.0 / med if med > 0 else 1.0


def _mmd2(A, B, g):
    m, n = len(A), len(B)
    Kaa, Kbb, Kab = (np.exp(-g * cdist(A, A, "sqeuclidean")),
                     np.exp(-g * cdist(B, B, "sqeuclidean")),
                     np.exp(-g * cdist(A, B, "sqeuclidean")))
    return float((Kaa.sum() - np.trace(Kaa)) / (m * (m - 1))
                 + (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1)) - 2 * Kab.mean())


def mmd_path(X, W, B=12, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    g = _gamma(Xs)
    a, b = Xs[W == 0], Xs[W == 1]
    p = X.shape[1]
    orig = _mmd2(a, b, g)
    paths = []
    for _ in range(B):
        pi = rng.permutation(p)
        curve = [_mmd2(a[:, pi[:s + 1]], b[:, pi[:s + 1]], g) for s in range(p)]
        paths.append({"feature_path": pi, "metric_path": np.array(curve)})
    return paths, orig


def porisk_path(X, Y, W, B=12, rng=None):
    """R-risk path with fixed LINEAR nuisances (no CV). R-risk falls as the tau
    model gains the features carrying the concept drift."""
    rng = np.random.default_rng() if rng is None else rng
    mu = LinearRegression().fit(X, Y).predict(X)
    e = LogisticRegression(max_iter=200).fit(X, W).predict_proba(X)[:, 1]
    e = np.clip(e, 0.02, 0.98)
    Yt, Wt = Y - mu, W - e
    mask = np.abs(Wt) > 1e-3
    z, sw = Yt[mask] / Wt[mask], Wt[mask] ** 2
    p = X.shape[1]

    def rrisk(cols):
        tau = Ridge(alpha=1.0).fit(X[mask][:, cols], z, sample_weight=sw)
        return float(np.mean((Yt - tau.predict(X[:, cols]) * Wt) ** 2))

    orig = rrisk(np.arange(p))
    paths = []
    for _ in range(B):
        pi = rng.permutation(p)
        curve = [rrisk(pi[:s + 1]) for s in range(p)]
        paths.append({"feature_path": pi, "metric_path": np.array(curve)})
    return paths, orig


# --------------------------------------------------------------------------- #
# Non-uniqueness summary (as in fsds_nonuniqueness)
# --------------------------------------------------------------------------- #
def summarize_path_nonuniqueness(paths, metric_orig, atol=1e-8):
    curves = np.stack([p["metric_path"] for p in paths], axis=0)
    b, p = curves.shape
    final = curves[:, -1]
    pair = np.mean([np.max(np.abs(curves[i] - curves[j]))
                    for i in range(b) for j in range(i + 1, b)])
    return {
        "n_paths": int(b), "n_steps": int(p), "metric_orig": float(metric_orig),
        "final_mean": float(final.mean()), "final_std": float(final.std()),
        "final_range": float(final.max() - final.min()),
        "final_all_equal_orig": bool(np.allclose(final, metric_orig, atol=atol)),
        "step_std_mean": float(curves.std(0).mean()),
        "step_std_max": float(curves.std(0).max()),
        "pairwise_curve_diff_mean": float(pair),
        "n_distinct_final_values": int(len(np.unique(np.round(final, 8)))),
    }


def _report(title, summ, metric_orig):
    print(f"\n[{title}]")
    print(f"  full-set metric (order-invariant TOTAL) = {metric_orig:+.5f}")
    print(f"  all paths end at the total?  final_all_equal_orig = "
          f"{summ['final_all_equal_orig']}  (range={summ['final_range']:.2e})")
    rel = summ["step_std_mean"] / (abs(metric_orig) + 1e-12)
    print(f"  incremental curves diverge:  step_std_mean = {summ['step_std_mean']:.5f} "
          f"({rel:.0%} of total),  step_std_max = {summ['step_std_max']:.5f}")
    print(f"  pairwise max curve gap (mean over path pairs) = "
          f"{summ['pairwise_curve_diff_mean']:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--rho", type=float, default=0.3)
    ap.add_argument("--b-paths", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    print("=" * 80)
    print("Path-order NON-UNIQUENESS of shift attribution (MMD / PO-risk)")
    print("=" * 80)
    print("total shift is order-invariant (all paths end equal); per-feature "
          "accrual is path-dependent -> not uniquely attributable.")

    rng = np.random.default_rng(args.seed)
    # concept drift (the emphasis) -- PO-/R-risk
    Xc, Yc, Wc = generate_concept_drift_dgp(args.n, args.p, delta_beta=0.6,
                                            rho=args.rho, seed=args.seed)
    pc, oc = porisk_path(Xc, Yc, Wc, B=args.b_paths, rng=rng)
    sc = summarize_path_nonuniqueness(pc, oc)
    _report("CONCEPT DRIFT  (PO-/R-risk path)", sc, oc)

    # covariate shift -- MMD
    Xv, Yv, Wv = generate_covariate_shift_dgp(args.n, args.p, gamma=0.6,
                                              rho=args.rho, seed=args.seed + 1)
    pv, ov = mmd_path(Xv, Wv, B=args.b_paths, rng=rng)
    sv = summarize_path_nonuniqueness(pv, ov)
    _report("COVARIATE SHIFT  (MMD path)", sv, ov)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("  * TOTAL shift is identifiable (order-invariant full-set metric).")
    print("  * PER-FEATURE contribution is NOT unique: incremental curves depend "
          "on the addition order (Shapley is just their average -- a proxy).")
    print("  * Therefore: no unique per-feature attribution at ANY level; only "
          "SUBSET localization is well-posed. Emphasis: concept drift.")
    ok = (sc["final_all_equal_orig"] and sv["final_all_equal_orig"]
          and sc["step_std_mean"] > 0 and sv["step_std_mean"] > 0)
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}  "
          f"(total order-invariant, path attribution non-unique)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
