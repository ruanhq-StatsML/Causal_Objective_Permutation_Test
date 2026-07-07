"""Minimal RBF MMD helper used by realdata_vimp_benchmark.compute_loco_mmd_batches."""
import numpy as np


def _rbf_gamma(X, Y, max_samples=400):
    n_x, n_y = X.shape[0], Y.shape[0]
    if n_x + n_y > max_samples:
        rng = np.random.default_rng(0)
        idx_x = rng.choice(n_x, size=min(n_x, max_samples // 2), replace=False)
        idx_y = rng.choice(n_y, size=min(n_y, max_samples // 2), replace=False)
        z = np.vstack([X[idx_x], Y[idx_y]])
    else:
        z = np.vstack([X, Y])
    diff = z[:, None, :] - z[None, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    positive = sq_dists[sq_dists > 0]
    if positive.size == 0:
        return 1.0
    return 1.0 / float(np.median(positive))


def rbf_mmd2(X, Y, gamma=None):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if gamma is None:
        gamma = _rbf_gamma(X, Y)
    XX = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    YY = np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    XY = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
    return float(
        np.mean(np.exp(-gamma * XX))
        + np.mean(np.exp(-gamma * YY))
        - 2.0 * np.mean(np.exp(-gamma * XY))
    )


class MMD:
    """RBF MMD wrapper matching benchmark_shapleyShift.py call pattern."""

    def __init__(self, compute_kernel="rbf"):
        if compute_kernel != "rbf":
            raise ValueError(f"unsupported kernel: {compute_kernel}")
        self.compute_kernel = compute_kernel
        self._gamma = None

    def __call__(self, X, Y):
        if self._gamma is None:
            self._gamma = _rbf_gamma(np.asarray(X), np.asarray(Y))
        stat = rbf_mmd2(X, Y, gamma=self._gamma)
        return stat, None
