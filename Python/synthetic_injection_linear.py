"""Inject a linear concept-drift shift on a random subset of features."""
import numpy as np


def synthetic_injection_linear(df1_X, df1_Y, df2_X, df2_Y, n_prop=0.3, seed=0):
    rng = np.random.default_rng(seed)
    p = df1_X.shape[1]
    n_shift = max(1, int(np.round(p * n_prop)))
    feature_ind = np.sort(rng.choice(p, n_shift, replace=False))
    beta = np.zeros(p, dtype=float)
    beta[feature_ind] = rng.uniform(0.5, 2.0, size=n_shift)

    df1 = np.hstack(
        [np.asarray(df1_X, dtype=float), np.asarray(df1_Y, dtype=float).reshape(-1, 1)]
    )
    df2_Y_shift = np.asarray(df2_Y, dtype=float).ravel() + np.asarray(df2_X, dtype=float) @ beta
    df2 = np.hstack([np.asarray(df2_X, dtype=float), df2_Y_shift.reshape(-1, 1)])
    return df1, df2, feature_ind
