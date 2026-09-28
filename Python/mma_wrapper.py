"""Thin MMA wrapper on the existing FSDS metrics.

``np.vstack`` -> ``FSDS_runner`` -> patch indices ``[[h, w], ...]`` and token
indices ``L`` -> perturb those tokens -> difference of MMD / PO-risk / HSIC.

The three class methods are the leave-one-token-out drops already used for
PO-risk (``obs - risk without j``) and LOGO-MMD (``mmd_full - mmd_without``).
HSIC uses the same trace formula. Post-hoc localization is the second stage:
inside the selected token set, perturb and recompute.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist, pdist

from graph_fsds.fsds_core import _po_risk


def _as_tokens(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        return X[:, :, None]
    if X.ndim != 3:
        raise ValueError("tokens must have shape (n, n_tokens, dim) or (n, p)")
    return X


def _flat(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float).reshape(X.shape[0], -1)


def _median_gamma(Z) -> float:
    """RBF gamma from the median heuristic on pooled pairwise distances."""
    d = pdist(Z, metric="sqeuclidean")
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return 1.0 / med if med > 0 else 1.0


def mmd2_unbiased(X, Y, gamma) -> float:
    """Unbiased estimator of squared MMD with an RBF kernel."""
    m, n = len(X), len(Y)
    Kxx = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    Kyy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    Kxy = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    s_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    s_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    s_xy = Kxy.mean()
    return float(s_xx + s_yy - 2.0 * s_xy)


def _rbf(x) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    d = cdist(z, z, "sqeuclidean")
    pos = d[d > 0]
    med = np.median(pos) if pos.size else 1.0
    return np.exp(-d / (med if med > 0 else 1.0))


def hsic(x, y) -> float:
    n = len(x)
    Kx, Ky = _rbf(x), _rbf(y)
    H = np.eye(n) - 1.0 / n
    return float(np.trace(Kx @ H @ Ky @ H) / (n - 1) ** 2)


def patch_of(token_index: int, grid) -> list:
    """Token layout is ``t * (n_h * n_w) + h * n_w + w``."""
    _n_t, n_h, n_w = grid
    spatial = int(token_index) % (n_h * n_w)
    h, w = divmod(spatial, n_w)
    return [int(h), int(w)]


def neighborhood_indices(adjacency, seeds) -> list:
    """Closed 1-hop neighborhood: the seed nodes plus their neighbors."""
    A = np.asarray(adjacency)
    chosen = {int(s) for s in np.atleast_1d(seeds)}
    for s in list(chosen):
        chosen.update(int(v) for v in np.flatnonzero(A[int(s)]))
    return sorted(chosen)


def perturb_neighborhood(x_ref, x_new, neigh) -> np.ndarray:
    """Put the reference features back on the neighborhood nodes of the new graph."""
    pert = np.array(x_new, copy=True, dtype=float)
    idx = np.asarray(list(neigh), dtype=int)
    pert[idx] = np.asarray(x_ref, dtype=float)[idx]
    return pert


def mmd_neighborhood_perturbation(x_ref, x_new, neigh) -> dict:
    """MMD drop from moving one neighborhood back to the reference snapshot.

    ``mmd2_unbiased`` compares the two node-feature clouds. Gamma is the median
    heuristic on the original pair and is held fixed, so the drop is the effect
    of moving those nodes, not a change of bandwidth. LOGO-MMD drops columns
    and therefore refits gamma; this perturbation keeps the coordinates.
    """
    ref = np.asarray(x_ref, dtype=float)
    new = np.asarray(x_new, dtype=float)
    gamma = _median_gamma(np.vstack([ref, new]))
    before = mmd2_unbiased(ref, new, gamma)
    after = mmd2_unbiased(ref, perturb_neighborhood(ref, new, neigh), gamma)
    return {
        "neighborhood": [int(i) for i in neigh],
        "gamma": float(gamma),
        "mmd_before": float(before),
        "mmd_after": float(after),
        "delta": float(before - after),
    }


def _perturb_tokens(X: np.ndarray, W: np.ndarray, L) -> np.ndarray:
    """Replace the query-batch tokens in ``L`` with the reference-batch mean."""
    pert = np.array(X, copy=True, dtype=float)
    ref = np.flatnonzero(np.asarray(W) == 0)
    qry = np.flatnonzero(np.asarray(W) == 1)
    idx = np.asarray(L, dtype=int)
    mu = pert[ref][:, idx, :].mean(axis=0)
    pert[qry[:, None], idx[None, :], :] = mu
    return pert


class MMD:
    def __call__(self, X, W) -> float:
        Z = _flat(_as_tokens(X))
        W = np.asarray(W, dtype=int).ravel()
        gamma = _median_gamma(Z)
        return mmd2_unbiased(Z[W == 0], Z[W == 1], gamma)

    def MMD_LOCO(self, X, W) -> np.ndarray:
        X = _as_tokens(X)
        full = self(X, W)
        vimp = np.empty(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            vimp[j] = full - self(np.delete(X, j, axis=1), W)
        return vimp


class HSIC:
    def __call__(self, X, W) -> float:
        return hsic(_flat(_as_tokens(X)), np.asarray(W, dtype=float).ravel())

    def HSIC_LOCO(self, X, W) -> np.ndarray:
        X = _as_tokens(X)
        full = self(X, W)
        vimp = np.empty(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            vimp[j] = full - self(np.delete(X, j, axis=1), W)
        return vimp


class PORisk:
    def __init__(self, n_folds: int = 2, domain_model: str = "logistic", seed: int = 2026):
        self.n_folds = n_folds
        self.domain_model = domain_model
        self.seed = seed

    def __call__(self, X, Y, W, seed: Optional[int] = None) -> float:
        return _po_risk(
            _flat(_as_tokens(X)),
            Y,
            W,
            n_folds=self.n_folds,
            domain_model=self.domain_model,
            clip_e=1e-2,
            seed=self.seed if seed is None else seed,
        )

    def PORisk_LOCO(self, X, Y, W) -> np.ndarray:
        X = _as_tokens(X)
        full = self(X, Y, W, self.seed)
        vimp = np.empty(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            vimp[j] = full - self(np.delete(X, j, axis=1), Y, W, self.seed + 11 + j)
        return vimp


def permutation_token_importance(X, W, rng, n_repeats: int = 1) -> np.ndarray:
    """MMD drop when token ``j`` is shuffled. Same difference as permutation importance."""
    X = _as_tokens(X)
    W = np.asarray(W, dtype=int).ravel()
    score = MMD()
    base = score(X, W)
    vimp = np.empty(X.shape[1], dtype=float)
    n = X.shape[0]
    for j in range(X.shape[1]):
        drops = np.empty(n_repeats, dtype=float)
        for _ in range(n_repeats):
            Xp = np.array(X, copy=True)
            Xp[:, j, :] = X[rng.permutation(n), j, :]
            drops[_] = base - score(Xp, W)
        vimp[j] = float(drops.mean())
    return vimp


def online_pfi(
    tokens,
    grid,
    ref_n: int = 8,
    recent_n: int = 6,
    step: int = 1,
    seed: int = 2026,
    n_repeats: int = 1,
) -> list:
    """Streaming permutation importance.

    The window is the one in ``pvalue_stream`` (RAP branch
    ``Python/online_drift_detectors.py``): a fixed reference batch, then the
    trailing recent batch. Each step shuffles one token and records the MMD drop.
    """
    tokens = _as_tokens(tokens)
    n = tokens.shape[0]
    if ref_n + recent_n > n:
        raise ValueError("ref_n + recent_n exceeds the stream length")
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(ref_n + recent_n, n + 1, step):
        idx = list(range(ref_n)) + list(range(m - recent_n, m))
        W = np.array([0] * ref_n + [1] * recent_n, dtype=int)
        vimp = permutation_token_importance(tokens[idx], W, rng, n_repeats=n_repeats)
        top = int(np.argmax(vimp))
        rows.append({
            "m": int(m),
            "top_token": top,
            "top_patch": patch_of(top, grid),
            "top_pfi": float(vimp[top]),
            "vimp": vimp,
        })
    return rows


def window_porisk(X, Y, W) -> float:
    """Full-set risk inside ``fsds_nonuniqueness.porisk_path``.

    Linear outcome fit, logistic propensity, ridge tau. No cross-fitting.
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

    X = _flat(_as_tokens(X))
    Y = np.asarray(Y, dtype=float).ravel()
    W = np.asarray(W, dtype=int).ravel()
    mu = LinearRegression().fit(X, Y).predict(X)
    e = LogisticRegression(max_iter=400).fit(X, W).predict_proba(X)[:, 1]
    e = np.clip(e, 0.02, 0.98)
    Yt = Y - mu
    Wt = W.astype(float) - e
    mask = np.abs(Wt) > 1e-3
    if int(mask.sum()) < 3:
        return float(np.mean(Yt ** 2))
    z = Yt[mask] / Wt[mask]
    sw = Wt[mask] ** 2
    tau = Ridge(alpha=1.0).fit(X[mask], z, sample_weight=sw)
    return float(np.mean((Yt - tau.predict(X) * Wt) ** 2))


def online_bootstrap_ci(
    X,
    Y,
    ref_n: int = 8,
    recent_n: int = 6,
    step: int = 2,
    n_boot: int = 40,
    seed: int = 2026,
) -> list:
    """Online PO-risk with the bootstrap percentile interval.

    The window is the ``pvalue_stream`` reference plus trailing recent batch.
    Resamples are the stratified bootstrap in ``_resample_indices``. The
    interval is the 2.5 and 97.5 percentiles used by ``resample_vimp``.
    """
    from fsds_vimp_inference import _resample_indices

    X = _as_tokens(X)
    Y = np.asarray(Y, dtype=float).ravel()
    n = X.shape[0]
    if ref_n + recent_n > n:
        raise ValueError("ref_n + recent_n exceeds the stream length")
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(ref_n + recent_n, n + 1, step):
        idx = np.array(list(range(ref_n)) + list(range(m - recent_n, m)))
        Xw, yw = X[idx], Y[idx]
        w = np.array([0] * ref_n + [1] * recent_n, dtype=int)
        point = window_porisk(Xw, yw, w)
        draws = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            bidx = _resample_indices(w, rng, "bootstrap", 1.0, 0.0)
            draws[b] = window_porisk(Xw[bidx], yw[bidx], w[bidx])
        rows.append({
            "m": int(m),
            "porisk": float(point),
            "ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)),
        })
    return rows


def plot_online_porisk(series, path: str, enter_m: int) -> None:
    """Path figure: PO-risk line and the online bootstrap band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(series), figsize=(6.2 * len(series), 5.2), sharey=False)
    if len(series) == 1:
        axes = [axes]
    for ax, (title, rows) in zip(axes, series):
        m = np.array([r["m"] for r in rows])
        y = np.array([r["porisk"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        ax.fill_between(m, lo, hi, color="#1f77b4", alpha=0.15, label="95% online bootstrap CI")
        ax.plot(m, y, color="#1f77b4", lw=1.8, label="online PO-risk")
        ax.axvline(enter_m, color="#d62728", ls="--", lw=1.2, label="query enters recent window")
        ax.set_title(title)
        ax.set_xlabel("stream step m")
        ax.set_ylabel("PO-risk")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def FSDS_runner(X, W, grid, top_k: int = 4):
    """Leave-one-token-out MMD. Returns patch coordinates and token indices ``L``."""
    score = MMD().MMD_LOCO(X, W)
    L = [int(i) for i in np.argsort(-score)[:top_k]]
    patches = [patch_of(i, grid) for i in L]
    return patches, L


def post_hoc_localization(X, Y, W, L, porisk: Optional[PORisk] = None) -> dict:
    """Stage B: perturb the selected tokens, then LOCO inside that set."""
    porisk = porisk or PORisk()
    mmd, hsic_fn = MMD(), HSIC()
    base = {"MMD": mmd(X, W), "HSIC": hsic_fn(X, W), "PORisk": porisk(X, Y, W)}
    pert = _perturb_tokens(X, W, L)
    after = {"MMD": mmd(pert, W), "HSIC": hsic_fn(pert, W), "PORisk": porisk(pert, Y, W)}
    delta = {name: base[name] - after[name] for name in base}
    sub = np.asarray(X, dtype=float)[:, list(L)]
    loco = {
        "MMD": mmd.MMD_LOCO(sub, W),
        "HSIC": hsic_fn.HSIC_LOCO(sub, W),
        "PORisk": porisk.PORisk_LOCO(sub, Y, W),
    }
    return {"base": base, "after": after, "delta": delta, "loco": loco}


def graph_neighborhood_delta(x_ref, x_new, y_ref, y_new, neigh) -> dict:
    """MMD / PO-risk / HSIC drop after the neighborhood perturbation."""
    y = np.concatenate([np.asarray(y_ref, dtype=float), np.asarray(y_new, dtype=float)])
    w = np.concatenate([
        np.zeros(len(x_ref), dtype=int),
        np.ones(len(x_new), dtype=int),
    ])

    def pack(x_query):
        return np.vstack([x_ref, x_query])

    base = pack(x_new)
    after = pack(perturb_neighborhood(x_ref, x_new, neigh))
    mmd_step = mmd_neighborhood_perturbation(x_ref, x_new, neigh)
    hsic_fn, porisk = HSIC(), window_porisk
    return {
        "neighborhood": list(neigh),
        "mmd": mmd_step,
        "delta": {
            "MMD": mmd_step["delta"],
            "PORisk": porisk(base, y, w) - porisk(after, y, w),
            "HSIC": hsic_fn(base, w) - hsic_fn(after, w),
        },
    }


def mma_wrapper(z_ref, z_query, Y, grid, top_k: int = 4, porisk: Optional[PORisk] = None) -> dict:
    X = np.vstack([z_ref, z_query])
    W = np.concatenate([
        np.zeros(len(z_ref), dtype=int),
        np.ones(len(z_query), dtype=int),
    ])
    Y = np.asarray(Y, dtype=float).ravel()
    patches, L = FSDS_runner(X, W, grid, top_k=top_k)
    localized = post_hoc_localization(X, Y, W, L, porisk=porisk)
    return {"patch_indices": patches, "L": L, **localized}
