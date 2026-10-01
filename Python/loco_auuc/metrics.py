"""AUUC / Qini / uplift@k — standard cumulative-gain curve (not the buggy *k scaling)."""

from __future__ import annotations

import numpy as np


def qini_curve(y: np.ndarray, treatment: np.ndarray, uplift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Cumulative Qini-style gain when targeting top fractions by predicted uplift.

    At prefix k (sorted by uplift desc):
      gain_k = (mean Y | T=1, in prefix) - (mean Y | T=0, in prefix) * (n_t + n_c in prefix)
    Returns (fractions in (0,1], gain_values).
    """
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(treatment, dtype=int).ravel()
    uplift = np.asarray(uplift, dtype=float).ravel()
    order = np.argsort(-uplift)
    y, t = y[order], t[order]
    n = len(y)

    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))

    rt = np.divide(y_t, n_t, out=np.zeros(n, dtype=float), where=n_t > 0)
    rc = np.divide(y_c, n_c, out=np.zeros(n, dtype=float), where=n_c > 0)
    gain = (rt - rc) * (n_t + n_c)
    frac = np.arange(1, n + 1, dtype=float) / n
    return frac, gain


def auuc(y: np.ndarray, treatment: np.ndarray, uplift: np.ndarray) -> float:
    """Area under cumulative gain curve, normalized by n."""
    frac, gain = qini_curve(y, treatment, uplift)
    area = np.trapezoid(gain, frac) if hasattr(np, "trapezoid") else np.trapz(gain, frac)
    return float(area / max(len(y), 1))


def auuc_random_baseline(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift: np.ndarray,
    *,
    n_draws: int = 50,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    scores = [auuc(y, treatment, rng.permutation(uplift)) for _ in range(n_draws)]
    return float(np.mean(scores))


def qini_coefficient(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift: np.ndarray,
    *,
    n_random: int = 50,
    seed: int = 42,
) -> float:
    return auuc(y, treatment, uplift) - auuc_random_baseline(
        y, treatment, uplift, n_draws=n_random, seed=seed
    )


def uplift_at_k(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift: np.ndarray,
    k_ratio: float = 0.1,
) -> float:
    order = np.argsort(-uplift)
    k = max(int(len(y) * k_ratio), 1)
    y_k = y[order][:k]
    t_k = treatment[order][:k]
    nt = max((t_k == 1).sum(), 1)
    nc = max((t_k == 0).sum(), 1)
    return float(y_k[t_k == 1].sum() / nt - y_k[t_k == 0].sum() / nc)


def evaluate_window(y: np.ndarray, treatment: np.ndarray, uplift: np.ndarray, *, seed: int = 42) -> dict:
    return {
        "auuc": auuc(y, treatment, uplift),
        "qini": qini_coefficient(y, treatment, uplift, seed=seed),
        "uplift_at_10pct": uplift_at_k(y, treatment, uplift, 0.1),
        "uplift_at_20pct": uplift_at_k(y, treatment, uplift, 0.2),
    }


def bootstrap_auuc(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(auuc(y[idx], treatment[idx], uplift[idx]))
    vals = np.asarray(vals)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "ci_lower": float(np.percentile(vals, 2.5)),
        "ci_upper": float(np.percentile(vals, 97.5)),
    }
