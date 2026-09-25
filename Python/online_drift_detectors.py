"""Online drift detectors for the clever-covariate RAP stream.

Two streaming signals, both computed from the finished-beat stream of one search:

* ``online_rf_perm_pvalue`` -- an *online RandomForest permutation* two-sample test
  (a classifier two-sample test, "onlineRFPerm"). At a given step it asks whether a
  RandomForest can separate a *reference* batch of early finished beats from a
  *recent* batch, using the clever-covariate features ``[e, H, |H|]``. The
  statistic is the classifier's out-of-bag accuracy; the null is built by permuting
  the batch labels ``n_perm`` times (the same permute-then-refit recipe as
  ``RRPerm``/``DRPerm`` in this repo), giving ``p = (1 + #{perm >= obs}) / (1 + n_perm)``.

* ``rolling_stats`` -- an online rolling mean / rolling standard deviation of the
  outcome (the "history" layer statistic from the ToT interpretation note).

The point of adding these is that the RAP switch no longer needs the ad-hoc
last-four-loss margin rule: the FSDS permutation p-value crossing ``alpha`` is a
calibrated drift trigger, so the same distribution-shift test this repository is
built around is what fires the RAP re-route. If scikit-learn is unavailable the
p-value falls back to a numpy standardized-mean-gap statistic under the identical
permutation null, so the pipeline still runs everywhere.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from rap_clever_covariate_guide import Beat, DEFAULT_CLIP, clever_covariate, clip_prior

try:  # RandomForest is the concrete "RF" learner, as in RRPerm/DRPerm.
    from sklearn.ensemble import RandomForestClassifier

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - exercised only when sklearn is missing
    _HAVE_SKLEARN = False

__all__ = [
    "rolling_stats",
    "beat_features",
    "online_rf_perm_pvalue",
    "pvalue_stream",
]


def rolling_stats(values: Sequence[float], window: int) -> Dict[str, List[float]]:
    """Online rolling mean and (population) rolling std over a trailing window."""
    means: List[float] = []
    stds: List[float] = []
    for i in range(len(values)):
        chunk = list(values[max(0, i - window + 1): i + 1])
        m = sum(chunk) / len(chunk)
        var = sum((v - m) ** 2 for v in chunk) / len(chunk)
        means.append(m)
        stds.append(var ** 0.5)
    return {"mean": means, "std": stds}


def beat_features(beats: Sequence[Beat], clip: Sequence[float] = DEFAULT_CLIP) -> np.ndarray:
    """Per-beat clever-covariate features ``[e, H, |H|]`` used by the C2ST."""
    rows = []
    for b in beats:
        e = clip_prior(b.prior, clip)
        h = clever_covariate(b.outcome, b.prior, clip)
        rows.append([e, h, abs(h)])
    return np.asarray(rows, dtype=float)


def _rf_oob_accuracy(X: np.ndarray, w: np.ndarray, seed: int) -> float:
    clf = RandomForestClassifier(
        n_estimators=80,
        oob_score=True,
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, w)
        score = getattr(clf, "oob_score_", None)
    if score is None or score != score:  # NaN guard for tiny samples
        return float((clf.predict(X) == w).mean())
    return float(score)


def _numpy_gap_statistic(X: np.ndarray, w: np.ndarray) -> float:
    """Fallback C2ST statistic: mean standardized gap between the two batches."""
    a = X[w == 0]
    b = X[w == 1]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    pooled = X.std(axis=0)
    pooled = np.where(pooled < 1e-9, 1.0, pooled)
    return float(np.mean(np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled))


def online_rf_perm_pvalue(
    X: np.ndarray,
    w: np.ndarray,
    n_perm: int = 80,
    seed: int = 2026,
) -> Dict[str, float]:
    """Permutation two-sample p-value separating batch ``w`` from features ``X``."""
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=int)
    rng = np.random.default_rng(seed)
    if _HAVE_SKLEARN:
        observed = _rf_oob_accuracy(X, w, seed)
        null = np.array([_rf_oob_accuracy(X, rng.permutation(w), seed + b + 1) for b in range(n_perm)])
    else:  # pragma: no cover - only when sklearn is missing
        observed = _numpy_gap_statistic(X, w)
        null = np.array([_numpy_gap_statistic(X, rng.permutation(w)) for _ in range(n_perm)])
    p_value = (1.0 + float(np.sum(null >= observed))) / (1.0 + n_perm)
    return {"statistic": float(observed), "p_value": float(p_value)}


def pvalue_stream(
    beats: Sequence[Beat],
    ref_n: int = 8,
    recent_n: int = 6,
    window: int = 4,
    n_perm: int = 80,
    alpha: float = 0.05,
    seed: int = 2026,
    clip: Sequence[float] = DEFAULT_CLIP,
    also_at: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Stream the online RF-permutation p-value and rolling stats over the beats.

    Returns per-step p-values (indexed by the number ``m`` of finished beats), the
    rolling mean/std of the outcome, and ``detect_index`` -- the first ``m`` at which
    the p-value drops below ``alpha`` and enough recent beats exist to be tested.
    """
    outcomes = [b.outcome for b in beats]
    roll = rolling_stats(outcomes, window)
    feats = beat_features(beats, clip)
    n = len(beats)
    pvals: List[float] = [float("nan")] * (n + 1)
    w = np.array([0] * ref_n + [1] * recent_n, dtype=int)

    def p_at(m: int) -> float:
        idx = list(range(0, ref_n)) + list(range(m - recent_n, m))
        return online_rf_perm_pvalue(feats[idx], w, n_perm=n_perm, seed=seed)["p_value"]

    # Walk forward and stop as soon as drift is detected; this keeps the RF cost
    # bounded to the steps before detection. We always report p at the drift point
    # and at the final step regardless of the early stop.
    detect_index = -1
    for m in range(ref_n + recent_n, n + 1):
        pvals[m] = p_at(m)
        if detect_index < 0 and pvals[m] < alpha:
            detect_index = m
            break
    extra = list(also_at or []) + [n]
    for m in extra:
        if ref_n + recent_n <= m <= n and pvals[m] != pvals[m]:  # NaN -> compute
            pvals[m] = p_at(m)
    return {
        "pvals": pvals,
        "roll_mean": roll["mean"],
        "roll_std": roll["std"],
        "detect_index": detect_index,
        "alpha": alpha,
        "learner": "random_forest" if _HAVE_SKLEARN else "numpy_gap",
    }
