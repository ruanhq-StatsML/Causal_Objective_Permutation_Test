#!/usr/bin/env python3
"""
Stationary-DGP false-alarm (type-I error) study for the online feature-blending
pipeline:

    sliding/decayed window (L_decay)
      -> Fisher pre-screen (p -> survivors, grouped into business blocks)
      -> incremental RF + OOB block-permutation importance
      -> per-feature / per-group z-scores
      -> recency-weighted Stouffer aggregation (weights = L_decay)
      -> group-level test + BH across groups (hierarchical gating)
      -> survivors -> meta-learner VI (R-risk / PO-risk, a la RRPerm/DRPerm)
      -> output: covariate component (domain/MMD) + concept component (R-risk)

Everything below is run under H0 (a *stationary* DGP: no covariate shift,
no concept drift, both batches share the same P(X) and P(Y|X)). A well
calibrated detector should fire at the nominal level alpha; anything that fires
much more often is leaking false alarms. We decompose the false-alarm rate (FAR)
by *driver*:

  Exp A (feature multiplicity): with p features and 10 business groups, compare
         - *-ANY        : reject if ANY feature is individually significant (no
                          multiplicity control)
         - *-BH         : Benjamini-Hochberg across features
         - *-GROUP-BH   : hierarchical gating -- BH across the 10 group-level
                          max-stat permutation p-values
         - *-MAXPERM    : single global statistic = max over features, calibrated
                          by a label-shuffle permutation null
         for both the Fisher pre-screen and the RF-OOB permutation screen.

  Exp B (window dependence from L_decay): a single feature observed over K
         windows, combined with recency-weighted Stouffer. Disjoint windows are
         independent; a *sliding* (overlapping) decayed window induces positive
         autocorrelation that naive Stouffer (independence assumption) ignores.

  Exp C (global permute-then-refit calibration): MMD, domain-AUC, R-risk,
         PO-risk -- each a single global statistic with a permutation / refit
         null. These are the "safe" detectors.

The DGP mirrors Python/shift_dgp_whole.py (m0 outcome surface) with
x_shift_type="none", concept_drift_type="none".
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from joblib import Parallel, delayed

    _HAVE_JOBLIB = True
except Exception:  # pragma: no cover
    _HAVE_JOBLIB = False


# --------------------------------------------------------------------------- #
# Stationary DGP (H0): both batches identically distributed.
# --------------------------------------------------------------------------- #
def m0(X: np.ndarray) -> np.ndarray:
    """Outcome surface copied from shift_dgp_whole.ShiftDGP.m0 (first 6 cols)."""
    return (
        0.5 * X[:, 0]
        + 0.75 * X[:, 1]
        - 0.25 * X[:, 2]
        + 0.25 * np.sin(X[:, 3])
        + 0.75 * X[:, 4] * X[:, 5]
    )


def gen_stationary(n_ref: int, n_new: int, p: int, noise_sd: float, rng: np.random.Generator):
    """Pooled (X, Y, W) with NO covariate shift and NO concept drift."""
    X = rng.normal(0.0, 1.0, size=(n_ref + n_new, max(p, 6)))
    X = X[:, :p] if p >= 6 else X
    if p < 6:  # keep m0 well defined
        X = np.hstack([X, rng.normal(0.0, 1.0, size=(n_ref + n_new, 6 - p))])
    Y = m0(X) + rng.normal(0.0, noise_sd, size=X.shape[0])
    W = np.concatenate([np.zeros(n_ref, dtype=int), np.ones(n_new, dtype=int)])
    perm = rng.permutation(X.shape[0])
    return X[perm], Y[perm], W[perm]


# --------------------------------------------------------------------------- #
# Multiplicity helpers.
# --------------------------------------------------------------------------- #
def bh_reject(pvals: np.ndarray, alpha: float) -> bool:
    """Benjamini-Hochberg: True if ANY hypothesis is rejected at FDR alpha."""
    p = np.sort(np.asarray(pvals, dtype=float))
    m = p.size
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p <= thresh
    return bool(passed.any())


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- #
# Exp A: Fisher pre-screen false alarms.
# --------------------------------------------------------------------------- #
def fisher_two_sample_t(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Per-feature |Welch t| between the two batches (a Fisher-style score)."""
    x0 = X[W == 0]
    x1 = X[W == 1]
    t, _ = stats.ttest_ind(x0, x1, axis=0, equal_var=False)
    return np.abs(np.nan_to_num(t))


def fisher_rules(X, W, p_groups, alpha, n_perm, rng) -> dict[str, bool]:
    """Derive ANY / BH / GROUP-BH / MAXPERM decisions from one label-shuffle null."""
    obs_t = fisher_two_sample_t(X, W)
    p = obs_t.size

    # Analytic per-feature two-sided p-values for ANY / BH.
    x0, x1 = X[W == 0], X[W == 1]
    _, pvals = stats.ttest_ind(x0, x1, axis=0, equal_var=False)
    pvals = np.nan_to_num(pvals, nan=1.0)

    # Label-shuffle permutation null for the max statistic (global & per-group).
    group_ids = np.repeat(np.arange(len(p_groups)), [len(g) for g in p_groups])
    null_global_max = np.empty(n_perm)
    null_group_max = np.empty((n_perm, len(p_groups)))
    for b in range(n_perm):
        Wp = rng.permutation(W)
        tb = fisher_two_sample_t(X, Wp)
        null_global_max[b] = tb.max()
        for gi, cols in enumerate(p_groups):
            null_group_max[b, gi] = tb[cols].max()

    obs_group_max = np.array([obs_t[cols].max() for cols in p_groups])
    group_pvals = np.array(
        [(1 + np.sum(null_group_max[:, gi] >= obs_group_max[gi])) / (n_perm + 1) for gi in range(len(p_groups))]
    )
    global_max_p = (1 + np.sum(null_global_max >= obs_t.max())) / (n_perm + 1)

    return {
        "Fisher-ANY": bool((pvals < alpha).any()),
        "Fisher-BH": bh_reject(pvals, alpha),
        "Fisher-GROUP-BH": bh_reject(group_pvals, alpha),
        "Fisher-MAXPERM": bool(global_max_p < alpha),
    }


# --------------------------------------------------------------------------- #
# Exp A: RF-OOB permutation-importance screen false alarms.
# --------------------------------------------------------------------------- #
def rf_oob_importance(X, W, n_estimators, rng_seed) -> tuple[np.ndarray, RandomForestClassifier]:
    """
    OOB permutation importance of a domain classifier (old vs new).
    importance_j = OOB-accuracy(unpermuted) - OOB-accuracy(feature j permuted),
    evaluated on each tree's own out-of-bag rows (no separate holdout -- the RF
    property that makes it the natural streaming choice).
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        bootstrap=True,
        oob_score=False,
        max_features="sqrt",
        n_jobs=1,
        random_state=rng_seed,
    )
    rf.fit(X, W)
    n, p = X.shape
    n_samples = n
    rs = np.random.RandomState(rng_seed + 7)
    importances = np.zeros(p)
    base_correct = np.zeros(n)
    base_count = np.zeros(n)
    perm_correct = np.zeros((p, n))
    for est in rf.estimators_:
        unsampled = _oob_indices(est, n_samples, n)
        if unsampled.size == 0:
            continue
        Xo = X[unsampled]
        yo = W[unsampled]
        pred = est.predict(Xo)
        base_correct[unsampled] += (pred == yo)
        base_count[unsampled] += 1
        for j in range(p):
            Xp = Xo.copy()
            Xp[:, j] = Xp[rs.permutation(Xo.shape[0]), j]
            perm_correct[j, unsampled] += (est.predict(Xp) == yo)
    mask = base_count > 0
    base_acc = base_correct[mask].sum() / base_count[mask].sum()
    for j in range(p):
        perm_acc = perm_correct[j, mask].sum() / base_count[mask].sum()
        importances[j] = base_acc - perm_acc
    return importances, rf


def _oob_indices(estimator, n_samples_bootstrap, n):
    from sklearn.ensemble._forest import _generate_unsampled_indices

    try:
        return _generate_unsampled_indices(estimator.random_state, n, n_samples_bootstrap)
    except Exception:
        return np.arange(n)


def rf_rules(X, W, p_groups, alpha, n_perm, n_estimators, rng, base_seed) -> dict[str, bool]:
    obs_imp, _ = rf_oob_importance(X, W, n_estimators, base_seed)
    p = obs_imp.size
    null_imp = np.empty((n_perm, p))
    for b in range(n_perm):
        Wp = rng.permutation(W)
        null_imp[b], _ = rf_oob_importance(X, Wp, n_estimators, base_seed + 1000 + b)

    # Per-feature permutation p-values (one-sided: importance > 0).
    per_feat_p = np.array([(1 + np.sum(null_imp[:, j] >= obs_imp[j])) / (n_perm + 1) for j in range(p)])

    # Global and per-group max-stat permutation nulls.
    null_global_max = null_imp.max(axis=1)
    global_max_p = (1 + np.sum(null_global_max >= obs_imp.max())) / (n_perm + 1)
    group_pvals = []
    for cols in p_groups:
        obs_g = obs_imp[cols].max()
        null_g = null_imp[:, cols].max(axis=1)
        group_pvals.append((1 + np.sum(null_g >= obs_g)) / (n_perm + 1))
    group_pvals = np.array(group_pvals)

    return {
        "RFPerm-ANY": bool((per_feat_p < alpha).any()),
        "RFPerm-BH": bh_reject(per_feat_p, alpha),
        "RFPerm-GROUP-BH": bh_reject(group_pvals, alpha),
        "RFPerm-MAXPERM": bool(global_max_p < alpha),
    }


# --------------------------------------------------------------------------- #
# Exp B: recency-weighted Stouffer over windows (single feature).
# --------------------------------------------------------------------------- #
def stouffer_window_rules(K, m, stride, decay, alpha, rng) -> dict[str, bool]:
    """
    Single stationary feature observed as a stream. Each window produces a
    two-sample drift z-score (current window vs reference window); the K z's are
    recency-weighted Stouffer-combined with weights w_k = decay**(K-1-k)
    (== L_decay forgetting). We contrast two realistic online protocols:

      DISJOINT : every window draws a fresh, independent reference and current
                 block  ->  the K z's are independent  ->  Stouffer's
                 independence assumption holds  ->  calibrated.

      SLIDING  : a single FIXED reference window is reused while the current
                 window slides with stride < m (overlapping)  ->  a shared
                 reference offset + overlap make the K z's POSITIVELY
                 correlated  ->  naive (independence) Stouffer under-estimates
                 the combined variance  ->  over-dispersed  ->  inflated FAR.
    """
    w = decay ** (np.arange(K)[::-1])

    def combine(zs):
        return float(np.sum(w * zs) / math.sqrt(np.sum(w * w)))

    # Disjoint: fresh independent reference + current each window.
    z_disjoint = np.array([_z(rng.normal(0, 1, m), rng.normal(0, 1, m)) for _ in range(K)])

    # Sliding: one fixed reference reused against overlapping current windows.
    ref = rng.normal(0, 1, m)
    total = m + (K - 1) * stride
    stream = rng.normal(0, 1, total)
    z_sliding = np.array([_z(ref, stream[i * stride : i * stride + m]) for i in range(K)])

    crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "Stouffer-DISJOINT": bool(abs(combine(z_disjoint)) > crit),
        "Stouffer-SLIDING": bool(abs(combine(z_sliding)) > crit),
    }


def _z(ref: np.ndarray, cur: np.ndarray) -> float:
    diff = cur.mean() - ref.mean()
    se = math.sqrt(cur.var(ddof=1) / cur.size + ref.var(ddof=1) / ref.size)
    return diff / se if se > 0 else 0.0


# --------------------------------------------------------------------------- #
# Exp C: global permute-then-refit detectors.
# --------------------------------------------------------------------------- #
def _rbf_gram(X: np.ndarray, gamma: float) -> np.ndarray:
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0, out=d2)
    return np.exp(-gamma * d2)


def mmd_rule(X, W, alpha, n_perm, rng) -> bool:
    n = X.shape[0]
    sub = X[rng.choice(n, size=min(n, 500), replace=False)]
    d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / med if med > 0 else 1.0
    K = _rbf_gram(X, gamma)

    def mmd2(idx0, idx1):
        n0, n1 = idx0.size, idx1.size
        k00 = (K[np.ix_(idx0, idx0)].sum() - np.trace(K[np.ix_(idx0, idx0)])) / (n0 * (n0 - 1))
        k11 = (K[np.ix_(idx1, idx1)].sum() - np.trace(K[np.ix_(idx1, idx1)])) / (n1 * (n1 - 1))
        k01 = K[np.ix_(idx0, idx1)].mean()
        return k00 + k11 - 2 * k01

    obs = mmd2(np.where(W == 0)[0], np.where(W == 1)[0])
    null = np.empty(n_perm)
    for b in range(n_perm):
        Wp = rng.permutation(W)
        null[b] = mmd2(np.where(Wp == 0)[0], np.where(Wp == 1)[0])
    pval = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return bool(pval < alpha)


def domain_auc_rule(X, W, alpha, n_perm, rng) -> bool:
    def cv_auc(Xx, Ww):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        aucs = []
        for tr, te in skf.split(Xx, Ww):
            clf = LogisticRegression(max_iter=200)
            clf.fit(Xx[tr], Ww[tr])
            aucs.append(roc_auc_score(Ww[te], clf.predict_proba(Xx[te])[:, 1]))
        return float(np.mean(aucs))

    obs = cv_auc(X, W)
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = cv_auc(X, rng.permutation(W))
    pval = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return bool(pval < alpha)


def _cross_nuisance(X, Y, W, n_folds, n_estimators, seed):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    mu = np.zeros(n)
    e = np.zeros(n)
    for k, te in enumerate(folds):
        tr = np.setdiff1d(np.arange(n), te)
        rfm = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=seed + k)
        rfm.fit(X[tr], Y[tr])
        mu[te] = rfm.predict(X[te])
        rfe = RandomForestClassifier(n_estimators=n_estimators, n_jobs=1, random_state=seed + 100 + k)
        rfe.fit(X[tr], W[tr])
        e[te] = rfe.predict_proba(X[te])[:, 1]
    return mu, np.clip(e, 0.01, 0.99)


def rrisk_rule(X, Y, W, alpha, n_perm, n_estimators, seed) -> bool:
    """R-risk permute-then-refit (concept-drift detector, RRPerm-style, light)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def rrisk(Xx, Yy, Ww, sd):
        mu, e = _cross_nuisance(Xx, Yy, Ww, 3, n_estimators, sd)
        yt, wt = Yy - mu, Ww - e
        mask = np.abs(wt) > 1e-3
        tau = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=1.0))])
        tau.fit(Xx[mask], (yt[mask] / wt[mask]), r__sample_weight=wt[mask] ** 2)
        th = tau.predict(Xx)
        return float(np.mean((yt - th * wt) ** 2))

    rng = np.random.default_rng(seed)
    obs = rrisk(X, Y, W, seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = rrisk(X, Y, rng.permutation(W), seed + 500 + b)
    pval = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return bool(pval < alpha)


def porisk_rule(X, Y, W, alpha, n_perm, n_estimators, seed) -> bool:
    """PO-risk permute-then-refit (DRPerm-style, light)."""

    def porisk(Xx, Yy, Ww, sd):
        mu, e = _cross_nuisance(Xx, Yy, Ww, 3, n_estimators, sd)
        pseudo = (Yy - mu) * (Ww - e)
        rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=sd + 1)
        rf.fit(Xx, pseudo)
        return float(np.mean(rf.predict(Xx) ** 2))

    rng = np.random.default_rng(seed + 13)
    obs = porisk(X, Y, W, seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = porisk(X, Y, rng.permutation(W), seed + 700 + b)
    pval = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return bool(pval < alpha)


# --------------------------------------------------------------------------- #
# Monte Carlo driver.
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    alpha: float = 0.05
    n_jobs: int = -1
    # Exp A (cheap: Fisher)
    fisher_p: int = 1000
    fisher_groups: int = 10
    fisher_n: int = 500
    fisher_reps: int = 400
    fisher_perm: int = 200
    # Exp A (expensive: RF-OOB-perm; sized for a ~4-core box)
    rf_p: int = 20
    rf_groups: int = 10
    rf_n: int = 400
    rf_reps: int = 80
    rf_perm: int = 40
    rf_trees: int = 50
    # Exp B (Stouffer windows)
    stouffer_K: int = 8
    stouffer_m: int = 150
    stouffer_stride: int = 60
    stouffer_decay: float = 0.85
    stouffer_reps: int = 1000
    # Exp C (global)
    global_p: int = 20
    global_n: int = 300
    global_reps: int = 200
    global_perm: int = 120
    meta_reps: int = 60
    meta_perm: int = 25
    meta_trees: int = 40
    noise_sd: float = 1.0
    base_seed: int = 20260101


def _groups(p: int, g: int) -> list[np.ndarray]:
    return [np.array(x) for x in np.array_split(np.arange(p), g)]


def run_fisher(cfg: Config, rep: int) -> dict[str, bool]:
    rng = np.random.default_rng(cfg.base_seed + rep)
    X, _, W = gen_stationary(cfg.fisher_n, cfg.fisher_n, cfg.fisher_p, cfg.noise_sd, rng)
    return fisher_rules(X, W, _groups(cfg.fisher_p, cfg.fisher_groups), cfg.alpha, cfg.fisher_perm, rng)


def run_rf(cfg: Config, rep: int) -> dict[str, bool]:
    rng = np.random.default_rng(cfg.base_seed + 10_000 + rep)
    X, _, W = gen_stationary(cfg.rf_n, cfg.rf_n, cfg.rf_p, cfg.noise_sd, rng)
    return rf_rules(X, W, _groups(cfg.rf_p, cfg.rf_groups), cfg.alpha, cfg.rf_perm, cfg.rf_trees, rng, cfg.base_seed + 10_000 + rep)


def run_stouffer(cfg: Config, rep: int) -> dict[str, bool]:
    rng = np.random.default_rng(cfg.base_seed + 20_000 + rep)
    return stouffer_window_rules(cfg.stouffer_K, cfg.stouffer_m, cfg.stouffer_stride, cfg.stouffer_decay, cfg.alpha, rng)


def run_global(cfg: Config, rep: int) -> dict[str, bool]:
    rng = np.random.default_rng(cfg.base_seed + 30_000 + rep)
    X, _, W = gen_stationary(cfg.global_n, cfg.global_n, cfg.global_p, cfg.noise_sd, rng)
    return {
        "MMD": mmd_rule(X, W, cfg.alpha, cfg.global_perm, rng),
        "domain-AUC": domain_auc_rule(X, W, cfg.alpha, cfg.global_perm, rng),
    }


def run_meta(cfg: Config, rep: int) -> dict[str, bool]:
    rng = np.random.default_rng(cfg.base_seed + 40_000 + rep)
    X, Y, W = gen_stationary(cfg.global_n, cfg.global_n, cfg.global_p, cfg.noise_sd, rng)
    seed = cfg.base_seed + 40_000 + rep
    return {
        "R-risk (RRPerm)": rrisk_rule(X, Y, W, cfg.alpha, cfg.meta_perm, cfg.meta_trees, seed),
        "PO-risk (DRPerm)": porisk_rule(X, Y, W, cfg.alpha, cfg.meta_perm, cfg.meta_trees, seed),
    }


def _mc(fn: Callable, cfg: Config, reps: int, label: str) -> list[dict[str, bool]]:
    print(f"  [{label}] running {reps} stationary replications ...", flush=True)
    if _HAVE_JOBLIB and cfg.n_jobs != 1:
        return Parallel(n_jobs=cfg.n_jobs)(delayed(fn)(cfg, r) for r in range(reps))
    return [fn(cfg, r) for r in range(reps)]


def summarize(results: list[dict[str, bool]], driver: str, alpha: float) -> pd.DataFrame:
    reps = len(results)
    rows = []
    for key in results[0]:
        k = int(sum(r[key] for r in results))
        far = k / reps
        lo, hi = wilson_ci(k, reps)
        calibrated = bool(lo <= alpha <= hi)
        verdict = "calibrated" if calibrated else ("inflated" if far > alpha else "conservative")
        rows.append(
            {
                "driver": driver,
                "detector": key,
                "false_alarm_rate": far,
                "ci_lo": lo,
                "ci_hi": hi,
                "reps": reps,
                "alpha": alpha,
                "calibrated": calibrated,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny sizes for a smoke test")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results_false_alarm"))
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    cfg = Config(n_jobs=args.n_jobs)
    if args.quick:
        cfg = Config(
            n_jobs=args.n_jobs,
            fisher_p=200, fisher_n=150, fisher_reps=30, fisher_perm=50,
            rf_p=12, rf_n=150, rf_reps=12, rf_perm=15, rf_trees=40,
            stouffer_reps=60,
            global_p=10, global_n=150, global_reps=20, global_perm=40,
            meta_reps=10, meta_perm=15, meta_trees=40,
        )

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[config]\n{json.dumps(cfg.__dict__, indent=2)}\n")

    frames = []
    frames.append(summarize(_mc(run_fisher, cfg, cfg.fisher_reps, "Exp A / Fisher"), "A: multiplicity (Fisher)", cfg.alpha))
    frames.append(summarize(_mc(run_rf, cfg, cfg.rf_reps, "Exp A / RF-OOB-perm"), "A: multiplicity (RF-OOB-perm)", cfg.alpha))
    frames.append(summarize(_mc(run_stouffer, cfg, cfg.stouffer_reps, "Exp B / Stouffer"), "B: window dependence (Stouffer)", cfg.alpha))
    frames.append(summarize(_mc(run_global, cfg, cfg.global_reps, "Exp C / MMD+domain-AUC"), "C: global permute-then-refit", cfg.alpha))
    frames.append(summarize(_mc(run_meta, cfg, cfg.meta_reps, "Exp C / meta-learner"), "C: global permute-then-refit", cfg.alpha))

    df = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(args.outdir, "false_alarm_rates.csv")
    df.to_csv(csv_path, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print("\n================ FALSE-ALARM RATES under a STATIONARY DGP ================")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[wrote] {csv_path}")

    try:
        _plot(df, cfg.alpha, os.path.join(args.outdir, "false_alarm_rates.png"))
        print(f"[wrote] {os.path.join(args.outdir, 'false_alarm_rates.png')}")
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped] {exc}")


def _plot(df: pd.DataFrame, alpha: float, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.copy()
    labels = df["detector"].tolist()
    far = df["false_alarm_rate"].values
    lo = far - df["ci_lo"].values
    hi = df["ci_hi"].values - far
    palette = {"calibrated": "#2e8b57", "inflated": "#d1495b", "conservative": "#4a7fb5"}
    colors = [palette.get(v, "#888888") for v in df.get("verdict", ["calibrated"] * len(df))]

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, far, xerr=[lo, hi], color=colors, alpha=0.88, capsize=3)
    ax.axvline(alpha, color="black", ls="--", lw=1.5, label=f"nominal alpha = {alpha}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("False-alarm rate (type-I error) under stationary H0")
    ax.set_title("Where false alarms come from in the online feature-blending pipeline")
    from matplotlib.patches import Patch

    handles = [
        plt.Line2D([0], [0], color="black", ls="--", lw=1.5, label=f"nominal alpha = {alpha}"),
        Patch(color=palette["calibrated"], label="calibrated (CI covers alpha)"),
        Patch(color=palette["inflated"], label="inflated (FAR > alpha)"),
        Patch(color=palette["conservative"], label="conservative (FAR < alpha)"),
    ]
    prev = None
    for yi, drv in zip(y, df["driver"]):
        if drv != prev:
            ax.axhline(yi + 0.5, color="0.85", lw=0.8)
            prev = drv
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
