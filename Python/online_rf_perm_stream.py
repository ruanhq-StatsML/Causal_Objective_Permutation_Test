#!/usr/bin/env python3
"""
OnlineRFPerm: a continuous-time (streaming) distribution-shift monitor built on
a random-forest domain classifier + permutation importance, reduced to ONE
calibrated global statistic per window (NO per-feature multiple testing).

Design (the agreed pipeline, realised for a stream):

    fixed reference window R (the trusted "existing batch")
      -> for each incoming window C_t:
           fit a domain classifier on (R vs C_t)
           -> permutation importance over features
           -> global statistic s_t = max_j importance_j         (ONE scalar)
      -> threshold tau calibrated ONCE on a reference-only null  (split R into
         two pseudo-batches; tau = (1-alpha) quantile of the null s)
      -> alarm_t = s_t > tau                                     (one decision)
    per-group importances are reported for ATTRIBUTION only, never for deciding.

Why this is the irreplaceable role for RF (demonstrated by `demo()`):
  Under a COVARIANCE-only shift (means unchanged, a correlation appears between
  two features) the Bayes-optimal domain discriminator is quadratic. A linear
  base learner (logistic) is blind to it -- its permutation importance stays at
  the null level -- while the RF, which can represent interactions/thresholds,
  lifts its statistic above tau. Under a plain mean shift both detect it; under
  a stationary stream neither fires (false-alarm control).

Run:
    python online_rf_perm_stream.py           # full demo + figure
    python online_rf_perm_stream.py --quick    # tiny smoke test
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Base learners (same permutation-importance protocol; only the family differs).
# --------------------------------------------------------------------------- #
def make_rf(trees: int, seed: int):
    return RandomForestClassifier(
        n_estimators=trees, max_features="sqrt", n_jobs=1, random_state=seed
    )


def make_logistic(seed: int):
    return Pipeline(
        [("s", StandardScaler()), ("lr", LogisticRegression(max_iter=500))]
    )


# --------------------------------------------------------------------------- #
# One global statistic for a (reference, current) pair: max permutation
# importance of a domain classifier, scored by held-out AUC.
# --------------------------------------------------------------------------- #
def domain_perm_stat(ref: np.ndarray, cur: np.ndarray, make_model: Callable, seed: int,
                     n_repeats: int = 5) -> tuple[float, np.ndarray]:
    X = np.vstack([ref, cur])
    y = np.concatenate([np.zeros(len(ref), int), np.ones(len(cur), int)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    model = make_model(seed)
    model.fit(Xtr, ytr)
    r = permutation_importance(
        model, Xte, yte, scoring="roc_auc", n_repeats=n_repeats, random_state=seed
    )
    imp = r.importances_mean
    return float(imp.max()), imp


def calibrate_threshold(ref: np.ndarray, make_model: Callable, alpha: float,
                        n_null: int, seed: int, n_repeats: int = 5) -> float:
    """Reference-only null: split R into two pseudo-batches (no shift)."""
    rng = np.random.default_rng(seed)
    n = len(ref)
    half = n // 2
    null = np.empty(n_null)
    for b in range(n_null):
        idx = rng.permutation(n)
        a, c = ref[idx[:half]], ref[idx[half : 2 * half]]
        null[b], _ = domain_perm_stat(a, c, make_model, seed + 100 + b, n_repeats)
    return float(np.quantile(null, 1 - alpha))


# --------------------------------------------------------------------------- #
# Stream generators (continuous time, changepoint at t_star).
# --------------------------------------------------------------------------- #
def window_stationary(m, p, rng):
    return rng.normal(0, 1, size=(m, p))


def window_mean_shift(m, p, rng, delta):
    X = rng.normal(0, 1, size=(m, p))
    X[:, :3] += delta
    return X


def window_cov_shift(m, p, rng, rho):
    """Means unchanged; inject correlation between feature 0 and 2 (quadratic)."""
    X = rng.normal(0, 1, size=(m, p))
    X[:, 2] = rho * X[:, 0] + math.sqrt(max(1e-6, 1 - rho * rho)) * X[:, 2]
    return X


# --------------------------------------------------------------------------- #
# Run one stream through one detector.
# --------------------------------------------------------------------------- #
@dataclass
class StreamResult:
    stats: np.ndarray
    tau: float
    t_star: int
    alarms: np.ndarray


def run_stream(ref, make_model, scenario, T, m, p, t_star, mag, alpha, seed,
               n_null, n_repeats):
    rng = np.random.default_rng(seed)
    tau = calibrate_threshold(ref, make_model, alpha, n_null, seed, n_repeats)
    stats = np.empty(T)
    for t in range(T):
        drifting = t >= t_star
        if scenario == "stationary" or not drifting:
            cur = window_stationary(m, p, rng)
        elif scenario == "mean":
            cur = window_mean_shift(m, p, rng, mag)
        elif scenario == "cov":
            cur = window_cov_shift(m, p, rng, mag)
        else:
            raise ValueError(scenario)
        stats[t], _ = domain_perm_stat(ref, cur, make_model, seed + 7 + t, n_repeats)
    return StreamResult(stats=stats, tau=tau, t_star=t_star, alarms=stats > tau)


# --------------------------------------------------------------------------- #
# Demo: RF vs logistic across stationary / mean / covariance streams.
# --------------------------------------------------------------------------- #
@dataclass
class DemoConfig:
    p: int = 12
    m: int = 300
    n_ref: int = 600
    T: int = 40
    t_star: int = 20
    trees: int = 80
    alpha: float = 0.05
    n_null: int = 80
    n_repeats: int = 5
    mean_mag: float = 0.6
    cov_mag: float = 0.75
    seed: int = 20260101


def run_demo(cfg: DemoConfig):
    rng = np.random.default_rng(cfg.seed)
    ref = rng.normal(0, 1, size=(cfg.n_ref, cfg.p))
    scenarios = [("stationary", 0.0), ("mean", cfg.mean_mag), ("cov", cfg.cov_mag)]
    models = {
        "RF (OnlineRFPerm)": lambda s: make_rf(cfg.trees, s),
        "Logistic": lambda s: make_logistic(s),
    }
    results = {}
    for si, (sc, mag) in enumerate(scenarios):
        for mi, (mname, mk) in enumerate(models.items()):
            res = run_stream(
                ref, mk, sc, cfg.T, cfg.m, cfg.p, cfg.t_star, mag, cfg.alpha,
                cfg.seed + 1000 * si + 137 * mi, cfg.n_null, cfg.n_repeats,
            )
            results[(sc, mname)] = res
    return results


def detection_summary(results, cfg: DemoConfig):
    rows = []
    for (sc, mname), res in results.items():
        pre = res.alarms[: cfg.t_star]
        post = res.alarms[cfg.t_star :]
        far = float(pre.mean()) if pre.size else float("nan")
        det = float(post.mean()) if post.size else float("nan")
        first = next((cfg.t_star + i for i, a in enumerate(post) if a), None)
        delay = (first - cfg.t_star) if first is not None else None
        rows.append(
            {
                "scenario": sc,
                "model": mname,
                "pre_change_FAR": round(far, 3),
                "post_change_detect_rate": round(det, 3),
                "detection_delay": delay,
            }
        )
    return rows


def plot_demo(results, cfg: DemoConfig, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = [("stationary", "A. stationary (no shift)"),
                 ("mean", "B. mean shift (linear)"),
                 ("cov", "C. covariance shift (interaction)")]
    colors = {"RF (OnlineRFPerm)": "#2e8b57", "Logistic": "#d1495b"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (sc, title) in zip(axes, scenarios):
        for mname in colors:
            res = results[(sc, mname)]
            ax.plot(res.stats, marker="o", ms=3, color=colors[mname], label=mname)
            ax.axhline(res.tau, color=colors[mname], ls=":", lw=1.2, alpha=0.8)
        if sc != "stationary":
            res = results[(sc, "RF (OnlineRFPerm)")]
            ax.axvline(res.t_star, color="black", ls="--", lw=1.2)
            ax.text(res.t_star + 0.3, ax.get_ylim()[1] * 0.92, "changepoint", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("stream window t")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("global statistic  s_t = max permutation importance\n(dotted = calibrated threshold tau)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("OnlineRFPerm stream: RF is irreplaceable under interaction/covariance drift "
                 "(panel C: only RF crosses tau)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results_online_rf_perm"))
    args = ap.parse_args()

    cfg = DemoConfig()
    if args.quick:
        cfg = DemoConfig(p=8, m=150, n_ref=300, T=16, t_star=8, trees=40, n_null=25, n_repeats=3)

    os.makedirs(args.outdir, exist_ok=True)
    print("[OnlineRFPerm stream] running demo ...", flush=True)
    results = run_demo(cfg)
    rows = detection_summary(results, cfg)

    import pandas as pd

    df = pd.DataFrame(rows).sort_values(["scenario", "model"]).reset_index(drop=True)
    csv_path = os.path.join(args.outdir, "online_rf_perm_detection.csv")
    df.to_csv(csv_path, index=False)
    print("\n============ OnlineRFPerm stream detection (continuous time) ============")
    print(df.to_string(index=False))
    print(f"\n[wrote] {csv_path}")

    png = os.path.join(args.outdir, "online_rf_perm_stream.png")
    try:
        plot_demo(results, cfg, png)
        print(f"[wrote] {png}")
    except Exception as exc:  # pragma: no cover
        print(f"[plot skipped] {exc}")


if __name__ == "__main__":
    main()
