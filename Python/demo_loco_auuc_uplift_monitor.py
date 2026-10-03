#!/usr/bin/env python3
"""Synthetic covariate + concept drift demo for LOCO-AUUC uplift monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from loco_auuc.data import split_ref_live
from loco_auuc.learners import make_learner
from loco_auuc.monitor import run_uplift_monitor

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def _simulate(n: int, seed: int, *, live: bool = False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    if live:
        X[:, 0] += 1.2  # covariate shift on f0
    t = rng.binomial(1, 0.5, size=n)
    logits = 0.4 * X[:, 1] + 0.2 * X[:, 2]
    if live:
        logits += 0.35 * X[:, 3]  # concept: new driver on f3
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p)
    return X, t, y


def _e_hat_fit(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(X, w)
    return clf.predict_proba(X)[:, 1]


def main() -> int:
    X_ref, t_ref, y_ref = _simulate(4000, 11, live=False)
    X_live, t_live, y_live = _simulate(1500, 22, live=True)

    (X_tr, t_tr, y_tr), (X_ev, t_ev, y_ev), (X_lv, t_lv, y_lv) = split_ref_live(
        X_ref, t_ref, y_ref, X_live, t_live, y_live, eval_frac=0.25, seed=42
    )
    names = [f"f{j}" for j in range(X_tr.shape[1])]
    groups = {"block_a": [0, 1, 2], "block_b": [3, 4, 5], "block_c": [6, 7]}

    report = run_uplift_monitor(
        X_tr,
        t_tr,
        y_tr,
        X_ev,
        t_ev,
        y_ev,
        X_lv,
        t_lv,
        y_lv,
        lambda: make_learner("xlearner", random_state=42),
        feature_names=names,
        groups=groups,
        fit_propensity_for_overlap=_e_hat_fit,
    )

    out = ART / "loco_auuc_uplift_monitor.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    loco = report["loco_auuc"]
    print("AUUC ref (full):", round(loco["auuc_ref_full"], 5))
    print("AUUC live (full):", round(loco["auuc_live_full"], 5))
    print("AUUC gap:", round(loco["auuc_gap"], 5))
    print("MMD² X:", round(report["mmd2_x_ref_vs_live"], 5))
    print("Diagnosis:", report["diagnosis"]["label"])
    for h in report["diagnosis"]["hints"]:
        print(" ", h)
    print("Group LOCO (drop_live):")
    for row in sorted(loco["loco_rows"], key=lambda r: -r["drop_live"]):
        print(f"  {row['feature']:10s} drop_ref={row['drop_ref']:.5f} drop_live={row['drop_live']:.5f}")
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
