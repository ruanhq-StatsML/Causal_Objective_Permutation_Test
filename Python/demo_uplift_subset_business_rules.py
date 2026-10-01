#!/usr/bin/env python3
"""Uplift subset localization + plain business rules (FSDS shift playbook → AUUC)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from loco_auuc.data import split_ref_live
from loco_auuc.learners import make_learner
from loco_auuc.subset_localization import run_uplift_subset_localization

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def _simulate(n: int, seed: int, *, live: bool = False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    if live:
        X[:, 0] += 1.2
    t = rng.binomial(1, 0.5, size=n)
    logits = 0.4 * X[:, 1] + 0.2 * X[:, 2]
    if live:
        logits += 0.35 * X[:, 3]
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p)
    return X, t, y


def _batch_e(X_ref, X_live):
    X = np.vstack([X_ref, X_live])
    w = np.array([0] * len(X_ref) + [1] * len(X_live), dtype=int)
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(X, w)
    return clf.predict_proba(X_live)[:, 1]


def main() -> int:
    X_ref, t_ref, y_ref = _simulate(4000, 11, live=False)
    X_live, t_live, y_live = _simulate(1500, 22, live=True)

    (X_tr, t_tr, y_tr), (X_ev, t_ev, y_ev), (X_lv, t_lv, y_lv) = split_ref_live(
        X_ref, t_ref, y_ref, X_live, t_live, y_live, eval_frac=0.25, seed=42
    )
    names = [f"f{j}" for j in range(X_tr.shape[1])]
    groups = {"block_a": [0, 1, 2], "block_b": [3, 4, 5], "block_c": [6, 7]}

    e_live = _batch_e(np.vstack([X_ev, X_tr[:200]]), X_lv)

    report = run_uplift_subset_localization(
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
        groups,
        feature_names=names,
        e_batch_live=e_live,
        overlap_ess=0.7,
    )

    out = ART / "uplift_subset_business_rules.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=== Uplift subset localization (FSDS-style) ===")
    print(f"AUUC_live={report['auuc_live_global']:.4f}  gap={report['auuc_gap']:.4f}  "
          f"MMD2={report['mmd2_x_global']:.4f}")
    print(f"Diagnosis: {report['diagnosis']['label']}")
    print("\nLOGO-MMD (covariate groups):")
    for row in report["logo_mmd"]:
        print(f"  {row['group']}: logo_drop={row['logo_mmd_drop']:.4f}")
    print("\nBusiness rules:")
    for br in report["business_rules"]:
        print(f"  [{br['action']}] P{br['priority']}: {br['rule']}")
    print("\nWrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
