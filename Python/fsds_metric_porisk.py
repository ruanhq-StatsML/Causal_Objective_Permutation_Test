#!/usr/bin/env python3
"""Understand how Y|X changes with PO-risk -- one model per target metric.

HSIC gives a symmetric association only. To understand whether and where the
relationship between a TARGET metric Y and the other metrics X shifts between
batches, we run the pseudo-outcome (PO-)risk with batch as the treatment: it
detects a change in P(Y|X) (concept drift), and its leave-one-out (LOCO) tells
which X the change loads on. Crucially this is ONE model per target Y (the
meta-learner takes all X jointly) -- not a model for every X-Y pair.

DGP: metrics X are drawn i.i.d. (no covariate shift). The target Y=gmv keeps a
stable part but, for the new batch only, gains a dependence on `rating`
(the rating->gmv relationship strengthens) -- a concept drift localized to
`rating`. PO-risk should fire and LOCO should point at `rating`.
"""
from __future__ import annotations

import argparse
import numpy as np

from fsds_merchant_prototype import (
    _cross_fit_mu, _cross_fit_e, po_statistic,
)

X_METRICS = ["delivery", "rating", "orders", "promo", "tenure"]


def generate(n=400, delta=1.5, seed=2026):
    rng = np.random.default_rng(seed)
    W = rng.binomial(1, 0.5, n)
    X = rng.normal(0, 1, (n, len(X_METRICS)))          # i.i.d. metrics, no cov shift
    d, r, o, p, _t = X.T
    Y = 0.5 * o + 0.4 * p + rng.normal(0, 0.6, n)       # stable mapping
    Y = Y + W * delta * r                               # concept drift on `rating`
    return X, Y, W


def _porisk(Xsub, Y, W, seed, n_folds=5):
    mu = _cross_fit_mu(Xsub, Y, n_folds, seed)
    e = _cross_fit_e(Xsub, W, n_folds, seed)
    return po_statistic(Xsub, Y - mu, W, e, seed=seed + 200)


def porisk_test(X, Y, W, n_perm=100, seed=2026):
    obs = _porisk(X, Y, W, seed)
    rng = np.random.default_rng(seed + 1)
    cnt = 0
    for _ in range(n_perm):
        Wp = rng.permutation(W)
        if _porisk(X, Y, Wp, seed + 7) >= obs:
            cnt += 1
    return obs, (1 + cnt) / (1 + n_perm)


def porisk_loco(X, Y, W, seed=2026):
    obs = _porisk(X, Y, W, seed)
    p = X.shape[1]
    rows = []
    for j in range(p):
        Xj = X[:, np.setdiff1d(np.arange(p), j)]
        loco = _porisk(Xj, Y, W, seed + 11 + j)
        rows.append((X_METRICS[j], obs - loco))        # drop when j removed
    return obs, sorted(rows, key=lambda r: -r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=1.5)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    X, Y, W = generate(delta=args.delta, seed=args.seed)

    print("=" * 72)
    print("PO-risk: does the target's Y|X relationship shift, and on which X?")
    print("=" * 72)
    print("target Y = gmv; X = " + ", ".join(X_METRICS))
    print("ground truth: Y|X concept drift localized to `rating`\n")

    obs, p = porisk_test(X, Y, W, n_perm=args.n_perm, seed=args.seed)
    print(f"[1] PO-risk test:  statistic={obs:.4f}  perm p={p:.4f}  "
          f"-> {'Y|X SHIFTED' if p < 0.05 else 'no shift'}")

    obs2, loco = porisk_loco(X, Y, W, seed=args.seed)
    print("\n[2] PO-risk LOCO (drop in PO-risk when the metric is removed):")
    for name, v in loco:
        flag = " <== driver" if name == "rating" else ""
        print(f"    {name:10s} loco_vimp={v:+.4f}{flag}")
    root = loco[0][0]
    print(f"\n  => Y|X change loads on: {root}")
    print("  note: ONE PO-risk model per target Y (all X jointly) -- "
          "not a model for every X-Y pair.")
    ok = (p < 0.05) and root == "rating"
    print(f"RESULT: {'PASS' if ok else 'CHECK'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
