#!/usr/bin/env python3
"""Automatic cross-level metric association + root-cause localization (prototype).

Business metrics live at different levels (order / user / merchant) with NO
explicit formula linking them. Given a shift in one metric we want to (a) learn
the association graph among metrics from data, and (b) root-cause the change to a
metric at another level (e.g. merchant GMV -> an order-level driver).

Prototype DGP: an order-level driver (delivery speed) improves for new merchants
and propagates DOWNSTREAM through a chain
    delivery(-) -> user_rating(+) -> n_orders(+) -> gmv(+),
while quantity / discount are unrelated noise. The observed symptom is "merchant
GMV up"; the true root cause is the order-level delivery metric.

Steps:
  1. Association graph: pairwise Spearman + HSIC (kernel dependence, no functional
     form assumed) -> weighted edges.
  2. Marginal shift score per metric: |cohen d| between batches + permutation p.
  3. Root-cause by CONDITIONAL SCREENING: a metric m is root-like if controlling
     for it removes the batch effect on the others (it screens off the shift),
     score(m) = mean_o [1 - |W-coef in o~W+m| / |W-coef in o~W|].
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.stats import spearmanr

METRICS = ["delivery_mins", "user_rating", "n_orders", "gmv", "quantity", "discount"]
LEVEL = {"delivery_mins": "order-ops", "quantity": "order-ops", "discount": "order-ops",
         "user_rating": "user", "n_orders": "merchant", "gmv": "merchant"}


def generate(n=300, shift=1.2, seed=2026):
    rng = np.random.default_rng(seed)
    W = rng.binomial(1, 0.5, n)
    delivery = rng.normal(0, 1, n) - shift * W                 # new merchants faster
    user_rating = -0.8 * delivery + rng.normal(0, 0.6, n)      # faster -> higher rating
    n_orders = 0.7 * user_rating - 0.4 * delivery + rng.normal(0, 0.6, n)
    gmv = 0.9 * n_orders + 0.3 * user_rating + rng.normal(0, 0.6, n)
    quantity = rng.normal(0, 1, n)                             # unrelated
    discount = rng.normal(0, 1, n)                             # unrelated
    M = np.column_stack([delivery, user_rating, n_orders, gmv, quantity, discount])
    return M, W


def _rbf(x):
    x = x.reshape(-1, 1)
    d = (x - x.T) ** 2
    med = np.median(d[d > 0]) or 1.0
    return np.exp(-d / med)


def hsic(x, y):
    n = len(x)
    Kx, Ky = _rbf(x), _rbf(y)
    H = np.eye(n) - 1.0 / n
    return float(np.trace(Kx @ H @ Ky @ H) / (n - 1) ** 2)


def association_graph(M, thr=0.1):
    p = M.shape[1]
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            rho = spearmanr(M[:, i], M[:, j]).correlation
            h = hsic(M[:, i], M[:, j])
            if abs(rho) > 0.15 or h > thr:
                edges.append((METRICS[i], METRICS[j], float(rho), float(h)))
    return edges


def marginal_shift(M, W, n_perm=500, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for j, name in enumerate(METRICS):
        a, b = M[W == 0, j], M[W == 1, j]
        pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) or 1.0
        d = (b.mean() - a.mean()) / pooled
        obs = abs(b.mean() - a.mean())
        cnt = sum(abs(M[rng.permutation(W) == 1, j].mean()
                      - M[rng.permutation(W) == 0, j].mean()) >= obs
                  for _ in range(n_perm))
        rows.append((name, float(d), (1 + cnt) / (1 + n_perm)))
    return sorted(rows, key=lambda r: -abs(r[1]))


def _resid(y, x):
    """Residual of y after regressing on [1, x] (removes x's linear signal)."""
    A = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _abscorr(a, b):
    c = np.corrcoef(a, b)[0, 1]
    return abs(c) if np.isfinite(c) else 0.0


def rootcause_screening(M, W, shifted_idx):
    """A metric m is root-like if it SCREENS OFF the batch effect on the other
    *shifted* metrics: regress each other shifted metric on m alone, then check
    whether the residual still carries W. Residual-vs-W correlation avoids the
    W--m collinearity that destabilizes a direct o~W+m coefficient; restricting to
    shifted metrics avoids dividing by the near-zero shift of noise metrics."""
    Wf = W.astype(float)
    rows = []
    for m in shifted_idx:
        reductions = []
        for o in shifted_idx:
            if o == m:
                continue
            base = _abscorr(M[:, o], Wf)
            scr = _abscorr(_resid(M[:, o], M[:, m]), Wf)
            if base > 1e-6:
                reductions.append(1.0 - scr / base)
        rows.append((METRICS[m], float(np.mean(reductions)) if reductions else 0.0))
    return sorted(rows, key=lambda r: -r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    M, W = generate(shift=args.shift, seed=args.seed)

    print("=" * 74)
    print("CROSS-LEVEL METRIC ASSOCIATION + ROOT-CAUSE (prototype)")
    print("=" * 74)
    print("true chain: delivery_mins(order) -> user_rating -> n_orders -> gmv(merchant)")
    print("symptom: merchant GMV shifted; find the root cause metric & its level\n")

    print("[1] Learned association graph (Spearman rho / HSIC), no explicit formula:")
    for u, v, rho, h in association_graph(M):
        print(f"    {u:14s} -- {v:14s}  rho={rho:+.2f}  hsic={h:.3f}")

    print("\n[2] Marginal shift per metric (|cohen d|, perm p) -- WHICH metrics moved:")
    ms = marginal_shift(M, W, seed=args.seed)
    for name, d, pv in ms:
        lvl = LEVEL[name]
        print(f"    {name:14s} [{lvl:9s}] cohen_d={d:+.2f}  p={pv:.3f}")
    shifted_idx = [METRICS.index(name) for name, d, pv in ms if pv < 0.05]

    print("\n[3] Root-cause by conditional screening over the SHIFTED metrics "
          "(higher = screens off the batch effect on the others):")
    rc = rootcause_screening(M, W, shifted_idx)
    for name, s in rc:
        print(f"    {name:14s} [{LEVEL[name]:9s}] screening_score={s:+.2f}")
    root = rc[0][0]
    print(f"\n  => symptom 'gmv (merchant)' root-caused to: {root} [{LEVEL[root]}]")
    ok = root == "delivery_mins"
    print(f"RESULT: {'PASS' if ok else 'CHECK'}  (true root = delivery_mins [order-ops])")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
