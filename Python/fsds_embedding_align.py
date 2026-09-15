#!/usr/bin/env python3
"""Set->merchant alignment: kernel mean embedding (RFF) loses less information
than summary aggregation.

A merchant is a SET of orders. Aligning order-level features to the merchant grain
is a set->vector pooling. Summary aggregation (mean / std / quantiles) keeps only
a few moments; the kernel mean embedding mu_P = (1/k) sum phi(x) with a
characteristic kernel is INJECTIVE (mu_P = mu_Q iff P = Q), so it retains the
whole per-merchant order distribution. Random Fourier Features give a finite-dim
approximation and are consistent with our MMD (MMD = distance of mean embeddings).

Demonstration: new merchants have a MEAN-PRESERVING shape shift on order-level
`delivery` (same mean, larger variance). Mean-aggregation misses it; the RFF mean
embedding catches it.
"""
from __future__ import annotations

import argparse
import numpy as np

from fsds_logo_mmd import _median_gamma, mmd2_unbiased


def generate(n_merch=200, k_orders=60, var_shift=2.2, seed=2026):
    rng = np.random.default_rng(seed)
    W = rng.binomial(1, 0.5, n_merch)
    orders = []          # list of per-merchant order arrays (1-D delivery)
    for m in range(n_merch):
        sd = np.sqrt(var_shift) if W[m] == 1 else 1.0    # mean-preserving variance shift
        orders.append(rng.normal(0.0, sd, k_orders))
    return orders, W


def rff(x, D=64, gamma=0.5, seed=0):
    """Random Fourier Features for the RBF kernel on scalar x -> D-dim map."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, np.sqrt(2 * gamma), D)
    b = rng.uniform(0, 2 * np.pi, D)
    return np.sqrt(2.0 / D) * np.cos(np.outer(x, w) + b)      # (len(x), D)


def merchant_embeddings(orders, D=64, gamma=0.5, seed=0):
    """Kernel mean embedding per merchant (mean of RFF over its orders)."""
    return np.array([rff(o, D=D, gamma=gamma, seed=seed).mean(0) for o in orders])


def _cohen_d(a, b):
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) or 1.0
    return (b.mean() - a.mean()) / pooled


def _perm_p(stat_fn, W, n_perm=300, seed=0):
    obs = stat_fn(W)
    rng = np.random.default_rng(seed)
    cnt = sum(stat_fn(rng.permutation(W)) >= obs for _ in range(n_perm))
    return obs, (1 + cnt) / (1 + n_perm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var-shift", type=float, default=2.2)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    orders, W = generate(var_shift=args.var_shift, seed=args.seed)

    # (1) summary aggregation: merchant mean (and std) of order-level delivery
    m_mean = np.array([o.mean() for o in orders])
    m_std = np.array([o.std() for o in orders])

    # (2) kernel mean embedding (RFF) per merchant
    E = merchant_embeddings(orders, D=args.D, seed=args.seed)

    print("=" * 74)
    print("Alignment by summary aggregation vs kernel mean embedding (RFF)")
    print("=" * 74)
    print("new merchants: order-level `delivery` has SAME mean, LARGER variance\n")

    d_mean = _cohen_d(m_mean[W == 0], m_mean[W == 1])
    d_std = _cohen_d(m_std[W == 0], m_std[W == 1])
    print(f"[summary] delivery__mean  batch shift  cohen_d = {d_mean:+.3f}  "
          "<- mean aggregation MISSES the shape shift")
    print(f"[summary] delivery__std   batch shift  cohen_d = {d_std:+.3f}  "
          "(only caught if you knew to add std)")

    def emb_mmd(w):
        g = _median_gamma(E)
        return mmd2_unbiased(E[w == 0], E[w == 1], g)
    mmd_obs, p = _perm_p(emb_mmd, W, seed=args.seed)
    print(f"[embedding] RFF mean-embedding MMD^2 = {mmd_obs:.4f}  perm p = {p:.3f}  "
          "<- CATCHES the shift (any moment), no need to pick which")

    ok = abs(d_mean) < 0.3 and p < 0.05
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}  "
          f"(mean-agg misses [|d|={abs(d_mean):.2f}], embedding detects [p={p:.3f}])")
    print("\nwhy: mean embedding is injective for a characteristic kernel -> it "
          "encodes the whole order distribution (minimal information loss);\n"
          "any fixed finite summary can be fooled by a shift orthogonal to it.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
