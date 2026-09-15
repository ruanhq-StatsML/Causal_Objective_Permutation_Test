#!/usr/bin/env python3
"""Exact Shapley-MMD attribution over ALL feature-group subsets.

LOGO only measures the marginal contribution of a group *removed from the full
set*. The principled attribution enumerates **all subsets** (coalitions) and
averages each group's marginal MMD contribution over them -- the Shapley value.
With K raw-attribute groups (here ~7 incl. the activity/count group), the power
set has 2^K subsets, small enough to **enumerate exactly** -- no BH gatekeeping,
no fixed taxonomy path.

Only the *ranking* of the Shapley values matters (kernel/bandwidth scale is
irrelevant to the order), so a single fixed bandwidth is used. Selection uses a
permutation null (shuffle the batch label, recompute Shapley); ranking stability
uses the stratified bootstrap.
"""
from __future__ import annotations

import argparse
import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

from fsds_merchant_prototype import raw_attr_of
from fsds_two_dimensional import generate_two_dim_data, aggregate_to_entity, DIMENSIONS

N_CAP = 400  # cap entities per axis to keep the O(N^2) subset sweeps snappy


def _gamma(Z):
    d = pdist(Z, "sqeuclidean")
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return 1.0 / med if med > 0 else 1.0


def _mmd_from_K(K, W):
    a, b = np.where(W == 0)[0], np.where(W == 1)[0]
    m, n = a.size, b.size
    Kxx, Kyy, Kxy = K[np.ix_(a, a)], K[np.ix_(b, b)], K[np.ix_(a, b)]
    return ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
            - 2.0 * Kxy.mean())


def _group_sqdist(feat_std, groups):
    return {g: cdist(feat_std[m].values, feat_std[m].values, "sqeuclidean")
            for g, m in groups.items()}


def _subset_kernels(D, gamma, gkeys):
    """Precompute the RBF kernel matrix of every non-empty subset (fixed rows)."""
    K = {}
    for r in range(1, len(gkeys) + 1):
        for combo in combinations(gkeys, r):
            S = frozenset(combo)
            K[S] = np.exp(-gamma * sum(D[g] for g in S))
    return K


def _shapley(subset_K, W, gkeys):
    Kn = len(gkeys)
    mmd = {frozenset(): 0.0}
    for S, Kmat in subset_K.items():
        mmd[S] = _mmd_from_K(Kmat, W)
    phi = {g: 0.0 for g in gkeys}
    for g in gkeys:
        others = [x for x in gkeys if x != g]
        for r in range(len(others) + 1):
            w = math.factorial(r) * math.factorial(Kn - r - 1) / math.factorial(Kn)
            for combo in combinations(others, r):
                S = frozenset(combo)
                phi[g] += w * (mmd[S | {g}] - mmd[S])
    return phi


def shapley_axis(orders, dim, n_perm=100, n_boot=50, seed=2026):
    cfg = DIMENSIONS[dim]
    feat = aggregate_to_entity(orders, cfg["key"], cfg["other_ids"])
    ent = orders[[cfg["key"], cfg["label"]]].drop_duplicates() \
        .set_index(cfg["key"])[cfg["label"]]
    W_all = ent.reindex(feat.index).values.astype(int)
    feat_std_all = ((feat - feat.mean()) / feat.std().replace(0, 1)).fillna(0.0)

    rng = np.random.default_rng(seed)
    if len(feat_std_all) > N_CAP:  # stratified subsample for the O(N^2) sweeps
        i0 = np.where(W_all == 0)[0]
        i1 = np.where(W_all == 1)[0]
        k0 = int(round(N_CAP * i0.size / len(W_all)))
        keep = np.concatenate([rng.choice(i0, k0, replace=False),
                               rng.choice(i1, N_CAP - k0, replace=False)])
        feat_std = feat_std_all.iloc[keep].reset_index(drop=True)
        W = W_all[keep]
    else:
        feat_std, W = feat_std_all.reset_index(drop=True), W_all

    names = list(feat_std.columns)
    gkeys = sorted({raw_attr_of(f) for f in names})
    groups = {g: [f for f in names if raw_attr_of(f) == g] for g in gkeys}

    gamma = _gamma(feat_std.values)
    D = _group_sqdist(feat_std, groups)
    subset_K = _subset_kernels(D, gamma, gkeys)

    phi = _shapley(subset_K, W, gkeys)                       # point estimate

    # permutation null (fixed rows -> reuse subset kernels) -> selection p
    null = {g: np.empty(n_perm) for g in gkeys}
    for b in range(n_perm):
        Wp = rng.permutation(W)
        pp = _shapley(subset_K, Wp, gkeys)
        for g in gkeys:
            null[g][b] = pp[g]
    pval = {g: (1.0 + np.sum(null[g] >= phi[g])) / (1.0 + n_perm) for g in gkeys}

    # bootstrap (resample rows -> recompute) -> ranking stability
    top_k = 2
    boot_phi = {g: np.empty(n_boot) for g in gkeys}
    in_top = {g: 0 for g in gkeys}
    i0, i1 = np.where(W == 0)[0], np.where(W == 1)[0]
    for b in range(n_boot):
        bi = np.concatenate([rng.choice(i0, i0.size, replace=True),
                             rng.choice(i1, i1.size, replace=True)])
        fb = feat_std.iloc[bi].reset_index(drop=True)
        fb = fb + rng.normal(0, 1e-3, fb.shape)
        Db = _group_sqdist(fb, groups)
        Kb = _subset_kernels(Db, _gamma(fb.values), gkeys)
        pb = _shapley(Kb, W[bi], gkeys)
        order = sorted(gkeys, key=lambda g: -pb[g])[:top_k]
        for g in gkeys:
            boot_phi[g][b] = pb[g]
            if g in order:
                in_top[g] += 1

    rows = [{"axis": dim, "attribute": g, "shapley": phi[g],
             "perm_p": pval[g], "boot_mean": float(boot_phi[g].mean()),
             "boot_ci_lo": float(np.percentile(boot_phi[g], 2.5)),
             "boot_ci_hi": float(np.percentile(boot_phi[g], 97.5)),
             "top2_freq": in_top[g] / n_boot} for g in gkeys]
    df = pd.DataFrame(rows).sort_values("shapley", ascending=False).reset_index(drop=True)
    return df, {"n_used": int(len(W)), "gamma": float(gamma)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    _, _, _, orders = generate_two_dim_data(cov_shift=args.cov_shift, seed=args.seed)
    truth = {"merchant": {"gmv", "user_rating"},
             "user": {"basket_size", "discount_rate"}}

    frames = []
    for dim in ("merchant", "user"):
        df, meta = shapley_axis(orders, dim, n_perm=args.n_perm,
                                n_boot=args.n_boot, seed=args.seed)
        frames.append(df)
        print("=" * 78)
        print(f"AXIS={dim}  (n_used={meta['n_used']})  "
              f"exact Shapley over all 2^{df.shape[0]} group subsets")
        print(f"  {'attribute':14s} {'shapley':>10s} {'perm_p':>8s} "
              f"{'top2_freq':>10s}")
        for _, r in df.iterrows():
            star = " <== truth" if r["attribute"] in truth[dim] else ""
            print(f"  {r['attribute']:14s} {r['shapley']:10.5f} {r['perm_p']:8.4f} "
                  f"{r['top2_freq']:10.2f}{star}")
        ranked = list(df["attribute"])[:2]
        ok = set(ranked) == truth[dim]
        print(f"  top-2 by Shapley = {ranked}  -> {'MATCH' if ok else 'NO MATCH'}\n")

    reg = pd.concat(frames, ignore_index=True)
    if args.csv:
        reg.to_csv(args.csv, index=False)
        print(f"saved -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
