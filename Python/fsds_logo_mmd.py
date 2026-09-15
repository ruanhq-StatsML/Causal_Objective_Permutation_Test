#!/usr/bin/env python3
"""LOGO-MMD attribution for merchant-dimension COVARIATE shift.

Motivation
----------
The synthetic merchants carry a covariate shift (P(X) differs between the
existing and new batch) with no concept drift.  PO-risk / pseudo-outcome
learners are insensitive to pure covariate shift, so the causal-objective test
does not fire here.  A kernel two-sample distance -- Maximum Mean Discrepancy
(MMD) -- *is* sensitive to P(X) shift, and **LOGO-MMD** (leave-one-group-out)
attributes the shift to the merchant feature groups that drive it.

This module reuses the relational generation + merchant-dimension aggregation
from ``fsds_merchant_prototype`` and adds:

* an unbiased RBF-kernel MMD^2 estimator (median-heuristic bandwidth),
* a permutation test for global covariate-shift detection,
* **LOGO-MMD**: drop each raw-attribute group and measure the fall in MMD,
* a per-group standalone MMD + permutation p-value cross-check (FS by MMD).
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

from fsds_merchant_prototype import (
    RAW_ATTRS, generate_relational_data, aggregate_to_merchant, raw_attr_of,
)

COV_SHIFT_ATTRS = ["gmv", "user_rating"]  # ground-truth covariate-shifted attrs


# --------------------------------------------------------------------------- #
# MMD
# --------------------------------------------------------------------------- #
def _median_gamma(Z):
    """RBF gamma from the median heuristic on pooled pairwise distances."""
    d = pdist(Z, metric="sqeuclidean")
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return 1.0 / med if med > 0 else 1.0


def mmd2_unbiased(X, Y, gamma):
    """Unbiased estimator of squared MMD with an RBF kernel."""
    m, n = len(X), len(Y)
    Kxx = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    Kyy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    Kxy = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    s_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    s_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    s_xy = Kxy.mean()
    return float(s_xx + s_yy - 2.0 * s_xy)


def mmd_permutation_test(Z, W, gamma, n_perm=500, seed=2026):
    """Permutation test of H0: batch 0 and batch 1 share the same P(X)."""
    rng = np.random.default_rng(seed)
    X, Y = Z[W == 0], Z[W == 1]
    observed = mmd2_unbiased(X, Y, gamma)
    m = (W == 0).sum()
    perm = np.empty(n_perm)
    for b in range(n_perm):
        idx = rng.permutation(len(Z))
        Zp = Z[idx]
        perm[b] = mmd2_unbiased(Zp[:m], Zp[m:], gamma)
    p_value = (1.0 + np.sum(perm >= observed)) / (1.0 + n_perm)
    return {"mmd2": observed, "p_value": p_value, "perm_mean": float(perm.mean())}


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def logo_mmd(feat_std, W, groups):
    """LEAVE-ONE-GROUP-OUT MMD attribution.

    For each raw-attribute group, drop all of its aggregated columns, recompute
    MMD on the remainder (with its own median-heuristic bandwidth), and report
    the fall in MMD.  A large drop => that group carried the covariate shift.
    """
    names = np.asarray(feat_std.columns)
    Z_full = feat_std.values
    g_full = _median_gamma(Z_full)
    mmd_full = mmd2_unbiased(Z_full[W == 0], Z_full[W == 1], g_full)

    rows = []
    for gname, members in groups.items():
        keep = np.array([nm not in set(members) for nm in names])
        Zsub = feat_std.values[:, keep]
        gsub = _median_gamma(Zsub)
        mmd_sub = mmd2_unbiased(Zsub[W == 0], Zsub[W == 1], gsub)
        rows.append({"attribute": gname, "n_features": len(members),
                     "logo_drop": mmd_full - mmd_sub, "mmd_without": mmd_sub,
                     "is_shift": gname in COV_SHIFT_ATTRS})
    df = pd.DataFrame(rows).sort_values("logo_drop", ascending=False).reset_index(drop=True)
    return df, mmd_full


def per_group_mmd(feat_std, W, groups, n_perm=300, seed=2026):
    """Standalone MMD of each group alone, with a permutation p-value.

    A direct 'feature-selection by MMD': which merchant attribute groups differ
    significantly between the two batches."""
    rows = []
    for gi, (gname, members) in enumerate(groups.items()):
        Zg = feat_std[members].values
        gg = _median_gamma(Zg)
        res = mmd_permutation_test(Zg, W, gg, n_perm=n_perm, seed=seed + gi)
        rows.append({"attribute": gname, "group_mmd2": res["mmd2"],
                     "p_value": res["p_value"], "is_shift": gname in COV_SHIFT_ATTRS})
    return pd.DataFrame(rows).sort_values("group_mmd2", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_covariate_shift_dataset(cov_shift=1.6, seed=2026):
    _, _, merchants, orders = generate_relational_data(
        seed=seed, cov_shift=cov_shift, cov_shift_attrs=COV_SHIFT_ATTRS)
    feat = aggregate_to_merchant(orders)
    feat = feat.reindex(merchants["merchant_id"].values)
    merchants = merchants.set_index("merchant_id").loc[feat.index].reset_index()
    W = merchants["W"].values.astype(int)
    # standardize on pooled data (RBF/MMD is scale sensitive)
    feat_std = (feat - feat.mean()) / feat.std().replace(0, 1)
    feat_std = feat_std.fillna(0.0)
    return merchants, feat_std, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plot", type=str, default="")
    args = ap.parse_args()

    merchants, feat_std, W = build_covariate_shift_dataset(
        cov_shift=args.cov_shift, seed=args.seed)
    feature_names = list(feat_std.columns)
    groups = {a: [f for f in feature_names if raw_attr_of(f) == a] for a in RAW_ATTRS}

    print("=" * 74)
    print("LOGO-MMD merchant-dimension COVARIATE-shift attribution")
    print("=" * 74)
    print(f"merchants (rows)      : {feat_std.shape[0]}  "
          f"(existing W=0: {(W == 0).sum()}, new W=1: {(W == 1).sum()})")
    print(f"merchant features     : {feat_std.shape[1]}  "
          f"(rich + rolling aggregations of {len(RAW_ATTRS)} raw attrs)")
    print(f"ground-truth shifted  : {COV_SHIFT_ATTRS}  (covariate / P(X) shift)")
    print()

    print("[1] Global MMD permutation test (all merchant features) ...")
    g_full = _median_gamma(feat_std.values)
    glob = mmd_permutation_test(feat_std.values, W, g_full,
                                n_perm=args.n_perm, seed=args.seed)
    print(f"    MMD^2={glob['mmd2']:.5f}  perm_mean={glob['perm_mean']:.5f}"
          f"  p_value={glob['p_value']:.4f}  "
          f"reject_H0={glob['p_value'] < 0.05}")
    print()

    print("[2] Headline attribution -- LOGO-MMD (leave-one-raw-attribute-out):")
    logo, mmd_full = logo_mmd(feat_std, W, groups)
    for _, r in logo.iterrows():
        flag = "  <== SHIFT" if r["is_shift"] else ""
        print(f"      {r['attribute']:14s} (n={int(r['n_features']):2d})  "
              f"MMD drop={r['logo_drop']:+.5f}{flag}")
    top_attrs = set(logo.head(len(COV_SHIFT_ATTRS))["attribute"])
    logo_ok = top_attrs == set(COV_SHIFT_ATTRS)
    print(f"    top-{len(COV_SHIFT_ATTRS)} = {sorted(top_attrs)}  "
          f"(expected {sorted(COV_SHIFT_ATTRS)}) -> {'MATCH' if logo_ok else 'NO MATCH'}")
    print()

    print("[3] Cross-check -- per-group standalone MMD + permutation p-value:")
    pg = per_group_mmd(feat_std, W, groups, n_perm=min(args.n_perm, 300),
                       seed=args.seed)
    for _, r in pg.iterrows():
        flag = "  <== SHIFT" if r["is_shift"] else ""
        print(f"      {r['attribute']:14s}  MMD^2={r['group_mmd2']:+.5f}  "
              f"p={r['p_value']:.4f}{flag}")
    print()

    ok = (glob["p_value"] < 0.05) and logo_ok
    print(f"[4] RESULT: {'PASS' if ok else 'CHECK'}  "
          f"(covariate shift {'detected' if glob['p_value'] < 0.05 else 'NOT detected'}"
          f"; LOGO-MMD attribution {'correct' if logo_ok else 'incorrect'})")

    if args.plot:
        _make_plot(logo, pg, args.plot)
        print(f"    saved chart -> {args.plot}")
    return 0 if ok else 1


def _make_plot(logo, pg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(14, 6))

    g = logo.iloc[::-1]
    axl.barh(g["attribute"], g["logo_drop"],
             color=["#d62728" if s else "#7f7f7f" for s in g["is_shift"]])
    axl.set_title("Headline: LOGO-MMD (leave-one-raw-attribute-out)")
    axl.set_xlabel("MMD drop when the group is removed  (MMD_full - MMD_without)")
    axl.axvline(0, color="k", lw=0.8)

    p = pg.iloc[::-1]
    axr.barh(p["attribute"], p["group_mmd2"],
             color=["#d62728" if s else "#7f7f7f" for s in p["is_shift"]])
    for y, (_, r) in enumerate(p.iterrows()):
        axr.text(r["group_mmd2"], y, f"  p={r['p_value']:.3f}",
                 va="center", fontsize=8)
    axr.set_title("Cross-check: per-group standalone MMD^2")
    axr.set_xlabel("group MMD^2 (batch 0 vs batch 1)")

    fig.suptitle("LOGO-MMD attribution of a merchant-dimension covariate shift  "
                 "(red = truly-shifted raw attribute)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
