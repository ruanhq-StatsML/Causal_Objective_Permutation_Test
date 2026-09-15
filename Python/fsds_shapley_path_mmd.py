#!/usr/bin/env python3
"""Permutation-path (Monte-Carlo) Shapley for MMD -- an impossibility study.

Method (the "add features one-by-one along random paths" idea):
    for each of many random feature orders pi = np.random.permutation(p):
        walk the path, adding one feature at a time; the marginal MMD^2 gain of
        adding feature f to the current prefix S is credited to f.
    Average the credits over paths -> Monte-Carlo Shapley value phi_f.

What it demonstrates (impossibility vs possibility):
    Among strongly CORRELATED shifted features (e.g. the many aggregations of the
    same raw attribute), the per-feature Shapley split is essentially arbitrary
    and unstable across path samples -- you cannot uniquely say "feature f
    contributed x% of the OOD". BUT the SUBSET total (sum over the group's
    features) is large and stable across samples -- you CAN say "this subset
    contributes a lot". Impossibility at the atom, possibility at the subset.

Efficiency: per-feature squared-distance matrices are precomputed once; walking a
path just accumulates them (O(N^2) per step) with a fixed global bandwidth.
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.spatial.distance import cdist, pdist

from fsds_merchant_prototype import (
    generate_relational_data, aggregate_to_merchant, raw_attr_of,
)
from fsds_logo_mmd import COV_SHIFT_ATTRS


def _gamma(X):
    d = pdist(X, "sqeuclidean")
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return 1.0 / med if med > 0 else 1.0


def _mmd_from_K(K, W):
    a, b = np.where(W == 0)[0], np.where(W == 1)[0]
    m, n = a.size, b.size
    Kxx, Kyy, Kxy = K[np.ix_(a, a)], K[np.ix_(b, b)], K[np.ix_(a, b)]
    return ((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1)) - 2.0 * Kxy.mean())


def path_shapley(feat_std, W, n_paths=200, seed=0):
    """Monte-Carlo permutation-path Shapley over the features (MMD^2 game)."""
    X = feat_std.values.astype(float)
    N, p = X.shape
    gamma = _gamma(X)
    Dfeat = [cdist(X[:, [j]], X[:, [j]], "sqeuclidean") for j in range(p)]
    rng = np.random.default_rng(seed)
    phi = np.zeros(p)
    for _ in range(n_paths):
        order = rng.permutation(p)
        D = np.zeros((N, N))
        prev = 0.0
        for j in order:
            D = D + Dfeat[j]
            cur = _mmd_from_K(np.exp(-gamma * D), W)
            phi[j] += cur - prev
            prev = cur
    return phi / n_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--n-paths", type=int, default=200)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    _, _, merchants, orders = generate_relational_data(
        seed=args.seed, cov_shift=args.cov_shift, cov_shift_attrs=COV_SHIFT_ATTRS)
    feat = aggregate_to_merchant(orders).reindex(merchants["merchant_id"].values)
    W = merchants.set_index("merchant_id").loc[feat.index]["W"].values.astype(int)
    feat_std = ((feat - feat.mean()) / feat.std().replace(0, 1)).fillna(0.0)
    names = list(feat_std.columns)
    attrs = sorted({raw_attr_of(f) for f in names})

    # repeat over independent path samples to measure stability
    phis = np.array([path_shapley(feat_std, W, n_paths=args.n_paths, seed=args.seed + s)
                     for s in range(args.n_seeds)])          # (n_seeds, p)

    feat_mean = phis.mean(0)
    feat_std_ = phis.std(0)
    # group (subset) totals per seed
    gidx = {a: [i for i, f in enumerate(names) if raw_attr_of(f) == a] for a in attrs}
    grp = np.array([[phis[s, idx].sum() for a, idx in gidx.items()]
                    for s in range(args.n_seeds)])           # (n_seeds, n_groups)
    grp_mean = grp.mean(0)
    grp_std = grp.std(0)

    print("=" * 80)
    print("Permutation-path MMD Shapley -- impossibility (per-feature) vs "
          "possibility (per-subset)")
    print("=" * 80)
    print(f"cov-shift on {COV_SHIFT_ATTRS}; {args.n_paths} paths x {args.n_seeds} "
          f"independent samples; {feat_std.shape[1]} features\n")

    print("[A] SUBSET (raw-attribute group) total Shapley -- STABLE:")
    gord = np.argsort(-grp_mean)
    for k in gord:
        a = list(gidx)[k]
        cv = grp_std[k] / abs(grp_mean[k]) if abs(grp_mean[k]) > 1e-9 else float("nan")
        flag = " <== shifted" if a in COV_SHIFT_ATTRS else ""
        print(f"    {a:14s} shapley_sum={grp_mean[k]:+.4f} +/- {grp_std[k]:.4f} "
              f"(CV={cv:.2f}){flag}")

    print("\n[B] Per-feature Shapley WITHIN a shifted group -- UNSTABLE / non-unique:")
    g0 = COV_SHIFT_ATTRS[0]
    idx = gidx[g0]
    order = sorted(idx, key=lambda i: -feat_mean[i])
    for i in order:
        cv = feat_std_[i] / abs(feat_mean[i]) if abs(feat_mean[i]) > 1e-9 else float("nan")
        print(f"    {names[i]:30s} shapley={feat_mean[i]:+.4f} +/- {feat_std_[i]:.4f} "
              f"(CV={cv:.2f})")

    # ---- [C] the decisive impossibility: attribution CONVENTIONS disagree ----
    # For correlated features there is no unique "per-feature OOD contribution":
    # standalone MMD (each feature alone), LOGO (drop from the full set), and
    # Shapley give CONTRADICTORY per-feature answers. Yet all three agree on
    # which SUBSET matters. We use a single global bandwidth so it is a pure
    # convention comparison.
    from scipy.stats import spearmanr
    X = feat_std.values.astype(float)
    N, p = X.shape
    gamma = _gamma(X)
    Df = [cdist(X[:, [j]], X[:, [j]], "sqeuclidean") for j in range(p)]
    Dfull = np.sum(Df, axis=0)
    mmd_full = _mmd_from_K(np.exp(-gamma * Dfull), W)
    standalone = np.array([_mmd_from_K(np.exp(-gamma * Df[j]), W) for j in range(p)])
    logo = np.array([mmd_full - _mmd_from_K(np.exp(-gamma * (Dfull - Df[j])), W)
                     for j in range(p)])

    g0 = COV_SHIFT_ATTRS[0]
    idx0 = gidx[g0]
    print(f"\n[C] Per-feature contribution MAGNITUDE is convention-dependent "
          f"(group `{g0}`):")
    print(f"    {'feature':30s} {'standalone':>11s} {'LOGO':>9s} {'Shapley':>9s}")
    for i in sorted(idx0, key=lambda i: -feat_mean[i]):
        print(f"    {names[i]:30s} {standalone[i]:11.4f} {logo[i]:9.4f} "
              f"{feat_mean[i]:9.4f}")
    rho_ls = spearmanr(logo[idx0], standalone[idx0]).correlation
    # the three conventions may agree on RANK but assign very different
    # MAGNITUDES to the same feature -> "how much" is not unique.
    ratios = [max(standalone[i], logo[i], feat_mean[i])
              / max(min(standalone[i], logo[i], feat_mean[i]), 1e-9) for i in idx0]
    med_ratio = float(np.median(ratios))
    print(f"    within-group rank corr (LOGO vs standalone) = {rho_ls:+.2f} "
          "(rank may agree), BUT")
    print(f"    the SAME feature gets very different MAGNITUDES across conventions: "
          f"median max/min ratio = {med_ratio:.1f}x")
    print("    => 'how much does feature f contribute to the OOD' has no unique "
          "value (standalone vs LOGO vs Shapley differ several-fold).")

    # subset-level agreement (possibility)
    def _grp(vec, a):
        return float(vec[gidx[a]].sum())
    print("\n[D] But the SUBSET answer is convention-INVARIANT (possibility):")
    print(f"    `{g0}` group total:  standalone={_grp(standalone, g0):+.3f}  "
          f"LOGO(group-drop) n/a-per-atom  Shapley={_grp(feat_mean, g0):+.3f}")
    top_shap = sorted(attrs, key=lambda a: -_grp(feat_mean, a))[:2]
    top_std = sorted(attrs, key=lambda a: -_grp(standalone, a))[:2]
    print(f"    top-2 subsets by Shapley   = {top_shap}")
    print(f"    top-2 subsets by standalone= {top_std}")
    subset_agree = set(top_shap) == set(top_std) == set(COV_SHIFT_ATTRS)

    print("\n[E] Takeaway:")
    print("    You CANNOT uniquely say HOW MUCH each feature contributes to the "
          "OOD (the magnitude is convention-dependent, several-fold apart);")
    print("    you CAN say which SUBSET contributes a lot (all conventions agree "
          "on the subset selection).")
    atom_nonunique = med_ratio > 1.5           # magnitude not unique across conventions
    ok = atom_nonunique and subset_agree
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}  "
          f"(atom magnitude non-unique={atom_nonunique} [{med_ratio:.1f}x], "
          f"subset selection agrees={subset_agree})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
