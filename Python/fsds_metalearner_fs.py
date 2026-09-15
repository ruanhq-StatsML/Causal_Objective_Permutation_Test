#!/usr/bin/env python3
"""CFPerm-style pipeline: meta-learner testing as feature selection, then
post-hoc subset localization -- both via the causal objective (R-risk).

The meta-learner (R-learner) is valid in the overlap regime: batch is a
"treatment" W, the outcome mechanism changes with W (concept drift), and the
propensity e(x) is non-degenerate. Two stages:

Stage A -- feature selection (the CFPerm logic).
    Cross-fit mu(x)=E[Y|x], e(x)=P(W=1|x); residualize Y_tilde, W_tilde; fit the
    R-learner tau and its R-risk. Leave-one-attribute-GROUP-out (robust to
    within-group correlation): dropping a group the meta-learner uses to explain
    the batch effect raises the R-risk. A permutation threshold (permute W)
    selects the causally-relevant attribute groups.

Stage B -- post-hoc subset localization.
    Within the selected groups, leave-one-FEATURE-out R-risk to rank the top
    driver features.

Simple selection throughout: one permutation threshold, then ranking. No MMD
here (MMD targets covariate P(X) shift; this pipeline targets concept drift).
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fsds_merchant_prototype import (
    build_merchant_dataset, _cross_fit_mu, _cross_fit_e, raw_attr_of, DRIFT_ATTRS,
)


def _rrisk(Xsub, Yt, Wt, clip=1e-3):
    mask = np.abs(Wt) > clip
    if mask.sum() < 5:
        return float(np.mean(Yt ** 2))
    tau = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    tau.fit(Xsub[mask], Yt[mask] / Wt[mask], ridge__sample_weight=Wt[mask] ** 2)
    return float(np.mean((Yt - tau.predict(Xsub) * Wt) ** 2))


def _loco(X, Yt, Wt, idx_map):
    """R-risk rise when each key's columns are removed (higher = more important)."""
    full = _rrisk(X, Yt, Wt)
    p = X.shape[1]
    return {k: _rrisk(X[:, np.setdiff1d(np.arange(p), idx)], Yt, Wt) - full
            for k, idx in idx_map.items()}


def _residualize(feat, Y, W, n_folds, seed):
    X = feat.values.astype(float)
    mu = _cross_fit_mu(X, np.asarray(Y, float), n_folds, seed)
    e = _cross_fit_e(X, np.asarray(W, int), n_folds, seed)
    return X, np.asarray(Y, float) - mu, np.asarray(W, int) - e, e


def stage_a_select_groups(feat, Yt, Wt, X, e, W, n_perm=80, seed=2026):
    names = list(feat.columns)
    attrs = sorted({raw_attr_of(f) for f in names})
    gidx = {a: np.array([i for i, f in enumerate(names) if raw_attr_of(f) == a])
            for a in attrs}
    vimp = _loco(X, Yt, Wt, gidx)
    rng = np.random.default_rng(seed + 1)
    null = []
    for _ in range(n_perm):
        null.extend(_loco(X, Yt, rng.permutation(np.asarray(W, int)) - e, gidx).values())
    thr = float(np.percentile(null, 95))
    ranked = sorted(attrs, key=lambda a: -vimp[a])
    selected = [a for a in ranked if vimp[a] > thr]
    return selected, vimp, thr, ranked


def stage_b_localize(feat, Yt, Wt, selected_groups):
    names = list(feat.columns)
    sel_features = [f for f in names if raw_attr_of(f) in selected_groups]
    X = feat[sel_features].values.astype(float)
    fidx = {f: np.array([i]) for i, f in enumerate(sel_features)}
    vimp = _loco(X, Yt, Wt, fidx)
    return sorted(vimp.items(), key=lambda kv: -kv[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=float, default=2.5)
    ap.add_argument("--n-perm", type=int, default=80)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    merchants, feat, Y, W = build_merchant_dataset(drift=args.drift, seed=args.seed)
    X, Yt, Wt, e = _residualize(feat, Y, W, 5, args.seed)

    print("=" * 78)
    print("STAGE A -- meta-learner testing as feature selection (R-risk group LOCO)")
    print("=" * 78)
    print(f"batch effect (concept drift) loads on {DRIFT_ATTRS}")
    groups, vimp, thr, ranked = stage_a_select_groups(
        feat, Yt, Wt, X, e, W, n_perm=args.n_perm, seed=args.seed)
    print(f"group LOCO VIMP (R-risk rise when removed); perm-95pct thr={thr:+.4f}")
    for a in ranked:
        flag = " <== drift attr" if a in DRIFT_ATTRS else ""
        mark = "*" if a in groups else " "
        print(f"  {mark} {a:14s} vimp={vimp[a]:+.4f}{flag}")
    print(f"\n-> selected groups: {groups}")
    if not groups:
        print("nothing selected; abort")
        return 1

    print("\n" + "=" * 78)
    print("STAGE B -- post-hoc subset localization (feature LOCO within selected)")
    print("=" * 78)
    drivers = stage_b_localize(feat, Yt, Wt, groups)
    print("top driver features:")
    for name, v in drivers[:10]:
        flag = " <== drift attr" if raw_attr_of(name) in DRIFT_ATTRS else ""
        print(f"    {name:32s} vimp={v:+.4f}{flag}")

    truth = set(DRIFT_ATTRS)
    top_attrs = {raw_attr_of(n) for n, _ in drivers[:6]}
    ok = truth.issubset(set(groups)) and truth.issubset(top_attrs)
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}  "
          f"(Stage-A groups={sorted(set(groups) & truth)}; "
          f"Stage-B top attrs={sorted(top_attrs & truth)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
