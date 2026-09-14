#!/usr/bin/env python3
"""Statistical inference layer for the LOGO-MMD feature-selection attribution.

The point-estimate attribution in ``fsds_logo_mmd.py`` ranks merchant feature
groups by their LOGO-MMD importance, but a ranking alone is not a statistically
justified selection.  This module turns it into an inferential procedure:

1.  **Permutation validity** -- the per-group MMD permutation test is an exact
    test of H0: P0(X_group) = P1(X_group) under exchangeability.
2.  **Multiple-testing control** -- the six group p-values are corrected with
    Holm (FWER) and Benjamini-Hochberg (FDR); the selected set then has an
    explicit error guarantee.
3.  **Hierarchical (gatekeeping) testing** -- groups are the gate; individual
    features are examined only inside groups that pass, which respects the
    within-group correlation and is more powerful than flat per-feature LOCO.
4.  **Stability selection** -- stratified subsampling gives the sampling
    distribution of each group's LOGO-MMD importance: a bootstrap/percentile CI,
    a standard error, and a selection frequency.  Groups whose CI excludes 0 and
    whose selection frequency exceeds a threshold are the justified drivers
    (Meinshausen & Buhlmann, 2010).

Run:  python Python/fsds_vimp_inference.py --plot out.png
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from fsds_merchant_prototype import RAW_ATTRS, raw_attr_of
from fsds_logo_mmd import (
    COV_SHIFT_ATTRS, build_covariate_shift_dataset, per_group_mmd,
    _median_gamma, mmd2_unbiased,
)


# --------------------------------------------------------------------------- #
# LOGO-MMD importance for an arbitrary row subset (used by the resampler)
# --------------------------------------------------------------------------- #
def _logo_drops(Z, W, names, groups):
    g_full = _median_gamma(Z)
    mmd_full = mmd2_unbiased(Z[W == 0], Z[W == 1], g_full)
    out = {}
    for gname, members in groups.items():
        keep = np.array([nm not in set(members) for nm in names])
        Zsub = Z[:, keep]
        gsub = _median_gamma(Zsub)
        out[gname] = mmd_full - mmd2_unbiased(Zsub[W == 0], Zsub[W == 1], gsub)
    return out


def _resample_indices(W, rng, method, frac, jitter):
    idx0, idx1 = np.where(W == 0)[0], np.where(W == 1)[0]
    if method == "subsample":
        b0 = rng.choice(idx0, size=max(2, int(round(frac * len(idx0)))), replace=False)
        b1 = rng.choice(idx1, size=max(2, int(round(frac * len(idx1)))), replace=False)
    else:
        # bootstrap (with replacement): jitter (added by the caller) breaks the
        # exact ties that would otherwise bias the RBF-MMD U-statistic.
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
    return np.concatenate([b0, b1])


def resample_vimp(feat_std, W, groups, n_resample=300, frac=0.8,
                  method="subsample", jitter=1e-3, seed=2026):
    """Stability-selection style inference for the LOGO-MMD group importances."""
    names = np.asarray(feat_std.columns)
    Z = feat_std.values
    rng = np.random.default_rng(seed)

    point = _logo_drops(Z, W, names, groups)
    draws = {g: np.empty(n_resample) for g in groups}
    for b in range(n_resample):
        idx = _resample_indices(W, rng, method, frac, jitter)
        Zb, Wb = Z[idx], W[idx]
        if method == "bootstrap" and jitter > 0:
            Zb = Zb + rng.normal(0.0, jitter, Zb.shape)
        d = _logo_drops(Zb, Wb, names, groups)
        for g in groups:
            draws[g][b] = d[g]

    rows = []
    for g in groups:
        arr = draws[g]
        rows.append({
            "attribute": g,
            "logo_drop": point[g],
            "boot_mean": float(arr.mean()),
            "boot_se": float(arr.std(ddof=1)),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "sel_freq": float(np.mean(arr > 0)),
            "is_shift": g in COV_SHIFT_ATTRS,
        })
    return pd.DataFrame(rows).sort_values("logo_drop", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Group-level multiple-testing selection
# --------------------------------------------------------------------------- #
def group_selection(feat_std, W, groups, alpha=0.05, n_perm=500, seed=2026):
    pg = per_group_mmd(feat_std, W, groups, n_perm=n_perm, seed=seed)
    pvals = pg["p_value"].values
    holm = multipletests(pvals, alpha=alpha, method="holm")
    bh = multipletests(pvals, alpha=alpha, method="fdr_bh")
    pg = pg.assign(p_holm=holm[1], reject_holm=holm[0],
                   p_bh=bh[1], reject_bh=bh[0])
    return pg


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--n-resample", type=int, default=300)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--method", choices=["subsample", "bootstrap"], default="subsample")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--pi-thr", type=float, default=0.9, help="stability threshold")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plot", type=str, default="")
    args = ap.parse_args()

    merchants, feat_std, W = build_covariate_shift_dataset(
        cov_shift=args.cov_shift, seed=args.seed)
    feature_names = list(feat_std.columns)
    groups = {a: [f for f in feature_names if raw_attr_of(f) == a] for a in RAW_ATTRS}

    print("=" * 78)
    print("LOGO-MMD VIMP inference  (multiple testing + hierarchical + stability)")
    print("=" * 78)
    print(f"merchants={feat_std.shape[0]}  features={feat_std.shape[1]}  "
          f"groups={len(groups)}  ground-truth shift={COV_SHIFT_ATTRS}")
    print(f"resample: method={args.method} frac={args.frac} B={args.n_resample}  "
          f"stability threshold pi>={args.pi_thr}")
    print()

    # ---- (A) group-level testing with multiple-testing control ----
    print("[A] Group-level MMD test + multiple-testing correction:")
    pg = group_selection(feat_std, W, groups, alpha=args.alpha,
                         n_perm=args.n_perm, seed=args.seed)
    print(f"    {'attribute':14s} {'MMD^2':>9s} {'p_raw':>7s} {'p_holm':>7s} "
          f"{'p_bh':>7s}  select")
    for _, r in pg.iterrows():
        star = " <== SHIFT" if r["is_shift"] else ""
        sel = "HOLM+BH" if r["reject_holm"] else ("BH" if r["reject_bh"] else "-")
        print(f"    {r['attribute']:14s} {r['group_mmd2']:9.4f} {r['p_value']:7.4f} "
              f"{r['p_holm']:7.4f} {r['p_bh']:7.4f}  {sel}{star}")
    holm_set = set(pg.loc[pg["reject_holm"], "attribute"])
    print(f"    Holm-selected (FWER<= {args.alpha}): {sorted(holm_set)}")
    print()

    # ---- (B) stability-selection VIMP with bootstrap/subsample CIs ----
    print(f"[B] LOGO-MMD VIMP inference ({args.method}, B={args.n_resample}):")
    vimp = resample_vimp(feat_std, W, groups, n_resample=args.n_resample,
                         frac=args.frac, method=args.method, seed=args.seed)
    print(f"    {'attribute':14s} {'drop':>8s} {'95% CI':>18s} {'sel_freq':>9s}  "
          f"stable")
    stable_set = set()
    for _, r in vimp.iterrows():
        ci = f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]"
        ci_excl0 = r["ci_lo"] > 0
        stable = (r["sel_freq"] >= args.pi_thr) and ci_excl0
        if stable:
            stable_set.add(r["attribute"])
        star = " <== SHIFT" if r["is_shift"] else ""
        print(f"    {r['attribute']:14s} {r['logo_drop']:+8.4f} {ci:>18s} "
              f"{r['sel_freq']:9.2f}  {'YES' if stable else 'no'}{star}")
    print(f"    Stability-selected (pi>= {args.pi_thr} & CI>0): {sorted(stable_set)}")
    print()

    # ---- (C) hierarchical drill-down inside selected groups ----
    print("[C] Hierarchical drill-down (per-feature MMD inside selected groups):")
    gate = holm_set or stable_set
    for g in sorted(gate):
        members = groups[g]
        sub = {f"{g}::{m}": [m] for m in members}
        fpg = per_group_mmd(feat_std, W, sub, n_perm=200, seed=args.seed + 1)
        fpg["feature"] = [k.split("::", 1)[1] for k in fpg["attribute"]]
        top = fpg.sort_values("group_mmd2", ascending=False).head(3)
        tops = ", ".join(f"{t.feature}(MMD={t.group_mmd2:.3f},p={t.p_value:.3f})"
                         for t in top.itertuples())
        print(f"    [{g}] top features: {tops}")
    print()

    # ---- validation ----
    truth = set(COV_SHIFT_ATTRS)
    ok = (holm_set == truth) and (stable_set == truth)
    print(f"[D] RESULT: {'PASS' if ok else 'CHECK'}  "
          f"(Holm={sorted(holm_set)}, stability={sorted(stable_set)}, "
          f"truth={sorted(truth)})")

    if args.plot:
        _make_plot(vimp, pg, args.pi_thr, args.plot)
        print(f"    saved chart -> {args.plot}")
    return 0 if ok else 1


def _make_plot(vimp, pg, pi_thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 6))

    v = vimp.iloc[::-1]
    err = np.vstack([v["logo_drop"] - v["ci_lo"], v["ci_hi"] - v["logo_drop"]])
    colors = ["#d62728" if s else "#7f7f7f" for s in v["is_shift"]]
    axl.barh(v["attribute"], v["logo_drop"], xerr=err, color=colors,
             error_kw={"ecolor": "k", "capsize": 4, "lw": 1})
    axl.axvline(0, color="k", lw=0.8)
    axl.set_title("LOGO-MMD importance with 95% resample CI")
    axl.set_xlabel("MMD drop (leave-one-group-out)   [error bar = bootstrap CI]")

    s = vimp.sort_values("sel_freq").iloc[::-1].iloc[::-1]
    axr.barh(s["attribute"], s["sel_freq"],
             color=["#d62728" if x else "#7f7f7f" for x in s["is_shift"]])
    axr.axvline(pi_thr, color="b", ls="--", lw=1, label=f"pi threshold {pi_thr}")
    axr.set_xlim(0, 1.02)
    axr.set_title("Stability-selection frequency  P(LOGO-MMD drop > 0)")
    axr.set_xlabel("selection frequency across resamples")
    axr.legend(loc="lower right")

    fig.suptitle("Statistically-justified LOGO-MMD attribution: multiple-testing "
                 "+ stability selection\n(red = truly-shifted raw attribute)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
