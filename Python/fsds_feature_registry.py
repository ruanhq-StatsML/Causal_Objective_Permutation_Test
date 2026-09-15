#!/usr/bin/env python3
"""Automated feature-store drift-attribution registry (MMD-LOGO driven).

This is the production-facing entry point: given the shared order table, it runs
the MMD-LOGO attribution for each entity axis and emits a flat **feature drift
registry** -- one row per (axis, feature) -- ready for a feature-store dashboard,
alerting, or governance (refresh / deprecate / gate-retrain).

Design choices (per review):
* **MMD-LOGO is the headline importance.** A single **global median bandwidth**
  is fixed once per axis on the full standardized feature matrix, so every LOGO
  drop is computed with the *same* kernel and the group importances are directly
  comparable (this also removes the earlier "different-dim bandwidth" caveat).
* **Bootstrap stability.** Group LOGO importance gets a bootstrap CI + selection
  frequency (stratified resampling with jitter to break RBF-MMD ties).
* **Cross-feature / cross-tree FDR.** Group significance is BH-controlled; the
  within-group feature level uses the Benjamini-Bogomolov deflated level
  q * R/m, so the FDR over selected features across the tree is controlled.

Per feature the registry records: selection verdict, group LOGO importance +
bootstrap CI/frequency, group & feature p (raw and adjusted), and the
characterization (cohen_d, KS split threshold, direction, per-batch quantiles).
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from fsds_merchant_prototype import raw_attr_of
from fsds_logo_mmd import _median_gamma, mmd2_unbiased, mmd_permutation_test
from fsds_localization_tree import agg_family, _batch_summary, _best_split
from fsds_two_dimensional import generate_two_dim_data, aggregate_to_entity, DIMENSIONS


# --------------------------------------------------------------------------- #
# MMD-LOGO with a fixed GLOBAL bandwidth (comparable drops)
# --------------------------------------------------------------------------- #
def logo_global(Z, W, names, groups, gamma):
    mmd_full = mmd2_unbiased(Z[W == 0], Z[W == 1], gamma)
    drops = {}
    for g, members in groups.items():
        keep = np.array([nm not in set(members) for nm in names])
        Zsub = Z[:, keep]
        drops[g] = mmd_full - mmd2_unbiased(Zsub[W == 0], Zsub[W == 1], gamma)
    return mmd_full, drops


def bootstrap_logo(feat_std, W, groups, n_boot=150, jitter=1e-3, seed=2026):
    """Stratified bootstrap of the global-bandwidth LOGO drops -> CI + freq."""
    names = np.asarray(feat_std.columns)
    Z = feat_std.values
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.where(W == 0)[0], np.where(W == 1)[0]
    draws = {g: np.empty(n_boot) for g in groups}
    for b in range(n_boot):
        bi = np.concatenate([rng.choice(idx0, idx0.size, replace=True),
                             rng.choice(idx1, idx1.size, replace=True)])
        Zb = Z[bi] + rng.normal(0.0, jitter, Z[bi].shape)
        Wb = W[bi]
        gb = _median_gamma(Zb)
        _, d = logo_global(Zb, Wb, names, groups, gb)
        for g in groups:
            draws[g][b] = d[g]
    return {g: {"boot_mean": float(a.mean()), "ci_lo": float(np.percentile(a, 2.5)),
                "ci_hi": float(np.percentile(a, 97.5)), "sel_freq": float(np.mean(a > 0))}
            for g, a in draws.items()}


# --------------------------------------------------------------------------- #
# Registry for one axis
# --------------------------------------------------------------------------- #
def build_axis_registry(orders, dim, q=0.1, n_perm=400, n_boot=150, seed=2026):
    cfg = DIMENSIONS[dim]
    feat = aggregate_to_entity(orders, cfg["key"], cfg["other_ids"])
    ent = orders[[cfg["key"], cfg["label"]]].drop_duplicates() \
        .set_index(cfg["key"])[cfg["label"]]
    W = ent.reindex(feat.index).values.astype(int)
    feat_std = ((feat - feat.mean()) / feat.std().replace(0, 1)).fillna(0.0)
    names = list(feat_std.columns)
    attrs = sorted({raw_attr_of(f) for f in names})
    groups = {a: [f for f in names if raw_attr_of(f) == a] for a in attrs}

    Z = feat_std.values
    gamma_global = _median_gamma(Z)                              # ONE bandwidth
    global_test = mmd_permutation_test(Z, W, gamma_global, n_perm=n_perm, seed=seed)

    # ---- group level: LOGO (global gamma) + significance (per-group perm) ----
    _, logo_drops = logo_global(Z, W, np.asarray(names), groups, gamma_global)
    boot = bootstrap_logo(feat_std, W, groups, n_boot=n_boot, seed=seed)
    grp_p = {}
    for gi, (g, members) in enumerate(groups.items()):
        Zg = feat_std[members].values
        grp_p[g] = mmd_permutation_test(Zg, W, _median_gamma(Zg),
                                        n_perm=n_perm, seed=seed + gi)["p_value"]
    g_names = list(groups)
    g_padj = dict(zip(g_names, multipletests([grp_p[g] for g in g_names],
                                             alpha=q, method="fdr_bh")[1]))
    sel_groups = [g for g in g_names if (g_padj[g] < q and logo_drops[g] > 0
                                         and boot[g]["sel_freq"] >= 0.9)]
    R, m = len(sel_groups), len(g_names)
    q_feat = q * R / m if R > 0 else 0.0

    # ---- feature level within selected groups: per-feature perm + BB-FDR ----
    feat_selected, feat_p, feat_padj = set(), {}, {}
    for g in sel_groups:
        members = groups[g]
        fp = [mmd_permutation_test(feat_std[[f]].values, W,
                                   _median_gamma(feat_std[[f]].values),
                                   n_perm=n_perm, seed=seed + 17 + i)["p_value"]
              for i, f in enumerate(members)]
        rej, padj = multipletests(fp, alpha=q_feat, method="fdr_bh")[:2]
        for i, f in enumerate(members):
            feat_p[f], feat_padj[f] = fp[i], padj[i]
            if rej[i]:
                feat_selected.add(f)

    # ---- assemble one row per feature ----
    rows = []
    for f in names:
        g = raw_attr_of(f)
        vals = feat[f].values
        summ = _batch_summary(vals, W)
        split = _best_split(vals, W)
        final = f in feat_selected
        priority = max(logo_drops[g], 0.0) * abs(summ["cohen_d"]) if final else 0.0
        rows.append({
            "axis": dim, "feature": f, "raw_attr": g, "agg_family": agg_family(f),
            "group_selected": g in sel_groups, "group_logo_vimp": logo_drops[g],
            "group_logo_ci_lo": boot[g]["ci_lo"], "group_logo_ci_hi": boot[g]["ci_hi"],
            "group_sel_freq": boot[g]["sel_freq"], "group_padj": g_padj[g],
            "feature_p": feat_p.get(f, np.nan), "feature_padj": feat_padj.get(f, np.nan),
            "selected": final, "cohen_d": summ["cohen_d"],
            "mean_batch0": summ["batch0"]["mean"], "mean_batch1": summ["batch1"]["mean"],
            "split_threshold": split["threshold"], "split_ks": split["ks_stat"],
            "direction": split["direction"], "priority": priority,
        })
    reg = pd.DataFrame(rows)
    meta = {"axis": dim, "label": cfg["label"], "n_entities": int(len(feat_std)),
            "n_features": len(names), "global_mmd2": global_test["mmd2"],
            "global_p": global_test["p_value"], "selected_groups": sel_groups,
            "q_feat_effective": q_feat}
    return reg, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--n-perm", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    _, _, _, orders = generate_two_dim_data(cov_shift=args.cov_shift, seed=args.seed)

    regs, metas = [], {}
    for dim in ("merchant", "user"):
        reg, meta = build_axis_registry(orders, dim, q=args.q, n_perm=args.n_perm,
                                        n_boot=args.n_boot, seed=args.seed)
        regs.append(reg)
        metas[dim] = meta
        print("=" * 84)
        print(f"AXIS={dim}  label={meta['label']}  entities={meta['n_entities']}  "
              f"global MMD^2={meta['global_mmd2']:.4f} p={meta['global_p']:.4f}")
        print(f"selected groups (BH & bootstrap-stable): {meta['selected_groups']}  "
              f"q_feat_eff={meta['q_feat_effective']:.4f}")
        sel = reg[reg["selected"]].sort_values("priority", ascending=False)
        print(f"selected features: {len(sel)}  (top by priority)")
        for _, r in sel.head(8).iterrows():
            print(f"  {r['feature']:30s} vimp(LOGO)={r['group_logo_vimp']:+.4f} "
                  f"d={r['cohen_d']:+.2f} split@{r['split_threshold']:.3g} "
                  f"[{r['direction']}] prio={r['priority']:.3f}")
        print()

    registry = pd.concat(regs, ignore_index=True)
    # governance summary
    print("=" * 84)
    print("FEATURE-STORE GOVERNANCE SUMMARY")
    for dim in ("merchant", "user"):
        sub = registry[registry["axis"] == dim]
        print(f"  [{dim}] {int(sub['selected'].sum())}/{len(sub)} features flagged as "
              f"drifting; raw signals = "
              f"{sorted(sub.loc[sub['selected'], 'raw_attr'].unique())}")

    if args.csv:
        registry.to_csv(args.csv, index=False)
        print(f"saved registry CSV -> {args.csv}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"meta": metas,
                       "registry": registry.to_dict(orient="records")}, fh, indent=2)
        print(f"saved registry JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
