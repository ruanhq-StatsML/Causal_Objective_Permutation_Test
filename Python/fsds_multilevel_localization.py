#!/usr/bin/env python3
"""Multi-level subset post-hoc localization of a merchant-dimension shift.

Generalizes the two-level FS into an arbitrary-depth **feature tree** and drills
down layer by layer, testing a *subset* of merchant features at every node with
an MMD permutation test and only descending into significant nodes:

    root (all features)
      +-- raw attribute group        (gmv, user_rating, quantity, ...)
            +-- aggregation family    (location / spread / mass / rolling)
                  +-- individual feature (leaf)

FDR is controlled down the tree in the Benjamini-Bogomolov (2014) style: the
children of a selected node are tested at the running level deflated by the
parent layer's selection ratio (R/m).  Descendants of non-selected nodes are
never tested, which is exactly the power gain of hierarchical localization.

Run:  python Python/fsds_multilevel_localization.py --plot out.png
"""
from __future__ import annotations

import argparse
import pandas as pd
from statsmodels.stats.multitest import multipletests

from fsds_merchant_prototype import RAW_ATTRS, raw_attr_of
from fsds_logo_mmd import (
    COV_SHIFT_ATTRS, build_covariate_shift_dataset, _median_gamma,
    mmd_permutation_test,
)

# aggregation-type sub-families (partition the 11 aggregations of each raw attr)
AGG_FAMILIES = {
    "location": ["mean", "median", "q25", "q75"],
    "spread": ["std", "roll5_std_avg"],
    "mass": ["min", "max", "sum"],
    "rolling": ["roll5_mean_last", "roll5_mean_avg"],
}


def build_feature_tree(feature_names):
    """Nested dict: internal node -> dict of children; leaf-family -> list of cols."""
    names = set(feature_names)
    tree = {}
    for a in RAW_ATTRS:
        node = {}
        for fam, suffixes in AGG_FAMILIES.items():
            cols = [f"{a}__{s}" for s in suffixes if f"{a}__{s}" in names]
            if cols:
                node[fam] = cols
        if node:
            tree[a] = node
    activity = [f for f in feature_names if raw_attr_of(f) == "count"]
    if activity:
        tree["activity"] = activity
    return tree


def _cols_under(node):
    if isinstance(node, list):
        return list(node)
    out = []
    for child in node.values():
        out += _cols_under(child)
    return out


def _subset_test(feat_std, W, cols, n_perm, seed):
    Z = feat_std[cols].values
    gamma = _median_gamma(Z)
    res = mmd_permutation_test(Z, W, gamma, n_perm=n_perm, seed=seed)
    return res["mmd2"], res["p_value"]


def localize(feat_std, W, name, node, alpha, n_perm, seed, depth, records):
    """Recursively test + select the children of an already-selected node."""
    indent = "    " + "  " * depth
    # leaf-family: children are individual features
    if isinstance(node, list):
        children = [(f, [f]) for f in node]
    else:
        children = [(cn, child) for cn, child in node.items()]

    pvals, mmds = [], []
    for cn, child in children:
        cols = child if isinstance(child, list) else _cols_under(child)
        m2, p = _subset_test(feat_std, W, cols, n_perm, seed + hash(cn) % 1000)
        pvals.append(p)
        mmds.append(m2)
    rej = multipletests(pvals, alpha=alpha, method="fdr_bh")
    R, m = int(rej[0].sum()), len(children)
    alpha_child = alpha * R / m if R > 0 else 0.0

    selected_leaves = []
    for (cn, child), p, padj, sel, m2 in zip(children, pvals, rej[1], rej[0], mmds):
        is_leaf = isinstance(child, list) and len(child) == 1
        kind = "leaf" if is_leaf else ("leaf-family" if isinstance(child, list)
                                       else "family")
        mark = "SELECT" if sel else "-"
        records.append({"depth": depth, "parent": name, "node": cn, "kind": kind,
                        "mmd2": m2, "p_raw": p, "p_adj": padj, "selected": bool(sel),
                        "raw_attr": (raw_attr_of(child[0]) if isinstance(child, list)
                                     and child else name)})
        print(f"{indent}{cn:22s} MMD^2={m2:+.4f} p={p:.4f} adj={padj:.4f}  {mark}")
        if sel:
            if is_leaf:
                selected_leaves.append(child[0])
            else:
                selected_leaves += localize(feat_std, W, cn, child, alpha_child,
                                            n_perm, seed + 1, depth + 1, records)
    if R > 0 and not isinstance(node, list):
        print(f"{indent}-> {R}/{m} selected; children FDR level -> {alpha_child:.4f}")
    return selected_leaves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--q", type=float, default=0.1, help="root FDR level")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plot", type=str, default="")
    args = ap.parse_args()

    merchants, feat_std, W = build_covariate_shift_dataset(
        cov_shift=args.cov_shift, seed=args.seed)
    feature_names = list(feat_std.columns)
    tree = build_feature_tree(feature_names)

    print("=" * 80)
    print("Multi-level subset post-hoc localization (hierarchical MMD + BB-FDR)")
    print("=" * 80)
    print(f"merchants={feat_std.shape[0]}  features={feat_std.shape[1]}  "
          f"root FDR q={args.q}  ground-truth shift={COV_SHIFT_ATTRS}")
    print()

    # ---- root / global gate ----
    m2_root, p_root = _subset_test(feat_std, W, feature_names, args.n_perm, args.seed)
    print(f"[root] all features: MMD^2={m2_root:.4f} p={p_root:.4f}  "
          f"-> {'REJECT H0, drill down' if p_root < args.q else 'no shift, stop'}")
    if p_root >= args.q:
        return 1
    print()

    print("[drill-down]  (only significant nodes are expanded)")
    records = []
    leaves = localize(feat_std, W, "root", tree, args.q, args.n_perm,
                      args.seed + 3, 0, records)
    print()

    # ---- summary + validation ----
    rec = pd.DataFrame(records)
    leaf_attrs = sorted({raw_attr_of(f) for f in leaves})
    leak = [f for f in leaves if raw_attr_of(f) not in COV_SHIFT_ATTRS]
    sel_families = rec[(rec["kind"] == "leaf-family") & rec["selected"]]
    print(f"[summary] selected attribute groups (L1): "
          f"{sorted(set(rec[(rec.depth == 0) & rec.selected].node))}")
    print(f"          selected aggregation families (L2): "
          f"{sorted(set(sel_families['parent'] + '/' + sel_families['node']))}")
    print(f"          selected leaf features (L3): {len(leaves)} "
          f"(attrs={leaf_attrs}, false-discoveries={len(leak)})")
    ok = set(leaf_attrs) == set(COV_SHIFT_ATTRS) and len(leak) == 0 and len(leaves) > 0
    print(f"[RESULT] {'PASS' if ok else 'CHECK'}  "
          f"(localized to {leaf_attrs}, expected {sorted(COV_SHIFT_ATTRS)})")

    if args.plot:
        _make_plot(rec, args.plot)
        print(f"    saved chart -> {args.plot}")
    return 0 if ok else 1


def _make_plot(rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, depth, title in zip(
            axes, [0, 1, 2],
            ["L1: raw-attribute groups", "L2: aggregation families "
             "(within selected)", "L3: individual features (within selected)"]):
        d = rec[rec["depth"] == depth].copy()
        d = d.sort_values("mmd2").tail(14)
        labels = (d["node"] if depth != 1
                  else (d["parent"] + "/" + d["node"]))
        colors = ["#d62728" if s else "#7f7f7f" for s in d["selected"]]
        ax.barh(labels, d["mmd2"], color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("subset MMD^2")
    fig.suptitle("Multi-level subset post-hoc localization  "
                 "(red = selected at that layer's BB-FDR level)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
