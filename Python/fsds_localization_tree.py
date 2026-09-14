#!/usr/bin/env python3
"""Multi-layer subset post-hoc localization of a merchant-dimension shift.

Once the global MMD test rejects H0: P0(X) = P1(X), this module drills the
attribution down through a hierarchy of nested feature subsets, re-testing MMD
on each subset and controlling the error with a Benjamini-Bogomolov style
hierarchical FDR (the effective level is deflated by R/m at every split, so the
FDR over the discovered leaves across the tree is controlled).

Layers
------
    root  (all merchant features)
      -> raw attribute        (gmv, user_rating, ... ; the item->merchant roll-up)
        -> aggregation family (summary stats vs rolling-window vs activity)
          -> feature          (gmv__mean, gmv__roll5_mean_avg, ...)  [leaf]

Each node carries its MMD test, its adjusted p-value / selection flag, and
observation-level statistics (per-batch merchant-level summary + the underlying
order-level distribution of the raw attribute).  The tree is printed and dumped
to JSON:  {relation, aggregation, global_test, root: {..., children: [...]}}.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from statsmodels.stats.multitest import multipletests

from fsds_merchant_prototype import (
    RAW_ATTRS, generate_relational_data, aggregate_to_merchant, raw_attr_of,
)
from fsds_logo_mmd import COV_SHIFT_ATTRS, _median_gamma, mmd_permutation_test


# --------------------------------------------------------------------------- #
# Partitioning of a node's feature set into child subsets
# --------------------------------------------------------------------------- #
def agg_family(fname):
    if raw_attr_of(fname) == "count":
        return "activity"
    suffix = fname.split("__", 1)[1]
    return "rolling" if "roll" in suffix else "summary"


def children_of(kind, cols):
    """Return {child_name: [feature cols]} and the child kind, or (None, None)."""
    if kind == "root":
        part = {}
        for f in cols:
            part.setdefault(raw_attr_of(f), []).append(f)
        return part, "attribute"
    if kind == "attribute":
        part = {}
        for f in cols:
            part.setdefault(agg_family(f), []).append(f)
        return part, "family"
    if kind == "family":
        return {f: [f] for f in cols}, "feature"
    return None, None


# --------------------------------------------------------------------------- #
# MMD test + observation-level statistics
# --------------------------------------------------------------------------- #
def _mmd_test(cols, W, feat_std, n_perm, seed):
    Z = feat_std[cols].values
    g = _median_gamma(Z)
    r = mmd_permutation_test(Z, W, g, n_perm=n_perm, seed=seed)
    return {"mmd2": float(r["mmd2"]), "p_value": float(r["p_value"]),
            "perm_mean": float(r["perm_mean"])}


def _batch_summary(values, W):
    a, b = values[W == 0], values[W == 1]
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) or 1.0
    return {
        "batch0": {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
                   "median": float(np.median(a))},
        "batch1": {"n": int(b.size), "mean": float(b.mean()), "std": float(b.std()),
                   "median": float(np.median(b))},
        "mean_shift": float(b.mean() - a.mean()),
        "cohen_d": float((b.mean() - a.mean()) / pooled),
    }


def _feature_obs_stats(fname, feat_raw, W, order_stats):
    obs = {"merchant_level": _batch_summary(feat_raw[fname].values, W)}
    ra = raw_attr_of(fname)
    if ra in order_stats:
        obs["order_level_raw_attr"] = {"attribute": ra, **order_stats[ra]}
    return obs


def _order_level_stats(orders):
    out = {}
    for a in RAW_ATTRS:
        g = orders.groupby("W")[a]
        m = g.mean().to_dict()
        s = g.std().to_dict()
        out[a] = {"batch0_order_mean": float(m.get(0, np.nan)),
                  "batch1_order_mean": float(m.get(1, np.nan)),
                  "batch0_order_std": float(s.get(0, np.nan)),
                  "batch1_order_std": float(s.get(1, np.nan))}
    return out


# --------------------------------------------------------------------------- #
# Recursive localization
# --------------------------------------------------------------------------- #
def _node(name, kind, depth, cols, test, selected, p_adjusted,
          feat_raw, W, order_stats, is_leaf):
    node = {
        "name": name, "kind": kind, "depth": depth, "n_features": len(cols),
        "selected": bool(selected),
        "p_adjusted": None if p_adjusted is None else float(p_adjusted),
        "stats": test,
        "children": [],
    }
    if is_leaf:
        node["observation_level"] = _feature_obs_stats(cols[0], feat_raw, W, order_stats)
    else:
        node["members"] = cols if len(cols) <= 12 else f"{len(cols)} features"
    return node


def expand(node, kind, cols, q_eff, W, feat_std, feat_raw, order_stats,
           n_perm, seed, max_depth):
    if node["depth"] >= max_depth or q_eff <= 0:
        return
    part, child_kind = children_of(kind, cols)
    if not part:
        return
    names = list(part)
    tests = [_mmd_test(part[cn], W, feat_std, n_perm, seed + 3 * i)
             for i, cn in enumerate(names)]
    pvals = np.array([t["p_value"] for t in tests])
    rej, padj = multipletests(pvals, alpha=q_eff, method="fdr_bh")[:2]
    R, m = int(rej.sum()), len(names)
    q_child = q_eff * R / m if R > 0 else 0.0

    order = np.argsort(pvals)
    for i in order:
        cn = names[i]
        is_leaf = (child_kind == "feature")
        child = _node(cn, child_kind, node["depth"] + 1, part[cn], tests[i],
                      bool(rej[i]), padj[i], feat_raw, W, order_stats, is_leaf)
        node["children"].append(child)
        if rej[i] and not is_leaf:
            expand(child, child_kind, part[cn], q_child, W, feat_std, feat_raw,
                   order_stats, n_perm, seed + 101 + i, max_depth)


def build_localization_tree(feat_std, feat_raw, W, orders, q=0.1, n_perm=400,
                            max_depth=3, seed=2026):
    cols = list(feat_std.columns)
    order_stats = _order_level_stats(orders)
    global_test = _mmd_test(cols, W, feat_std, n_perm, seed)
    root = _node("merchant_feature_space", "root", 0, cols, global_test,
                 selected=True, p_adjusted=None, feat_raw=feat_raw, W=W,
                 order_stats=order_stats, is_leaf=False)
    root["members"] = f"{len(cols)} features"
    if global_test["p_value"] < q:  # only localize after a global rejection
        expand(root, "root", cols, q, W, feat_std, feat_raw, order_stats,
               n_perm, seed + 1, max_depth)
    return {
        "relation_path": ["merchant", "item", "order"],
        "aggregation": "order -> item -> merchant (rich + rolling)",
        "method": "multi-layer subset post-hoc localization; MMD permutation "
                  "test per subset; Benjamini-Bogomolov hierarchical FDR",
        "fdr_level_q": q,
        "global_test": global_test,
        "root": root,
    }


# --------------------------------------------------------------------------- #
# Pretty ASCII print + leaf collection
# --------------------------------------------------------------------------- #
def print_tree(node, prefix=""):
    tag = "SELECT" if node["selected"] else "  --  "
    padj = "" if node["p_adjusted"] is None else f" padj={node['p_adjusted']:.4f}"
    d = ""
    if "observation_level" in node:
        c = node["observation_level"]["merchant_level"]["cohen_d"]
        d = f" cohen_d={c:+.2f}"
    print(f"{prefix}[{tag}] {node['name']:22s} "
          f"MMD2={node['stats']['mmd2']:+.4f} p={node['stats']['p_value']:.4f}"
          f"{padj}{d}")
    for ch in node["children"]:
        print_tree(ch, prefix + "    ")


def selected_leaves(node, acc):
    if not node["children"] and node["kind"] == "feature" and node["selected"]:
        acc.append(node["name"])
    for ch in node["children"]:
        selected_leaves(ch, acc)
    return acc


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _load(cov_shift, seed):
    _, _, merchants, orders = generate_relational_data(
        seed=seed, cov_shift=cov_shift, cov_shift_attrs=COV_SHIFT_ATTRS)
    feat = aggregate_to_merchant(orders)
    feat = feat.reindex(merchants["merchant_id"].values)
    merchants = merchants.set_index("merchant_id").loc[feat.index].reset_index()
    W = merchants["W"].values.astype(int)
    feat_std = ((feat - feat.mean()) / feat.std().replace(0, 1)).fillna(0.0)
    return merchants, feat, feat_std, W, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--n-perm", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    merchants, feat, feat_std, W, orders = _load(args.cov_shift, args.seed)
    tree = build_localization_tree(feat_std, feat, W, orders, q=args.q,
                                   n_perm=args.n_perm, max_depth=args.max_depth,
                                   seed=args.seed)

    print("=" * 78)
    print("Multi-layer subset post-hoc localization tree  (MMD + hierarchical FDR)")
    print("=" * 78)
    print(f"merchants={feat_std.shape[0]}  features={feat_std.shape[1]}  "
          f"q={args.q}  ground-truth shift={COV_SHIFT_ATTRS}")
    gt = tree["global_test"]
    print(f"global MMD^2={gt['mmd2']:.4f}  p={gt['p_value']:.4f}  "
          f"reject_H0={gt['p_value'] < args.q}")
    print()
    print_tree(tree["root"])
    print()

    leaves = selected_leaves(tree["root"], [])
    leak = [f for f in leaves if raw_attr_of(f) not in COV_SHIFT_ATTRS]
    sel_attrs = sorted({raw_attr_of(f) for f in leaves})
    print(f"selected leaf features: {len(leaves)}  across attributes {sel_attrs}")
    print(f"false discoveries (non-shifted attrs): {len(leak)}")
    ok = (gt["p_value"] < args.q) and set(sel_attrs) == set(COV_SHIFT_ATTRS) \
        and len(leak) == 0 and len(leaves) > 0
    print(f"RESULT: {'PASS' if ok else 'CHECK'}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(tree, fh, indent=2)
        print(f"saved JSON tree -> {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
