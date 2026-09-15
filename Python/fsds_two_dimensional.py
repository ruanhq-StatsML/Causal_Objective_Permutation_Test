#!/usr/bin/env python3
"""Two-dimensional FSDS: run the localization method along TWO entity axes.

The same order table is rolled up along two different entity dimensions --
merchant and buyer(user) -- and the multi-layer MMD localization is run on each,
so feature selection is done on a *second* dimension, not just the merchant one.

    orders (shared observation table)
      |-- roll up by merchant_id  --> merchant features  --> localize (W_merchant)
      |-- roll up by user_id      --> buyer   features   --> localize (W_user)

Each dimension has its own batch label and its own injected covariate shift:
    merchant axis (W_merchant): shift on {gmv, user_rating}
    buyer    axis (W_user):     shift on {basket_size, discount_rate}
Because orders mix both batch types across the *other* axis, the aggregation
decouples the two shifts, so each dimension should recover only its own drivers.

Output JSON:
    {"structure": ..., "dimensions": {"merchant": {...,"tree":<node>},
                                       "user": {...,"tree":<node>}}}
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from fsds_merchant_prototype import RAW_ATTRS, raw_attr_of, _random_merchant_names
from fsds_localization_tree import build_localization_tree, print_tree, selected_leaves


# --------------------------------------------------------------------------- #
# Generic aggregation of order-level attributes to an arbitrary entity axis
# --------------------------------------------------------------------------- #
def aggregate_to_entity(orders, key, other_id_cols, roll_window=5):
    """Rich + rolling aggregation of the raw order attributes to ``key``.

    Feature names are ``<attr>__<agg>`` (so ``raw_attr_of`` maps them back), plus
    ``n_orders`` and ``n_unique_<other>`` activity counts.
    """
    o = orders.sort_values([key, "order_number"]).reset_index(drop=True)
    g = o.groupby(key, sort=True)

    summary = {}
    for attr in RAW_ATTRS:
        s = g[attr]
        summary[f"{attr}__mean"] = s.mean()
        summary[f"{attr}__std"] = s.std().fillna(0.0)
        summary[f"{attr}__min"] = s.min()
        summary[f"{attr}__max"] = s.max()
        summary[f"{attr}__median"] = s.median()
        summary[f"{attr}__q25"] = s.quantile(0.25)
        summary[f"{attr}__q75"] = s.quantile(0.75)
        summary[f"{attr}__sum"] = s.sum()

    roll_feats = {}
    for attr in RAW_ATTRS:
        rmean = (o.groupby(key)[attr].rolling(roll_window, min_periods=1)
                 .mean().reset_index(level=0))
        rstd = (o.groupby(key)[attr].rolling(roll_window, min_periods=1)
                .std().reset_index(level=0))
        rmean.columns = [key, "v"]
        rstd.columns = [key, "v"]
        gm = rmean.groupby(key)["v"]
        gs = rstd.groupby(key)["v"]
        roll_feats[f"{attr}__roll{roll_window}_mean_last"] = gm.last()
        roll_feats[f"{attr}__roll{roll_window}_mean_avg"] = gm.mean()
        roll_feats[f"{attr}__roll{roll_window}_std_avg"] = gs.mean().fillna(0.0)

    counts = {"n_orders": g.size()}
    for c in other_id_cols:
        counts[f"n_unique_{c.replace('_id', '')}"] = g[c].nunique()

    feat = pd.concat([pd.DataFrame(summary), pd.DataFrame(roll_feats),
                      pd.DataFrame(counts)], axis=1)
    return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# --------------------------------------------------------------------------- #
# Data with two independent batch axes and two injected shifts
# --------------------------------------------------------------------------- #
def generate_two_dim_data(n_merchants=250, n_users=800, n_orders=20000,
                          items_per_merchant=10, cov_shift=1.6, seed=2026):
    rng = np.random.default_rng(seed)

    merchant_ids = np.array([f"M{100000 + i}" for i in range(n_merchants)])
    W_merch = rng.binomial(1, 0.5, n_merchants)
    merchants = pd.DataFrame({"merchant_id": merchant_ids, "W_merchant": W_merch,
                              "merchant_name": _random_merchant_names(n_merchants, seed + 1)})

    n_items = n_merchants * items_per_merchant
    item_ids = np.array([f"I{500000 + i}" for i in range(n_items)])
    items = pd.DataFrame({"item_id": item_ids,
                          "merchant_id": np.repeat(merchant_ids, items_per_merchant)})

    user_ids = np.array([f"U{200000 + i}" for i in range(n_users)])
    W_user = rng.binomial(1, 0.5, n_users)
    users = pd.DataFrame({"user_id": user_ids, "W_user": W_user})

    orders = pd.DataFrame({
        "order_number": rng.integers(100000000, 200000000, n_orders).astype(str),
        "item_id": rng.choice(item_ids, n_orders),
        "user_id": rng.choice(user_ids, n_orders),
    })
    orders = orders.merge(items, on="item_id", how="left")
    orders = orders.merge(merchants[["merchant_id", "W_merchant"]], on="merchant_id", how="left")
    orders = orders.merge(users, on="user_id", how="left")

    n = len(orders)
    orders["gmv"] = np.exp(rng.normal(3.0, 0.6, n))
    orders["quantity"] = rng.poisson(2.0, n) + 1
    orders["discount_rate"] = rng.beta(2, 8, n)
    orders["delivery_mins"] = rng.gamma(4.0, 8.0, n)
    orders["user_rating"] = np.clip(rng.normal(4.2, 0.7, n), 1, 5)
    orders["basket_size"] = (rng.poisson(3.0, n) + 1).astype(float)

    # merchant-axis shift (on the new merchants' orders)
    mm = orders["W_merchant"].values == 1
    orders.loc[mm, "gmv"] *= np.exp(cov_shift * 0.35)
    orders.loc[mm, "user_rating"] = np.clip(
        orders.loc[mm, "user_rating"].values - cov_shift * 0.6, 1, 5)

    # buyer-axis shift (on the new buyers' orders)
    uu = orders["W_user"].values == 1
    orders.loc[uu, "basket_size"] = orders.loc[uu, "basket_size"].values * (1 + cov_shift * 0.4)
    orders.loc[uu, "discount_rate"] = np.clip(
        orders.loc[uu, "discount_rate"].values + cov_shift * 0.06, 0, 1)

    return merchants, users, items, orders


DIMENSIONS = {
    "merchant": {"key": "merchant_id", "label": "W_merchant",
                 "other_ids": ["user_id", "item_id"],
                 "ground_truth": ["gmv", "user_rating"]},
    "user": {"key": "user_id", "label": "W_user",
             "other_ids": ["merchant_id", "item_id"],
             "ground_truth": ["basket_size", "discount_rate"]},
}


def run_dimension(orders, dim, q, n_perm, seed):
    cfg = DIMENSIONS[dim]
    feat = aggregate_to_entity(orders, cfg["key"], cfg["other_ids"])
    ent = orders[[cfg["key"], cfg["label"]]].drop_duplicates().set_index(cfg["key"])[cfg["label"]]
    W = ent.reindex(feat.index).values.astype(int)
    feat_std = ((feat - feat.mean()) / feat.std().replace(0, 1)).fillna(0.0)
    orders_dim = orders.assign(W=orders[cfg["label"]])
    tree = build_localization_tree(feat_std, feat, W, orders_dim, q=q,
                                   n_perm=n_perm, seed=seed)
    leaves = selected_leaves(tree["root"], [])
    sel_attrs = sorted({raw_attr_of(f) for f in leaves})
    leak = [f for f in leaves if raw_attr_of(f) not in cfg["ground_truth"]]
    return {"feat_std": feat_std, "W": W, "tree": tree, "leaves": leaves,
            "sel_attrs": sel_attrs, "leak": leak, "cfg": cfg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--n-perm", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    merchants, users, items, orders = generate_two_dim_data(
        cov_shift=args.cov_shift, seed=args.seed)

    out = {"structure": "two-dimensional FSDS over a shared order table "
                        "(rolled up along the merchant and buyer axes)",
           "dimensions": {}}
    all_ok = True
    for dim in ("merchant", "user"):
        res = run_dimension(orders, dim, args.q, args.n_perm, args.seed)
        cfg = res["cfg"]
        gt = res["tree"]["global_test"]
        print("=" * 80)
        print(f"DIMENSION = {dim.upper()}   (label {cfg['label']}, "
              f"rows={res['feat_std'].shape[0]}, features={res['feat_std'].shape[1]})")
        print(f"ground-truth shift = {cfg['ground_truth']}   "
              f"global MMD^2={gt['mmd2']:.4f} p={gt['p_value']:.4f} "
              f"reject={gt['p_value'] < args.q}")
        print("=" * 80)
        print_tree(res["tree"]["root"])
        ok = (gt["p_value"] < args.q) and set(res["sel_attrs"]) == set(cfg["ground_truth"]) \
            and len(res["leak"]) == 0 and len(res["leaves"]) > 0
        all_ok = all_ok and ok
        print(f"\n  selected leaves={len(res['leaves'])} across {res['sel_attrs']} "
              f"| false discoveries={len(res['leak'])} | {'PASS' if ok else 'CHECK'}\n")
        out["dimensions"][dim] = {
            "label": cfg["label"], "ground_truth": cfg["ground_truth"],
            "selected_attributes": res["sel_attrs"],
            "n_selected_leaves": len(res["leaves"]),
            "false_discoveries": len(res["leak"]),
            "tree": res["tree"],
        }

    print("=" * 80)
    print(f"TWO-DIMENSIONAL RESULT: {'PASS' if all_ok else 'CHECK'}  "
          f"(each dimension recovers only its own drivers)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"saved two-dimensional JSON -> {args.json}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
