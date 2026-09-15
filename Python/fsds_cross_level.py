#!/usr/bin/env python3
"""Two-level (conditional) subset localization -- with a results table.

Ground truth: new merchants (W=1) have a GMV covariate shift that is concentrated
in ONE product vertical (item category c*). The procedure should:
  Level 1 (merchant axis): localize the responsible raw-attribute subset -> gmv.
  Level 2 (product-vertical, CONDITIONAL on the gmv signal): among new vs existing
           merchants, localize the responsible category subset -> {c*}.

Level-1 uses MMD detection + leave-one-group-out (LOGO) ablation + bootstrap
stability (no multiple-testing). Level-2 uses a per-vertical new-vs-existing GMV
gap with a merchant-block permutation p-value (labels live at the merchant
level). Prints a two-level results table.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from fsds_logo_mmd import _median_gamma, mmd2_unbiased


N_CATEGORIES = 6
TRUE_VERTICAL = 2                       # c*: the vertical carrying the shift


def generate(n_merchants=200, items_per_merchant=12, n_orders=24000,
             shift=1.2, seed=2026):
    rng = np.random.default_rng(seed)
    mids = np.array([f"M{i}" for i in range(n_merchants)])
    W = rng.binomial(1, 0.5, n_merchants)
    n_items = n_merchants * items_per_merchant
    item_merchant = np.repeat(mids, items_per_merchant)
    item_W = np.repeat(W, items_per_merchant)          # merchant W per item
    item_cat = rng.integers(0, N_CATEGORIES, n_items)

    o_item = rng.integers(0, n_items, n_orders)
    merchant = item_merchant[o_item]
    cat = item_cat[o_item]
    Wm = item_W[o_item]
    gmv = np.exp(rng.normal(3.0, 0.6, n_orders))
    quantity = (rng.poisson(2.0, n_orders) + 1).astype(float)
    # shift concentrated in vertical c* for new merchants
    mask = (Wm == 1) & (cat == TRUE_VERTICAL)
    gmv[mask] *= np.exp(shift)

    orders = pd.DataFrame({"merchant_id": merchant, "W": Wm, "category": cat,
                           "gmv": gmv, "quantity": quantity})
    return orders, mids, W


# --------------------------------------------------------------------------- #
# Level 1: merchant-axis LOGO localization
# --------------------------------------------------------------------------- #
def _agg_merchant(orders):
    g = orders.groupby("merchant_id", sort=True)
    feat = {}
    for a in ["gmv", "quantity"]:
        feat[f"{a}__mean"] = g[a].mean()
        feat[f"{a}__std"] = g[a].std().fillna(0.0)
        feat[f"{a}__median"] = g[a].median()
        feat[f"{a}__q75"] = g[a].quantile(0.75)
    return pd.DataFrame(feat)


def _mmd_perm(Z, W, gamma, n_perm, seed):
    obs = mmd2_unbiased(Z[W == 0], Z[W == 1], gamma)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        Wp = rng.permutation(W)
        if mmd2_unbiased(Z[Wp == 0], Z[Wp == 1], gamma) >= obs:
            cnt += 1
    return obs, (1 + cnt) / (1 + n_perm)


def level1(orders, mids, Wm, n_perm=400, n_boot=100, seed=2026):
    feat = _agg_merchant(orders).reindex(mids)
    W = Wm.astype(int)
    fs = ((feat - feat.mean()) / feat.std().replace(0, 1)).fillna(0.0)
    names = list(fs.columns)
    groups = {"gmv": [c for c in names if c.startswith("gmv__")],
              "quantity": [c for c in names if c.startswith("quantity__")]}
    Z = fs.values
    gamma = _median_gamma(Z)
    mmd_full, p_full = _mmd_perm(Z, W, gamma, n_perm, seed)

    rows = []
    rng = np.random.default_rng(seed + 1)
    i0, i1 = np.where(W == 0)[0], np.where(W == 1)[0]
    for gname, cols in groups.items():
        keep = [c for c in names if c not in set(cols)]
        Zk = fs[keep].values
        logo = mmd_full - mmd2_unbiased(Zk[W == 0], Zk[W == 1], _median_gamma(Zk))
        # bootstrap stability of this group's standalone MMD > 0
        Zg = fs[cols].values
        pos = 0
        for _ in range(n_boot):
            bi = np.concatenate([rng.choice(i0, i0.size, True),
                                 rng.choice(i1, i1.size, True)])
            Zb = Zg[bi] + rng.normal(0, 1e-3, Zg[bi].shape)
            if mmd2_unbiased(Zb[W[bi] == 0], Zb[W[bi] == 1], _median_gamma(Zb)) > 0:
                pos += 1
        rows.append({"group": gname, "logo_vimp": logo, "stability": pos / n_boot})
    df = pd.DataFrame(rows).sort_values("logo_vimp", ascending=False)
    selected = list(df[(df.logo_vimp > 0) & (df.stability >= 0.9)]["group"])
    return df, selected, mmd_full, p_full


# --------------------------------------------------------------------------- #
# Level 2: conditional product-vertical localization (new vs existing GMV)
# --------------------------------------------------------------------------- #
def level2(orders, attr="gmv", n_perm=500, seed=2026):
    """Per-vertical new-vs-existing gap on the flagged attribute, with a
    merchant-block permutation p-value (labels are at the merchant level)."""
    # merchant-level W for block permutation
    m_w = orders.groupby("merchant_id")["W"].first()
    merchants = m_w.index.values
    w_m = m_w.values.astype(int)
    o_m_idx = np.searchsorted(merchants, orders["merchant_id"].values)
    cats = orders["category"].values
    a = orders[attr].values

    def gaps(w_order):
        out = {}
        for c in range(N_CATEGORIES):
            sel = cats == c
            g0 = a[sel & (w_order == 0)]
            g1 = a[sel & (w_order == 1)]
            out[c] = abs(g1.mean() - g0.mean()) if len(g0) and len(g1) else 0.0
        return out

    obs = gaps(w_m[o_m_idx])
    rng = np.random.default_rng(seed)
    ge = {c: 0 for c in range(N_CATEGORIES)}
    for _ in range(n_perm):
        wp_m = rng.permutation(w_m)                 # block permute at merchant level
        gp = gaps(wp_m[o_m_idx])
        for c in range(N_CATEGORIES):
            if gp[c] >= obs[c]:
                ge[c] += 1
    rows = [{"vertical": c, "newvs_existing_gap": obs[c],
             "perm_p": (1 + ge[c]) / (1 + n_perm)} for c in range(N_CATEGORIES)]
    df = pd.DataFrame(rows).sort_values("newvs_existing_gap", ascending=False)
    selected = list(df[df.perm_p < 0.05]["vertical"])
    return df, selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    orders, mids, W = generate(shift=args.shift, seed=args.seed)

    print("=" * 74)
    print("TWO-LEVEL CONDITIONAL SUBSET LOCALIZATION")
    print("=" * 74)
    print(f"ground truth: GMV shift concentrated in vertical c*={TRUE_VERTICAL} "
          f"for new merchants\n")

    l1, sel1, mmd1, p1 = level1(orders, mids, W, seed=args.seed)
    print(f"[Level 1: merchant axis]  global MMD^2={mmd1:.4f}  p={p1:.4f}")
    print(l1.to_string(index=False))
    print(f"  -> selected attribute subset: {sel1}\n")

    l2, sel2 = level2(orders, attr="gmv", seed=args.seed)
    print("[Level 2: product vertical | conditional on gmv, new vs existing]")
    print(l2.to_string(index=False))
    print(f"  -> selected vertical subset: {sel2}")

    ok = ("gmv" in sel1) and (TRUE_VERTICAL in sel2) and set(sel2) == {TRUE_VERTICAL}
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}  "
          f"(L1 -> gmv; L2 -> vertical {sel2}, truth {{{TRUE_VERTICAL}}})")
    if args.csv:
        l1.assign(level="L1_merchant").to_csv(args.csv, index=False)
        l2.assign(level="L2_vertical").to_csv(args.csv.replace(".csv", "_l2.csv"),
                                              index=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
