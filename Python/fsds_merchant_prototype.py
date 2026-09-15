#!/usr/bin/env python3
"""FSDS (Feature-Selection Distribution-Shift) attribution prototype.

Hierarchy / relational schema
-----------------------------
    user  --places-->  order  --for-->  item  --sold_by-->  merchant

Each conversion (a user buying an item) generates one *order* with an
``order_number``.  The item -> merchant edge is what lets us roll everything up
to the **merchant dimension**: order -> item -> merchant.

What this prototype does
------------------------
1.  Generates the relational tables (users, items, merchants, orders) with a
    handful of raw order-level attributes.
2.  Splits merchants into an *existing* batch (W = 0) and a *new* batch (W = 1)
    and injects a distribution shift on a KNOWN subset of the raw attributes for
    the new batch (this is the ground truth for the attribution).
3.  Aggregates the order-level attributes up to the merchant dimension using
    *rich* aggregations (mean / std / min / max / median / quantiles / sum) plus
    *rolling-window* aggregations over each merchant's order stream -- not just a
    one-pass standardized mean.
4.  Runs the causal-objective distribution-shift test (PO-risk permute-then-refit,
    mirroring ``DRPerm.py``) to detect whether the two batches differ.
5.  Runs **PO-risk + LOCO** (mirroring ``R_risk_loco.py``) to attribute the shift
    to individual merchant features, and checks that the aggregations derived
    from the truly-shifted raw attributes rank at the top.

It is self-contained (numpy / pandas / scikit-learn) but uses the repo's own
``ModelRegistry`` rf_regressor as the outcome / tau learner so it exercises the
project code.  Run it with the repo's ``Python/`` dir on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier

from model_registry_class import ModelRegistry  # repo code

# --------------------------------------------------------------------------- #
# Outcome / tau learner: the repo's rf_regressor (works out of the box).
# Propensity learner: sklearn RF classifier (the repo's classifier adapters
# depend on a helper module that is not vendored into this slice of the repo).
# --------------------------------------------------------------------------- #
_RF = ModelRegistry(ntree=200, nthread=1).as_r_style_dict()["rf_regressor"]

RAW_ATTRS = ["gmv", "quantity", "discount_rate", "delivery_mins",
             "user_rating", "basket_size"]
# Ground-truth raw attributes whose relationship to the outcome drifts for the
# new batch (concept drift). PO-risk is designed to pick these up; pure P(X)
# covariate shift with an unchanged Y|X is better detected via the propensity /
# R-risk path instead.
DRIFT_ATTRS = ["gmv", "user_rating"]


# --------------------------------------------------------------------------- #
# 1. Relational data generation
# --------------------------------------------------------------------------- #
def _random_merchant_names(n, seed=0):
    """Cheap pronounceable English-ish names. A vLLM ``generate`` call with a
    SamplingParams temperature could drop in here to emit fancier strings."""
    rng = np.random.default_rng(seed)
    heads = ["Sun", "Blue", "Green", "Iron", "Golden", "Silver", "Red", "Star",
             "Moon", "Swift", "Prime", "Urban", "Fresh", "Noble", "Bright",
             "Crisp", "Cloud", "Maple", "Cedar", "Coral"]
    tails = ["Mart", "Bazaar", "Goods", "Trading", "Supply", "Foods", "Grocer",
             "Depot", "Emporium", "Kitchen", "Works", "Collective", "Market",
             "Provisions", "Outlet", "Traders", "Wholesale", "Pantry"]
    names = []
    for _ in range(n):
        names.append(f"{rng.choice(heads)} {rng.choice(tails)}")
    # de-duplicate by suffixing a short id where needed
    seen, out = {}, []
    for nm in names:
        seen[nm] = seen.get(nm, 0) + 1
        out.append(nm if seen[nm] == 1 else f"{nm} {seen[nm]}")
    return out


def generate_relational_data(n_merchants=250, n_users=4000, n_orders=20000,
                             items_per_merchant=10, seed=2026,
                             cov_shift=0.0, cov_shift_attrs=None):
    """Build users / items / merchants / orders and return them as DataFrames.

    By default order-level attributes are drawn from the SAME distribution for
    both batches (no covariate shift); the existing-vs-new difference is then
    injected as a concept drift in the merchant outcome (see
    ``build_merchant_dataset``, used by the PO-risk path).

    Set ``cov_shift`` > 0 to instead inject a genuine **covariate shift**: for
    new-batch (W = 1) merchants the raw ``cov_shift_attrs`` are shifted at the
    order level, which propagates into every aggregation of those attributes.
    This is the regime the LOGO-MMD path targets.
    """
    if cov_shift_attrs is None:
        cov_shift_attrs = ["gmv", "user_rating"]
    rng = np.random.default_rng(seed)

    # --- merchants (with batch label W and English names) ---
    merchant_ids = np.array([f"M{100000 + i}" for i in range(n_merchants)])
    W = rng.binomial(1, 0.5, size=n_merchants)
    merchants = pd.DataFrame({
        "merchant_id": merchant_ids,
        "merchant_name": _random_merchant_names(n_merchants, seed=seed + 1),
        "W": W,
    })

    # --- items: each item belongs to exactly one merchant ---
    n_items = n_merchants * items_per_merchant
    item_ids = np.array([f"I{500000 + i}" for i in range(n_items)])
    item_merchant = np.repeat(merchant_ids, items_per_merchant)
    items = pd.DataFrame({"item_id": item_ids, "merchant_id": item_merchant})

    # --- users ---
    user_ids = np.array([f"U{200000 + i}" for i in range(n_users)])

    # --- orders: each row is a (user buys item) conversion ---
    order_number = rng.integers(100000000, 200000000, n_orders).astype(str)
    order_item = rng.choice(item_ids, size=n_orders)
    order_user = rng.choice(user_ids, size=n_orders)

    orders = pd.DataFrame({
        "order_number": order_number,
        "user_id": order_user,
        "item_id": order_item,
    })
    # attach the merchant of each ordered item (item -> merchant edge)
    orders = orders.merge(items, on="item_id", how="left")
    orders = orders.merge(merchants[["merchant_id", "W"]], on="merchant_id",
                          how="left")

    # --- raw order-level attributes ---
    n = len(orders)
    orders["gmv"] = np.exp(rng.normal(3.0, 0.6, n))            # order value
    orders["quantity"] = rng.poisson(2.0, n) + 1
    orders["discount_rate"] = rng.beta(2, 8, n)
    orders["delivery_mins"] = rng.gamma(4.0, 8.0, n)
    orders["user_rating"] = np.clip(rng.normal(4.2, 0.7, n), 1, 5)
    orders["basket_size"] = rng.poisson(3.0, n) + 1

    # --- optional covariate shift on new-batch (W=1) merchants' orders ---
    if cov_shift > 0:
        new_mask = orders["W"].values == 1
        if "gmv" in cov_shift_attrs:
            orders.loc[new_mask, "gmv"] *= np.exp(cov_shift * 0.35)
        if "user_rating" in cov_shift_attrs:
            orders.loc[new_mask, "user_rating"] = np.clip(
                orders.loc[new_mask, "user_rating"].values - cov_shift * 0.6, 1, 5)
        for a in cov_shift_attrs:
            if a in ("gmv", "user_rating"):
                continue
            orders.loc[new_mask, a] = orders.loc[new_mask, a].values * (
                1.0 + cov_shift * 0.25)

    return users_df(user_ids), items, merchants, orders


def users_df(user_ids):
    return pd.DataFrame({"user_id": user_ids})


# --------------------------------------------------------------------------- #
# 2. Aggregate order-level features up to the merchant dimension
# --------------------------------------------------------------------------- #
def aggregate_to_merchant(orders, roll_window=5):
    """Rich + rolling aggregation of order-level attributes to the merchant.

    Returns a merchant-indexed feature frame.  Feature names are
    ``<attr>__<aggregation>`` so each feature maps back to its raw attribute.
    """
    orders = orders.sort_values(["merchant_id", "order_number"]).reset_index(drop=True)
    g = orders.groupby("merchant_id", sort=True)

    frames = []

    # ---- rich per-attribute summary statistics (not just the mean) ----
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
    frames.append(pd.DataFrame(summary))

    # ---- rolling-window aggregations over each merchant's order stream ----
    # (captures recency / drift within the merchant, not just a flat mean)
    roll_feats = {}
    for attr in RAW_ATTRS:
        rmean = (orders.groupby("merchant_id")[attr]
                 .rolling(roll_window, min_periods=1).mean()
                 .reset_index(level=0))
        rstd = (orders.groupby("merchant_id")[attr]
                .rolling(roll_window, min_periods=1).std()
                .reset_index(level=0))
        rmean.columns = ["merchant_id", "v"]
        rstd.columns = ["merchant_id", "v"]
        gm = rmean.groupby("merchant_id")["v"]
        gs = rstd.groupby("merchant_id")["v"]
        roll_feats[f"{attr}__roll{roll_window}_mean_last"] = gm.last()
        roll_feats[f"{attr}__roll{roll_window}_mean_avg"] = gm.mean()
        roll_feats[f"{attr}__roll{roll_window}_std_avg"] = gs.mean().fillna(0.0)
    frames.append(pd.DataFrame(roll_feats))

    # ---- merchant-level activity / diversity counts ----
    counts = pd.DataFrame({
        "n_orders": g.size(),
        "n_unique_users": g["user_id"].nunique(),
        "n_unique_items": g["item_id"].nunique(),
    })
    frames.append(counts)

    feat = pd.concat(frames, axis=1)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feat


def raw_attr_of(feature_name):
    for a in RAW_ATTRS:
        if feature_name.startswith(a + "__"):
            return a
    return "count"  # n_orders / n_unique_* activity features


def is_drift_feature(feature_name):
    return raw_attr_of(feature_name) in DRIFT_ATTRS


# --------------------------------------------------------------------------- #
# 3. PO-risk statistic + permute-then-refit test (mirrors DRPerm.py)
# --------------------------------------------------------------------------- #
def _folds(n, n_folds, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, n_folds)


def _cross_fit_mu(X, Y, n_folds, seed):
    n = X.shape[0]
    mu = np.zeros(n)
    for k, te in enumerate(_folds(n, n_folds, seed)):
        tr = np.setdiff1d(np.arange(n), te)
        fit = _RF["fit"](X[tr], Y[tr], seed=seed + k)
        mu[te] = _RF["predict"](fit, X[te])
    return mu


def _cross_fit_e(X, W, n_folds, seed, clip_e=0.01):
    n = X.shape[0]
    e = np.zeros(n)
    for k, te in enumerate(_folds(n, n_folds, seed + 7)):
        tr = np.setdiff1d(np.arange(n), te)
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                     random_state=seed + k, n_jobs=1).fit(X[tr], W[tr])
        pos = list(clf.classes_).index(1)
        e[te] = clf.predict_proba(X[te])[:, pos]
    return np.clip(e, clip_e, 1 - clip_e)


def po_statistic(X, resid_y, W, e_hat, seed):
    """PO-risk statistic = mean(tau^2), tau fit on the pseudo-outcome."""
    pseudo = resid_y * (W - e_hat)
    tau_fit = _RF["fit"](X, pseudo, seed=seed)
    tau = _RF["predict"](tau_fit, X)
    return float(np.mean(tau ** 2))


def po_risk_test(X, Y, W, n_folds=5, n_perm=200, seed=2026, clip_e=0.01, alpha=0.05):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    W = np.asarray(W, dtype=int).ravel()
    rng = np.random.default_rng(seed)

    mu = _cross_fit_mu(X, Y, n_folds, seed)
    e = _cross_fit_e(X, W, n_folds, seed, clip_e)
    resid_y = Y - mu
    observed = po_statistic(X, resid_y, W, e, seed=seed + 200)

    perm = np.empty(n_perm)
    for b in range(n_perm):
        w_perm = rng.permutation(W)
        e_perm = _cross_fit_e(X, w_perm, n_folds, seed + 300 + b, clip_e)
        perm[b] = po_statistic(X, resid_y, w_perm, e_perm, seed=seed + 400 + b)
    p_value = (1.0 + np.sum(perm >= observed)) / (1.0 + n_perm)
    return {"statistic": observed, "p_value": p_value,
            "reject": bool(p_value < alpha), "perm_mean": float(perm.mean())}


# --------------------------------------------------------------------------- #
# 4. PO-risk + LOCO attribution (mirrors R_risk_loco.py)
# --------------------------------------------------------------------------- #
def po_risk_loco(X, Y, W, feature_names, n_folds=5, seed=2026, clip_e=0.01,
                 n_jobs=-1):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    W = np.asarray(W, dtype=int).ravel()
    p = X.shape[1]

    mu_full = _cross_fit_mu(X, Y, n_folds, seed)
    e_full = _cross_fit_e(X, W, n_folds, seed, clip_e)
    observed = po_statistic(X, Y - mu_full, W, e_full, seed=seed + 200)

    def _loco(j):
        cols = np.setdiff1d(np.arange(p), j)
        Xj = X[:, cols]
        s = seed + 1000 + j
        mu_j = _cross_fit_mu(Xj, Y, n_folds, s)
        e_j = _cross_fit_e(Xj, W, n_folds, s, clip_e)
        return po_statistic(Xj, Y - mu_j, W, e_j, seed=s + 200)

    loco_stats = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_loco)(j) for j in range(p))
    loco_stats = np.asarray(loco_stats)

    # Removing a drift-driving feature lowers the detected PO-risk, so a large
    # (observed - loco) drop => that feature carried the distribution shift.
    vimp = observed - loco_stats
    order = np.argsort(-vimp)
    return pd.DataFrame({
        "feature": np.asarray(feature_names)[order],
        "loco_vimp": vimp[order],
        "loco_statistic": loco_stats[order],
        "is_drift": [is_drift_feature(f) for f in np.asarray(feature_names)[order]],
    }).reset_index(drop=True), observed


def po_risk_logo(X, Y, W, feature_names, groups, n_folds=5, seed=2026,
                 clip_e=0.01):
    """Leave-One-(attribute)-Group-Out attribution.

    Drops ALL aggregations derived from a raw attribute at once, which answers
    "which raw signal drove the drift" and is robust to the within-group
    correlation that dilutes single-feature LOCO.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    W = np.asarray(W, dtype=int).ravel()
    names = np.asarray(feature_names)

    mu_full = _cross_fit_mu(X, Y, n_folds, seed)
    e_full = _cross_fit_e(X, W, n_folds, seed, clip_e)
    observed = po_statistic(X, Y - mu_full, W, e_full, seed=seed + 200)

    rows = []
    for gi, (gname, members) in enumerate(groups.items()):
        keep = np.array([n not in set(members) for n in names])
        Xg = X[:, keep]
        s = seed + 5000 + gi
        mu_g = _cross_fit_mu(Xg, Y, n_folds, s)
        e_g = _cross_fit_e(Xg, W, n_folds, s, clip_e)
        loco = po_statistic(Xg, Y - mu_g, W, e_g, seed=s + 200)
        rows.append({"attribute": gname, "n_features": len(members),
                     "logo_vimp": observed - loco, "logo_statistic": loco,
                     "is_drift": gname in DRIFT_ATTRS})
    df = pd.DataFrame(rows).sort_values("logo_vimp", ascending=False).reset_index(drop=True)
    return df, observed


# --------------------------------------------------------------------------- #
# 5. Driver
# --------------------------------------------------------------------------- #
def build_merchant_dataset(drift=2.5, seed=2026):
    """Aggregate to merchant level and attach a merchant outcome with a KNOWN
    concept drift: for new-batch merchants (W = 1) the outcome gains an extra
    dependence on the DRIFT_ATTRS aggregations, so tau(X) (the batch CATE) is a
    function of exactly those features."""
    _, _, merchants, orders = generate_relational_data(seed=seed)
    feat = aggregate_to_merchant(orders)
    feat = feat.reindex(merchants["merchant_id"].values)  # align order
    merchants = merchants.set_index("merchant_id").loc[feat.index].reset_index()

    W = merchants["W"].values.astype(int)
    rng = np.random.default_rng(seed + 99)
    Xs = (feat - feat.mean()) / (feat.std().replace(0, 1))  # standardize for Y only

    # stable component: same relationship for both batches (drives mu, not tau)
    base = (0.8 * Xs["quantity__mean"] + 0.6 * Xs["basket_size__mean"]
            + 0.4 * Xs["discount_rate__mean"] + 0.3 * Xs["delivery_mins__mean"])
    # concept-drift component: only the new batch's outcome depends on gmv &
    # user_rating aggregations -> these become the ground-truth drift drivers.
    drift_signal = (Xs["gmv__mean"] + 0.7 * Xs["gmv__roll5_mean_last"]
                    - Xs["user_rating__mean"] - 0.5 * Xs["user_rating__std"])
    Y = (base + W * drift * drift_signal + rng.normal(0, 1.0, len(feat))).values
    return merchants, feat, Y, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=float, default=2.5)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plot", type=str, default="")
    args = ap.parse_args()

    merchants, feat, Y, W = build_merchant_dataset(drift=args.drift, seed=args.seed)
    feature_names = list(feat.columns)
    X = feat.values
    groups = {a: [f for f in feature_names if raw_attr_of(f) == a]
              for a in RAW_ATTRS}

    print("=" * 74)
    print("FSDS merchant-dimension attribution prototype  (PO-risk + LOCO)")
    print("=" * 74)
    print(f"merchants (rows)      : {X.shape[0]}  "
          f"(existing W=0: {(W == 0).sum()}, new W=1: {(W == 1).sum()})")
    print(f"merchant features     : {X.shape[1]}  "
          f"(rich + rolling aggregations of {len(RAW_ATTRS)} raw attrs)")
    print(f"ground-truth drift    : {DRIFT_ATTRS}  (concept drift on new batch)")
    print("sample merchant names : "
          + ", ".join(merchants["merchant_name"].head(4)))
    print()

    print("[1] Global PO-risk permute-then-refit test ...")
    test = po_risk_test(X, Y, W, n_perm=args.n_perm, seed=args.seed)
    print(f"    statistic={test['statistic']:.5f}  perm_mean={test['perm_mean']:.5f}"
          f"  p_value={test['p_value']:.4f}  reject_H0={test['reject']}")
    print()

    print("[2] Headline attribution -- leave-one-raw-attribute-out (PO-risk):")
    logo, observed = po_risk_logo(X, Y, W, feature_names, groups, seed=args.seed)
    for _, r in logo.iterrows():
        flag = "  <== DRIFT" if r["is_drift"] else ""
        print(f"      {r['attribute']:14s} (n={int(r['n_features']):2d})  "
              f"vimp={r['logo_vimp']:+.5f}{flag}")
    top_attrs = set(logo.head(len(DRIFT_ATTRS))["attribute"])
    group_ok = top_attrs == set(DRIFT_ATTRS)
    print(f"    top-{len(DRIFT_ATTRS)} attributes = {sorted(top_attrs)}  "
          f"(expected {sorted(DRIFT_ATTRS)}) -> {'MATCH' if group_ok else 'NO MATCH'}")
    print()

    print("[3] Detail -- per-feature PO-risk + LOCO (top 12):")
    ranking, _ = po_risk_loco(X, Y, W, feature_names, seed=args.seed)
    for _, r in ranking.head(12).iterrows():
        flag = "  <== drift attr" if r["is_drift"] else ""
        print(f"      {r['feature']:32s}  vimp={r['loco_vimp']:+.5f}{flag}")
    n_drift_total = int(ranking["is_drift"].sum())
    hits = int(ranking.head(n_drift_total)["is_drift"].sum())
    prec = hits / n_drift_total
    print(f"    precision@{n_drift_total} (drift-derived features in top-"
          f"{n_drift_total}) = {prec:.2f}")
    print()

    ok = test["reject"] and group_ok
    print(f"[4] RESULT: {'PASS' if ok else 'CHECK'}  "
          f"(drift {'detected' if test['reject'] else 'NOT detected'}; "
          f"headline attribution {'correct' if group_ok else 'incorrect'})")

    if args.plot:
        _make_plot(ranking, logo, args.plot)
        print(f"    saved chart -> {args.plot}")
    return 0 if ok else 1


def _make_plot(ranking, logo, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axg, axf) = plt.subplots(1, 2, figsize=(15, 7))

    g = logo.iloc[::-1]
    axg.barh(g["attribute"], g["logo_vimp"],
             color=["#d62728" if s else "#7f7f7f" for s in g["is_drift"]])
    axg.set_title("Headline: leave-one-raw-attribute-out (PO-risk)")
    axg.set_xlabel("group importance  (observed  -  leave-group-out PO-risk)")
    axg.axvline(0, color="k", lw=0.8)

    top = ranking.head(18).iloc[::-1]
    axf.barh(top["feature"], top["loco_vimp"],
             color=["#d62728" if s else "#7f7f7f" for s in top["is_drift"]])
    axf.set_title("Detail: per-feature PO-risk + LOCO (top 18)")
    axf.set_xlabel("LOCO importance  (observed  -  leave-one-out PO-risk)")
    axf.axvline(0, color="k", lw=0.8)

    fig.suptitle("PO-risk + LOCO attribution of a merchant-dimension concept "
                 "drift  (red = derived from a truly-drifting raw attribute)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
