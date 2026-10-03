#!/usr/bin/env python3
"""Feature-store delivery demo: relational dataset + multi-layer attribution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PY = Path(__file__).resolve().parent
if str(PY) not in sys.path:
    sys.path.insert(0, str(PY))

from feature_store.cross_level import TRUE_VERTICAL, level1, level2, generate as cross_generate  # noqa: E402
from feature_store.logo_mmd import (  # noqa: E402
    build_covariate_shift_dataset,
    logo_mmd,
    mmd_permutation_test,
    per_group_mmd,
    _median_gamma,
)
from feature_store.merchant_prototype import (  # noqa: E402
    RAW_ATTRS,
    aggregate_to_merchant,
    aggregate_to_user,
    generate_relational_data,
    raw_attr_of,
)
from feature_store.multilevel_localization import (  # noqa: E402
    build_feature_tree,
    localize,
    _subset_test,
)

ROOT = PY.parent
DATA = ROOT / "delivery" / "data" / "feature_store"
ART = ROOT / "artifacts" / "feature_store"
DATA.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)


def export_relational_tables(seed: int = 2026, cov_shift: float = 1.6) -> dict:
    users, items, merchants, orders = generate_relational_data(
        n_merchants=200,
        n_orders=15000,
        seed=seed,
        cov_shift=cov_shift,
        cov_shift_attrs=["gmv", "user_rating"],
    )
    users.to_csv(DATA / "users.csv", index=False)
    items.to_csv(DATA / "items.csv", index=False)
    merchants.to_csv(DATA / "merchants.csv", index=False)
    orders.to_csv(DATA / "orders.csv", index=False)

    feat_m = aggregate_to_merchant(orders).reindex(merchants["merchant_id"])
    feat_u = aggregate_to_user(orders)
    Wm = merchants.set_index("merchant_id").loc[feat_m.index, "W"].values.astype(int)
    feat_m.assign(W=Wm).reset_index().to_csv(DATA / "merchant_features.csv", index=False)
    feat_u.to_csv(DATA / "user_features.csv")
    return {
        "n_users": int(len(users)),
        "n_merchants": int(len(merchants)),
        "n_orders": int(len(orders)),
        "merchant_feature_dim": int(feat_m.shape[1]),
        "user_feature_dim": int(feat_u.shape[1]),
    }


def run_merchant_logo(feat_std, W, n_perm: int, seed: int) -> dict:
    names = list(feat_std.columns)
    groups = {a: [f for f in names if raw_attr_of(f) == a] for a in RAW_ATTRS}
    g_full = _median_gamma(feat_std.values)
    glob = mmd_permutation_test(feat_std.values, W, g_full, n_perm=n_perm, seed=seed)
    logo_df, _ = logo_mmd(feat_std, W, groups)
    pg = per_group_mmd(feat_std, W, groups, n_perm=min(n_perm, 200), seed=seed)
    return {
        "global_mmd": glob,
        "logo_mmd_top": logo_df.head(6).to_dict(orient="records"),
        "per_group_mmd_top": pg.head(6).to_dict(orient="records"),
    }


def run_multilevel_tree(feat_std, W, q: float, n_perm: int, seed: int) -> dict:
    feature_names = list(feat_std.columns)
    tree = build_feature_tree(feature_names)
    m2_root, p_root = _subset_test(feat_std, W, feature_names, n_perm, seed)
    records = []
    leaves = []
    if p_root < q:
        leaves = localize(feat_std, W, "root", tree, q, n_perm, seed + 3, 0, records)
    rec = pd.DataFrame(records) if records else pd.DataFrame()
    return {
        "root_mmd2": float(m2_root),
        "root_p": float(p_root),
        "selected_leaves": leaves[:20],
        "n_tree_records": int(len(rec)),
        "l1_selected": sorted(set(rec[(rec.depth == 0) & rec.selected].node)) if len(rec) else [],
    }


def run_cross_level(seed: int) -> dict:
    orders, mids, Wm = cross_generate(n_merchants=120, n_orders=8000, seed=seed)
    l1_df, sel1, mmd1, p1 = level1(orders, mids, Wm, n_perm=150, n_boot=40, seed=seed)
    l2_df, sel2 = level2(orders, attr="gmv", n_perm=150, seed=seed)
    return {
        "truth_vertical": int(TRUE_VERTICAL),
        "level1_selected_attrs": sel1,
        "level1_global_mmd2": float(mmd1),
        "level1_global_p": float(p1),
        "level1_table": l1_df.to_dict(orient="records"),
        "level2_selected_verticals": sel2,
        "level2_table": l2_df.to_dict(orient="records"),
    }


def main() -> int:
    seed = 7451
    n_perm = 200
    table_meta = export_relational_tables(seed=seed)
    merchants, feat_std, W = build_covariate_shift_dataset(cov_shift=1.6, seed=seed)

    payload = {
        "dataset_dir": str(DATA),
        "relational": table_meta,
        "schema": "user -> order -> item -> merchant (W on merchant)",
        "merchant_logo_mmd": run_merchant_logo(feat_std, W, n_perm, seed),
        "merchant_multilevel_tree": run_multilevel_tree(feat_std, W, q=0.1, n_perm=n_perm, seed=seed),
        "cross_level_merchant_vertical": run_cross_level(seed + 1),
    }

    out_json = ART / "feature_store_attribution_report.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"artifact": str(out_json), "dataset": str(DATA), "summary": table_meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
