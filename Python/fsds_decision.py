#!/usr/bin/env python3
"""Structured insights v2: localization + a retrain-decision layer.

The localization says *which* features shifted; this module adds the *so-what*
as measurable numbers by attaching a toy downstream task and evaluating:

* covariate impact  = importance-weighted new-batch loss - in-distribution loss
                      (label-free-estimable via the propensity density ratio
                      w(x)=e(x)/(1-e(x)); reliable only when the weight ESS is
                      high);
* concept  impact   = true new-batch loss - importance-weighted estimate
                      (the excess the covariate reweighting cannot explain =
                      concept drift, needs labels);
* exposure          = downstream-model importance of the shifted features.

Decision rule per axis:
    concept impact CI > 0          -> retrain      (mapping changed)
    weight ESS too low             -> collect_labels (reweighting unreliable)
    covariate impact large & exposed -> reweight
    else                           -> monitor

Each shifted group/feature becomes a schema-v2 structured insight carrying
shift_type / exposure / impact / decision / priority / owner + evidence.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from fsds_two_dimensional import generate_two_dim_data, run_dimension, DIMENSIONS

AXIS_ENTITY = {"merchant": "merchants", "user": "buyers"}
OWNER = {"merchant": "supply", "user": "demand"}


def _safe_d(d):
    """Clamp cohen_d for display: near-zero-variance features can blow it up."""
    if not np.isfinite(d):
        return float("nan")
    return float(np.clip(d, -99.0, 99.0))


# --------------------------------------------------------------------------- #
# Toy downstream task (merchant axis: concept drift; user axis: covariate only)
# --------------------------------------------------------------------------- #
def downstream_target(feat, W, axis, seed):
    Xs = ((feat - feat.mean()) / feat.std().replace(0, 1)).fillna(0.0)
    rng = np.random.default_rng(seed + 4)
    base = (0.9 * Xs.get("quantity__mean", 0) + 0.6 * Xs.get("delivery_mins__mean", 0)
            + 0.5 * Xs.get("n_orders", 0))
    if axis == "merchant":  # concept drift: mapping changes for the new batch
        drift = W * 2.5 * (Xs.get("gmv__mean", 0) - Xs.get("user_rating__mean", 0))
        y = base + drift + rng.normal(0, 1.0, len(feat))
    else:                   # same mapping both batches (pure covariate shift)
        y = (base + 0.7 * Xs.get("basket_size__mean", 0)
             + rng.normal(0, 1.0, len(feat)))
    return np.asarray(y, dtype=float)


# --------------------------------------------------------------------------- #
# Retrain evaluation
# --------------------------------------------------------------------------- #
def evaluate_axis(feat, W, y, seed, n_boot=300):
    X = feat.values.astype(float)
    old, new = W == 0, W == 1
    reg = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                random_state=seed, n_jobs=1)
    # in-distribution loss on the old batch (cross-fitted)
    kf = KFold(5, shuffle=True, random_state=seed)
    pred_old_cv = cross_val_predict(reg, X[old], y[old], cv=kf)
    res_old = (y[old] - pred_old_cv) ** 2
    model = reg.fit(X[old], y[old])                      # deploy on old
    res_new = (y[new] - model.predict(X[new])) ** 2

    # propensity density ratio w(x)=e/(1-e), cross-fitted
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                 random_state=seed, n_jobs=1)
    e = cross_val_predict(clf, X, W, cv=StratifiedKFold(5, shuffle=True,
                          random_state=seed), method="predict_proba")[:, 1]
    auc = float(roc_auc_score(W, e))                     # batch separability
    e = np.clip(e, 0.02, 0.98)
    w_old = (e[old] / (1 - e[old]))
    w_old = w_old / w_old.mean()
    ess = float((w_old.sum() ** 2) / (w_old ** 2).sum())
    ess_frac = ess / old.sum()

    def _stats(rng):
        io = rng.integers(0, old.sum(), old.sum())
        ino = rng.integers(0, new.sum(), new.sum())
        l_old = res_old[io].mean()
        l_iw = np.average(res_old[io], weights=w_old[io])
        l_new = res_new[ino].mean()
        return l_old, l_iw, l_new

    l_old = res_old.mean()
    l_iw = np.average(res_old, weights=w_old)
    l_new = res_new.mean()
    rng = np.random.default_rng(seed + 1)
    cov, con = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        a, i, n = _stats(rng)
        cov[b] = i - a
        con[b] = n - i
    return {
        "l_old": float(l_old), "l_iw": float(l_iw), "l_new": float(l_new),
        "covariate_impact": float(l_iw - l_old),
        "covariate_ci": [float(np.percentile(cov, 2.5)), float(np.percentile(cov, 97.5))],
        "concept_impact": float(l_new - l_iw),
        "concept_ci": [float(np.percentile(con, 2.5)), float(np.percentile(con, 97.5))],
        "weight_ess_frac": ess_frac, "propensity_auc": auc,
        "importances": dict(zip(feat.columns, model.feature_importances_)),
    }


def decide(ev, tol_frac=0.15, auc_max=0.9):
    """Overlap is gated by the propensity AUC (batch separability): if AUC is
    high the batches barely overlap, so importance weighting -- and therefore the
    covariate/concept split -- is invalid regardless of the (clip-inflated) ESS.
    In that regime only the observed new-batch loss is trustworthy."""
    con_lo = ev["concept_ci"][0]
    rel_cov = ev["covariate_impact"] / max(ev["l_old"], 1e-9)
    rel_new = (ev["l_new"] - ev["l_old"]) / max(ev["l_old"], 1e-9)
    auc = ev["propensity_auc"]
    if auc > auc_max:                                 # batches (near-)separable
        if rel_new > tol_frac:
            return "retrain", ("loss degraded; covariate/concept unidentifiable "
                               f"(batches separable, AUC={auc:.2f} -> collect labels)")
        return "monitor", f"separable (AUC={auc:.2f}) but no loss degradation"
    if con_lo > 0:                                    # strict CI: costly decision
        return "retrain", "concept drift: excess loss beyond covariate reweighting"
    if rel_cov > tol_frac:
        return "reweight", f"covariate shift degrades loss by {rel_cov:.0%} (in support)"
    return "monitor", "shift present but no material performance impact"


# --------------------------------------------------------------------------- #
# Assemble schema-v2 insights
# --------------------------------------------------------------------------- #
def _selected_groups(root):
    out = []
    for attr in root["children"]:
        if not attr["selected"]:
            continue
        leaves = []
        stack = list(attr["children"])
        while stack:
            n = stack.pop()
            if n["kind"] == "feature":
                if n["selected"]:
                    leaves.append(n)
            else:
                stack.extend(n["children"])
        if leaves:
            out.append((attr, leaves))
    return out


def build(orders, q=0.1, n_perm=200, seed=2026):
    insights = []
    for axis in ("merchant", "user"):
        res = run_dimension(orders, axis, q, n_perm, seed)
        feat, W, tree = res["feat_std"], res["W"], res["tree"]
        # recover raw (unstandardized) feature frame for the task + importances
        cfg = DIMENSIONS[axis]
        from fsds_two_dimensional import aggregate_to_entity
        raw = aggregate_to_entity(orders, cfg["key"], cfg["other_ids"])
        ent = orders[[cfg["key"], cfg["label"]]].drop_duplicates() \
            .set_index(cfg["key"])[cfg["label"]]
        raw = raw.reindex(feat.index if hasattr(feat, "index") else raw.index)
        raw = raw.loc[~raw.index.duplicated()]
        W = ent.reindex(raw.index).values.astype(int)
        y = downstream_target(raw, W, axis, seed)
        ev = evaluate_axis(raw, W, y, seed)
        decision, reason = decide(ev)
        imp = ev["importances"]
        total_imp = sum(imp.values()) or 1.0

        gt = tree["global_test"]
        if ev["propensity_auc"] > 0.9:
            shift_type = "unidentifiable_separable"
        elif ev["concept_ci"][0] > 0:
            shift_type = "concept"
        else:
            shift_type = "covariate"
        insights.append({
            "axis": axis, "scope": "axis", "owner": OWNER[axis],
            "global_mmd2": gt["mmd2"], "global_p": gt["p_value"],
            "shift_type": shift_type,
            "impact": {"l_old": ev["l_old"], "l_iw": ev["l_iw"], "l_new": ev["l_new"],
                       "covariate_impact": ev["covariate_impact"],
                       "covariate_ci": ev["covariate_ci"],
                       "concept_impact": ev["concept_impact"],
                       "concept_ci": ev["concept_ci"],
                       "weight_ess_frac": ev["weight_ess_frac"]},
            "decision": decision, "reason": reason,
            "text": f"[{axis}] {OWNER[axis]}-side shift; "
                    f"loss {ev['l_old']:.3f}->{ev['l_new']:.3f} "
                    f"(covariate {ev['covariate_impact']:+.3f}, "
                    f"concept {ev['concept_impact']:+.3f}, "
                    f"AUC={ev['propensity_auc']:.2f}) -> DECISION: {decision} "
                    f"({reason}).",
        })

        for attr, leaves in _selected_groups(tree["root"]):
            g = attr["name"]
            exposure = sum(imp.get(f, 0.0) for f in [ln["name"] for ln in leaves]) / total_imp
            strongest = max(leaves, key=lambda n: n["stats"]["mmd2"])  # mmd2 avoids
            d = _safe_d(strongest["observation_level"]["entity_level"]["cohen_d"])
            priority = exposure * abs(ev["concept_impact"] if shift_type == "concept"
                                      else ev["covariate_impact"])
            insights.append({
                "axis": axis, "scope": "group", "target": g, "owner": OWNER[axis],
                "shift_type": shift_type, "n_features": len(leaves),
                "model_exposure": float(exposure), "priority": float(priority),
                "decision": decision, "group_adj_p": attr.get("p_adjusted"),
                "text": f"  [{axis}] `{g}` shifted ({len(leaves)} feats, "
                        f"strongest {strongest['name']} d={d:+.2f}); "
                        f"model_exposure={exposure:.2f}, priority={priority:.3f} "
                        f"-> {decision}.",
                "features": [{
                    "feature": ln["name"],
                    "cohen_d": ln["observation_level"]["entity_level"]["cohen_d"],
                    "split": ln["observation_level"]["split"]["threshold"],
                    "direction": ln["observation_level"]["split"]["direction"],
                    "model_importance_share": imp.get(ln["name"], 0.0) / total_imp,
                } for ln in sorted(leaves, key=lambda n: -imp.get(n["name"], 0.0))],
            })
    return insights


def _scenario(kind, n=600, p=8, seed=0):
    """Controlled scenarios WITH support overlap to exercise every decision."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    W = np.r_[np.zeros(n // 2), np.ones(n - n // 2)].astype(int)
    X = rng.normal(size=(n, p))
    beta = np.r_[np.ones(3), np.zeros(p - 3)]        # model uses first 3 features
    if kind == "covariate":         # mild P(X) shift on used features (overlap)
        X[W == 1, 0] += 0.8
        X[W == 1, 1] += 0.8
        y = X @ beta + rng.normal(0, 1, n)
    elif kind == "concept":         # same P(X), mapping changes for new batch
        y = X @ beta + W * (X @ np.r_[np.zeros(3), 2.0, 2.0, np.zeros(p - 5)]) \
            + rng.normal(0, 1, n)
    else:                           # shift only on UNUSED features -> monitor
        X[W == 1, 5] += 1.5
        X[W == 1, 6] += 1.5
        y = X @ beta + rng.normal(0, 1, n)
    feat = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    return feat, W, y


def validate():
    print("Decision-logic self-check (controlled, with support overlap):")
    for kind, expect in [("covariate", "reweight"), ("concept", "retrain"),
                         ("irrelevant", "monitor")]:
        feat, W, y = _scenario(kind, seed=7)
        ev = evaluate_axis(feat, W, y, seed=7, n_boot=200)
        dec, reason = decide(ev)
        ok = "OK" if dec == expect else f"!! expected {expect}"
        print(f"  {kind:11s}: AUC={ev['propensity_auc']:.2f} ESS={ev['weight_ess_frac']:.2f} "
              f"cov={ev['covariate_impact']:+.3f} con={ev['concept_impact']:+.3f} "
              f"-> {dec:14s} [{ok}]  ({reason})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="run the controlled decision-logic self-check and exit")
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    if args.validate:
        validate()
        return 0

    _, _, _, orders = generate_two_dim_data(cov_shift=args.cov_shift, seed=args.seed)
    insights = build(orders, q=args.q, n_perm=args.n_perm, seed=args.seed)

    for ins in insights:
        if ins["scope"] == "axis":
            print("\n" + "=" * 80 + "\n" + ins["text"])
        else:
            print(ins["text"])
            for f in ins["features"][:4]:
                print(f"       - {f['feature']:28s} d={f['cohen_d']:+.2f} "
                      f"split@{f['split']:.3g} imp_share={f['model_importance_share']:.2f}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(insights, fh, indent=2)
        print(f"\nsaved schema-v2 insights -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
