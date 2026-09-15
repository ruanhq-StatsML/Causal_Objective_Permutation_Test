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
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from fsds_two_dimensional import generate_two_dim_data, run_dimension, DIMENSIONS
from fsds_logo_mmd import _median_gamma, mmd2_unbiased

AXIS_ENTITY = {"merchant": "merchants", "user": "buyers"}
OWNER = {"merchant": "supply", "user": "demand"}
SCHEMA_VERSION = "fsds-insight/1.0"


def _iid(dedup_key):
    return hashlib.sha1(dedup_key.encode()).hexdigest()[:12]


def _candidate_rule(feature, split):
    op = ">" if split["direction"] == "new_batch_higher" else "<"
    return {"feature": feature, "op": op,
            "threshold": round(float(split["threshold"]), 4),
            "implies": "new_batch", "ks": round(float(split["ks_stat"]), 3)}


def _trend(dedup_key, priority, prev):
    if prev is None:
        return {"status": "unknown", "delta_priority": None, "first_seen": None}
    if dedup_key not in prev:
        return {"status": "new", "delta_priority": None, "first_seen": "this_run"}
    prev_p = prev[dedup_key].get("priority")
    first = prev[dedup_key].get("trend", {}).get("first_seen") or "previous_run"
    if prev_p is None or priority is None:
        return {"status": "continuing", "delta_priority": None, "first_seen": first}
    delta = priority - prev_p
    rel = delta / prev_p if prev_p > 1e-9 else 0.0
    status = ("worsening" if rel > 0.1 else "improving" if rel < -0.1 else "stable")
    return {"status": status, "delta_priority": float(delta), "first_seen": first}


def _group_stability(feat_std, W, members, n_boot=60, seed=0):
    """Bootstrap fraction of resamples with a positive group MMD (real number)."""
    Z = feat_std[members].values
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(W == 0)[0], np.where(W == 1)[0]
    pos = 0
    for _ in range(n_boot):
        bi = np.concatenate([rng.choice(i0, i0.size, True), rng.choice(i1, i1.size, True)])
        Zb = Z[bi] + rng.normal(0, 1e-3, Z[bi].shape)
        Wb = W[bi]
        if mmd2_unbiased(Zb[Wb == 0], Zb[Wb == 1], _median_gamma(Zb)) > 0:
            pos += 1
    return pos / n_boot


def _safe_d(d):
    """Clamp cohen_d for display: near-zero-variance features can blow it up."""
    if not np.isfinite(d):
        return float("nan")
    return float(np.clip(d, -99.0, 99.0))


def _arrow(direction):
    return "\u2191" if direction == "new_batch_higher" else "\u2193"


def _pct(old, new):
    return 100.0 * (new - old) / abs(old) if abs(old) > 1e-9 else float("nan")


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


def build(orders, q=0.1, n_perm=200, seed=2026, prev=None):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
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
        auc = ev["propensity_auc"]
        if auc > 0.9:
            shift_type = "unidentifiable_separable"
        elif ev["concept_ci"][0] > 0:
            shift_type = "concept"
        else:
            shift_type = "covariate"
        dk = f"{axis}:axis"
        pr_axis = abs(ev["l_new"] - ev["l_old"]) / max(ev["l_old"], 1e-9)
        axis_ins = {
            "schema_version": SCHEMA_VERSION, "insight_id": _iid(dk), "dedup_key": dk,
            "generated_at": now, "axis": axis, "scope": "axis", "owner": OWNER[axis],
            "finding": "distribution_shift", "shift_type": shift_type,
            "global_mmd2": gt["mmd2"], "global_p": gt["p_value"],
            "impact": {"l_old": ev["l_old"], "l_iw": ev["l_iw"], "l_new": ev["l_new"],
                       "covariate_impact": ev["covariate_impact"],
                       "covariate_ci": ev["covariate_ci"],
                       "concept_impact": ev["concept_impact"],
                       "concept_ci": ev["concept_ci"],
                       "weight_ess_frac": ev["weight_ess_frac"]},
            "confidence": {"significance_p": gt["p_value"],
                           "identifiability_auc": auc,
                           "label_free_identifiable": bool(auc <= 0.9)},
            "priority": float(pr_axis), "decision": decision, "reason": reason,
            "trend": _trend(dk, pr_axis, prev),
            "conclusion": None,          # filled after the group loop (two-level)
            "text": f"[{axis}] {OWNER[axis]}-side shift; "
                    f"loss {ev['l_old']:.3f}->{ev['l_new']:.3f} "
                    f"(covariate {ev['covariate_impact']:+.3f}, "
                    f"concept {ev['concept_impact']:+.3f}, "
                    f"AUC={auc:.2f}) -> DECISION: {decision} ({reason}).",
        }
        insights.append(axis_ins)

        group_insights = []
        for attr, leaves in _selected_groups(tree["root"]):
            g = attr["name"]
            names_g = [ln["name"] for ln in leaves]
            exposure = sum(imp.get(f, 0.0) for f in names_g) / total_imp
            strongest = max(leaves, key=lambda n: n["stats"]["mmd2"])
            el = strongest["observation_level"]["entity_level"]
            d = _safe_d(el["cohen_d"])
            direction = strongest["observation_level"]["split"]["direction"]
            priority = exposure * abs(ev["concept_impact"] if shift_type == "concept"
                                      else ev["covariate_impact"])
            stab = _group_stability(feat, W, names_g, seed=seed + 3)
            dk = f"{axis}:group:{g}"
            feats = [{
                "feature": ln["name"],
                "cohen_d": ln["observation_level"]["entity_level"]["cohen_d"],
                "median_old": ln["observation_level"]["entity_level"]["batch0"]["median"],
                "median_new": ln["observation_level"]["entity_level"]["batch1"]["median"],
                "pct_change": _pct(
                    ln["observation_level"]["entity_level"]["batch0"]["median"],
                    ln["observation_level"]["entity_level"]["batch1"]["median"]),
                "split": ln["observation_level"]["split"]["threshold"],
                "direction": ln["observation_level"]["split"]["direction"],
                "model_importance_share": imp.get(ln["name"], 0.0) / total_imp,
                "candidate_rule": _candidate_rule(
                    ln["name"], ln["observation_level"]["split"]),
            } for ln in sorted(leaves, key=lambda n: -imp.get(n["name"], 0.0))]
            # Level-2 conclusion for this group
            arrow = _arrow(direction)
            pc = _pct(el["batch0"]["median"], el["batch1"]["median"])
            g_concl = (f"new {AXIS_ENTITY[axis]} have {arrow} `{g}` "
                       f"(median {el['batch0']['median']:.3g}->{el['batch1']['median']:.3g}, "
                       f"{pc:+.0f}%; {len(leaves)} features, stability {stab:.2f}); "
                       f"rule e.g. {feats[0]['candidate_rule']['feature']} "
                       f"{feats[0]['candidate_rule']['op']} "
                       f"{feats[0]['candidate_rule']['threshold']}.")
            group_insights.append({
                "schema_version": SCHEMA_VERSION, "insight_id": _iid(dk),
                "dedup_key": dk, "generated_at": now,
                "axis": axis, "scope": "group", "target": g, "owner": OWNER[axis],
                "finding": "distribution_shift", "shift_type": shift_type,
                "direction": direction, "n_features": len(leaves),
                "model_exposure": float(exposure), "priority": float(priority),
                "decision": decision,
                "confidence": {"significance_p": attr.get("p_adjusted"),
                               "stability_freq": float(stab),
                               "identifiability_auc": auc},
                "trend": _trend(dk, float(priority), prev),
                "conclusion": g_concl,
                "text": f"  [{axis}] `{g}` shifted ({len(leaves)} feats, "
                        f"strongest {strongest['name']} d={d:+.2f}); "
                        f"exposure={exposure:.2f}, priority={priority:.3f}, "
                        f"stability={stab:.2f} -> {decision}.",
                "features": feats,
            })
        insights.extend(group_insights)

        # ---- Level-1 (group) synthesis into an axis-level conclusion ----
        drivers = sorted(group_insights, key=lambda gi: -gi["priority"])
        parts = [f"`{gi['target']}` {_arrow(gi['direction'])}" for gi in drivers]
        n_feats = sum(gi["n_features"] for gi in group_insights)
        axis_ins["conclusion"] = (
            f"{OWNER[axis].capitalize()}-side {shift_type} shift "
            f"(global p={gt['p_value']:.4f}): "
            f"level-1 localizes to {len(group_insights)} attribute group(s) "
            f"[{', '.join(parts)}]; level-2 selects {n_feats} features. "
            f"Decision: {decision.upper()} ({reason}).")
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
    ap.add_argument("--prev", type=str, default="",
                    help="previous insights JSON to diff for trend fields")
    args = ap.parse_args()

    if args.validate:
        validate()
        return 0

    prev = None
    if args.prev:
        with open(args.prev) as fh:
            prev = {i["dedup_key"]: i for i in json.load(fh)}

    _, _, _, orders = generate_two_dim_data(cov_shift=args.cov_shift, seed=args.seed)
    insights = build(orders, q=args.q, n_perm=args.n_perm, seed=args.seed, prev=prev)

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
