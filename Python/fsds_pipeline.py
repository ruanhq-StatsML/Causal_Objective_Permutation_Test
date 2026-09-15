#!/usr/bin/env python3
"""End-to-end FSDS pipeline (prototype orchestrator).

One command runs the whole flow on the two-axis marketplace data and prints an
executive summary plus the machine-readable structured insights:

    generate -> global detection -> Shapley-MMD ranking
             -> two-stage Westfall-Young localization -> retrain decision
             -> fsds-insight/1.0 objects

It reuses the individual modules (no logic duplicated): the localization +
decision + hardened insights come from ``fsds_decision.build``; the coalition
ranking comes from ``fsds_shapley_mmd.shapley_axis``.
"""
from __future__ import annotations

import argparse
import json

from fsds_two_dimensional import generate_two_dim_data
from fsds_decision import build as build_insights
from fsds_shapley_mmd import shapley_axis

SIDE = {"merchant": "supply-side", "user": "demand-side"}


def _ar(direction):
    return "\u2191" if direction == "new_batch_higher" else "\u2193"


def run(orders, q=0.1, n_perm=200, seed=2026):
    insights = build_insights(orders, q=q, n_perm=n_perm, seed=seed)
    by_axis = {}
    for ins in insights:
        by_axis.setdefault(ins["axis"], {"axis": None, "groups": []})
        if ins["scope"] == "axis":
            by_axis[ins["axis"]]["axis"] = ins
        elif ins["scope"] == "group":
            by_axis[ins["axis"]]["groups"].append(ins)

    out = {"axes": {}}
    for axis in ("merchant", "user"):
        sh_df, _ = shapley_axis(orders, axis, n_perm=80, n_boot=40, seed=seed)
        shap = [(r.attribute, float(r.shapley), float(r.perm_p))
                for r in sh_df.itertuples()]
        ax = by_axis[axis]["axis"]
        groups = sorted(by_axis[axis]["groups"], key=lambda g: -g["priority"])
        out["axes"][axis] = {"global": ax, "shapley_ranking": shap, "groups": groups}
    return out


def print_summary(out):
    print("=" * 84)
    print("FSDS PIPELINE  --  EXECUTIVE SUMMARY")
    print("=" * 84)
    for axis in ("merchant", "user"):
        a = out["axes"][axis]
        g = a["global"]
        detected = g["global_p"] < 0.1
        print(f"\n[{axis.upper()} | {SIDE[axis]}]  owner={g['owner']}")
        print(f"  Global : MMD^2={g['global_mmd2']:.4f}  p={g['global_p']:.4f}  "
              f"-> {'SHIFT DETECTED' if detected else 'no shift'}")
        top = ", ".join(f"{n}({s:.3f}, p={p:.3f})" for n, s, p in a["shapley_ranking"][:3])
        print(f"  Shapley drivers (top-3): {top}")
        imp = g["impact"]
        auc = g["confidence"]["identifiability_auc"]
        print(f"  Decision: {g['decision'].upper()}  "
              f"(ESS={imp['weight_ess_frac']:.2f} AUC={auc:.2f}; "
              f"cov={imp['covariate_impact']:+.3f} con={imp['concept_impact']:+.3f})")
        print(f"    reason: {g['reason']}")
        print(f"  CONCLUSION: {g['conclusion']}")
        # ---- Level 1: which attribute groups ----
        print("  Level-1 (attribute groups, by priority):")
        for grp in a["groups"]:
            c = grp["confidence"]
            print(f"    * {grp['target']:14s} {_ar(grp['direction'])} "
                  f"feats={grp['n_features']:2d} exposure={grp['model_exposure']:.2f} "
                  f"stability={c['stability_freq']:.2f} priority={grp['priority']:.3f} "
                  f"trend={grp['trend']['status']}")
        # ---- Level 2: which features within each selected group ----
        print("  Level-2 (features within selected groups):")
        for grp in a["groups"]:
            print(f"    [{grp['target']}] {grp['conclusion']}")
            for f in grp["features"][:3]:
                r = f["candidate_rule"]
                print(f"        - {f['feature']:26s} {_ar(f['direction'])} "
                      f"median {f['median_old']:.3g}->{f['median_new']:.3g} "
                      f"({f['pct_change']:+.0f}%)  rule: {r['feature']} {r['op']} "
                      f"{r['threshold']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    _, _, _, orders = generate_two_dim_data(cov_shift=args.cov_shift, seed=args.seed)
    out = run(orders, q=args.q, n_perm=args.n_perm, seed=args.seed)
    print_summary(out)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nsaved pipeline output -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
