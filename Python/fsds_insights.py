#!/usr/bin/env python3
"""Turn the two-level localization into STRUCTURED INSIGHTS.

The localization tree tells us *which* features shifted; a product/dashboard
needs *insights*: where (axis -> attribute group -> feature), what (direction +
magnitude), how sure (adjusted p / stability), and so-what (severity + suggested
action). This module walks the per-axis localization trees and emits, for each
selected group and feature, a structured insight object plus a rendered
human-readable sentence -- a two-level structure (group headline -> feature
drill-down).
"""
from __future__ import annotations

import argparse
import json

from fsds_two_dimensional import generate_two_dim_data, run_dimension

AXIS_ENTITY = {"merchant": "merchants", "user": "buyers"}
AXIS_SIDE = {"merchant": "supply-side", "user": "demand-side"}


def _severity(p_adj, abs_d):
    if (p_adj is not None and p_adj < 0.01) and abs_d >= 2.0:
        return "high"
    if (p_adj is not None and p_adj < 0.05) and abs_d >= 0.5:
        return "medium"
    return "low"


def _dir_phrase(direction):
    return "higher" if direction == "new_batch_higher" else "lower"


def _iter_selected(root):
    """Yield (attribute_node, [selected feature leaf nodes]) for selected groups."""
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
            yield attr, leaves


def _feature_insight(axis, group, leaf):
    ol = leaf["observation_level"]
    ent = ol["entity_level"]
    split = ol["split"]
    d = ent["cohen_d"]
    direction = split["direction"]
    text = (f"`{leaf['name']}`: new-batch median "
            f"{_median(ol, 'batch0')}\u2192{_median(ol, 'batch1')} "
            f"({_dir_phrase(direction)}, cohen_d={d:+.2f}); "
            f"split at {split['threshold']:.3g} separates batches "
            f"(KS={split['ks_stat']:.2f}).")
    return {
        "axis": axis, "scope": "feature", "target": leaf["name"], "parent": group,
        "finding": "distribution_shift", "direction": _dir_phrase(direction),
        "magnitude": {"cohen_d": d, "mean_batch0": ent["batch0"]["mean"],
                      "mean_batch1": ent["batch1"]["mean"], "mmd2": leaf["stats"]["mmd2"]},
        "confidence": {"p_adjusted": leaf.get("p_adjusted")},
        "characterization": {"split_threshold": split["threshold"],
                             "ks_stat": split["ks_stat"],
                             "quantiles": ol.get("quantiles")},
        "severity": _severity(leaf.get("p_adjusted"), abs(d)),
        "text": text,
    }


def _median(ol, batch):
    return round(ol["entity_level"][batch]["median"], 3)


def _group_insight(axis, attr, leaves):
    # dominant direction / magnitude from the strongest feature
    strongest = max(leaves, key=lambda n: abs(n["observation_level"]["entity_level"]["cohen_d"]))
    d = strongest["observation_level"]["entity_level"]["cohen_d"]
    direction = strongest["observation_level"]["split"]["direction"]
    p_adj = attr.get("p_adjusted")
    sev = _severity(p_adj, abs(d))
    action = (f"Covariate shift on {AXIS_SIDE[axis]} `{attr['name']}`: "
              f"new {AXIS_ENTITY[axis]} show {_dir_phrase(direction)} {attr['name']}. "
              f"Monitor `{attr['name']}`; if it feeds downstream models, reweight or "
              f"retrain; check the data pipeline for a collection / definition change.")
    text = (f"[{axis}] Shift localized to `{attr['name']}` "
            f"({len(leaves)} features selected): new {AXIS_ENTITY[axis]} have "
            f"{_dir_phrase(direction)} `{attr['name']}` "
            f"(strongest {strongest['name']}, cohen_d={d:+.2f}). "
            f"Confidence {sev} (group adj p={p_adj:.4f}).")
    return {
        "axis": axis, "scope": "group", "target": attr["name"], "parent": None,
        "finding": "distribution_shift", "direction": _dir_phrase(direction),
        "magnitude": {"group_mmd2": attr["stats"]["mmd2"], "max_abs_cohen_d": abs(d)},
        "confidence": {"p_adjusted": p_adj},
        "severity": sev, "suggested_action": action,
        "n_features_selected": len(leaves), "text": text,
    }


def build_insights(orders, q=0.1, n_perm=300, seed=2026):
    insights = []
    for axis in ("merchant", "user"):
        res = run_dimension(orders, axis, q, n_perm, seed)
        root = res["tree"]["root"]
        gt = res["tree"]["global_test"]
        insights.append({"axis": axis, "scope": "axis",
                         "finding": "global_test",
                         "global_mmd2": gt["mmd2"], "global_p": gt["p_value"],
                         "reject": gt["p_value"] < q,
                         "text": f"[{axis}] Global {AXIS_SIDE[axis]} distribution "
                                 f"shift: MMD^2={gt['mmd2']:.4f}, p={gt['p_value']:.4f} "
                                 f"-> {'DETECTED' if gt['p_value'] < q else 'none'}."})
        for attr, leaves in _iter_selected(root):
            g = _group_insight(axis, attr, leaves)
            insights.append(g)
            for leaf in sorted(leaves, key=lambda n: n["observation_level"]
                               ["entity_level"]["cohen_d"], reverse=True):
                insights.append(_feature_insight(axis, attr["name"], leaf))
    return insights


def render(insights):
    lines = []
    for ins in insights:
        if ins["scope"] == "axis":
            lines.append("\n" + "=" * 78)
            lines.append(ins["text"])
        elif ins["scope"] == "group":
            lines.append("\n  " + ins["text"])
            lines.append("    ACTION: " + ins["suggested_action"])
        else:
            lines.append("      - [" + ins["severity"] + "] " + ins["text"])
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cov-shift", type=float, default=1.6)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    _, _, _, orders = generate_two_dim_data(cov_shift=args.cov_shift, seed=args.seed)
    insights = build_insights(orders, q=args.q, n_perm=args.n_perm, seed=args.seed)
    print(render(insights))
    n_group = sum(i["scope"] == "group" for i in insights)
    n_feat = sum(i["scope"] == "feature" for i in insights)
    print(f"\n\nsummary: {n_group} group-level + {n_feat} feature-level insights "
          f"across 2 axes.")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(insights, fh, indent=2)
        print(f"saved structured insights -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
