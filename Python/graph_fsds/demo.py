"""
End-to-end demo: FSDS applied to graph embeddings.

Run it with::

    python -m graph_fsds.demo                 # all scenarios, quick settings
    python -m graph_fsds.demo --scenario concept_drift --figures
    python -m graph_fsds.demo --n-perm 200 --full

For each synthetic scenario it:
    1. builds two graph snapshots (existing vs new batch),
    2. computes hierarchical embeddings (fit on existing, transform new),
    3. measures cluster stability (ARI / NMI / modularity),
    4. runs the conditional hierarchical FSDS cascade,
    5. prints the attribution report and (optionally) writes figures.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# allow "python Python/graph_fsds/demo.py" as well as "-m graph_fsds.demo"
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph_fsds import dgp, visualize
    from graph_fsds.cluster_stability import cluster_stability
    from graph_fsds.graph_embedding import HierarchicalGraphEmbedding
    from graph_fsds.hierarchical_fsds import ConditionalHierarchicalFSDS
else:
    from . import dgp, visualize
    from .cluster_stability import cluster_stability
    from .graph_embedding import HierarchicalGraphEmbedding
    from .hierarchical_fsds import ConditionalHierarchicalFSDS


def run_scenario(scenario, *, n_perm=40, n_hops=2, seed=2026, figures=False,
                 out_dir=".", verbose=False, strength=1.0):
    exist, new = dgp.make_scenario(scenario, seed=seed, strength=strength)

    embedder = HierarchicalGraphEmbedding(n_hops=n_hops, standardize=True)
    he_exist = embedder.fit_transform(exist.graph, exist.features, exist.communities,
                                      nodelist=exist.nodelist)
    he_new = embedder.transform(new.graph, new.features, exist.communities,
                                nodelist=new.nodelist)

    stab = cluster_stability(exist.graph, new.graph, nodelist=exist.nodelist)

    fsds = ConditionalHierarchicalFSDS(
        risk="po", n_perm=n_perm, n_folds=5, alpha=0.05,
        seed=seed, verbose=verbose)
    result = fsds.run(he_exist, he_new, exist.labels, new.labels, stability=stab)

    print(f"\n########## scenario: {scenario} ##########")
    print("cluster stability :", stab.summary())
    print(result.report())

    figs = {}
    if figures:
        figs = visualize.visualize_attribution(
            result, exist.graph, new.graph,
            prefix=os.path.join(out_dir, f"fsds_{scenario}"))
        print("figures:", figs)
    return {"scenario": scenario, "stability": stab, "result": result, "figures": figs}


def main(argv=None):
    ap = argparse.ArgumentParser(description="FSDS on graph embeddings demo")
    ap.add_argument("--scenario", default="all",
                    choices=["all"] + dgp.SCENARIOS)
    ap.add_argument("--n-perm", type=int, default=40)
    ap.add_argument("--n-hops", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--figures", action="store_true",
                    help="write attribution figures (PNG)")
    ap.add_argument("--full", action="store_true",
                    help="use a heavier permutation budget (n_perm=200)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.full:
        args.n_perm = max(args.n_perm, 200)

    os.makedirs(args.out_dir, exist_ok=True)
    scenarios = dgp.SCENARIOS if args.scenario == "all" else [args.scenario]

    t0 = time.time()
    summary = []
    for sc in scenarios:
        res = run_scenario(sc, n_perm=args.n_perm, n_hops=args.n_hops, seed=args.seed,
                           figures=args.figures, out_dir=args.out_dir,
                           verbose=args.verbose, strength=args.strength)
        summary.append(res)

    print("\n" + "=" * 68)
    print("SUMMARY  (expected drift vs detected)")
    print("=" * 68)
    expected = {
        "null": "none", "covariate_shift": "covariate", "concept_drift": "concept",
        "community_shift": "covariate", "structure_shift": "covariate/none",
    }
    for res in summary:
        sc = res["scenario"]
        r = res["result"]
        n_shift = len(r.shifted_communities)
        print(f"{sc:<16} expect={expected.get(sc,'?'):<14} "
              f"detected_type={r.overall_drift_type:<9} "
              f"primary={r.primary_level:<9} shifted_comms={n_shift} "
              f"stability={res['stability'].stability_score():.2f}")
    print(f"\nelapsed: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
