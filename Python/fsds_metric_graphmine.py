#!/usr/bin/env python3
"""FSDS + graph-mining: turn the metric association graph into structure.

Pipeline (no explicit metric relationships, no causal claim):
  1. Build a weighted metric graph with HSIC edges (non-linear dependence).
  2. Graph-mine the structure:
       - connected components  -> metric clusters (chain vs isolated noise);
       - maximum-weight spanning tree -> the linkage BACKBONE (denoise);
       - eigenvector / degree centrality -> hub metrics;
  3. Localize a change: given a flagged (shifted) target metric, run
     Random-Walk-with-Restart / personalized PageRank on the HSIC graph seeded at
     the target -> rank the associated metrics that carry it; combine with each
     node's own batch-shift score.

Uses scipy.sparse.csgraph (connected components, MST) + a numpy power-iteration
PPR; no networkx dependency.
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from fsds_metric_graph import generate, hsic, METRICS, LEVEL


def hsic_adjacency(M):
    p = M.shape[1]
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            A[i, j] = A[j, i] = hsic(M[:, i], M[:, j])
    return A


def shift_scores(M, W):
    out = np.zeros(M.shape[1])
    for j in range(M.shape[1]):
        a, b = M[W == 0, j], M[W == 1, j]
        pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) or 1.0
        out[j] = abs(b.mean() - a.mean()) / pooled
    return out


def eig_centrality(A, iters=200):
    v = np.ones(A.shape[0])
    for _ in range(iters):
        v = A @ v
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
    return v


def personalized_pagerank(A, seed, alpha=0.15, iters=300):
    row = A.sum(1, keepdims=True)
    P = np.divide(A, row, out=np.zeros_like(A), where=row > 0)
    n = A.shape[0]
    e = np.zeros(n)
    e[seed] = 1.0
    r = e.copy()
    for _ in range(iters):
        r = (1 - alpha) * (P.T @ r) + alpha * e
    return r / r.sum()


def draw_graph(A, Athr, mst, sh, ppr, target, path):
    """PNG illustration: HSIC edges (grey), MST backbone (blue, bold), nodes
    coloured by batch shift and sized by RWR propagation from the target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = len(METRICS)
    ang = 2 * np.pi * np.arange(p) / p + np.pi / 2
    pos = np.column_stack([np.cos(ang), np.sin(ang)])
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    amax = A.max() or 1.0
    for i in range(p):
        for j in range(i + 1, p):
            if Athr[i, j] > 0:
                ax.plot(*zip(pos[i], pos[j]), color="#9aa0a6",
                        lw=0.6 + 5 * A[i, j] / amax, alpha=0.6, zorder=1)
    for i in range(p):
        for j in range(p):
            if mst[i, j] > 0:
                ax.plot(*zip(pos[i], pos[j]), color="#1f77b4", lw=3.2,
                        alpha=0.9, zorder=2)

    sizes = 500 + 6000 * ppr
    sc = ax.scatter(pos[:, 0], pos[:, 1], s=sizes, c=sh, cmap="Reds",
                    vmin=0, vmax=max(sh.max(), 1e-6), edgecolors="k",
                    linewidths=1.3, zorder=3)
    ti = METRICS.index(target)
    ax.scatter([pos[ti, 0]], [pos[ti, 1]], s=sizes[ti] + 500, marker="*",
               facecolors="none", edgecolors="#111", linewidths=2.0, zorder=4)
    for i, name in enumerate(METRICS):
        ax.annotate(f"{name}\n[{LEVEL[name]}]", pos[i], ha="center", va="center",
                    fontsize=8, zorder=5)

    fig.colorbar(sc, ax=ax, shrink=0.7, label="batch shift (|cohen d|)")
    ax.plot([], [], color="#1f77b4", lw=3.2, label="MST linkage backbone")
    ax.plot([], [], color="#9aa0a6", lw=2, label="HSIC edge (width $\\propto$ HSIC)")
    ax.scatter([], [], s=200, marker="*", facecolors="none", edgecolors="#111",
               label=f"flagged target ({target})")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.set_title("FSDS $+$ graph-mining: metric linkage graph\n"
                 "node size $\\propto$ RWR propagation from the flagged target; "
                 "colour $\\propto$ batch shift")
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved graph illustration -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=1.2)
    ap.add_argument("--tau", type=float, default=0.01, help="HSIC edge threshold")
    ap.add_argument("--target", type=str, default="gmv")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plot", type=str, default="")
    args = ap.parse_args()

    M, W = generate(shift=args.shift, seed=args.seed)
    A = hsic_adjacency(M)
    Athr = A * (A > args.tau)

    print("=" * 74)
    print("FSDS + GRAPH-MINING on the metric association graph")
    print("=" * 74)

    # (1) connected components
    ncomp, labels = connected_components(csr_matrix(Athr > 0), directed=False)
    print(f"\n[1] Connected components ({ncomp}):")
    for c in range(ncomp):
        members = [METRICS[i] for i in range(len(METRICS)) if labels[i] == c]
        print(f"    component {c}: {members}")

    # (2) max-weight spanning backbone (MST on 1/weight within each component)
    inv = np.where(Athr > 0, 1.0 / (Athr + 1e-9), 0.0)
    mst = minimum_spanning_tree(csr_matrix(inv)).toarray()
    print("\n[2] Linkage backbone (max-weight spanning tree edges):")
    for i in range(len(METRICS)):
        for j in range(len(METRICS)):
            if mst[i, j] > 0:
                print(f"    {METRICS[i]:14s} -- {METRICS[j]:14s}  HSIC={A[i, j]:.3f}")

    # (3) centrality + shift
    cen = eig_centrality(Athr)
    sh = shift_scores(M, W)
    print("\n[3] Node centrality (eigenvector) and batch-shift score:")
    for i in np.argsort(-cen):
        print(f"    {METRICS[i]:14s} [{LEVEL[METRICS[i]]:9s}] "
              f"centrality={cen[i]:.2f}  shift={sh[i]:.2f}")

    # (4) localize a flagged target via RWR / personalized PageRank
    t = METRICS.index(args.target)
    ppr = personalized_pagerank(Athr, t)
    print(f"\n[4] Localization for flagged target `{args.target}` "
          "(RWR proximity x shift):")
    order = [i for i in np.argsort(-ppr) if i != t]
    for i in order:
        tag = "  <== associated & shifted" if (ppr[i] > 1e-3 and sh[i] > 0.5) else ""
        print(f"    {METRICS[i]:14s} [{LEVEL[METRICS[i]]:9s}] "
              f"ppr={ppr[i]:.3f}  shift={sh[i]:.2f}{tag}")

    assoc = {METRICS[i] for i in order if ppr[i] > 1e-3 and sh[i] > 0.5}
    print(f"\n  => `{args.target}` change is associated with (across levels): "
          f"{sorted(assoc)}")
    truth = {"delivery_mins", "user_rating", "n_orders"}
    ok = truth.issubset(assoc) and "quantity" not in assoc and "discount" not in assoc
    print(f"RESULT: {'PASS' if ok else 'CHECK'}  (recovered the linked shifted "
          f"chain, isolated noise metrics)")
    if args.plot:
        draw_graph(A, Athr, mst, sh, ppr, args.target, args.plot)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
