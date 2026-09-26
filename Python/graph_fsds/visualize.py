"""
Visualisation of hierarchical FSDS attribution on graphs.

Three figures make the attribution legible:

    * node attribution      : the two snapshots side by side, nodes coloured by
                              their per-node drift score.
    * community attribution : bar chart of per-community test statistics /
                              rejection.
    * global VIMP           : which embedding dimensions drive the pooled shift.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402


def visualize_node_attribution(
    G_exist, G_new, node_attribution, nodelist, out_path="fsds_node_attribution.png",
    title="FSDS node-level drift attribution",
):
    attr = {nodelist[i]: float(node_attribution[i]) for i in range(len(nodelist))}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pos = nx.spring_layout(G_exist, seed=42)
    vmax = max(1e-6, float(np.max(node_attribution)))
    for ax, G, name in [(axes[0], G_exist, "existing batch"),
                        (axes[1], G_new, "new batch")]:
        colors = [attr.get(n, 0.0) for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5)
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=colors, cmap="RdYlGn_r",
            vmin=0.0, vmax=vmax, node_size=60)
        ax.set_title(name)
        ax.axis("off")
    fig.colorbar(nodes, ax=axes, fraction=0.025, pad=0.02, label="drift score")
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_community_attribution(
    community_results, out_path="fsds_community_attribution.png",
    title="FSDS community-level drift attribution",
):
    if not community_results:
        return None
    cids = sorted(community_results.keys())
    stats = [community_results[c].statistic for c in cids]
    reject = [community_results[c].significant for c in cids]
    colors = ["#d62728" if r else "#7fb069" for r in reject]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(c) for c in cids], stats, color=colors)
    ax.set_xlabel("community id")
    ax.set_ylabel("test statistic")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#d62728"),
               plt.Rectangle((0, 0), 1, 1, color="#7fb069")]
    ax.legend(handles, ["shifted", "stable"], loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def visualize_global_vimp(
    vimp, feature_names, top_k=12, out_path="fsds_global_vimp.png",
    title="FSDS global feature (embedding-dim) importance",
):
    vimp = np.asarray(vimp, dtype=float)
    order = np.argsort(-vimp)[:top_k][::-1]
    names = [feature_names[i] if i < len(feature_names) else f"dim_{i}" for i in order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(order))))
    ax.barh(names, vimp[order], color="#3b7dd8")
    ax.set_xlabel("combined domain + LOCO importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def visualize_attribution(result, G_exist, G_new, prefix="fsds"):
    """Write all available figures for a :class:`HierarchicalDrift` result."""
    paths = {}
    if result.node_attribution is not None:
        paths["node"] = visualize_node_attribution(
            G_exist, G_new, result.node_attribution, result.nodelist,
            out_path=f"{prefix}_node_attribution.png")
    if result.community_results:
        paths["community"] = visualize_community_attribution(
            result.community_results, out_path=f"{prefix}_community_attribution.png")
    if result.global_result is not None and result.global_result.vimp is not None:
        paths["global"] = visualize_global_vimp(
            result.global_result.vimp, result.feature_names,
            out_path=f"{prefix}_global_vimp.png")
    return paths
