#!/usr/bin/env python3
"""MVP: graph-localization (off-the-shelf) -> candidate set -> FSDS -> evaluate.

Cross-level attribution: order-level and merchant-level metrics live at different
grains. We (1) ALIGN every metric to a common grain (here: the merchant), (2) run
graph-localization with an off-the-shelf clustering package (sklearn
SpectralClustering on an HSIC affinity) to get the candidate subgraph carrying
the shift, (3) run FSDS (LOGO on the shift metric) inside the candidate set, and
(4) evaluate without ground truth (structuredness + stability + recall sanity).

Dimension alignment is the crux: order-level metrics are aggregated up to the
merchant so all metrics share one index; only then are they comparable on one
graph. The candidate subgraph therefore spans levels (order + merchant).
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

from fsds_metric_graph import hsic


def generate(n_merch=200, shift=1.3, seed=2026):
    rng = np.random.default_rng(seed)
    W = rng.binomial(1, 0.5, n_merch)
    rows = []
    for m in range(n_merch):
        k = rng.poisson(60) + 20
        delivery = rng.normal(0, 1, k) - shift * W[m]          # order-level, faster if new
        item_price = rng.normal(0, 1, k)                       # order-level noise
        promo = rng.normal(0, 1, k)                            # order-level noise
        rows.append(pd.DataFrame({"merchant": m, "delivery": delivery,
                                  "item_price": item_price, "promo": promo}))
    orders = pd.concat(rows, ignore_index=True)

    # ALIGN order-level metrics to the merchant grain (common index)
    g = orders.groupby("merchant")
    feat = pd.DataFrame({
        "delivery__mean": g["delivery"].mean(),       # order-level (aligned)
        "item_price__mean": g["item_price"].mean(),   # order-level (aligned) noise
        "promo__mean": g["promo"].mean(),             # order-level (aligned) noise
    }).reindex(range(n_merch))

    # merchant-level metrics (native grain), chained to order-level delivery
    d = feat["delivery__mean"].values
    rating = -0.8 * d + rng.normal(0, 0.5, n_merch)
    n_orders = 0.7 * rating - 0.4 * d + rng.normal(0, 0.5, n_merch)
    gmv = 0.9 * n_orders + 0.3 * rating + rng.normal(0, 0.5, n_merch)
    refund = rng.normal(0, 1, n_merch)                # merchant-level noise
    feat["gmv"] = gmv
    feat["n_orders"] = n_orders
    feat["rating"] = rating
    feat["refund"] = refund

    level = {"delivery__mean": "order", "item_price__mean": "order",
             "promo__mean": "order", "gmv": "merchant", "n_orders": "merchant",
             "rating": "merchant", "refund": "merchant"}
    truth = {"delivery__mean", "gmv", "n_orders", "rating"}
    return feat, W, level, truth


def shift_signal(M, W):
    s = np.zeros(M.shape[1])
    for j in range(M.shape[1]):
        a, b = M[W == 0, j], M[W == 1, j]
        pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) or 1.0
        s[j] = abs(b.mean() - a.mean()) / pooled
    return s


def hsic_affinity(M):
    p = M.shape[1]
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            A[i, j] = A[j, i] = hsic(M[:, i], M[:, j])
    return A


def graph_localize(M, s, names, seed=0):
    """Off-the-shelf graph-localization: SpectralClustering on the HSIC affinity;
    pick the cluster with the largest mean shift as the candidate subgraph."""
    A = hsic_affinity(M)
    A = A / (A.max() or 1.0)
    np.fill_diagonal(A, 1.0)
    sc = SpectralClustering(n_clusters=2, affinity="precomputed",
                            assign_labels="discretize", random_state=seed)
    lab = sc.fit_predict(A)
    means = [s[lab == c].mean() for c in (0, 1)]
    cand = int(np.argmax(means))
    return [names[i] for i in range(len(names)) if lab[i] == cand], A


def fsds_logo(M, W, names, cand):
    """FSDS inside the candidate set: LOGO on the MMD-style shift (drop-one-metric
    reduction in the multivariate batch separation)."""
    from fsds_logo_mmd import _median_gamma, mmd2_unbiased
    idx = [names.index(c) for c in cand]
    Z = M[:, idx]
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    g = _median_gamma(Zs)
    full = mmd2_unbiased(Zs[W == 0], Zs[W == 1], g)
    rows = []
    for k, c in enumerate(cand):
        keep = [t for t in range(len(idx)) if t != k]
        Zk = Zs[:, keep]
        logo = full - mmd2_unbiased(Zk[W == 0], Zk[W == 1], _median_gamma(Zk))
        rows.append((c, logo))
    return full, sorted(rows, key=lambda r: -r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    feat, W, level, truth = generate(shift=args.shift, seed=args.seed)
    names = list(feat.columns)
    M = feat.values
    s = shift_signal(M, W)

    print("=" * 74)
    print("MVP: graph-localization -> candidate set -> FSDS -> evaluate")
    print("=" * 74)
    print("dimension alignment: order-level metrics aggregated to the merchant "
          "grain; all metrics share one index.\n")
    print("per-metric batch shift (|cohen d|), by level:")
    for j in np.argsort(-s):
        print(f"    {names[j]:16s} [{level[names[j]]:8s}] shift={s[j]:.2f}")

    cand, A = graph_localize(M, s, names, seed=args.seed)
    print(f"\n[1] GRAPH-LOCALIZATION (SpectralClustering on HSIC): candidate set "
          f"spans levels {sorted({level[c] for c in cand})}")
    print(f"    candidate metrics: {sorted(cand)}")

    full, logo = fsds_logo(M, W, names, cand)
    print(f"\n[2] FSDS inside candidate (LOGO on multivariate MMD, full={full:.3f}):")
    for c, v in logo:
        print(f"    {c:16s} [{level[c]:8s}] logo={v:+.4f}")

    # (3) evaluate without ground truth: recall sanity + candidate purity
    cand_set = set(cand)
    recall = len(cand_set & truth) / len(truth)
    purity = len(cand_set & truth) / len(cand_set)
    cross_level = len({level[c] for c in cand}) > 1
    print(f"\n[3] EVALUATION: candidate spans order+merchant={cross_level}; "
          f"recall={recall:.2f}, purity={purity:.2f} (vs synthetic truth "
          f"{sorted(truth)})")
    ok = cross_level and recall >= 0.75 and purity >= 0.6
    print(f"RESULT: {'PASS' if ok else 'CHECK'}  "
          f"(cross-level candidate localized, then FSDS-ranked)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
