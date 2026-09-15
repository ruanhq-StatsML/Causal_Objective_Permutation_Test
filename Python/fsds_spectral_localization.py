#!/usr/bin/env python3
"""Spectral localization of a distribution shift on the metric graph.

Each metric/feature is a graph node (nodes are NOT collapsed -> no information
loss) and the per-node batch shift is a graph signal s. With the (normalized)
graph Laplacian L on an HSIC affinity:

  * Dirichlet energy  s^T L s = sum_{i<j} w_ij (s_i - s_j)^2  and the Rayleigh
    quotient  R(s)=s^T L s / s^T s  measure whether the shift is SMOOTH on the
    graph (co-shifting neighbours = structured / localized) vs high-frequency
    noise.
  * Low-frequency reconstruction  s_low = U_k U_k^T s  (k smallest eigenvalues)
    keeps only the smooth part; the localized subset = nodes with large |s_low|
    (equivalently a heat-kernel diffusion exp(-t L) s).

Because there is no ground truth beyond business, effectiveness is judged by a
concise, label-free 4-check battery: structured?, concentrated?, stable?,
calibrated?  (see `evaluate`).
"""
from __future__ import annotations

import argparse
import numpy as np

from fsds_metric_graph import hsic


def generate(n=300, n_noise=16, shift=1.4, seed=2026):
    """A 4-node shifted CHAIN (delivery->rating->orders->gmv) plus n_noise
    unrelated noise metrics; returns metric matrix, batch W, names, truth set."""
    rng = np.random.default_rng(seed)
    W = rng.binomial(1, 0.5, n)
    delivery = rng.normal(0, 1, n) - shift * W
    rating = -0.8 * delivery + rng.normal(0, 0.6, n)
    orders = 0.7 * rating - 0.4 * delivery + rng.normal(0, 0.6, n)
    gmv = 0.9 * orders + 0.3 * rating + rng.normal(0, 0.6, n)
    chain = np.column_stack([delivery, rating, orders, gmv])
    noise = rng.normal(0, 1, (n, n_noise))
    M = np.column_stack([chain, noise])
    names = ["delivery", "rating", "orders", "gmv"] + [f"noise_{i:02d}"
                                                       for i in range(n_noise)]
    truth = {"delivery", "rating", "orders", "gmv"}
    return M, W, names, truth


def hsic_affinity(M, tau=0.01):
    p = M.shape[1]
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            A[i, j] = A[j, i] = hsic(M[:, i], M[:, j])
    return A * (A > tau)


def shift_signal(M, W):
    s = np.zeros(M.shape[1])
    for j in range(M.shape[1]):
        a, b = M[W == 0, j], M[W == 1, j]
        pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) or 1.0
        s[j] = abs(b.mean() - a.mean()) / pooled
    return s


def _laplacian(A):
    d = A.sum(1)
    dinv = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
    return np.eye(len(A)) - (dinv[:, None] * A * dinv[None, :])   # normalized L


def rayleigh(A, s):
    s = s / (np.linalg.norm(s) + 1e-12)
    return float(s @ _laplacian(A) @ s)                          # in [0, 2]


def structuredness_p(A, s, n_perm=500, seed=0):
    obs = rayleigh(A, s)
    rng = np.random.default_rng(seed)
    cnt = sum(rayleigh(A, rng.permutation(s)) <= obs for _ in range(n_perm))
    return obs, (1 + cnt) / (1 + n_perm)


def low_freq(A, s, k=3):
    w, U = np.linalg.eigh(_laplacian(A))
    Uk = U[:, :k]
    s_low = Uk @ (Uk.T @ s)
    energy = float((Uk.T @ s) @ (Uk.T @ s) / (s @ s + 1e-12))
    return s_low, energy


def localized_idx(A, s, k=3):
    s_low, _ = low_freq(A, s, k)
    mag = np.abs(s_low)
    return {i for i in range(len(mag)) if mag[i] > mag.mean() + mag.std()}


# --------------------------------------------------------------------------- #
# Concise, label-free evaluation battery
# --------------------------------------------------------------------------- #
def evaluate(M, W, tau=0.01, k=3, n_boot=40, n_null=30, seed=2026):
    A = hsic_affinity(M, tau)
    s = shift_signal(M, W)
    R, p_struct = structuredness_p(A, s, seed=seed)
    _, conc = low_freq(A, s, k)
    base = localized_idx(A, s, k)

    rng = np.random.default_rng(seed + 1)
    n = len(W)
    jac = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        st = localized_idx(hsic_affinity(M[idx], tau), shift_signal(M[idx], W[idx]), k)
        u = base | st
        jac.append(len(base & st) / len(u) if u else 1.0)

    fp = 0
    for b in range(n_null):
        _, pn = structuredness_p(A, shift_signal(M, rng.permutation(W)),
                                 n_perm=200, seed=seed + b)
        fp += (pn < 0.05)
    return {"rayleigh": R, "p_structured": p_struct, "concentration": conc,
            "stability_jaccard": float(np.mean(jac)), "null_fpr": fp / n_null,
            "localized_idx": base}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=float, default=1.4)
    ap.add_argument("--n-noise", type=int, default=16)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plot", type=str, default="")
    args = ap.parse_args()

    M, W, names, truth = generate(n_noise=args.n_noise, shift=args.shift, seed=args.seed)
    A = hsic_affinity(M)
    s = shift_signal(M, W)
    s_low, _ = low_freq(A, s, args.k)

    print("=" * 74)
    print(f"SPECTRAL LOCALIZATION on a {len(names)}-node metric graph "
          f"(4 shifted chain + {args.n_noise} noise)")
    print("=" * 74)

    ev = evaluate(M, W, k=args.k, seed=args.seed)
    loc = sorted(names[i] for i in ev["localized_idx"])
    print("CONCISE EVALUATION BATTERY (no ground truth needed):")
    print(f"  (1) structured?    R={ev['rayleigh']:.3f}  perm p={ev['p_structured']:.3f}"
          "   (small R & p -> shift aligned with graph)")
    print(f"  (2) concentrated?  low-freq energy ratio={ev['concentration']:.2f}"
          "   (high -> a few modes / a subgraph)")
    print(f"  (3) stable?        bootstrap Jaccard={ev['stability_jaccard']:.2f}"
          "   (high -> same subset recurs)")
    print(f"  (4) calibrated?    null (permuted-batch) FPR={ev['null_fpr']:.2f}"
          "   (should be ~<=0.05)")
    print(f"\n  localized subset: {loc}")
    print(f"  (sanity vs synthetic truth {sorted(truth)}: "
          f"recall={len(set(loc) & truth)}/{len(truth)}, "
          f"false-positives={len(set(loc) - truth)})")
    ok = (ev["p_structured"] < 0.05 and ev["concentration"] > 0.6
          and ev["stability_jaccard"] > 0.6 and ev["null_fpr"] <= 0.1)
    print(f"RESULT: {'PASS' if ok else 'CHECK'}")

    if args.plot:
        _plot(s, s_low, names, truth, args.plot)
    return 0 if ok else 1


def _plot(s, s_low, names, truth, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(-np.abs(s_low))[:12]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(len(order))
    col = ["#d62728" if names[i] in truth else "#7f7f7f" for i in order]
    a1.bar(x, s[order], color=col)
    a1.set_title("Raw batch-shift signal (top nodes)")
    a2.bar(x, np.abs(s_low[order]), color=col)
    a2.set_title("Low-frequency (spectral) reconstruction $|s_{low}|$")
    for a in (a1, a2):
        a.set_xticks(x)
        a.set_xticklabels([names[i] for i in order], rotation=40, ha="right", fontsize=7)
    fig.suptitle("Spectral localization: the smooth low-frequency part keeps the "
                 "connected shifted subgraph (red) and suppresses isolated noise")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"saved -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
