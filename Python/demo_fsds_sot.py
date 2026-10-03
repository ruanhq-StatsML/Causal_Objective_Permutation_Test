#!/usr/bin/env python3
"""Runnable demo: FSDS-SoT attribution -> budget plan -> economics."""

from __future__ import annotations

import json
import sys

import numpy as np

from fsds_sot import FSDSSoT


def synthetic_sot_batch(n: int, branches: int, p: int, seed: int, drift_strength: float):
    rng = np.random.default_rng(seed)
    X_list = []
    branch_list = []
    quality_list = []
    for _ in range(n):
        base = rng.normal(size=(branches, p))
        # Inject drift on first 20% of features for "live" batch later
        scalars = np.array([branches, rng.integers(800, 2000), rng.normal()], dtype=float)
        feat = np.concatenate([base.mean(axis=0), scalars])
        X_list.append(feat)
        branch_list.append(base)
        quality_list.append(rng.uniform(0.55, 0.95, size=branches))
    return np.vstack(X_list), np.stack(branch_list), quality_list


def main() -> int:
    branches, p = 6, 32
    X0, B0, Q0 = synthetic_sot_batch(120, branches, p, seed=1, drift_strength=0.0)
    X1, B1, Q1 = synthetic_sot_batch(120, branches, p, seed=2, drift_strength=0.0)
    # Live batch: shift first 6 dims + inflate branch 2 & 4
    X1[:, :6] += np.random.default_rng(3).normal(scale=1.5, size=(X1.shape[0], 6))
    B1[:, 2, :] += 0.8
    B1[:, 4, :] += 0.5

    controller = FSDSSoT(seed=2026)
    report, plan, econ = controller.fit_plan(
        X0,
        X1,
        B1.mean(axis=0),
        branch_quality=Q1[0],
        total_token_budget=3000,
        latency_cap_tokens=480,
    )

    out = {
        "attribution": {
            "domain_auc": report.domain_auc,
            "mmd2": report.mmd2,
            "overlap_ok": bool(report.overlap_ok),
            "overlap_ess": report.overlap_ess,
            "top_features": report.top_features,
            "decomposability": {
                "allow_parallel": bool(report.decomposability.allow_parallel),
                "parallelizability": report.decomposability.parallelizability,
                "redundancy_rate": report.decomposability.redundancy_rate,
                "n_clusters": report.decomposability.n_clusters,
                "clusters": report.decomposability.sequential_clusters,
            },
        },
        "plan": {
            "use_parallel": plan.use_parallel,
            "latency_cap_tokens": plan.latency_cap_tokens,
            "branches": [
                {
                    "id": b.branch_id,
                    "expansion_tokens": b.expansion_tokens,
                    "check_budget": b.check_budget,
                    "model_tier": b.model_tier,
                    "shift_score": round(b.shift_score, 4),
                    "quality_score": round(b.quality_score, 4),
                }
                for b in plan.branch_budgets
            ],
        },
        "economics": econ.to_dict(),
    }

    print(json.dumps(out, indent=2))
    print("\n--- Summary ---")
    print(f"Parallel allowed: {plan.use_parallel}")
    print(f"Latency: {econ.baseline_latency_tokens} -> {econ.fsds_latency_tokens} tokens "
          f"({econ.latency_reduction_pct:.1f}% reduction on critical path)")
    print(f"Cost: ${econ.baseline_compute_cost_usd:.4f} -> "
          f"${econ.fsds_compute_cost_usd + econ.check_cost_usd:.4f} "
          f"(net save ${econ.net_savings_usd:.4f}, ROI {econ.roi_multiple:.1f}x vs FSDS overhead)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ".")
    raise SystemExit(main())
