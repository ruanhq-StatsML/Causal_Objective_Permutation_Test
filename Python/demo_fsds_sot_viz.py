#!/usr/bin/env python3
"""Run FSDS-SoT demo and write an embedding visualization dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from demo_fsds_sot import synthetic_sot_batch
from fsds_sot import FSDSSoT
from fsds_sot.viz import plot_sot_dashboard


def main() -> int:
    branches, p = 6, 32
    X0, B0, _ = synthetic_sot_batch(120, branches, p, seed=1, drift_strength=0.0)
    X1, B1, Q1 = synthetic_sot_batch(120, branches, p, seed=2, drift_strength=0.0)
    X1[:, :6] += np.random.default_rng(3).normal(scale=1.5, size=(X1.shape[0], 6))
    B1[:, 2, :] += 0.8
    B1[:, 4, :] += 0.5

    branch_emb = B1.mean(axis=0)
    controller = FSDSSoT(seed=2026)
    report, plan, econ = controller.fit_plan(
        X0, X1, branch_emb, branch_quality=Q1[0], total_token_budget=3000, latency_cap_tokens=480
    )

    ref = branch_emb.mean(axis=0, keepdims=True)
    shift = np.linalg.norm(branch_emb - ref, axis=1)

    out = Path(__file__).resolve().parent / "artifacts" / "fsds_sot_dashboard.png"
    plot_sot_dashboard(
        branch_emb,
        branch_shift_scores=shift,
        branch_quality=Q1[0],
        feature_vimp=report.covariate_vimp,
        out_path=out,
        title="FSDS-SoT: groups + budget drivers + batch drift",
    )
    print(f"Wrote {out}")
    print(f"Clusters: {report.decomposability.sequential_clusters}")
    for b in plan.branch_budgets:
        print(
            f"  branch {b.branch_id}: L={b.expansion_tokens} tier={b.model_tier} "
            f"checks={b.check_budget}"
        )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
