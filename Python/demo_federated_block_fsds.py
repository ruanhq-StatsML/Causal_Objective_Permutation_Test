#!/usr/bin/env python3
"""Demo: federated FSDS on synthetic vertical blocks (minimal uplink)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from DRPerm import DRPerm  # noqa: E402
from fsds_sot.federated_block_fsds import (  # noqa: E402
    local_block_fsds,
    merge_block_reports,
    simulate_vertical_partitions,
)

ART = ROOT.parent / "artifacts"


def _make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_ref, n_live, d = 800, 400, 12
    X_ref = rng.normal(size=(n_ref, d))
    X_live = rng.normal(size=(n_live, d))
    # covariate shift on block B
    X_live[:, 4:8] += 1.0
    Y_ref = rng.binomial(1, 1 / (1 + np.exp(-0.3 * X_ref[:, 0])), size=n_ref)
    Y_live = rng.binomial(1, 1 / (1 + np.exp(-0.3 * X_live[:, 0] + 0.5 * X_live[:, 6])), size=n_live)
    slices = {"block_A": slice(0, 4), "block_B": slice(4, 8), "block_C": slice(8, 12)}
    return X_ref, X_live, Y_ref, Y_live, slices


def _po_pvalue(X_ref, y_ref, X_live, y_live, seed: int = 2026) -> float:
    X = np.vstack([X_ref, X_live])
    Y = np.concatenate([y_ref, y_live])
    W = np.array([0] * len(X_ref) + [1] * len(X_live), dtype=int)
    out = DRPerm(X, Y, W, n_perm=40, seed=seed, return_detail=True)
    return float(out.get("p_value", 1.0))


def main() -> int:
    X_ref, X_live, y_ref, y_live, slices = _make_data()
    po_p = _po_pvalue(X_ref, y_ref, X_live, y_live)
    ref_parts, live_parts = simulate_vertical_partitions(X_ref, X_live, slices)

    reports = []
    for bid, Xr in ref_parts.items():
        Xl = live_parts[bid]
        # PO-risk on full X only at server in production; demo passes global p to each block for typing
        reports.append(
            local_block_fsds(
                Xr,
                Xl,
                block_id=bid,
                po_pvalue=po_p if bid == "block_B" else None,
            )
        )

    fed = merge_block_reports(reports, X_ref, X_live)
    out = ART / "federated_block_fsds_report.json"
    ART.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(fed.to_dict(), f, indent=2)

    print("=== Federated block FSDS (synthetic) ===")
    print(f"Global MMD2={fed.global_mmd2:.4f}  domain_auc={fed.global_domain_auc:.4f}")
    print("Block rank (logo proxy):", fed.merged_rank)
    for b in fed.blocks:
        print(f"  {b.block_id}: type={b.drift_type} mmd2={b.mmd2:.4f} auc={b.domain_auc:.4f}")
    print("Server actions:")
    for a in fed.server_actions:
        print(" ", a)
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
