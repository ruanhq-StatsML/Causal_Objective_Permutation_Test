#!/usr/bin/env python3
"""
Enterprise narrative: federated FSDS across Legal / Finance / Product agent blocks.

Simulates vertical silos (no raw PII centralization) + impact receipts for GTM.

Run: cd Python && python3 demo_federated_legal_agent_blocks.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from DRPerm import DRPerm
from fsds_sot.federated_block_fsds import (
    local_block_fsds,
    merge_block_reports,
    simulate_vertical_partitions,
)
from fsds_sot.impact_receipt import build_federated_block_receipt

ROOT = Path(__file__).resolve().parent
ART = ROOT.parent / "artifacts"
DOCS = ROOT.parent / "docs"


def _enterprise_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_ref, n_live, d = 600, 350, 15
    X_ref = rng.normal(size=(n_ref, d))
    X_live = rng.normal(size=(n_live, d))
    # Regulatory change: legal clause embeddings shift (cols 0:5)
    X_live[:, 0:5] += rng.normal(scale=1.4, size=(n_live, 5))
    # Mild product traffic mix (cols 10:15)
    X_live[:, 10:15] += 0.35
    y_ref = rng.binomial(1, 0.5, size=n_ref)
    y_live = rng.binomial(1, 1 / (1 + np.exp(-0.2 * X_live[:, 2])), size=n_live)
    slices = {
        "legal_agent_block": slice(0, 5),
        "finance_agent_block": slice(5, 10),
        "product_agent_block": slice(10, 15),
    }
    return X_ref, X_live, y_ref, y_live, slices


def _po_p(X_ref, y_ref, X_live, y_live) -> float:
    X = np.vstack([X_ref, X_live])
    Y = np.concatenate([y_ref, y_live])
    W = np.array([0] * len(X_ref) + [1] * len(X_live), dtype=int)
    out = DRPerm(X, Y, W, n_perm=36, seed=2026, return_detail=True)
    return float(out.get("p_value", 1.0))


def main() -> int:
    X_ref, X_live, y_ref, y_live, slices = _enterprise_data()
    po_p = _po_p(X_ref, y_ref, X_live, y_live)
    ref_parts, live_parts = simulate_vertical_partitions(X_ref, X_live, slices)

    reports = []
    for bid in slices:
        reports.append(
            local_block_fsds(
                ref_parts[bid],
                live_parts[bid],
                block_id=bid,
                po_pvalue=po_p if bid == "legal_agent_block" else None,
            )
        )

    fed = merge_block_reports(reports, X_ref, X_live)
    action_for: dict[str, str] = {}
    for a in fed.server_actions:
        for bid in slices:
            if f"[{bid}]" in a:
                action_for[bid] = a
    receipts = [
        build_federated_block_receipt(
            b.to_dict(),
            fed.global_mmd2,
            action_hint=action_for.get(b.block_id, ""),
            scenario="cross_silo_sales_copilot",
        )
        for b in fed.blocks
    ]

    # Simulated revenue unlock: campaigns allowed when legal block monitored (not blocked blind)
    legal = next(b for b in fed.blocks if b.block_id == "legal_agent_block")
    revenue_unlock_usd = 0.0
    if legal.drift_type in ("feature_drift", "compound"):
        revenue_unlock_usd = 125_000.0  # illustrative quarter unlock vs full centralization ban

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "narrative": (
            "Multi-agent sales copilot: Legal reviews clauses, Finance checks margin, Product personalizes. "
            "Federated FSDS detects regulatory drift on legal block only; finance/product stay stable — "
            "OFS targets legal features, not global model shutdown."
        ),
        "global": {
            "mmd2": fed.global_mmd2,
            "domain_auc": fed.global_domain_auc,
            "po_pvalue_global": po_p,
        },
        "block_rank": fed.merged_rank,
        "blocks": [b.to_dict() for b in fed.blocks],
        "server_actions": fed.server_actions,
        "impact_receipts": receipts,
        "business_impact_sim": {
            "revenue_unlock_usd_quarter": revenue_unlock_usd,
            "compliance_rationale": "Actions tied to block-level MMD/AUC envelopes, not discretionary agent mute",
            "uplink_bytes_order": "O(|F_k|) per silo per round",
        },
    }

    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "federated_legal_agent_blocks.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    narrative_md = DOCS / "FSDS_FEDERATED_LEGAL_AGENT_BLOCKS.md"
    narrative_md.write_text(
        "\n".join(
            [
                "# Federated FSDS — Legal / Finance / Product agent blocks",
                "",
                payload["narrative"],
                "",
                "## Run",
                "",
                "```bash",
                "cd Python && python3 demo_federated_legal_agent_blocks.py",
                "```",
                "",
                f"Artifact: `artifacts/federated_legal_agent_blocks.json`.",
                "",
                "## Business impact",
                "",
                f"- Top drift block: `{fed.merged_rank[0][0]}`",
                f"- Simulated revenue unlock (quarter): **${revenue_unlock_usd:,.0f}** when monitor enables controlled rollout vs hard stop",
                "- Each block ships an **impact receipt** for audit (`impact_receipts[]`).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("=== Federated legal-agent blocks ===")
    for b in fed.blocks:
        print(f"  {b.block_id}: {b.drift_type} mmd2={b.mmd2:.4f}")
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
