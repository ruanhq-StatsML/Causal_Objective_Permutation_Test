#!/usr/bin/env python3
"""One-page PNG: multi-agent, LangGraph, federated legal, uplift OPEX economics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def _load(name: str) -> dict:
    p = ART / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ma = _load("multi_agent_fsds_economics.json")
    lg = _load("langgraph_fsds_hook.json")
    fed = _load("federated_legal_agent_blocks.json")
    uplift = _load("uplift_two_layer_benchmark.json")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("FSDS business impact pack (sim + demos)", fontsize=14, fontweight="bold")

    # Panel A: multi-agent MC cost
    ax = axes[0, 0]
    mc = ma.get("monte_carlo_eval") or {}
    u = mc.get("uniform_debate_swarm", {})
    f = mc.get("fsds_routed_debate", {})
    if u and f:
        ax.bar(
            ["Uniform swarm", "FSDS routed"],
            [u.get("mean_cost_usd", 0), f.get("mean_cost_usd", 0)],
            color=["#94a3b8", "#2563eb"],
        )
        ax.set_ylabel("$/debate episode")
        ax.set_title("Multi-agent debate")
    else:
        ax.text(0.5, 0.5, "Run demo_fsds_multi_agent_economics.py", ha="center", va="center")

    # Panel B: debate early-stop
    es = _load("multi_agent_debate_early_stop.json")
    ax = axes[0, 1]
    if es.get("dispersion_series"):
        ax.plot(es["dispersion_series"], "o-", color="#059669")
        ax.axhline(0.14, color="#dc2626", ls="--", label="stop target")
        ax.set_xlabel("Debate round")
        ax.set_ylabel("Inter-role dispersion")
        ax.set_title(f"Early stop @ round {es.get('stop_round')}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Run demo_fsds_multi_agent_debate_early_stop.py", ha="center", va="center")

    # Panel C: federated blocks
    ax = axes[1, 0]
    blocks = fed.get("blocks") or []
    if blocks:
        ids = [b["block_id"].replace("_agent_block", "")[:8] for b in blocks]
        mmd = [b["mmd2"] for b in blocks]
        ax.barh(ids, mmd, color="#7c3aed")
        ax.set_xlabel("Block MMD²")
        ax.set_title("Federated silos (legal shift)")
    else:
        ax.text(0.5, 0.5, "Run demo_federated_legal_agent_blocks.py", ha="center", va="center")

    # Panel D: uplift OPEX by dataset
    ax = axes[1, 1]
    runs = uplift.get("runs") or []
    if runs:
        names = [r["dataset"][:12] for r in runs]
        opex = [r.get("economics_summary", {}).get("estimated_opex_net_usd", 0) for r in runs]
        ax.bar(names, opex, color="#ea580c")
        ax.set_ylabel("OPEX net (sim USD)")
        ax.set_title("Uplift cap rules (excl. RELEARN CAPEX)")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    else:
        ax.text(0.5, 0.5, "Run run_uplift_subset_benchmark.py", ha="center", va="center")

    plt.tight_layout()
    ART.mkdir(exist_ok=True)
    out = ART / "fsds_business_impact_dashboard_en.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print("Wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
