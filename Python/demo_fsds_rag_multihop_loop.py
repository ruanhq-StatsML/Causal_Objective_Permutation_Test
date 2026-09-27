#!/usr/bin/env python3
"""Multi-hop RAG closed-loop: reground hop with high shift until batch stabilizes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fsds_sot.agentic_dgp import episodes_to_arrays, generate_agentic_batch
from fsds_sot.applications import build_rag_multihop_trace, fit_agent_plan
from fsds_sot.closed_loop import run_self_iteration, summarize_loop


def _trace_batch(n: int, p: int, seed: int, drift: float) -> np.ndarray:
    ref = generate_agentic_batch(n, seed=seed, live=False)
    X, _, _, _ = episodes_to_arrays(ref)
    if drift > 0:
        X = X.copy()
        X[:, :8] += np.random.default_rng(seed + 99).normal(scale=drift, size=(X.shape[0], 8))
    return X


def main() -> int:
    rng = np.random.default_rng(42)
    d = 24
    ref = generate_agentic_batch(60, seed=1, live=False)
    X_ref, _, _, _ = episodes_to_arrays(ref)

    hops = ["hop0_retrieve", "hop1_refine", "hop2_answer"]
    hop_embs = rng.normal(size=(3, d))
    hop_embs[1, :10] += 2.0  # hop1 shifted vs ref corpus

    seg = build_rag_multihop_trace(hops, hop_embs, grounding=[0.92, 0.48, 0.75])

    # Simulated live windows: drift decays after each REGROUND-style fix
    drifts = [1.4, 0.9, 0.45, 0.08]
    windows = [_trace_batch(60, X_ref.shape[1], 10 + i, drift) for i, drift in enumerate(drifts)]

    applied = []

    def apply(_plan, rung, intervention):
        applied.append(intervention)

    state, outcomes = run_self_iteration(
        X_ref,
        windows,
        seg.embeddings,
        seg.quality,
        apply_intervention=apply,
        stabilize_threshold=0.15,
    )
    summary = summarize_loop(state, outcomes)

    _, plan, econ = fit_agent_plan(seg, X_ref, windows[0], total_token_budget=1800, latency_cap_tokens=350)

    out = {
        "pattern": "rag_multihop",
        "segments": list(seg.segment_names),
        "hop_grounding": seg.quality.tolist(),
        "interventions": applied,
        "loop": summary.to_dict(),
        "segment_budgets_hop1": [
            {"hop": hops[b.branch_id], "L": b.expansion_tokens, "checks": b.check_budget}
            for b in plan.branch_budgets
        ],
        "economics_first_window": econ.to_dict(),
    }
    print(json.dumps(out, indent=2))
    return 0 if summary.stabilized else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
