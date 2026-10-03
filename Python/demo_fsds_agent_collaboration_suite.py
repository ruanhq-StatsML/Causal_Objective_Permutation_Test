#!/usr/bin/env python3
"""
Multi-scenario agent collaboration bench: uniform vs FSDS per pattern + stacked levers.

Run: cd Python && python3 demo_fsds_agent_collaboration_suite.py
Writes: ../artifacts/agent_collaboration_scenarios.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from fsds_sot.applications import (
    AgentPattern,
    adaptive_sample_count,
    build_langgraph_node_trace,
    build_multi_agent_trace,
    build_rag_multihop_trace,
    build_react_trace,
    build_self_consistency_trace,
    chain_dispersion,
    debate_early_stop_round,
    fit_agent_plan,
    pattern_playbook,
)
from fsds_sot.incremental_value import simulate_episode_outcome, uniform_plan
from fsds_sot.pipeline import FSDSSoT
from fsds_sot.pricing import DEFAULT_CHECK_COST_USD, DEFAULT_COST_PER_1K_MEDIUM_USD

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
D = 24


def _mc_pattern(
    *,
    name: str,
    pattern: AgentPattern,
    build_episode: Callable[[np.random.Generator], Tuple[Any, np.ndarray, np.ndarray]],
    trace_batch: Callable[[int, bool, int], np.ndarray],
    need: np.ndarray,
    min_tokens: np.ndarray | None = None,
    n_eval: int = 24,
    seed: int = 2026,
) -> Dict[str, Any]:
    ctrl = FSDSSoT(seed=seed)
    X0 = trace_batch(28, False, seed)
    X1 = trace_batch(28, True, seed + 1)
    min_t = min_tokens if min_tokens is not None else np.zeros(len(need), dtype=int)

    def _run(use_fsds: bool) -> Dict[str, float]:
        rng = np.random.default_rng(seed + 99)
        succ, costs, toks, checks = [], [], [], []
        for i in range(n_eval):
            rng_i = np.random.default_rng(seed + i)
            seg, q, shift = build_episode(rng_i)
            Z = seg.embeddings
            if use_fsds:
                _, plan, _ = ctrl.fit_plan(
                    X0,
                    X1,
                    Z,
                    branch_quality=q,
                    total_token_budget=1800,
                    latency_cap_tokens=480,
                    need_weights=need,
                    min_tokens_by_branch=min_t,
                    budget_kappa=0.88,
                    entropy_lambda=0.06,
                )
            else:
                plan = uniform_plan(shift, q, latency_cap_tokens=480, total_token_budget=1800)
            s, _, tot, c, ch = simulate_episode_outcome(plan, need, q, uniform_L=480, rng=rng)
            succ.append(s)
            costs.append(c)
            toks.append(tot)
            checks.append(ch)
        return {
            "mean_success": float(np.mean(succ)),
            "mean_cost_usd": float(np.mean(costs)),
            "mean_total_tokens": float(np.mean(toks)),
            "mean_checks": float(np.mean(checks)),
        }

    uni = _run(False)
    fds = _run(True)
    delta_s = fds["mean_success"] - uni["mean_success"]
    delta_c = uni["mean_cost_usd"] - fds["mean_cost_usd"]
    pct = 100.0 * delta_c / max(uni["mean_cost_usd"], 1e-9)
    return {
        "scenario_id": name,
        "pattern": pattern.value,
        "playbook": pattern_playbook().get(pattern.value, ""),
        "X": "trace_features per episode (batch REF/LIVE for covariate attribution)",
        "Y": "binary task success (MC simulator)",
        "Z": "segment/role/hop/node embeddings → branch budgets",
        "W": "REF vs LIVE trace batches",
        "collaboration_note": _collab_note(pattern),
        "uniform": uni,
        "fsds": fds,
        "incremental": {
            "success_delta_pp": round(delta_s * 100, 2),
            "cost_save_usd_per_episode": round(delta_c, 5),
            "cost_reduction_pct": round(pct, 2),
            "checks_delta": round(fds["mean_checks"] - uni["mean_checks"], 2),
        },
    }


def _collab_note(pattern: AgentPattern) -> str:
    notes = {
        AgentPattern.MULTI_AGENT: "Three roles (pro/con/judge) share one FSDS plan; shift on con triggers role-level budget split.",
        AgentPattern.REACT: "Thought→action→observation chain; collaboration = sequential segments under one topology gate.",
        AgentPattern.RAG_BRANCH: "Multi-hop retrieve; hops collaborate via parallel chunk budget when decomposable.",
        AgentPattern.SELF_CONSISTENCY: "N chain samples; dispersion drives adaptive N (fewer samples when chains agree).",
        AgentPattern.LANGGRAPH: "Checkpoint nodes collaborate via graph trace batch; retry only drifted nodes in prod hook.",
    }
    return notes.get(pattern, "")


def main() -> int:
    scenarios: List[Dict[str, Any]] = []

    # --- ReAct ---
    def react_ep(rng: np.random.Generator):
        t, a, o = rng.normal(size=(3, D))
        o[:8] += rng.normal(scale=1.2, size=8)
        seg = build_react_trace(t, a, o, tool_ok=[0.82, 0.78, 0.48])
        q = np.array([0.7, 0.72, 0.55])
        shift = np.linalg.norm(seg.embeddings - seg.embeddings.mean(axis=0, keepdims=True), axis=1)
        shift = shift / (shift.max() + 1e-9)
        return seg, q, shift

    def react_batch(n: int, live: bool, seed: int) -> np.ndarray:
        rows = []
        for i in range(n):
            rng = np.random.default_rng(seed + i)
            t, a, o = rng.normal(size=(3, D))
            if live:
                o[:8] += 1.0
            rows.append(build_react_trace(t, a, o).trace_features)
        return np.stack(rows)

    scenarios.append(
        _mc_pattern(
            name="react_observation_drift",
            pattern=AgentPattern.REACT,
            build_episode=react_ep,
            trace_batch=react_batch,
            need=np.array([0.45, 0.5, 0.88]),
            min_tokens=np.array([0, 0, 280]),
        )
    )

    # --- RAG multihop ---
    def rag_ep(rng: np.random.Generator):
        hops = rng.normal(size=(4, D))
        hops[2] += rng.normal(scale=1.5, size=D)
        seg = build_rag_multihop_trace(
            ("retrieve_1", "retrieve_2", "reason", "answer"),
            hops,
            grounding=np.array([0.92, 0.85, 0.7, 0.75]),
        )
        q = np.array([0.9, 0.82, 0.65, 0.78])
        shift = np.linalg.norm(seg.embeddings - seg.embeddings.mean(axis=0, keepdims=True), axis=1)
        shift = shift / (shift.max() + 1e-9)
        return seg, q, shift

    def rag_batch(n: int, live: bool, seed: int) -> np.ndarray:
        rows = []
        for i in range(n):
            rng = np.random.default_rng(seed + i)
            hops = rng.normal(size=(4, D))
            if live:
                hops[2] += 0.8
            rows.append(
                build_rag_multihop_trace(
                    ("retrieve_1", "retrieve_2", "reason", "answer"), hops
                ).trace_features
            )
        return np.stack(rows)

    scenarios.append(
        _mc_pattern(
            name="rag_multihop_hop3_drift",
            pattern=AgentPattern.RAG_BRANCH,
            build_episode=rag_ep,
            trace_batch=rag_batch,
            need=np.array([0.55, 0.6, 0.9, 0.85]),
            min_tokens=np.array([0, 0, 300, 200]),
        )
    )

    # --- Self-consistency ---
    def sc_ep(rng: np.random.Generator):
        n_chain = 5
        chains = rng.normal(size=(n_chain, D))
        seg = build_self_consistency_trace(chains, chain_scores=np.linspace(0.6, 0.82, n_chain))
        q = np.linspace(0.62, 0.84, n_chain)
        shift = np.linalg.norm(seg.embeddings - seg.embeddings.mean(axis=0, keepdims=True), axis=1)
        shift = shift / (shift.max() + 1e-9)
        return seg, q, shift

    def sc_batch(n: int, live: bool, seed: int) -> np.ndarray:
        rows = []
        for i in range(n):
            rng = np.random.default_rng(seed + i)
            chains = rng.normal(size=(5, D))
            if live:
                chains += rng.normal(scale=0.3, size=(5, D))
            rows.append(build_self_consistency_trace(chains).trace_features)
        return np.stack(rows)

    scenarios.append(
        _mc_pattern(
            name="self_consistency_chain_dispersion",
            pattern=AgentPattern.SELF_CONSISTENCY,
            build_episode=sc_ep,
            trace_batch=sc_batch,
            need=np.full(5, 0.75),
        )
    )

    # --- Multi-agent debate (existing style) ---
    def ma_ep(rng: np.random.Generator):
        embs = rng.normal(size=(3, D))
        embs[1] += rng.normal(scale=1.1, size=D)
        seg = build_multi_agent_trace(("pro", "con", "judge"), embs, role_quality=np.array([0.68, 0.6, 0.86]))
        q = np.array([0.68, 0.6, 0.86])
        shift = np.linalg.norm(embs - embs.mean(axis=0, keepdims=True), axis=1)
        shift = shift / (shift.max() + 1e-9)
        return seg, q, shift

    def ma_batch(n: int, live: bool, seed: int) -> np.ndarray:
        rows = []
        for i in range(n):
            rng = np.random.default_rng(seed + i)
            embs = rng.normal(size=(3, D))
            if live:
                embs[1] += 1.2
            rows.append(build_multi_agent_trace(("pro", "con", "judge"), embs).trace_features)
        return np.stack(rows)

    scenarios.append(
        _mc_pattern(
            name="multi_agent_debate_con_drift",
            pattern=AgentPattern.MULTI_AGENT,
            build_episode=ma_ep,
            trace_batch=ma_batch,
            need=np.array([0.5, 0.55, 0.92]),
            min_tokens=np.array([0, 0, 320]),
        )
    )

    # --- LangGraph ---
    def lg_ep(rng: np.random.Generator):
        nodes = rng.normal(size=(5, D))
        nodes[3] += rng.normal(scale=1.3, size=D)
        seg = build_langgraph_node_trace(
            ("plan", "retrieve", "reason", "tool_call", "respond"),
            nodes,
            node_success=np.array([0.9, 0.78, 0.8, 0.5, 0.88]),
        )
        q = np.array([0.9, 0.78, 0.8, 0.5, 0.88])
        shift = np.linalg.norm(nodes - nodes.mean(axis=0, keepdims=True), axis=1)
        shift = shift / (shift.max() + 1e-9)
        return seg, q, shift

    def lg_batch(n: int, live: bool, seed: int) -> np.ndarray:
        rows = []
        for i in range(n):
            rng = np.random.default_rng(seed + i)
            nodes = rng.normal(size=(5, D))
            if live:
                nodes[3] += 1.0
            rows.append(
                build_langgraph_node_trace(
                    ("plan", "retrieve", "reason", "tool_call", "respond"), nodes
                ).trace_features
            )
        return np.stack(rows)

    scenarios.append(
        _mc_pattern(
            name="langgraph_tool_call_drift",
            pattern=AgentPattern.LANGGRAPH,
            build_episode=lg_ep,
            trace_batch=lg_batch,
            need=np.array([0.4, 0.55, 0.7, 0.95, 0.8]),
            min_tokens=np.array([0, 0, 0, 320, 180]),
        )
    )

    # --- Stacked collaboration: FSDS role budget + debate early-stop ---
    ma = scenarios[-2]  # multi_agent row
    rounds_rng = np.random.default_rng(7)
    base = rounds_rng.normal(size=(3, D))
    round_embs = []
    for r in range(5):
        round_embs.append(base + rounds_rng.normal(scale=max(0.03, 0.5 - 0.12 * r), size=(3, D)))
    R = np.stack(round_embs)
    stop_r = debate_early_stop_round(R, max_rounds=5, dispersion_target=0.38, min_rounds=2)
    cost_round = 3 * 380 * (DEFAULT_COST_PER_1K_MEDIUM_USD / 1000.0) + DEFAULT_CHECK_COST_USD
    macro_save = (5 - stop_r) * cost_round
    micro_save = ma["incremental"]["cost_save_usd_per_episode"]
    combined = micro_save + macro_save / 5.0  # amortize one debate's round save per episode metric
    scenarios.append(
        {
            "scenario_id": "stacked_multi_agent_fsds_plus_early_stop",
            "pattern": "multi_agent_debate+early_stop",
            "playbook": "Micro: per-role FSDS budgets; macro: dispersion early-stop on round tensors.",
            "X": ma["X"],
            "Y": ma["Y"],
            "Z": "round×role embeddings (R,B,d) for stop; final-round Z for FSDS",
            "W": ma["W"],
            "collaboration_note": "Agents collaborate across rounds; stop when centroids stabilize; FSDS trims per-role spend on LIVE shift.",
            "uniform": ma["uniform"],
            "fsds": ma["fsds"],
            "incremental": ma["incremental"],
            "stacked_levers": {
                "fsds_cost_reduction_pct": ma["incremental"]["cost_reduction_pct"],
                "early_stop_rounds": stop_r,
                "early_stop_max_rounds": 5,
                "macro_round_cost_save_usd_per_debate": round(macro_save, 5),
                "combined_story_usd_per_debate": round(
                    ma["fsds"]["mean_cost_usd"] * (1 - ma["incremental"]["cost_reduction_pct"] / 100)
                    + cost_round * stop_r,
                    5,
                ),
                "illustrative_combined_save_vs_uniform_full_debate": round(
                    ma["uniform"]["mean_cost_usd"] - (cost_round * stop_r + ma["fsds"]["mean_cost_usd"] * 0.6),
                    5,
                ),
            },
        }
    )

    # Self-consistency adaptive N vignette
    chains = np.stack([np.random.default_rng(i).normal(size=D) for i in range(8)])
    disp = chain_dispersion(chains)
    n_adapt = adaptive_sample_count(chains, n_min=3, n_max=12, dispersion_target=0.12)
    scenarios.append(
        {
            "scenario_id": "self_consistency_adaptive_N",
            "pattern": AgentPattern.SELF_CONSISTENCY.value,
            "collaboration_note": "Parallel chains collaborate via vote/dispersion; low dispersion → fewer paid samples.",
            "metrics": {
                "chain_dispersion": round(disp, 4),
                "adaptive_n": n_adapt,
                "n_chains_provided": 8,
            },
            "incremental": {
                "cost_reduction_pct": round(max(0, (8 - n_adapt) / 8) * 100, 1),
                "note": "Illustrative sample-count lever; not MC dollar in this vignette.",
            },
        }
    )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "FSDSSoT.fit_plan via fit_agent_plan; uniform_plan baseline; MC n_eval=24, batch=28",
        "scenarios": scenarios,
    }
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "agent_collaboration_scenarios.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out.relative_to(ROOT)} ({len(scenarios)} scenarios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
