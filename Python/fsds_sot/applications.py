"""
Adapters: map agent-reasoning traces to FSDS-SoT pipeline inputs.

Each pattern exposes segment embeddings + optional per-segment quality for
Object 1 (topology) and Object 2 (budget).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .pipeline import FSDSSoT


class AgentPattern(str, Enum):
    SOT = "skeleton_of_thought"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    MULTI_AGENT = "multi_agent_debate"
    RAG_BRANCH = "rag_per_chunk"
    SELF_CONSISTENCY = "self_consistency"


@dataclass
class SegmentTrace:
    """One reasoning episode as segments."""

    pattern: AgentPattern
    segment_names: Tuple[str, ...]
    embeddings: np.ndarray  # (B, d)
    quality: np.ndarray  # (B,)
    trace_features: np.ndarray  # (p,) episode-level scalars for batch VIMP


def _normalize_rows(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n = np.where(n <= 1e-12, 1.0, n)
    return Z / n


def build_react_trace(
    thought_emb: np.ndarray,
    action_emb: np.ndarray,
    observation_emb: np.ndarray,
    *,
    tool_ok: Optional[Sequence[float]] = None,
    extra_steps: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> SegmentTrace:
    """ReAct: segments = Thought, Action, Observation (+ optional repeats)."""
    parts = [("thought", thought_emb), ("action", action_emb), ("observation", observation_emb)]
    if extra_steps:
        parts.extend(extra_steps)
    names = tuple(n for n, _ in parts)
    Z = _normalize_rows(np.stack([p for _, p in parts], axis=0))
    if tool_ok is None:
        q = np.array([0.7, 0.75, 0.6][: len(names)], dtype=float)
    else:
        q = np.asarray(tool_ok, dtype=float)
    scalars = np.array([len(names), float(q.mean()), float(q.min())], dtype=float)
    trace = np.concatenate([Z.mean(axis=0), scalars])
    return SegmentTrace(AgentPattern.REACT, names, Z, q, trace)


def build_plan_execute_trace(
    plan_step_embs: np.ndarray,
    execution_quality: Optional[np.ndarray] = None,
) -> SegmentTrace:
    """Plan-and-execute: each row = one plan step embedding."""
    Z = _normalize_rows(plan_step_embs)
    B = Z.shape[0]
    names = tuple(f"plan_step_{i}" for i in range(B))
    q = execution_quality if execution_quality is not None else np.full(B, 0.72)
    q = np.asarray(q, dtype=float)
    trace = np.concatenate([Z.mean(axis=0), np.array([B, q.std(), q.min()])])
    return SegmentTrace(AgentPattern.PLAN_EXECUTE, names, Z, q, trace)


def chain_dispersion(chain_embs: np.ndarray) -> float:
    """Self-consistency: mean pairwise cosine distance in sample chain embeddings."""
    Z = _normalize_rows(chain_embs)
    if Z.shape[0] < 2:
        return 0.0
    sim = Z @ Z.T
    np.fill_diagonal(sim, 0.0)
    return float(1.0 - sim.mean())


def adaptive_sample_count(
    chain_embs: np.ndarray,
    *,
    n_min: int = 3,
    n_max: int = 15,
    dispersion_target: float = 0.12,
) -> int:
    """Stop / continue sampling: more chains while dispersion above target."""
    d = chain_dispersion(chain_embs)
    if d <= dispersion_target:
        return max(n_min, chain_embs.shape[0])
    extra = int(np.ceil((d - dispersion_target) / 0.04))
    return int(np.clip(chain_embs.shape[0] + extra, n_min, n_max))


def build_self_consistency_trace(
    chain_embs: np.ndarray,
    chain_scores: Optional[np.ndarray] = None,
) -> SegmentTrace:
    """Each row = one sampled reasoning chain final-state embedding."""
    Z = _normalize_rows(chain_embs)
    names = tuple(f"chain_{i}" for i in range(Z.shape[0]))
    if chain_scores is None:
        q = np.linspace(0.55, 0.85, Z.shape[0])
    else:
        q = np.asarray(chain_scores, dtype=float)
    disp = chain_dispersion(Z)
    trace = np.concatenate([Z.mean(axis=0), np.array([Z.shape[0], disp, q.std()])])
    return SegmentTrace(AgentPattern.SELF_CONSISTENCY, names, Z, q, trace)


def build_rag_multihop_trace(
    hop_names: Sequence[str],
    hop_chunk_embs: np.ndarray,
    *,
    grounding: Optional[np.ndarray] = None,
) -> SegmentTrace:
    """
    Multi-hop RAG: one segment per hop (aggregated chunk embedding for that hop).
    hop_chunk_embs: (H, d)
    """
    Z = _normalize_rows(hop_chunk_embs)
    names = tuple(hop_names)
    if grounding is None:
        q = np.linspace(0.9, 0.5, len(names))[::-1]
    else:
        q = np.asarray(grounding, dtype=float)
    trace = np.concatenate([Z.mean(axis=0), np.array([len(names), float(q.min()), float(q.std())])])
    return SegmentTrace(AgentPattern.RAG_BRANCH, names, Z, q, trace)


def build_multi_agent_trace(
    role_names: Sequence[str],
    utterance_embs: np.ndarray,
    role_quality: Optional[np.ndarray] = None,
) -> SegmentTrace:
    """Debate / multi-agent: one segment per role utterance."""
    Z = _normalize_rows(utterance_embs)
    names = tuple(role_names)
    q = role_quality if role_quality is not None else np.full(len(names), 0.68)
    trace = np.concatenate([Z.mean(axis=0), np.array([len(names), np.std(Z), 0.0])])
    return SegmentTrace(AgentPattern.MULTI_AGENT, names, Z, np.asarray(q, dtype=float), trace)


def fit_agent_plan(
    segment: SegmentTrace,
    X_old: np.ndarray,
    X_new: np.ndarray,
    *,
    controller: Optional[FSDSSoT] = None,
    **plan_kw,
):
    """Run FSDS plan on any segment trace (same Object 1+2 as SoT)."""
    ctrl = controller or FSDSSoT()
    return ctrl.fit_plan(
        X_old,
        X_new,
        segment.embeddings,
        branch_quality=segment.quality,
        **plan_kw,
    )


def pattern_playbook() -> dict[str, str]:
    """One-line ops hint per pattern."""
    return {
        AgentPattern.SOT.value: "Parallel skeleton points; merge/order then per-branch L/tier.",
        AgentPattern.REACT.value: "Budget Observation vs Thought; re-tool if obs shift without tool_ok.",
        AgentPattern.PLAN_EXECUTE.value: "Re-plan step with high PO-risk; sequential inside coupled steps.",
        AgentPattern.MULTI_AGENT.value: "Down-weight redundant roles; merge by concept dedup.",
        AgentPattern.RAG_BRANCH.value: "Per-chunk L and retrieve k; parallel chunk expand.",
        AgentPattern.SELF_CONSISTENCY.value: "Dispersion on chain embeddings → adaptive sample count N.",
    }
