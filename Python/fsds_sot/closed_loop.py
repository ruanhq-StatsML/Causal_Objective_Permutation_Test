"""
Closed-loop FSDS iteration (measure → attribute → act → re-measure → escalate).

Hook `apply_intervention` and window collectors to production; demos use simulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .pipeline import FSDSSoT, SoTAttributionReport, SoTPlan


class LadderRung(IntEnum):
    REALLOCATE = 0
    RETOPOLOGY = 1
    REGROUND = 2
    RECONFIGURE = 3
    RELEARN = 4


@dataclass
class LoopState:
    rung: LadderRung = LadderRung.REALLOCATE
    iteration: int = 0
    baseline_drift: Optional[float] = None
    stabilized: bool = False
    history: list[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LoopOutcome:
    accepted: bool
    escalated: bool
    report: SoTAttributionReport
    plan: SoTPlan
    rung_used: LadderRung
    drift_proxy: float
    intervention: Dict[str, Any]


def drift_proxy(report: SoTAttributionReport) -> float:
    return float(report.mmd2 + (1.0 - min(report.overlap_ess, 1.0)) + report.domain_auc * 0.1)


def decay_reference(X_old: np.ndarray, X_new: np.ndarray, *, alpha: float = 0.85) -> np.ndarray:
    """Exponential blend of reference window after accepted intervention."""
    a = float(np.clip(alpha, 0.0, 1.0))
    n = min(X_old.shape[0], X_new.shape[0])
    if n <= 0:
        return X_old
    blend = a * X_old[:n] + (1.0 - a) * X_new[:n]
    if X_old.shape[0] > n:
        blend = np.vstack([blend, X_old[n:]])
    return blend


def suggest_intervention(plan: SoTPlan, report: SoTAttributionReport, rung: LadderRung) -> Dict[str, Any]:
    """Map ladder rung + plan to operational hints (for logging / policy hooks)."""
    top_shift_branch = int(np.argmax([b.expansion_tokens for b in plan.branch_budgets]))
    actions: Dict[str, Any] = {
        "rung": rung.name,
        "parallel": bool(plan.use_parallel),
        "n_branches": len(plan.branch_budgets),
        "top_shift_branch": top_shift_branch,
        "domain_auc": report.domain_auc,
        "overlap_ok": bool(report.overlap_ok),
    }
    if rung == LadderRung.REALLOCATE:
        actions["type"] = "reallocate"
        actions["budgets"] = [
            {"branch": b.branch_id, "L": b.expansion_tokens, "tier": b.model_tier}
            for b in plan.branch_budgets
        ]
    elif rung == LadderRung.RETOPOLOGY:
        actions["type"] = "retopology"
        actions["clusters"] = plan.clusters
    elif rung == LadderRung.REGROUND:
        actions["type"] = "reground"
        actions["verify_branch"] = top_shift_branch
    elif rung == LadderRung.RECONFIGURE:
        actions["type"] = "reconfigure"
        actions["hint"] = "rollback_or_prompt_swap"
    else:
        actions["type"] = "relearn"
        actions["hint"] = "agod_or_data_collection"
    return actions


def iterate_once(
    X_old: np.ndarray,
    X_new: np.ndarray,
    branch_embeddings: np.ndarray,
    branch_quality: Optional[np.ndarray],
    state: LoopState,
    *,
    controller: Optional[FSDSSoT] = None,
    apply_intervention: Optional[Callable[[SoTPlan, LadderRung, Dict[str, Any]], None]] = None,
    drift_improved: Callable[[float, float], bool] = lambda prev, new: new < prev * 0.92,
    stabilize_threshold: float = 0.12,
    **plan_kw,
) -> Tuple[LoopOutcome, LoopState]:
    """
    One closed-loop pass: attribute → plan → act → record.

    First window sets baseline; later windows accept when drift proxy drops vs baseline
    or falls below ``stabilize_threshold``.
    """
    ctrl = controller or FSDSSoT()
    report, plan, _econ = ctrl.fit_plan(
        X_old,
        X_new,
        branch_embeddings,
        branch_quality=branch_quality,
        **plan_kw,
    )
    proxy = drift_proxy(report)
    intervention = suggest_intervention(plan, report, state.rung)

    if state.baseline_drift is None:
        state.baseline_drift = proxy
        accepted = proxy <= stabilize_threshold
    else:
        accepted = drift_improved(state.baseline_drift, proxy) or proxy <= stabilize_threshold

    if accepted:
        state.stabilized = True
        state.baseline_drift = min(state.baseline_drift, proxy)
    elif report.overlap_ok and state.rung < LadderRung.RELEARN:
        state.rung = LadderRung(state.rung + 1)

    escalated = not accepted and state.iteration > 0 and int(state.rung) > 0

    if apply_intervention is not None and not accepted:
        apply_intervention(plan, state.rung, intervention)

    state.iteration += 1
    state.history.append(
        {
            "iteration": state.iteration,
            "drift_proxy": proxy,
            "rung": int(state.rung),
            "accepted": bool(accepted),
            "stabilized": bool(state.stabilized),
            "domain_auc": float(report.domain_auc),
        }
    )

    outcome = LoopOutcome(
        accepted=bool(accepted),
        escalated=bool(escalated),
        report=report,
        plan=plan,
        rung_used=state.rung,
        drift_proxy=proxy,
        intervention=intervention,
    )
    return outcome, state


def run_self_iteration(
    X_ref: np.ndarray,
    live_windows: List[np.ndarray],
    branch_embeddings: np.ndarray,
    branch_quality: Optional[np.ndarray],
    *,
    max_iters: Optional[int] = None,
    on_reference_update: Optional[Callable[[np.ndarray], None]] = None,
    **iterate_kw,
) -> Tuple[LoopState, List[LoopOutcome]]:
    """
    Multi-window closed loop. Updates reference via ``decay_reference`` after accept.
    """
    state = LoopState()
    outcomes: List[LoopOutcome] = []
    X_old = X_ref
    windows = live_windows[: max_iters or len(live_windows)]

    for X_new in windows:
        out, state = iterate_once(
            X_old,
            X_new,
            branch_embeddings,
            branch_quality,
            state,
            **iterate_kw,
        )
        outcomes.append(out)
        if out.accepted:
            X_old = decay_reference(X_old, X_new)
            if on_reference_update is not None:
                on_reference_update(X_old)
            break
    return state, outcomes


@dataclass
class LoopSummary:
    iterations: int
    stabilized: bool
    final_drift: float
    rung_final: str
    drift_series: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "stabilized": bool(self.stabilized),
            "final_drift": self.final_drift,
            "rung_final": self.rung_final,
            "drift_series": self.drift_series,
        }


def summarize_loop(state: LoopState, outcomes: List[LoopOutcome]) -> LoopSummary:
    series = [o.drift_proxy for o in outcomes]
    last = outcomes[-1] if outcomes else None
    return LoopSummary(
        iterations=state.iteration,
        stabilized=state.stabilized,
        final_drift=last.drift_proxy if last else 0.0,
        rung_final=last.rung_used.name if last else LadderRung.REALLOCATE.name,
        drift_series=series,
    )
