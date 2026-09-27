"""
Closed-loop FSDS iteration skeleton (Algorithm: measure → attribute → act → re-measure).

Hook `apply_intervention` and `collect_next_window` to production; demo uses placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Tuple

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
    history: list[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LoopOutcome:
    accepted: bool
    escalated: bool
    report: SoTAttributionReport
    plan: SoTPlan
    rung_used: LadderRung
    drift_proxy: float


def _drift_proxy(report: SoTAttributionReport) -> float:
    return float(report.mmd2 + (1.0 - min(report.overlap_ess, 1.0)) + report.domain_auc * 0.1)


def iterate_once(
    X_old: np.ndarray,
    X_new: np.ndarray,
    branch_embeddings: np.ndarray,
    branch_quality: Optional[np.ndarray],
    state: LoopState,
    *,
    controller: Optional[FSDSSoT] = None,
    apply_intervention: Optional[Callable[[SoTPlan, LadderRung], None]] = None,
    drift_improved: Callable[[float, float], bool] = lambda prev, new: new < prev * 0.95,
) -> Tuple[LoopOutcome, LoopState]:
    """
    One closed-loop pass: attribute → plan → (optional) act → record for re-measure.

    Caller runs again on the *next* window with updated X_new to complete the loop.
    """
    ctrl = controller or FSDSSoT()
    report, plan, _econ = ctrl.fit_plan(
        X_old,
        X_new,
        branch_embeddings,
        branch_quality=branch_quality,
    )
    proxy = _drift_proxy(report)

    prev_proxy = state.history[-1]["drift_proxy"] if state.history else proxy + 1.0
    accepted = drift_improved(prev_proxy, proxy) and report.overlap_ok

    if apply_intervention is not None:
        apply_intervention(plan, state.rung)

    escalated = False
    if not accepted and report.overlap_ok:
        if state.rung < LadderRung.RELEARN:
            state.rung = LadderRung(state.rung + 1)
            escalated = True
        else:
            escalated = True  # exhausted → abstain upstream

    state.iteration += 1
    state.history.append(
        {
            "iteration": state.iteration,
            "drift_proxy": proxy,
            "rung": int(state.rung),
            "accepted": accepted,
            "domain_auc": report.domain_auc,
        }
    )

    outcome = LoopOutcome(
        accepted=accepted,
        escalated=escalated,
        report=report,
        plan=plan,
        rung_used=state.rung,
        drift_proxy=proxy,
    )
    return outcome, state
