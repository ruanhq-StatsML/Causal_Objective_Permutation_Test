"""rap_clever_covariate_guide.

Skill that turns the causal-inference *clever covariate*

    H = (Y - e) / (e * (1 - e))

into a natural-language planning hint and injects it into a RAP
(Reasoning-via-Planning) prompt so the LLM can *read* the anomaly signal and
adjust its exploration on the next planning step.

Beyond the raw per-step covariate, this skill also wires in the two online
monitoring primitives that this repository already relies on for streaming
distribution-shift detection (see ``mab_benchmark_core.empirical_pval_local``):

* an **online rolling empirical p-value** computed over the streaming sequence
  of monitored statistics (permute-then-refit style, upper-tail), and
* the derived **anomaly score** ``1 - p`` that the MAB benchmark uses to drive
  drift-adaptive exploration.

The clever covariate ``H`` supplies the *direction and magnitude* of the
surprise (a semantic signal), while the online rolling p-value supplies the
*statistical confidence* that the surprise is a genuine regime change rather
than noise. Fusing both yields the "numeric + semantic" double guidance:

* ``H`` tells the planner *which way* the last outcome deviated,
* the rolling p-value / anomaly score *gates* how strongly to react, and
* a regime-shift alarm (p < alpha) recommends re-planning.

Production value
----------------
* Fewer wasted expansions -> lower token cost. The planner only escalates
  exploration when the deviation is statistically notable, instead of
  overreacting to a single noisy outcome.
* Faster escape from bad branches -> higher success rate. A sustained,
  significant anomaly raises the alarm and steers the planner to safe branches
  / re-planning.
* Zero training. Pure prompt engineering on top of statistics the pipeline
  already computes, so the marginal engineering cost is close to zero.
* Complementary to numeric UCB injection: the same signal now also reaches the
  semantic (prompt) layer that generates candidate actions.

This module is dependency-light (only ``numpy`` + stdlib) and framework
agnostic, so it can be dropped into any RAP / MCTS planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

__all__ = [
    "clever_covariate",
    "OnlineRollingStatistic",
    "RAPCleverCovariateSkill",
    "RAPWithCovariate",
]


def clever_covariate(prior: float, outcome: float, eps: float = 1e-3) -> float:
    """Return the TMLE clever covariate ``H = (Y - e) / (e * (1 - e))``.

    Parameters
    ----------
    prior:
        Prior success probability ``e`` for the step (clipped to ``[eps, 1-eps]``
        to keep the denominator well-conditioned).
    outcome:
        Observed outcome ``Y`` (typically 0/1 but any real value is accepted).
    eps:
        Clipping tolerance for the prior.
    """
    e = float(np.clip(prior, eps, 1.0 - eps))
    return (float(outcome) - e) / (e * (1.0 - e))


class OnlineRollingStatistic:
    """Streaming empirical p-value + anomaly score monitor.

    Mirrors the upper-tail ``empirical_pval`` used throughout this repo's
    online distribution-shift detection: at step ``i`` (after ``burnin``
    observations) the p-value is

        p_i = (#{ history >= s_i } + 1) / (len(history) + 1)

    which is the online, permute-then-refit style calibration of how extreme
    the current statistic ``s_i`` is relative to its own recent history. Larger
    statistics are treated as more anomalous (upper tail). The anomaly score is
    ``1 - p``.

    A finite ``window`` turns this into a rolling monitor (only the last
    ``window`` statistics form the reference), which is what makes it robust to
    slow drift in long-running streams.
    """

    def __init__(self, burnin: int = 5, window: Optional[int] = None) -> None:
        if burnin < 0:
            raise ValueError("burnin must be non-negative")
        if window is not None and window <= 0:
            raise ValueError("window must be positive when provided")
        self.burnin = burnin
        self.window = window
        self.history: List[float] = []

    def update(self, statistic: float) -> Dict[str, float]:
        """Ingest one statistic and return ``{pval, anomaly_score, n}``.

        The current statistic is scored against the *existing* history and then
        appended, so the monitor never peeks at the current value when
        calibrating it (matching the offline ``empirical_pval`` semantics).
        """
        s = float(statistic)
        ref = self.history
        if self.window is not None and len(ref) > self.window:
            ref = ref[-self.window:]

        if len(self.history) < self.burnin:
            pval = 0.5
        else:
            cnt = sum(1 for v in ref if v >= s)
            pval = (cnt + 1) / (len(ref) + 1)

        self.history.append(s)
        if self.window is not None and len(self.history) > self.window:
            self.history = self.history[-self.window:]

        return {"pval": float(pval), "anomaly_score": float(1.0 - pval), "n": float(len(ref))}


_GUIDANCE_BLOCK = (
    "- aggressive: outcome much better than expected -> explore boldly, "
    "consider higher-risk/higher-reward branches.\n"
    "- conservative: outcome much worse than expected -> be cautious, prefer "
    "safe and proven branches, reduce exploration.\n"
    "- keep: outcome matched expectation (deviation not statistically notable) "
    "-> continue with the current strategy."
)


@dataclass
class RAPCleverCovariateSkill:
    """Skill: ``rap_clever_covariate_guide``.

    Fuses the clever covariate ``H`` with an online rolling empirical p-value
    and renders a RAP planning prompt with an explicit adjustment hint.

    Parameters
    ----------
    threshold:
        ``|H|`` above which a single-step deviation is considered large enough
        to escalate exploration even before the online monitor fires.
    alpha:
        Significance level for the online rolling p-value. ``p < alpha`` raises
        a regime-shift alarm and recommends re-planning.
    burnin, window:
        Configuration forwarded to the internal :class:`OnlineRollingStatistic`.
    """

    name: str = "rap_clever_covariate_guide"
    version: str = "0.2.0"
    threshold: float = 0.5
    alpha: float = 0.05
    burnin: int = 5
    window: Optional[int] = None
    monitor: OnlineRollingStatistic = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.monitor = OnlineRollingStatistic(burnin=self.burnin, window=self.window)

    def _decide(self, H: float, anomaly_score: float) -> Dict[str, Any]:
        """Return ``{hint, replan}`` from fused numeric + statistical signals."""
        alarm = anomaly_score >= (1.0 - self.alpha)
        strong = alarm or (abs(H) >= self.threshold)
        if not strong:
            hint = "keep"
        elif H > 0:
            hint = "aggressive"
        else:
            hint = "conservative"
        return {"hint": hint, "replan": bool(alarm)}

    def invoke(
        self,
        state: str,
        prior: float,
        outcome: float,
        history: Optional[List[str]] = None,
        monitor_stat: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute the covariate + online signals and build the RAP prompt.

        Parameters
        ----------
        state:
            Natural-language description of the current reasoning state.
        prior:
            Prior success probability ``e`` for the step.
        outcome:
            Observed outcome ``Y``.
        history:
            Recent planning steps (rendered into the prompt).
        monitor_stat:
            Statistic fed to the online rolling p-value monitor. Defaults to
            ``|H|`` so that large per-step surprises accumulate an anomaly
            signal; pass an explicit value (e.g. a validation loss or an
            OnlineRFPerm test statistic) to monitor a different quantity.

        Returns
        -------
        dict with keys ``prompt``, ``H``, ``hint``, ``pval``, ``anomaly_score``,
        ``replan`` and ``monitor_stat``.
        """
        e = float(np.clip(prior, 1e-3, 1.0 - 1e-3))
        H = clever_covariate(prior, outcome)
        stat = abs(H) if monitor_stat is None else float(monitor_stat)
        mon = self.monitor.update(stat)
        decision = self._decide(H, mon["anomaly_score"])

        prompt = self._render_prompt(
            state=state,
            e=e,
            outcome=outcome,
            H=H,
            stat=stat,
            mon=mon,
            decision=decision,
            history=history,
        )
        return {
            "prompt": prompt,
            "H": H,
            "hint": decision["hint"],
            "replan": decision["replan"],
            "pval": mon["pval"],
            "anomaly_score": mon["anomaly_score"],
            "monitor_stat": stat,
        }

    def _render_prompt(
        self,
        state: str,
        e: float,
        outcome: float,
        H: float,
        stat: float,
        mon: Dict[str, float],
        decision: Dict[str, Any],
        history: Optional[List[str]],
    ) -> str:
        history_text = "\n".join(history) if history else "None"
        alarm_txt = "RAISED" if decision["replan"] else "clear"
        replan_line = (
            "- Regime-shift alarm RAISED: the deviation is statistically "
            "significant (p < alpha). Consider re-planning: reset exploration, "
            "prune the current branch and re-expand from a safe node.\n"
            if decision["replan"]
            else ""
        )
        return (
            "[Clever Covariate Signal]\n"
            f"Previous prior (e):        {e:.3f}\n"
            f"Observed outcome (Y):      {outcome}\n"
            f"Clever covariate (H):      {H:.3f}\n"
            "\n"
            "[Online Rolling Monitor]  (OnlineRFPerm permute-then-refit statistic)\n"
            f"Monitored statistic:       {stat:.3f}\n"
            f"Rolling empirical p-value: {mon['pval']:.3f}  (reference n={int(mon['n'])})\n"
            f"Anomaly score (1 - p):     {mon['anomaly_score']:.3f}\n"
            f"Regime-shift alarm:        {alarm_txt}  (alpha = {self.alpha})\n"
            "\n"
            f"Adjustment hint: {decision['hint']}\n"
            "\n"
            f"{_GUIDANCE_BLOCK}\n"
            f"{replan_line}"
            "\n"
            "[Current State]\n"
            f"{state}\n"
            "\n"
            "[Recent History]\n"
            f"{history_text}\n"
            "\n"
            "Based on the above, generate the next planning step or candidate actions."
        )


class RAPWithCovariate:
    """Minimal RAP/MCTS expansion wrapper that injects the covariate prompt.

    ``llm`` must expose a ``generate(prompt: str) -> str`` method (or be any
    callable). ``parse_actions`` maps the raw LLM output + parent node to a list
    of child nodes; a trivial default splits on lines.
    """

    def __init__(
        self,
        llm: Any,
        skill: RAPCleverCovariateSkill,
        parse_actions: Optional[Callable[[str, Any], List[Any]]] = None,
    ) -> None:
        self.llm = llm
        self.skill = skill
        self.parse_actions = parse_actions or self._default_parse_actions

    @staticmethod
    def _default_parse_actions(actions: str, node: Any) -> List[Any]:
        return [line.strip() for line in str(actions).splitlines() if line.strip()]

    def _generate(self, prompt: str) -> str:
        if hasattr(self.llm, "generate"):
            return self.llm.generate(prompt)
        return self.llm(prompt)

    def expand(
        self,
        node: Any,
        prior: float,
        outcome: float,
        monitor_stat: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Expand ``node`` using a covariate-aware planning prompt.

        Returns ``{children, hint, replan, H, anomaly_score}`` so the caller can
        also feed ``hint`` / ``replan`` back into numeric UCB or pruning logic.
        """
        res = self.skill.invoke(
            state=getattr(node, "state", str(node)),
            prior=prior,
            outcome=outcome,
            history=getattr(node, "history", None),
            monitor_stat=monitor_stat,
        )
        actions = self._generate(res["prompt"])
        children = self.parse_actions(actions, node)
        return {
            "children": children,
            "hint": res["hint"],
            "replan": res["replan"],
            "H": res["H"],
            "anomaly_score": res["anomaly_score"],
        }


def _demo() -> None:
    skill = RAPCleverCovariateSkill(threshold=0.5, alpha=0.05, burnin=3)
    # Prior e=0.3, last step succeeded (Y=1) -> positive surprise -> aggressive.
    res = skill.invoke(
        state="Solving a 3-step arithmetic reasoning problem; step 1 done.",
        prior=0.3,
        outcome=1,
        history=["step1: decompose the problem", "step2: compute sub-goal"],
    )
    print("hint:", res["hint"], "| H:", round(res["H"], 3), "| anomaly:", round(res["anomaly_score"], 3))
    print(res["prompt"])


if __name__ == "__main__":
    _demo()
