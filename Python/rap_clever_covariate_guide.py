"""rap_clever_covariate_guide: turn the clever covariate into an RAP planning hint.

This module is the MVP that connects the FSDS / causal-objective machinery in this
repository to Reasoning-via-Planning (RAP) style search (MCTS / Tree-of-Thoughts).

The clever covariate is the influence-function weight from the doubly-robust /
TMLE pseudo-outcome learner used elsewhere in this repo (see ``tarnet.py``)::

        H = (Y - e) / (e * (1 - e))

where ``e`` is the *prior* (the stable score the search assigned to the committed
step, i.e. the propensity of that arm being taken / succeeding) and ``Y`` is the
episode outcome (``1`` solved, ``0`` not). In the causal setting ``H`` re-weights
the residual; here we reuse it as an *online anomaly score*:

* ``|H|`` is the size of the departure from the prior.
* ``sign(H)`` says whether the departure was a success (``H > 0``) or a failure
  (``H < 0``).

The novelty of this skill is not the number -- it is *where the number is read*.
In vanilla RAP, ``H`` (or a UCB variant of it) only nudges numeric node statistics.
Here we write ``H`` and its adjustment direction into the *planning prompt* as
natural language, so the LLM that proposes the next candidate actions can consume
the anomaly signal semantically. The companion paper shows that the recovery in
success rate appears only when the *next* episode's ranking consumes the sign, not
from printing ``H`` into a frozen search -- so this module keeps the sign front and
centre and exposes the channels that actually move (activation rate, prior of the
chosen action, conservative share) rather than ``|H|`` alone.

Causal-inference note on why this is well defined online: the current step cannot
read its own ``Y`` (the outcome is written back onto every committed step only
after the search returns). The sliding window therefore holds only *finished*
beats whose ``Y`` is already known, so ranking the next step by the sign never
leaks the current label into its own feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

__all__ = [
    "DEFAULT_CLIP",
    "DEFAULT_THRESHOLD",
    "clip_prior",
    "clever_covariate",
    "adjustment_hint",
    "hint_prior_band",
    "calibrate_threshold",
    "Beat",
    "WindowStats",
    "CovariateSignal",
    "RAPCleverCovariateSkill",
    "RAPWithCovariate",
]

DEFAULT_CLIP = (0.001, 0.999)
DEFAULT_THRESHOLD = 0.5

_HINT_LEGEND = (
    "- aggressive: the outcome was much better than the prior expected. Explore "
    "more boldly; consider higher-risk / higher-reward branches.\n"
    "- conservative: the outcome was much worse than the prior expected. Be more "
    "cautious; prefer safe, already-proven branches and reduce exploration.\n"
    "- keep: the outcome matched the prior. Continue with the current strategy."
)


def clip_prior(prior: float, clip: Sequence[float] = DEFAULT_CLIP) -> float:
    """Clip the prior score into the open interval ``clip`` to keep ``H`` finite."""
    low, high = clip
    return float(min(max(float(prior), low), high))


def clever_covariate(
    outcome: float,
    prior: float,
    clip: Sequence[float] = DEFAULT_CLIP,
) -> float:
    """Return the clever covariate ``H = (Y - e) / (e * (1 - e))``.

    For a binary outcome this collapses to the two immediate values used in the
    paper: a success gives ``H = 1 / e > 0`` and a failure gives
    ``H = -1 / (1 - e) < 0``. When ``e < 1/2`` a success has the larger magnitude.
    """
    e = clip_prior(prior, clip)
    return (float(outcome) - e) / (e * (1.0 - e))


def adjustment_hint(h_value: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Map a clever-covariate value to ``aggressive`` / ``conservative`` / ``keep``."""
    if h_value > threshold:
        return "aggressive"
    if h_value < -threshold:
        return "conservative"
    return "keep"


def hint_prior_band(threshold: float) -> Dict[str, object]:
    """Translate a threshold into the prior band it acts on for a binary outcome.

    For ``Y in {0, 1}`` the covariate is two-valued: a success gives ``1 / e`` and a
    failure gives ``-1 / (1 - e)``. Hence a success fires ``aggressive`` iff
    ``e < 1 / threshold`` and a failure fires ``conservative`` iff
    ``e > 1 - 1 / threshold``. Because ``|H| >= 1`` for every binary beat, any
    ``threshold < 1`` never yields ``keep`` (the sign always fires); a
    ``threshold > 1`` opens a dead-zone around ``e = 0.5`` where outcomes that
    matched a middling prior are kept. This helper returns those cut points so the
    threshold can be reasoned about as a surprise band rather than a magic number.
    """
    if threshold <= 0:
        return {
            "success_fires_below_e": 1.0,
            "failure_fires_above_e": 0.0,
            "always_fires_for_binary": True,
        }
    success_max = 1.0 / threshold
    failure_min = 1.0 - 1.0 / threshold
    return {
        # a success (Y=1) is 'aggressive' when its prior e is below this
        "success_fires_below_e": min(success_max, 1.0),
        # a failure (Y=0) is 'conservative' when its prior e is above this
        "failure_fires_above_e": max(failure_min, 0.0),
        # threshold < 1 => |H| >= 1 always exceeds it, so 'keep' never happens
        "always_fires_for_binary": threshold < 1.0,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile without a numpy dependency."""
    xs = sorted(values)
    if not xs:
        raise ValueError("cannot take a quantile of an empty sequence")
    if len(xs) == 1:
        return xs[0]
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(xs) - 1)
    low = int(pos)
    high = min(low + 1, len(xs) - 1)
    frac = pos - low
    return xs[low] * (1.0 - frac) + xs[high] * frac


def calibrate_threshold(
    reference_beats: Sequence["Beat"],
    quantile: float = 0.9,
    clip: Sequence[float] = DEFAULT_CLIP,
    minimum: float = DEFAULT_THRESHOLD,
) -> float:
    """Pick a threshold from the null (no-drift) distribution of ``|H|``.

    This is the permute-then-refit / null-calibration idea used elsewhere in the
    repo, applied to the anomaly score: on a reference window with no drift, set
    the threshold at the ``quantile`` of ``|H|`` so that ``keep`` covers the null
    fluctuations and roughly a ``1 - quantile`` fraction of null beats fire a hint.
    The result is floored at ``minimum`` so a degenerate reference window cannot
    silence the skill entirely. This matters most for continuous / soft outcomes,
    where ``|H|`` is genuinely spread out; for sparse binary outcomes every
    ``|H| >= 1`` and the calibrated threshold naturally lands at or above 1.
    """
    if not reference_beats:
        return float(minimum)
    mags = [abs(clever_covariate(b.outcome, b.prior, clip)) for b in reference_beats]
    return float(max(minimum, _quantile(mags, quantile)))


@dataclass
class Beat:
    """One finished, committed search step whose outcome ``Y`` is already known.

    ``prior`` is the stable score the search assigned to the committed action and
    ``outcome`` is the episode label written back after the search returned.
    """

    prior: float
    outcome: float
    label: Optional[str] = None

    def h(self, clip: Sequence[float] = DEFAULT_CLIP) -> float:
        return clever_covariate(self.outcome, self.prior, clip)


@dataclass
class WindowStats:
    """Summary of a sliding window of finished beats.

    These are exactly the channels the companion paper reports as the ones that
    *move* when the sign is read: the activation (success) rate, the mean prior of
    the chosen actions, the conservative share, plus the anomaly magnitude and the
    absolute residual that are shown *not* to record the recovery.
    """

    n: int = 0
    activation_rate: float = 0.0
    mean_prior: float = 0.0
    conservative_share: float = 0.0
    aggressive_share: float = 0.0
    keep_share: float = 0.0
    mean_abs_h: float = 0.0
    mean_abs_residual: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "n": self.n,
            "activation_rate": self.activation_rate,
            "mean_prior": self.mean_prior,
            "conservative_share": self.conservative_share,
            "aggressive_share": self.aggressive_share,
            "keep_share": self.keep_share,
            "mean_abs_h": self.mean_abs_h,
            "mean_abs_residual": self.mean_abs_residual,
        }


def summarize_window(
    beats: Sequence[Beat],
    threshold: float = DEFAULT_THRESHOLD,
    clip: Sequence[float] = DEFAULT_CLIP,
) -> WindowStats:
    """Compute :class:`WindowStats` over a sequence of finished beats."""
    n = len(beats)
    if n == 0:
        return WindowStats()
    hs = [b.h(clip) for b in beats]
    hints = [adjustment_hint(h, threshold) for h in hs]
    return WindowStats(
        n=n,
        activation_rate=sum(b.outcome for b in beats) / n,
        mean_prior=sum(clip_prior(b.prior, clip) for b in beats) / n,
        conservative_share=sum(1 for hint in hints if hint == "conservative") / n,
        aggressive_share=sum(1 for hint in hints if hint == "aggressive") / n,
        keep_share=sum(1 for hint in hints if hint == "keep") / n,
        mean_abs_h=sum(abs(h) for h in hs) / n,
        mean_abs_residual=sum(abs(b.outcome - clip_prior(b.prior, clip)) for b in beats) / n,
    )


@dataclass
class CovariateSignal:
    """Structured result of one skill invocation."""

    prompt: str
    H: float
    hint: str
    prior: float
    outcome: float
    abs_anomaly: float
    window: Optional[WindowStats] = None

    def as_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "prompt": self.prompt,
            "H": self.H,
            "hint": self.hint,
            "prior": self.prior,
            "outcome": self.outcome,
            "abs_anomaly": self.abs_anomaly,
        }
        if self.window is not None:
            out["window"] = self.window.as_dict()
        return out


class RAPCleverCovariateSkill:
    """Skill ``rap_clever_covariate_guide``.

    Computes the clever covariate ``H`` from the observed prior/outcome, maps it to
    an adjustment direction, and renders a natural-language block that can be
    prepended to an RAP planning prompt so the LLM reads the anomaly explicitly.
    """

    name = "rap_clever_covariate_guide"
    version = "0.1.0"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        clip: Sequence[float] = DEFAULT_CLIP,
    ) -> None:
        low, high = clip
        if not 0.0 < low < high < 1.0:
            raise ValueError("clip must satisfy 0 < low < high < 1")
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)
        self.clip = (float(low), float(high))

    def signal_block(
        self,
        prior: float,
        outcome: float,
        window: Optional[WindowStats] = None,
    ) -> str:
        """Render only the ``[Clever Covariate Signal]`` block (no state/history)."""
        e = clip_prior(prior, self.clip)
        h_value = clever_covariate(outcome, prior, self.clip)
        hint = adjustment_hint(h_value, self.threshold)
        lines = [
            "[Clever Covariate Signal]",
            f"Prior on committed step (e): {e:.3f}",
            f"Observed outcome (Y): {outcome:g}",
            f"Clever covariate H = (Y - e) / (e(1 - e)): {h_value:.3f}",
            f"Anomaly magnitude |H|: {abs(h_value):.3f}",
            f"Adjustment hint: {hint}",
        ]
        if window is not None and window.n > 0:
            lines.append(
                "Recent window ({n} finished steps): activation rate "
                "{act:.3f}, mean prior {mp:.3f}, conservative share {cs:.3f}.".format(
                    n=window.n,
                    act=window.activation_rate,
                    mp=window.mean_prior,
                    cs=window.conservative_share,
                )
            )
        lines.append("")
        lines.append(_HINT_LEGEND)
        return "\n".join(lines)

    def build_prompt(
        self,
        state: str,
        prior: float,
        outcome: float,
        history: Optional[Sequence[str]] = None,
        window: Optional[WindowStats] = None,
        base_prompt: Optional[str] = None,
    ) -> str:
        """Assemble the full planning prompt with the signal block prepended."""
        history_text = "\n".join(history) if history else "None"
        sections = [
            self.signal_block(prior, outcome, window),
            "",
            "[Current State]",
            state,
            "",
            "[Recent History]",
            history_text,
            "",
        ]
        if base_prompt:
            sections.append(base_prompt)
        else:
            sections.append(
                "Based on the above, generate the next planning step or candidate "
                "actions. Let the adjustment hint modulate how widely you explore."
            )
        return "\n".join(sections)

    def invoke(
        self,
        state: str,
        prior: float,
        outcome: float,
        history: Optional[Sequence[str]] = None,
        window_beats: Optional[Sequence[Beat]] = None,
        base_prompt: Optional[str] = None,
    ) -> CovariateSignal:
        """Compute ``H`` and return the hint plus the assembled planning prompt."""
        e = clip_prior(prior, self.clip)
        h_value = clever_covariate(outcome, prior, self.clip)
        hint = adjustment_hint(h_value, self.threshold)
        window = (
            summarize_window(window_beats, self.threshold, self.clip)
            if window_beats
            else None
        )
        prompt = self.build_prompt(
            state=state,
            prior=prior,
            outcome=outcome,
            history=history,
            window=window,
            base_prompt=base_prompt,
        )
        return CovariateSignal(
            prompt=prompt,
            H=h_value,
            hint=hint,
            prior=e,
            outcome=float(outcome),
            abs_anomaly=abs(h_value),
            window=window,
        )


@dataclass
class RAPWithCovariate:
    """Minimal illustration of wiring the skill into an RAP expansion step.

    ``llm`` is any callable ``str -> str`` (or ``-> Sequence``) that proposes the
    next candidate actions from a prompt. ``parse_actions`` turns that raw output
    into whatever node representation the caller uses. Both default to identity-ish
    stubs so the wiring can be unit-tested without a real model.
    """

    llm: Callable[[str], object]
    skill: RAPCleverCovariateSkill = field(default_factory=RAPCleverCovariateSkill)
    parse_actions: Callable[[object], object] = field(
        default_factory=lambda: (lambda raw: raw)
    )

    def expand(
        self,
        state: str,
        prior: float,
        outcome: float,
        history: Optional[Sequence[str]] = None,
        window_beats: Optional[Sequence[Beat]] = None,
        base_prompt: Optional[str] = None,
    ) -> Dict[str, object]:
        """Build the hinted prompt, query the LLM, and parse candidate actions."""
        signal = self.skill.invoke(
            state=state,
            prior=prior,
            outcome=outcome,
            history=history,
            window_beats=window_beats,
            base_prompt=base_prompt,
        )
        raw = self.llm(signal.prompt)
        actions = self.parse_actions(raw)
        return {"signal": signal, "raw": raw, "actions": actions}
