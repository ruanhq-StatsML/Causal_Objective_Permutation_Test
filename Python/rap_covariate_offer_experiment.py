"""Reproducible offer-search experiment for the clever-covariate RAP skill.

This harness is deterministic (activations are modular functions of the 0-based
user index), so every number printed here is reconstructable by hand and does not
depend on a random seed. It exercises :class:`RAPCleverCovariateSkill` on three
one-step search streams:

1. ``offer``    -- the exact setup from the companion paper (validation target:
                   post ``Y`` moves 0.500 -> 0.167 with the sign unread, and
                   0.500 -> 0.625 when the next episode reads the sign).
2. ``triage``   -- a support-ticket triage router (new dataset).
3. ``retrieval``-- a retrieval-augmented QA reranker (new dataset).

Each stream has three actions with a deterministic activation rule and a stable
score ``e`` (the prior). A judge commits one action per user; before the drift it
commits ``judge_pre`` and after the drift the mix flips it to ``judge_post``. The
"sign not read" policy always follows the judge. The "sign read" policy follows
the judge through a warmup, then permanently switches to the highest stable-score
action once the mean loss of the last ``window`` finished beats exceeds the
pre-drift reference loss by ``margin`` -- exactly the rule described in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from rap_clever_covariate_guide import (
    Beat,
    RAPCleverCovariateSkill,
    adjustment_hint,
    clip_prior,
)


@dataclass
class Action:
    name: str
    activate: Callable[[int], int]
    stable: float


@dataclass
class OfferConfig:
    name: str
    n_users: int
    drift_index: int
    actions: Dict[str, Action]
    judge_pre: str
    judge_post: str
    warmup: int = 12
    window: int = 4
    margin: float = 0.15


def _best_stable(actions: Dict[str, Action]) -> str:
    return max(actions.values(), key=lambda a: a.stable).name


def _phase_metrics(beats: List[Beat], skill: RAPCleverCovariateSkill) -> Dict[str, float]:
    if not beats:
        return {}
    n = len(beats)
    hs = [b.h(skill.clip) for b in beats]
    hints = [adjustment_hint(h, skill.threshold) for h in hs]
    succ = [abs(h) for h, b in zip(hs, beats) if b.outcome == 1]
    fail = [abs(h) for h, b in zip(hs, beats) if b.outcome == 0]
    return {
        "Y": sum(b.outcome for b in beats) / n,
        "mean_prior": sum(clip_prior(b.prior, skill.clip) for b in beats) / n,
        "mean_abs_H": sum(abs(h) for h in hs) / n,
        "succ_abs_H": (sum(succ) / len(succ)) if succ else float("nan"),
        "fail_abs_H": (sum(fail) / len(fail)) if fail else float("nan"),
        "conservative_share": sum(1 for x in hints if x == "conservative") / n,
        "abs_residual": sum(abs(b.outcome - clip_prior(b.prior, skill.clip)) for b in beats) / n,
    }


def _run_policy(cfg: OfferConfig, read_sign: bool, skill: RAPCleverCovariateSkill) -> Tuple[List[Beat], int]:
    """Return (beats, switch_index). switch_index = -1 if never switched."""
    pre_ref_fail = 1.0 - _pre_activation(cfg, cfg.judge_pre)
    switch_to = _best_stable(cfg.actions)
    beats: List[Beat] = []
    switched = False
    switch_index = -1
    for i in range(cfg.n_users):
        if read_sign and not switched and i >= cfg.warmup:
            recent = beats[-cfg.window:]
            if len(recent) == cfg.window:
                recent_loss = sum(1 for b in recent if b.outcome == 0) / cfg.window
                if recent_loss - pre_ref_fail > cfg.margin:
                    switched = True
                    switch_index = i
        if read_sign and switched:
            committed = cfg.actions[switch_to]
        else:
            committed = cfg.actions[cfg.judge_pre if i < cfg.drift_index else cfg.judge_post]
        y = committed.activate(i)
        beats.append(Beat(prior=committed.stable, outcome=float(y), label=committed.name))
    return beats, switch_index


def _pre_activation(cfg: OfferConfig, action_name: str) -> float:
    act = cfg.actions[action_name].activate
    pre = range(0, cfg.drift_index)
    return sum(act(i) for i in pre) / len(pre)


def run(cfg: OfferConfig, skill: RAPCleverCovariateSkill) -> Dict[str, object]:
    d = cfg.drift_index
    judge_beats, _ = _run_policy(cfg, read_sign=False, skill=skill)
    read_beats, switch_index = _run_policy(cfg, read_sign=True, skill=skill)
    post_n = cfg.n_users - d
    judge_pre = _phase_metrics(judge_beats[:d], skill)
    judge_post = _phase_metrics(judge_beats[d:], skill)
    read_pre = _phase_metrics(read_beats[:d], skill)
    read_post = _phase_metrics(read_beats[d:], skill)
    extra_vs_judge = round((read_post["Y"] - judge_post["Y"]) * post_n)
    extra_vs_pre = round((read_post["Y"] - judge_pre["Y"]) * post_n)
    return {
        "name": cfg.name,
        "post_n": post_n,
        "switch_index": switch_index,
        "read_switch_to": _best_stable(cfg.actions),
        "judge": {"pre": judge_pre, "post": judge_post},
        "read": {"pre": read_pre, "post": read_post},
        "extra_vs_judge": extra_vs_judge,
        "extra_vs_pre": extra_vs_pre,
    }


def build_configs() -> List[OfferConfig]:
    offer = OfferConfig(
        name="offer",
        n_users=40,
        drift_index=16,
        actions={
            "habit": Action("habit", lambda i: 1 if i % 2 == 0 else 0, 0.50),
            "growth": Action("growth", lambda i: 0 if i % 3 == 0 else 1, 0.67),
            "click": Action("click", lambda i: 1 if i % 5 == 0 else 0, 0.20),
        },
        judge_pre="habit",
        judge_post="click",
    )
    # Support-ticket triage router: a milder shortfall (0.250), and the best
    # available branch tops out at 0.55, so reading the sign recovers the
    # shortfall back to the baseline but cannot exceed it.
    triage = OfferConfig(
        name="triage",
        n_users=40,
        drift_index=16,
        actions={
            "template": Action("template", lambda i: 1 if i % 2 == 0 else 0, 0.50),
            "escalate": Action("escalate", lambda i: 1 if i % 2 == 1 else 0, 0.55),
            "autoclose": Action("autoclose", lambda i: 1 if i % 4 == 0 else 0, 0.25),
        },
        judge_pre="template",
        judge_post="autoclose",
    )
    # Retrieval-augmented QA reranker: a deep shortfall (0.125) and a strong
    # recovery, because the best branch (deep, 0.72) activates two-thirds of
    # users, so reading the sign lifts Y clearly above the baseline.
    retrieval = OfferConfig(
        name="retrieval",
        n_users=40,
        drift_index=16,
        actions={
            "shallow": Action("shallow", lambda i: 1 if i % 2 == 0 else 0, 0.50),
            "deep": Action("deep", lambda i: 0 if i % 4 == 0 else 1, 0.75),
            "cached": Action("cached", lambda i: 1 if i % 7 == 0 else 0, 0.14),
        },
        judge_pre="shallow",
        judge_post="cached",
    )
    return [offer, triage, retrieval]


def _fmt(x: float) -> str:
    return "nan" if x != x else f"{x:.3f}"


def main() -> None:
    skill = RAPCleverCovariateSkill(threshold=0.5)
    for cfg in build_configs():
        r = run(cfg, skill)
        jp, jpo = r["judge"]["pre"], r["judge"]["post"]
        rp, rpo = r["read"]["pre"], r["read"]["post"]
        print("=" * 68)
        print(f"dataset = {r['name']}  (post users = {r['post_n']})")
        print(f"  switch to '{r['read_switch_to']}' at user index {r['switch_index']}")
        print("  [sign NOT read]  pre Y = {a}  post Y = {b}  extra vs judge = 0".format(
            a=_fmt(jp["Y"]), b=_fmt(jpo["Y"])))
        print("  [sign read]      pre Y = {a}  post Y = {b}  extra vs judge = +{c}  extra vs pre = +{d}".format(
            a=_fmt(rp["Y"]), b=_fmt(rpo["Y"]), c=r["extra_vs_judge"], d=r["extra_vs_pre"]))
        print("  |H|:            pre {a} -> post {b} (judge) ; read post {c}".format(
            a=_fmt(jp["mean_abs_H"]), b=_fmt(jpo["mean_abs_H"]), c=_fmt(rpo["mean_abs_H"])))
        print("  post |H| split (judge): success {a} / failure {b}".format(
            a=_fmt(jpo["succ_abs_H"]), b=_fmt(jpo["fail_abs_H"])))
        print("  prior on committed: judge post {a} ; read post {b}".format(
            a=_fmt(jpo["mean_prior"]), b=_fmt(rpo["mean_prior"])))
        print("  conservative share: pre {a} ; judge post {b} ; read post {c}".format(
            a=_fmt(jp["conservative_share"]), b=_fmt(jpo["conservative_share"]),
            c=_fmt(rpo["conservative_share"])))
        print("  abs residual: pre {a} ; judge post {b} ; read post {c}".format(
            a=_fmt(jp["abs_residual"]), b=_fmt(jpo["abs_residual"]),
            c=_fmt(rpo["abs_residual"])))


if __name__ == "__main__":
    main()
