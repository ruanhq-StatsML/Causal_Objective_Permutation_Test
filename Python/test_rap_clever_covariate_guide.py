"""Tests for the rap_clever_covariate_guide skill and the offer experiment."""
import math

import pytest

from rap_clever_covariate_guide import (
    Beat,
    RAPCleverCovariateSkill,
    RAPWithCovariate,
    adjustment_hint,
    clever_covariate,
    clip_prior,
    summarize_window,
)


def test_clip_prior_bounds():
    assert clip_prior(0.0) == pytest.approx(0.001)
    assert clip_prior(1.0) == pytest.approx(0.999)
    assert clip_prior(0.4) == pytest.approx(0.4)


def test_binary_covariate_closed_form():
    # success -> 1/e, failure -> -1/(1-e)
    assert clever_covariate(1, 0.2) == pytest.approx(1.0 / 0.2)
    assert clever_covariate(0, 0.2) == pytest.approx(-1.0 / 0.8)
    assert clever_covariate(1, 0.5) == pytest.approx(2.0)
    assert clever_covariate(0, 0.5) == pytest.approx(-2.0)


def test_success_dominates_when_prior_below_half():
    e = 0.2
    assert abs(clever_covariate(1, e)) > abs(clever_covariate(0, e))


def test_general_formula_matches_definition():
    y, e = 0.7, 0.3
    expected = (y - e) / (e * (1 - e))
    assert clever_covariate(y, e) == pytest.approx(expected)


def test_hints():
    assert adjustment_hint(5.0) == "aggressive"
    assert adjustment_hint(-1.25) == "conservative"
    assert adjustment_hint(0.0) == "keep"
    assert adjustment_hint(0.5) == "keep"  # boundary is not strictly greater
    assert adjustment_hint(-0.5) == "keep"


def test_skill_invoke_offer_success():
    skill = RAPCleverCovariateSkill(threshold=0.5)
    res = skill.invoke(state="s", prior=0.3, outcome=1, history=["a", "b"])
    assert res.hint == "aggressive"
    assert res.H == pytest.approx(1.0 / 0.3)
    assert res.abs_anomaly == pytest.approx(1.0 / 0.3)
    assert "[Clever Covariate Signal]" in res.prompt
    assert "[Current State]" in res.prompt
    assert "aggressive" in res.prompt


def test_skill_invoke_failure_is_conservative():
    skill = RAPCleverCovariateSkill()
    res = skill.invoke(state="s", prior=0.6, outcome=0)
    assert res.hint == "conservative"
    assert res.H < 0


def test_window_channels_mirror_outcome():
    # 1 success + 3 failures at e=0.2: conservative share = failure rate = 0.75
    beats = [
        Beat(prior=0.2, outcome=1),
        Beat(prior=0.2, outcome=0),
        Beat(prior=0.2, outcome=0),
        Beat(prior=0.2, outcome=0),
    ]
    w = summarize_window(beats)
    assert w.n == 4
    assert w.activation_rate == pytest.approx(0.25)
    assert w.conservative_share == pytest.approx(0.75)
    assert w.conservative_share == pytest.approx(1 - w.activation_rate)
    assert w.mean_abs_h == pytest.approx((5.0 + 3 * 1.25) / 4)


def test_invoke_includes_window_summary():
    skill = RAPCleverCovariateSkill()
    beats = [Beat(prior=0.5, outcome=1), Beat(prior=0.5, outcome=0)]
    res = skill.invoke(state="s", prior=0.2, outcome=0, window_beats=beats)
    assert res.window is not None
    assert res.window.n == 2
    assert "Recent window" in res.prompt


def test_bad_clip_raises():
    with pytest.raises(ValueError):
        RAPCleverCovariateSkill(clip=(0.5, 0.4))
    with pytest.raises(ValueError):
        RAPCleverCovariateSkill(threshold=-1.0)


def test_rap_with_covariate_wiring():
    captured = {}

    def fake_llm(prompt):
        captured["prompt"] = prompt
        return "action_a\naction_b"

    engine = RAPWithCovariate(llm=fake_llm)
    out = engine.expand(state="node-state", prior=0.3, outcome=1, history=["h1"])
    assert "action_a" in out["actions"]
    assert out["signal"].hint == "aggressive"
    assert "[Clever Covariate Signal]" in captured["prompt"]


def test_as_dict_roundtrip():
    skill = RAPCleverCovariateSkill()
    res = skill.invoke(state="s", prior=0.2, outcome=1)
    d = res.as_dict()
    assert d["hint"] == "aggressive"
    assert d["H"] == pytest.approx(5.0)


def test_experiment_reproduces_offer_numbers():
    from rap_covariate_offer_experiment import build_configs, run

    skill = RAPCleverCovariateSkill(threshold=0.5)
    cfgs = {c.name: c for c in build_configs()}
    offer = run(cfgs["offer"], skill)
    assert offer["judge"]["post"]["Y"] == pytest.approx(0.167, abs=1e-3)
    assert offer["read"]["post"]["Y"] == pytest.approx(0.625, abs=1e-3)
    assert offer["extra_vs_judge"] == 11
    assert offer["extra_vs_pre"] == 3
    assert offer["switch_index"] == 17
    assert offer["judge"]["post"]["succ_abs_H"] == pytest.approx(5.0, abs=1e-3)
    assert offer["judge"]["post"]["fail_abs_H"] == pytest.approx(1.25, abs=1e-3)
    assert offer["judge"]["post"]["mean_abs_H"] == pytest.approx(1.875, abs=1e-3)


def test_experiment_triage_is_bounded_and_retrieval_is_strong():
    from rap_covariate_offer_experiment import build_configs, run

    skill = RAPCleverCovariateSkill(threshold=0.5)
    cfgs = {c.name: c for c in build_configs()}
    triage = run(cfgs["triage"], skill)
    retrieval = run(cfgs["retrieval"], skill)
    # triage recovers the shortfall but not above the baseline
    assert triage["read"]["post"]["Y"] == pytest.approx(0.5, abs=1e-3)
    assert triage["extra_vs_pre"] == 0
    # retrieval clears the baseline by a wide margin
    assert retrieval["read"]["post"]["Y"] == pytest.approx(0.75, abs=1e-3)
    assert retrieval["extra_vs_pre"] == 6


def test_no_nan_in_prior_channels():
    skill = RAPCleverCovariateSkill()
    res = skill.invoke(state="s", prior=0.0, outcome=1)
    assert not math.isnan(res.H)
    assert res.prior == pytest.approx(0.001)
