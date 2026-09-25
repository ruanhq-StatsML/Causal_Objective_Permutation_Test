"""Tests for the rap_clever_covariate_guide skill and the offer experiment."""
import math

import pytest

from rap_clever_covariate_guide import (
    Beat,
    RAPCleverCovariateSkill,
    RAPWithCovariate,
    adjustment_hint,
    calibrate_threshold,
    clever_covariate,
    clip_prior,
    hint_prior_band,
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


def test_hint_prior_band_binary_always_fires_below_one():
    # threshold < 1 => |H| >= 1 always exceeds it, so 'keep' never happens
    band = hint_prior_band(0.5)
    assert band["always_fires_for_binary"] is True
    assert band["success_fires_below_e"] == pytest.approx(1.0)
    assert band["failure_fires_above_e"] == pytest.approx(0.0)


def test_hint_prior_band_dead_zone_for_threshold_above_one():
    band = hint_prior_band(2.0)
    # success fires only when e < 0.5, failure only when e > 0.5
    assert band["success_fires_below_e"] == pytest.approx(0.5)
    assert band["failure_fires_above_e"] == pytest.approx(0.5)
    assert band["always_fires_for_binary"] is False
    # a middling-prior success/failure is now 'keep'
    assert adjustment_hint(clever_covariate(1, 0.6), threshold=2.0) == "keep"
    assert adjustment_hint(clever_covariate(0, 0.4), threshold=2.0) == "keep"


def test_calibrate_threshold_from_reference_window():
    # a null window of middling priors: |H| ~ 2, so a high quantile lands near 2
    ref = [Beat(prior=0.5, outcome=1), Beat(prior=0.5, outcome=0)] * 8
    tau = calibrate_threshold(ref, quantile=0.9)
    assert tau == pytest.approx(2.0, abs=1e-6)
    # empty reference falls back to the minimum, never silences the skill
    assert calibrate_threshold([], minimum=0.5) == pytest.approx(0.5)


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
    offer = run(cfgs["offer"], skill, online=False)
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
    triage = run(cfgs["triage"], skill, online=False)
    retrieval = run(cfgs["retrieval"], skill, online=False)
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


def test_rolling_stats():
    from online_drift_detectors import rolling_stats

    r = rolling_stats([1.0, 1.0, 0.0, 0.0], window=2)
    assert r["mean"] == pytest.approx([1.0, 1.0, 0.5, 0.0])
    assert r["std"][-1] == pytest.approx(0.0)
    assert r["std"][2] == pytest.approx(0.5)


def test_online_rf_perm_pvalue_separates_shift():
    from online_drift_detectors import beat_features, online_rf_perm_pvalue

    # reference beats at prior 0.5, recent beats at prior 0.2 -> separable
    ref = [Beat(prior=0.5, outcome=1), Beat(prior=0.5, outcome=0)] * 4
    recent = [Beat(prior=0.2, outcome=0)] * 8
    X = beat_features(ref + recent)
    w = [0] * len(ref) + [1] * len(recent)
    shifted = online_rf_perm_pvalue(X, w, n_perm=40, seed=1)

    # no shift: reference vs a second reference-like batch -> not significant
    ref2 = [Beat(prior=0.5, outcome=1), Beat(prior=0.5, outcome=0)] * 4
    X0 = beat_features(ref + ref2)
    null = online_rf_perm_pvalue(X0, w, n_perm=40, seed=1)

    assert shifted["p_value"] < 0.1
    assert null["p_value"] > shifted["p_value"]


def test_pvalue_stream_detects_after_drift():
    from online_drift_detectors import pvalue_stream

    # 8 pre-drift beats (e=0.5) then 8 post-drift beats (e=0.2, all failures)
    beats = [Beat(prior=0.5, outcome=float(i % 2 == 0)) for i in range(8)]
    beats += [Beat(prior=0.2, outcome=0.0) for _ in range(8)]
    out = pvalue_stream(beats, ref_n=6, recent_n=6, n_perm=40, alpha=0.05)
    assert out["detect_index"] >= 8  # detection only once recent window is post-drift
    assert out["pvals"][len(beats)] <= out["pvals"][out["detect_index"]] + 1e-9
