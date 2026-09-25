"""Tests for rap_clever_covariate_guide."""
from __future__ import annotations

import numpy as np

from rap_clever_covariate_guide import (
    OnlineRollingStatistic,
    RAPCleverCovariateSkill,
    RAPWithCovariate,
    clever_covariate,
)


def empirical_pval_local(seq, burnin=5):
    """Reference upper-tail rolling p-value, copied from mab_benchmark_core.

    Inlined so the test only depends on numpy (the heavy modules in this repo
    pull in scipy / xgboost / sklearn, which are not installed in CI).
    """
    pvals = [0.5] * len(seq)
    for i in range(burnin, len(seq)):
        hist = seq[:i]
        cnt = sum(1 for v in hist if v >= seq[i])
        pvals[i] = (cnt + 1) / (len(hist) + 1)
    return pvals


def test_clever_covariate_formula():
    # H = (Y - e) / (e (1 - e)) with e = 0.3, Y = 1.
    assert clever_covariate(0.3, 1) == (1 - 0.3) / (0.3 * 0.7)
    # Y = 0 gives negative covariate.
    assert clever_covariate(0.3, 0) < 0


def test_clever_covariate_clips_extreme_prior():
    # Prior of 0 / 1 must not blow up the denominator.
    assert np.isfinite(clever_covariate(0.0, 1))
    assert np.isfinite(clever_covariate(1.0, 0))


def test_online_rolling_matches_offline_empirical_pval():
    rng = np.random.default_rng(0)
    seq = rng.normal(size=40).tolist()
    burnin = 5
    offline = empirical_pval_local(seq, burnin=burnin)

    monitor = OnlineRollingStatistic(burnin=burnin)
    online = [monitor.update(s)["pval"] for s in seq]

    assert np.allclose(online, offline)


def test_anomaly_score_is_one_minus_pval():
    monitor = OnlineRollingStatistic(burnin=0)
    out = monitor.update(1.0)
    assert abs(out["anomaly_score"] - (1.0 - out["pval"])) < 1e-12


def test_hint_aggressive_on_positive_surprise():
    skill = RAPCleverCovariateSkill(threshold=0.5, burnin=0)
    res = skill.invoke(state="s", prior=0.2, outcome=1)
    assert res["hint"] == "aggressive"
    assert res["H"] > 0


def test_hint_conservative_on_negative_surprise():
    skill = RAPCleverCovariateSkill(threshold=0.5, burnin=0)
    res = skill.invoke(state="s", prior=0.8, outcome=0)
    assert res["hint"] == "conservative"
    assert res["H"] < 0


def test_hint_keep_when_deviation_small_and_not_significant():
    # Prior matches outcome expectation closely and monitor is in burn-in.
    skill = RAPCleverCovariateSkill(threshold=5.0, alpha=0.01, burnin=10)
    res = skill.invoke(state="s", prior=0.5, outcome=1)
    assert res["hint"] == "keep"
    assert res["replan"] is False


def test_regime_shift_alarm_triggers_replan():
    # Feed a burst of large statistics so the newest becomes extreme -> p small.
    skill = RAPCleverCovariateSkill(threshold=100.0, alpha=0.5, burnin=1)
    seed_stats = [0.0, 0.0, 0.0, 0.0]
    for s in seed_stats:
        skill.monitor.update(s)
    res = skill.invoke(state="s", prior=0.5, outcome=1, monitor_stat=10.0)
    assert res["replan"] is True
    assert res["anomaly_score"] >= 1.0 - skill.alpha


def test_prompt_contains_key_sections():
    skill = RAPCleverCovariateSkill()
    res = skill.invoke(state="my-state", prior=0.3, outcome=1, history=["a", "b"])
    prompt = res["prompt"]
    for token in [
        "[Clever Covariate Signal]",
        "Clever covariate (H)",
        "[Online Rolling Monitor]",
        "Rolling empirical p-value",
        "Adjustment hint:",
        "[Current State]",
        "my-state",
        "[Recent History]",
    ]:
        assert token in prompt


def test_rap_with_covariate_expand():
    class DummyNode:
        state = "node-state"
        history = ["h1"]

    class DummyLLM:
        def generate(self, prompt):
            assert "Adjustment hint:" in prompt
            return "action A\naction B\n"

    wrapper = RAPWithCovariate(DummyLLM(), RAPCleverCovariateSkill(burnin=0))
    out = wrapper.expand(DummyNode(), prior=0.3, outcome=1)
    assert out["children"] == ["action A", "action B"]
    assert out["hint"] in {"aggressive", "conservative", "keep"}


def test_simulation_fused_alarm_beats_clever_h():
    from rap_covariate_simulation import SimConfig, run

    cfg = SimConfig(n_seeds=40)
    summary = run(cfg)
    # The rolling p-value alarm keeps pre-shift false alarms near alpha, while
    # |H| alone fires on essentially every step.
    assert summary["fused"]["false_alarm_rate"] < 0.2
    assert summary["clever_H"]["false_alarm_rate"] > 0.8
    # It still detects the known regime shift quickly.
    assert summary["fused"]["detection_rate"] > 0.9
    assert summary["fused"]["detection_latency"] < 10


def test_window_makes_monitor_rolling():
    monitor = OnlineRollingStatistic(burnin=0, window=3)
    for s in [100.0, 100.0, 100.0]:
        monitor.update(s)
    # A value equal to the windowed reference stays calibrated within the window.
    out = monitor.update(1.0)
    assert 0.0 <= out["pval"] <= 1.0
    assert len(monitor.history) <= 3
