"""Reproducible simulation for rap_clever_covariate_guide.

We construct a streaming planning problem with a *known* regime shift and
measure how three exploration-control strategies behave:

We compare the *expensive* trigger -- the regime-shift / re-plan alarm, which
is what causes the planner to reset exploration and re-expand (extra token
cost). Three alarm rules:

* ``fixed``      -- always raise the alarm (upper bound on token cost).
* ``clever_H``   -- raise the alarm whenever the clever covariate |H| >= tau
                    (the covariate used on its own, no statistical gate).
* ``fused``      -- the ``rap_clever_covariate_guide`` skill's ``replan`` flag:
                    raise the alarm when the online rolling empirical p-value of
                    the drift statistic is significant (p < alpha).

The clever covariate still sets the *direction* of the hint in the skill; here
we isolate whether the online rolling p-value is a better *alarm* than |H|.

The stream: at each step the planner receives a continuous reward
``r_t ~ N(mu_regime, sigma)``. Before the shift ``mu = +1`` (a good branch),
after the shift ``mu = -1`` (the branch degraded). The planner's prior success
probability ``e_t`` is a lagging EWMA of past outcomes, so right after the shift
it stays optimistic -- exactly the miscalibration the monitor should catch. The
drift statistic fed to the online p-value is the per-step loss ``s_t = -r_t``
(higher = worse), matching this repo's optimal-MSE monitor.

Metrics (averaged over many seeds):

* ``escalation_rate``     -- fraction of all steps flagged (token-cost proxy).
* ``false_alarm_rate``    -- fraction of *pre-shift* steps flagged (waste).
* ``detection_rate``      -- fraction of seeds that flag at all post-shift.
* ``detection_latency``   -- mean steps from the shift to the first post-shift
                             flag (over seeds that detect).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from rap_clever_covariate_guide import OnlineRollingStatistic, clever_covariate


@dataclass
class SimConfig:
    T: int = 400
    shift_at: int = 200
    mu_pre: float = 1.0
    mu_post: float = -1.0
    sigma: float = 1.0
    ewma_gamma: float = 0.9
    tau: float = 0.5
    alpha: float = 0.05
    burnin: int = 20
    window: int = 50
    n_seeds: int = 400
    seed0: int = 2026


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _run_single(cfg: SimConfig, seed: int) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    monitor = OnlineRollingStatistic(burnin=cfg.burnin, window=cfg.window)

    flags = {"fixed": [], "clever_H": [], "fused": []}
    for t in range(cfg.T):
        mu = cfg.mu_pre if t < cfg.shift_at else cfg.mu_post
        r = rng.normal(mu, cfg.sigma)
        y = 1 if r > 0 else 0

        # Lagging EWMA prior over success; initialised optimistic.
        if t == 0:
            m = 1.0
        m = cfg.ewma_gamma * m + (1.0 - cfg.ewma_gamma) * y if t > 0 else m
        e = float(np.clip(_sigmoid(4.0 * (m - 0.5)), 1e-3, 1.0 - 1e-3))

        H = clever_covariate(e, y)
        loss = -r  # higher loss => more anomalous (upper tail)
        mon = monitor.update(loss)
        alarm = mon["pval"] < cfg.alpha

        flags["fixed"].append(True)
        flags["clever_H"].append(bool(abs(H) >= cfg.tau))
        flags["fused"].append(bool(alarm))

    out: Dict[str, Dict[str, float]] = {}
    for name, seq in flags.items():
        arr = np.asarray(seq, dtype=bool)
        pre = arr[: cfg.shift_at]
        post = arr[cfg.shift_at:]
        detected_idx = np.flatnonzero(post)
        out[name] = {
            "escalation_rate": float(arr.mean()),
            "false_alarm_rate": float(pre.mean()),
            "detected": float(detected_idx.size > 0),
            "latency": float(detected_idx[0]) if detected_idx.size else np.nan,
        }
    return out


def run(cfg: SimConfig | None = None) -> Dict[str, Dict[str, float]]:
    cfg = cfg or SimConfig()
    per_method: Dict[str, Dict[str, List[float]]] = {
        m: {"escalation_rate": [], "false_alarm_rate": [], "detected": [], "latency": []}
        for m in ("fixed", "clever_H", "fused")
    }
    for s in range(cfg.n_seeds):
        res = _run_single(cfg, cfg.seed0 + s)
        for m, metrics in res.items():
            for k, v in metrics.items():
                per_method[m][k].append(v)

    summary: Dict[str, Dict[str, float]] = {}
    for m, metrics in per_method.items():
        lat = np.asarray(metrics["latency"], dtype=float)
        summary[m] = {
            "escalation_rate": float(np.mean(metrics["escalation_rate"])),
            "false_alarm_rate": float(np.mean(metrics["false_alarm_rate"])),
            "detection_rate": float(np.mean(metrics["detected"])),
            "detection_latency": float(np.nanmean(lat)) if np.any(~np.isnan(lat)) else np.nan,
        }
    return summary


def _format_table(summary: Dict[str, Dict[str, float]]) -> str:
    header = f"{'method':<10} {'escalation':>11} {'false_alarm':>12} {'detect_rate':>12} {'latency':>9}"
    lines = [header, "-" * len(header)]
    labels = {"fixed": "fixed", "clever_H": "clever_H", "fused": "fused"}
    for m in ("fixed", "clever_H", "fused"):
        s = summary[m]
        lat = "-" if np.isnan(s["detection_latency"]) else f"{s['detection_latency']:.2f}"
        lines.append(
            f"{labels[m]:<10} {s['escalation_rate']:>11.3f} {s['false_alarm_rate']:>12.3f} "
            f"{s['detection_rate']:>12.3f} {lat:>9}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = SimConfig()
    summary = run(cfg)
    print(f"Simulation: T={cfg.T}, shift_at={cfg.shift_at}, seeds={cfg.n_seeds}, "
          f"tau={cfg.tau}, alpha={cfg.alpha}, window={cfg.window}\n")
    print(_format_table(summary))
