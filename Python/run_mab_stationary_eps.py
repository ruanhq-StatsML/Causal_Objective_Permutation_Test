"""Stationary-bandit epsilon sweep (no distribution shift) for contrast with shift DGPs."""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_MAB import CONFIG, PROTOTYPE_EPSILON_SWEEP_POLICIES
from run_mab_benchmark import _cfg_to_id, _expand_policy_grid, _policy_uses_context
from run_mab_scale_eps import build_variance_explore_analysis
from mab_benchmark_core import (
    context_dim,
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)

DEFAULT_EPS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]


def run_stationary_eps(
    n_repeats: int = 3,
    feature_dim: int = 30,
    total_samples: int = 20000,
    ref_samples: int = 5000,
    batch_size: int = 25,
    epsilon_grid: List[float] | None = None,
    output_prefix: str = "mab_stationary_d30_eps",
) -> pd.DataFrame:
    if epsilon_grid is None:
        epsilon_grid = DEFAULT_EPS
    policies_cfg = copy.deepcopy(PROTOTYPE_EPSILON_SWEEP_POLICIES)
    policies_cfg["Epsilon_Greedy"]["epsilon"] = [float(x) for x in epsilon_grid]
    policy_jobs = _expand_policy_grid(policies_cfg)

    run_config = copy.deepcopy(CONFIG)
    n_arms = run_config["model_pool"]["n_arms"]
    rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    for rep in range(n_repeats):
        data_cfg = dict(run_config["data"])
        data_cfg.update(
            {
                "name": "stationary_d30",
                "dgp": "stationary",
                "feature_dim": feature_dim,
                "total_samples": total_samples,
                "ref_samples": ref_samples,
                "batch_size": batch_size,
                "shift_magnitude": 0.0,
                "noise_scale": float(run_config["data"].get("noise_scale", 2.5)),
                "random_seed": int(run_config["data"]["random_seed"]) + rep * 31,
            }
        )
        t0 = time.time()
        print(
            f"[stationary] rep={rep + 1}/{n_repeats} d={feature_dim} "
            f"n={total_samples} ref={ref_samples} batch={batch_size} dgp=stationary",
            flush=True,
        )
        data = generate_dgp_from_config(data_cfg, n_arms=n_arms)
        cache = precompute_batch_cache(data, run_config)
        print(
            f"[stationary] ready in {time.time() - t0:.1f}s "
            f"n_batches={data['n_batches']} shift_batch={data['shift_batch_index']} "
            f"dgp={data.get('dgp')}",
            flush=True,
        )

        for policy_name, policy_cfg in policy_jobs:
            cfg_id = _cfg_to_id(policy_cfg)
            policy = make_policy(
                policy_name,
                n_arms,
                context_dim(run_config["context"]["n_base_features"]),
                dict(policy_cfg),
            )
            result = run_experiment(
                policy,
                data,
                run_config,
                use_context=_policy_uses_context(policy_name),
                batch_cache=cache,
            )
            rows.append(
                {
                    "dgp_name": "stationary",
                    "dgp": "stationary",
                    "cooling_preset": "stationary",
                    "rep": rep,
                    "feature_dim": feature_dim,
                    "total_samples": total_samples,
                    "ref_samples": ref_samples,
                    "batch_size": batch_size,
                    "n_batches": result["n_batches"],
                    "shift_batch_index": data["shift_batch_index"],
                    "policy": policy_name,
                    "config_id": cfg_id,
                    "final_regret": result["final_regret"],
                    "mean_reward": result["mean_reward"],
                    "hp_epsilon": policy_cfg.get("epsilon"),
                }
            )
            for batch_idx, cum_reg in enumerate(result["cum_regret"]):
                curve_rows.append(
                    {
                        "cooling_preset": "stationary",
                        "rep": rep,
                        "policy": policy_name,
                        "config_id": cfg_id,
                        "batch_index": batch_idx,
                        "cum_regret": float(cum_reg),
                    }
                )

        pd.DataFrame(rows).to_csv(f"{output_prefix}_metrics.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    curves_df = pd.DataFrame(curve_rows)
    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    curves_df.to_csv(f"{output_prefix}_cum_regret.csv", index=False)
    summary = (
        metrics_df.groupby(["policy", "config_id"])["final_regret"]
        .agg(["mean", "std", "min", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "final_regret_mean",
                "std": "final_regret_std",
                "min": "final_regret_best",
                "count": "n_repeats",
            }
        )
        .sort_values("final_regret_mean")
    )
    summary.to_csv(f"{output_prefix}_summary.csv", index=False)
    print(f"[stationary] saved {output_prefix}_summary.csv", flush=True)
    build_variance_explore_analysis(metrics_df, curves_df, output_prefix)

    # Overlay comparison plot vs shift run if available
    shift_path = "mab_scale_d30_higheps_variance_vs_explore.csv"
    stat_path = f"{output_prefix}_variance_vs_explore.csv"
    if os.path.exists(shift_path) and os.path.exists(stat_path):
        shift = pd.read_csv(shift_path)
        stat = pd.read_csv(stat_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].errorbar(
            shift["epsilon"] * 100, shift["final_regret_mean"], yerr=shift["final_regret_std"],
            fmt="o-", capsize=3, label="with shift (messy)", color="#E08214",
        )
        axes[0].errorbar(
            stat["epsilon"] * 100, stat["final_regret_mean"], yerr=stat["final_regret_std"],
            fmt="s-", capsize=3, label="stationary", color="#1F77B4",
        )
        axes[0].set_xlabel("Exploration traffic ε (%)")
        axes[0].set_ylabel("Final regret (mean ± std)")
        axes[0].set_title("Stationary vs shift: regret")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(shift["epsilon"] * 100, shift["final_regret_cv"], "o-", label="with shift", color="#E08214")
        axes[1].plot(stat["epsilon"] * 100, stat["final_regret_cv"], "s-", label="stationary", color="#1F77B4")
        axes[1].set_xlabel("Exploration traffic ε (%)")
        axes[1].set_ylabel("CV (std/mean)")
        axes[1].set_title("Stationary vs shift: robustness")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{output_prefix}_vs_shift.png", dpi=180)
        plt.close(fig)
        print(f"[stationary] saved {output_prefix}_vs_shift.png", flush=True)

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stationary epsilon-greedy sweep")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=30)
    parser.add_argument("--total-samples", type=int, default=20000)
    parser.add_argument("--ref-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--epsilon-grid", nargs="+", type=float, default=None)
    parser.add_argument("--output-prefix", default="mab_stationary_d30_eps")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_stationary_eps(
        n_repeats=args.n_repeats,
        feature_dim=args.feature_dim,
        total_samples=args.total_samples,
        ref_samples=args.ref_samples,
        batch_size=args.batch_size,
        epsilon_grid=args.epsilon_grid,
        output_prefix=args.output_prefix,
    )
