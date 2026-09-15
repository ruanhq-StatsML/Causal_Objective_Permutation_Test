"""Batch-size sensitivity: ε-greedy (2/5/10%) + other methods on messy & stationary."""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_MAB import CONFIG
from run_mab_benchmark import _cfg_to_id, _expand_policy_grid, _policy_uses_context
from mab_benchmark_core import (
    context_dim,
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)

DEFAULT_BATCHES = [5, 10, 20, 30, 50, 100, 200, 500]

# Compact method grid for ~1–2h wall-clock at d=30.
POLICIES = {
    "Epsilon_Greedy": {"epsilon": [0.02, 0.05, 0.10]},
    "AdaptiveEpsilonGreedy": {
        "base_epsilon": [0.05],
        "max_epsilon": [0.5],
        "gamma": [0.9],
        "anomaly_sensitivity": [0.3],
    },
    "UCB": {},
    "Random": {},
    "LinUCB_Vanilla": {
        "alpha": [0.1],
        "lambda_reg": [50],
        "use_momentum": [False],
    },
    "Thompson_Sampling": {"prior_lambda": [0.1]},
}


def _dgp_cfg(setting: str) -> Dict[str, Any]:
    if setting == "stationary":
        return {"name": "stationary_batch_sens", "dgp": "stationary", "shift_magnitude": 0.0}
    if setting in ("messy", "shift"):
        # nonlinear_messy (same family as messy_d500_n100k / scale_eps)
        return {
            "name": "messy_batch_sens",
            "dgp": "nonlinear_messy",
            "shift_magnitude": 0.5,
        }
    if setting in ("ultra",):
        return {
            "name": "ultra_batch_sens",
            "dgp": "nonlinear_messy_ultra",
            "shift_magnitude": 0.5,
            "shift_interval": 2000,
            "mini_shift_interval": 400,
            "covariate_drift_strength": 0.45,
        }
    raise ValueError(setting)


def build_batch_sensitivity_table(metrics_df: pd.DataFrame, output_prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (batch, policy, cfg_id), msub in metrics_df.groupby(
        ["batch_size", "policy", "config_id"]
    ):
        mean = float(msub["final_regret"].mean())
        std = float(msub["final_regret"].std(ddof=1)) if len(msub) > 1 else 0.0
        rows.append(
            {
                "batch_size": int(batch),
                "policy": policy,
                "config_id": cfg_id,
                "final_regret_mean": mean,
                "final_regret_std": std,
                "n_repeats": int(len(msub)),
                "n_batches": int(msub["n_batches"].iloc[0]),
                "setting": msub["setting"].iloc[0],
                "dgp": msub["dgp"].iloc[0],
            }
        )
    out = pd.DataFrame(rows).sort_values(["policy", "config_id", "batch_size"])
    path = f"{output_prefix}_batch_sensitivity.csv"
    out.to_csv(path, index=False)
    print(f"[batch-sens] saved {path} ({len(out)} rows)", flush=True)

    # Plot: mean±std vs batch for each policy (best/only config)
    fig, ax = plt.subplots(figsize=(12, 7))
    for (policy, cfg_id), sub in out.groupby(["policy", "config_id"]):
        sub = sub.sort_values("batch_size")
        ax.errorbar(
            sub["batch_size"],
            sub["final_regret_mean"],
            yerr=sub["final_regret_std"],
            fmt="o-",
            capsize=3,
            label=f"{policy}|{cfg_id}",
            linewidth=1.6,
        )
    ax.set_xscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("Final regret (mean ± sd)")
    ax.set_title(f"Batch-size sensitivity — {output_prefix}")
    ax.legend(fontsize=6.5, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_batch_sensitivity.png", dpi=160)
    plt.close(fig)
    print(f"[batch-sens] saved {output_prefix}_batch_sensitivity.png", flush=True)
    return out


def run_batch_sensitivity(
    setting: str = "messy",
    batch_sizes: List[int] | None = None,
    feature_dim: int = 30,
    total_samples: int = 15000,
    ref_samples: int = 5000,
    n_repeats: int = 2,
    output_prefix: str | None = None,
) -> pd.DataFrame:
    if batch_sizes is None:
        batch_sizes = list(DEFAULT_BATCHES)
    if output_prefix is None:
        output_prefix = f"mab_batch_sens_{setting}_d{feature_dim}"

    dgp_spec = _dgp_cfg(setting)
    policy_jobs = _expand_policy_grid(POLICIES)
    run_config = copy.deepcopy(CONFIG)
    n_arms = run_config["model_pool"]["n_arms"]
    rows: List[Dict[str, Any]] = []

    print(
        f"[batch-sens] setting={setting} dgp={dgp_spec['dgp']} d={feature_dim} "
        f"n={total_samples} ref={ref_samples} batches={batch_sizes} "
        f"jobs={len(policy_jobs)} reps={n_repeats}",
        flush=True,
    )

    for batch_size in batch_sizes:
        batch_size = int(batch_size)
        # Need at least a few stream batches.
        if total_samples - ref_samples < batch_size * 5:
            print(f"[batch-sens] skip batch={batch_size}: stream too short", flush=True)
            continue
        for rep in range(n_repeats):
            data_cfg = dict(run_config["data"])
            data_cfg.update(dgp_spec)
            data_cfg.update(
                {
                    "feature_dim": feature_dim,
                    "total_samples": total_samples,
                    "ref_samples": ref_samples,
                    "batch_size": batch_size,
                    "noise_scale": float(run_config["data"].get("noise_scale", 2.5)),
                    "random_seed": int(run_config["data"]["random_seed"])
                    + rep * 53
                    + batch_size * 3,
                }
            )
            if dgp_spec["dgp"] != "stationary":
                data_cfg["shift_point_index"] = total_samples // 2

            t0 = time.time()
            print(
                f"[batch-sens] batch={batch_size} rep={rep + 1}/{n_repeats} generating...",
                flush=True,
            )
            data = generate_dgp_from_config(data_cfg, n_arms=n_arms)
            cache = precompute_batch_cache(data, run_config)
            print(
                f"[batch-sens] batch={batch_size} rep={rep} ready in {time.time() - t0:.1f}s "
                f"n_batches={data['n_batches']}",
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
                row = {
                    "setting": setting,
                    "dgp_name": dgp_spec["name"],
                    "dgp": dgp_spec["dgp"],
                    "rep": rep,
                    "feature_dim": feature_dim,
                    "total_samples": total_samples,
                    "ref_samples": data["ref_samples"],
                    "batch_size": batch_size,
                    "n_batches": result["n_batches"],
                    "shift_batch_index": data["shift_batch_index"],
                    "policy": policy_name,
                    "config_id": cfg_id,
                    "final_regret": result["final_regret"],
                    "mean_reward": result["mean_reward"],
                }
                for k, v in policy_cfg.items():
                    row[f"hp_{k}"] = v
                rows.append(row)

            pd.DataFrame(rows).to_csv(f"{output_prefix}_metrics.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    summary = (
        metrics_df.groupby(["batch_size", "policy", "config_id"])["final_regret"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "final_regret_mean",
                "std": "final_regret_std",
                "count": "n_repeats",
            }
        )
        .sort_values(["batch_size", "final_regret_mean"])
    )
    summary.to_csv(f"{output_prefix}_summary.csv", index=False)
    build_batch_sensitivity_table(metrics_df, output_prefix)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["messy", "stationary", "ultra", "shift"], required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--feature-dim", type=int, default=30)
    parser.add_argument("--total-samples", type=int, default=15000)
    parser.add_argument("--ref-samples", type=int, default=5000)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_batch_sensitivity(
        setting=args.setting,
        batch_sizes=args.batch_sizes,
        feature_dim=args.feature_dim,
        total_samples=args.total_samples,
        ref_samples=args.ref_samples,
        n_repeats=args.n_repeats,
        output_prefix=args.output_prefix,
    )
