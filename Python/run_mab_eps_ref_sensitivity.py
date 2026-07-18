"""Epsilon-greedy sensitivity analysis vs reference-set size on ultra-messy DGP."""
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
from mab_benchmark_core import (
    context_dim,
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)

DEFAULT_REF_SIZES = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
DEFAULT_EPS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

ULTRA_DGP = {
    "name": "nonlinear_messy_ultra_ref_sens",
    "dgp": "nonlinear_messy_ultra",
    "shift_magnitude": 0.5,
    "shift_interval": 2000,
    "mini_shift_interval": 400,
    "covariate_drift_strength": 0.45,
}


def build_ref_sensitivity_table(metrics_df: pd.DataFrame, output_prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (ref, cfg_id), msub in metrics_df.groupby(["ref_samples", "config_id"]):
        eps = float(msub["hp_epsilon"].iloc[0])
        final_mean = float(msub["final_regret"].mean())
        final_std = float(msub["final_regret"].std(ddof=0)) if len(msub) > 1 else 0.0
        final_cv = final_std / final_mean if final_mean > 1e-9 else np.nan
        rows.append(
            {
                "ref_samples": int(ref),
                "config_id": cfg_id,
                "epsilon": eps,
                "explore_traffic_ratio": eps,
                "final_regret_mean": final_mean,
                "final_regret_std": final_std,
                "final_regret_cv": final_cv,
                "n_repeats": int(len(msub)),
                "n_batches": int(msub["n_batches"].iloc[0]),
                "feature_dim": int(msub["feature_dim"].iloc[0]),
                "total_samples": int(msub["total_samples"].iloc[0]),
                "batch_size": int(msub["batch_size"].iloc[0]),
                "shift_batch_index": int(msub["shift_batch_index"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows).sort_values(["epsilon", "ref_samples"])
    out_path = f"{output_prefix}_ref_sensitivity.csv"
    out.to_csv(out_path, index=False)
    print(f"[ref-sens] saved {out_path} ({len(out)} rows)", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for eps, sub in out.groupby("epsilon"):
        sub = sub.sort_values("ref_samples")
        x = sub["ref_samples"].to_numpy()
        axes[0].errorbar(
            x, sub["final_regret_mean"], yerr=sub["final_regret_std"],
            fmt="o-", capsize=3, label=f"ε={eps:g}", linewidth=1.6,
        )
        axes[1].plot(x, sub["final_regret_std"], "s-", label=f"ε={eps:g}", linewidth=1.6)
        axes[2].plot(x, sub["final_regret_cv"], "^-", label=f"ε={eps:g}", linewidth=1.6)

    axes[0].set_xlabel("reference size n_ref")
    axes[0].set_ylabel("Final regret (mean ± std)")
    axes[0].set_title("Regret vs reference size")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xlabel("reference size n_ref")
    axes[1].set_ylabel("Std across repeats")
    axes[1].set_title("Repeat variance vs reference size")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)

    axes[2].set_xlabel("reference size n_ref")
    axes[2].set_ylabel("CV (std/mean)")
    axes[2].set_title("Robustness (CV) vs reference size")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(f"{output_prefix}_ref_sensitivity.png", dpi=180)
    plt.close(fig)
    print(f"[ref-sens] saved {output_prefix}_ref_sensitivity.png", flush=True)
    return out


def run_ref_sensitivity(
    ref_sizes: List[int],
    epsilon_grid: List[float],
    feature_dim: int = 30,
    total_samples: int = 15000,
    batch_size: int = 100,
    n_repeats: int = 3,
    output_prefix: str = "mab_eps_ref_sensitivity",
) -> pd.DataFrame:
    policies_cfg = copy.deepcopy(PROTOTYPE_EPSILON_SWEEP_POLICIES)
    policies_cfg["Epsilon_Greedy"]["epsilon"] = [float(x) for x in epsilon_grid]
    policy_jobs = _expand_policy_grid(policies_cfg)

    run_config = copy.deepcopy(CONFIG)
    n_arms = run_config["model_pool"]["n_arms"]
    rows: List[Dict[str, Any]] = []

    for ref in ref_sizes:
        ref = int(ref)
        # Keep enough stream length for a meaningful bandit path.
        if ref >= total_samples - batch_size * 5:
            print(f"[ref-sens] skip ref={ref}: too large for n={total_samples}", flush=True)
            continue
        for rep in range(n_repeats):
            data_cfg = dict(run_config["data"])
            data_cfg.update(ULTRA_DGP)
            data_cfg["feature_dim"] = int(feature_dim)
            data_cfg["total_samples"] = int(total_samples)
            data_cfg["ref_samples"] = ref
            data_cfg["batch_size"] = int(batch_size)
            data_cfg["shift_point_index"] = int(total_samples) // 2
            data_cfg["random_seed"] = int(run_config["data"]["random_seed"]) + rep * 23 + ref

            t0 = time.time()
            print(
                f"[ref-sens] n_ref={ref} rep={rep + 1}/{n_repeats} "
                f"d={feature_dim} n={total_samples} batch={batch_size} "
                f"dgp=nonlinear_messy_ultra",
                flush=True,
            )
            data = generate_dgp_from_config(data_cfg, n_arms=n_arms)
            cache = precompute_batch_cache(data, run_config)
            print(
                f"[ref-sens] n_ref={ref} rep={rep} ready in {time.time() - t0:.1f}s "
                f"n_batches={data['n_batches']} shift_batch={data['shift_batch_index']} "
                f"actual_ref={data['ref_samples']}",
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
                        "dgp_name": ULTRA_DGP["name"],
                        "dgp": "nonlinear_messy_ultra",
                        "rep": rep,
                        "feature_dim": int(feature_dim),
                        "total_samples": int(total_samples),
                        "ref_samples": int(data["ref_samples"]),
                        "requested_ref_samples": ref,
                        "batch_size": int(batch_size),
                        "n_batches": result["n_batches"],
                        "shift_batch_index": data["shift_batch_index"],
                        "policy": policy_name,
                        "config_id": cfg_id,
                        "final_regret": result["final_regret"],
                        "mean_reward": result["mean_reward"],
                        "hp_epsilon": policy_cfg.get("epsilon"),
                    }
                )

            metrics_df = pd.DataFrame(rows)
            metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    summary = (
        metrics_df.groupby(["ref_samples", "policy", "config_id"])["final_regret"]
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
        .sort_values(["ref_samples", "final_regret_mean"])
    )
    summary.to_csv(f"{output_prefix}_summary.csv", index=False)
    print(f"[ref-sens] saved {output_prefix}_summary.csv", flush=True)
    build_ref_sensitivity_table(metrics_df, output_prefix)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epsilon-greedy sensitivity vs reference size")
    parser.add_argument("--ref-sizes", nargs="+", type=int, default=DEFAULT_REF_SIZES)
    parser.add_argument("--epsilon-grid", nargs="+", type=float, default=DEFAULT_EPS)
    parser.add_argument("--feature-dim", type=int, default=30)
    parser.add_argument("--total-samples", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--output-prefix", default="mab_eps_ref_sensitivity")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.analyze_only:
        metrics_df = pd.read_csv(f"{args.output_prefix}_metrics.csv")
        build_ref_sensitivity_table(metrics_df, args.output_prefix)
    else:
        run_ref_sensitivity(
            ref_sizes=args.ref_sizes,
            epsilon_grid=args.epsilon_grid,
            feature_dim=args.feature_dim,
            total_samples=args.total_samples,
            batch_size=args.batch_size,
            n_repeats=args.n_repeats,
            output_prefix=args.output_prefix,
        )
