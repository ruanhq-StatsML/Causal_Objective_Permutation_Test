"""Epsilon-Greedy sensitivity w.r.t. batch size on nonlinear_messy DGP."""
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
from run_mab_benchmark import _cfg_to_id, _expand_policy_grid
from mab_benchmark_core import (
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)

MESSY_DGP = {
    "name": "nonlinear_messy",
    "dgp": "nonlinear_messy",
    "shift_magnitude": 0.5,
    "shift_point_index": 10_000,
}

DEFAULT_BATCH_SIZES = [10, 25, 50, 75, 100, 200, 250, 500]
DEFAULT_EPS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

EPS_POLICY_GRID = {
    "Epsilon_Greedy": {
        "epsilon": DEFAULT_EPS,
    },
}


def build_batch_sensitivity_summary(metrics_df: pd.DataFrame, output_prefix: str) -> pd.DataFrame:
    summary = (
        metrics_df.groupby(["batch_size", "config_id", "hp_epsilon"])["final_regret"]
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
    )
    summary.to_csv(f"{output_prefix}_by_batch_epsilon.csv", index=False)

    best_per_batch = (
        summary.sort_values("final_regret_mean")
        .groupby("batch_size", as_index=False)
        .first()
        .sort_values("batch_size")
    )
    best_per_batch.to_csv(f"{output_prefix}_best_per_batch.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for eps in sorted(metrics_df["hp_epsilon"].unique()):
        sub = summary[summary["hp_epsilon"] == eps].sort_values("batch_size")
        if sub.empty:
            continue
        axes[0].errorbar(
            sub["batch_size"],
            sub["final_regret_mean"],
            yerr=sub["final_regret_std"],
            fmt="o-",
            capsize=3,
            label=f"ε={eps:g}",
            alpha=0.85,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Batch size (samples per batch)")
    axes[0].set_ylabel("Final regret (mean ± std)")
    axes[0].set_title("ε-Greedy regret vs batch size")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc="best")

    axes[1].bar(
        best_per_batch["batch_size"].astype(str),
        best_per_batch["final_regret_mean"],
        yerr=best_per_batch["final_regret_std"],
        color="#E08214",
        capsize=4,
    )
    axes[1].set_xlabel("Batch size")
    axes[1].set_ylabel("Best regret (mean ± std)")
    axes[1].set_title("Best ε per batch size")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_batch_sensitivity.png", dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(
        best_per_batch["batch_size"],
        best_per_batch["hp_epsilon"],
        "s-",
        color="#1F77B4",
        linewidth=2,
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Best ε")
    ax2.set_title("Optimal exploration rate vs batch size")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{output_prefix}_best_epsilon_vs_batch.png", dpi=180)
    plt.close(fig2)

    return best_per_batch


def run_eps_batch_sensitivity(
    n_repeats: int = 5,
    feature_dim: int = 30,
    total_samples: int = 20_000,
    ref_samples: int = 5_000,
    batch_sizes: List[int] | None = None,
    epsilon_grid: List[float] | None = None,
    output_prefix: str = "mab_eps_batch_sensitivity",
) -> pd.DataFrame:
    batch_sizes = batch_sizes or list(DEFAULT_BATCH_SIZES)
    policies_cfg = copy.deepcopy(EPS_POLICY_GRID)
    if epsilon_grid is not None:
        policies_cfg["Epsilon_Greedy"]["epsilon"] = [float(x) for x in epsilon_grid]
    policy_jobs = _expand_policy_grid(policies_cfg)
    print(
        f"[eps-batch] {len(batch_sizes)} batch sizes × {len(policy_jobs)} ε configs × {n_repeats} repeats",
        flush=True,
    )

    run_config = copy.deepcopy(CONFIG)
    rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    for batch_size in batch_sizes:
        data_cfg = dict(run_config["data"])
        data_cfg.update(MESSY_DGP)
        data_cfg["feature_dim"] = feature_dim
        data_cfg["total_samples"] = total_samples
        data_cfg["ref_samples"] = ref_samples
        data_cfg["batch_size"] = int(batch_size)
        n_arms = run_config["model_pool"]["n_arms"]
        n_batches_est = max(1, int(np.ceil((total_samples - ref_samples) / max(1, batch_size))))

        for rep in range(n_repeats):
            data_cfg_run = dict(data_cfg)
            data_cfg_run["random_seed"] = int(data_cfg["random_seed"]) + rep * 13 + batch_size
            t0 = time.time()
            print(
                f"[eps-batch] batch_size={batch_size} rep={rep + 1}/{n_repeats} "
                f"(d={feature_dim}, n={total_samples}, ~{n_batches_est} batches)",
                flush=True,
            )
            data = generate_dgp_from_config(data_cfg_run, n_arms=n_arms)
            shift_batch = int(data["shift_batch_index"])
            cache = precompute_batch_cache(data, run_config)

            for policy_name, policy_cfg in policy_jobs:
                cfg_id = _cfg_to_id(policy_cfg)
                t1 = time.time()
                policy = make_policy(policy_name, n_arms, 53, dict(policy_cfg))
                result = run_experiment(
                    policy,
                    data,
                    run_config,
                    use_context=False,
                    batch_cache=cache,
                )
                rows.append(
                    {
                        "dgp_name": MESSY_DGP["name"],
                        "rep": rep,
                        "batch_size": batch_size,
                        "n_batches": result["n_batches"],
                        "shift_batch_index": shift_batch,
                        "feature_dim": feature_dim,
                        "total_samples": total_samples,
                        "ref_samples": ref_samples,
                        "policy": policy_name,
                        "config_id": cfg_id,
                        "final_regret": result["final_regret"],
                        "mean_reward": result["mean_reward"],
                        **{f"hp_{k}": v for k, v in policy_cfg.items()},
                    }
                )
                for batch_idx, cum_reg in enumerate(result["cum_regret"]):
                    curve_rows.append(
                        {
                            "rep": rep,
                            "batch_size": batch_size,
                            "config_id": cfg_id,
                            "batch_index": batch_idx,
                            "cum_regret": float(cum_reg),
                            "hp_epsilon": policy_cfg["epsilon"],
                        }
                    )
                print(
                    f"[eps-batch] bs={batch_size} rep={rep} ε={policy_cfg['epsilon']} "
                    f"regret={result['final_regret']:.2f} ({time.time()-t1:.1f}s)",
                    flush=True,
                )

    metrics_df = pd.DataFrame(rows)
    curves_df = pd.DataFrame(curve_rows)
    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    curves_df.to_csv(f"{output_prefix}_cum_regret.csv", index=False)
    best = build_batch_sensitivity_summary(metrics_df, output_prefix)

    print("\n=== BEST ε PER BATCH SIZE ===", flush=True)
    print(best.to_string(index=False), flush=True)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ε-Greedy batch-size sensitivity (nonlinear_messy)")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--feature-dim", type=int, default=30)
    parser.add_argument("--total-samples", type=int, default=20_000)
    parser.add_argument("--ref-samples", type=int, default=5_000)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BATCH_SIZES,
    )
    parser.add_argument("--output-prefix", default="mab_eps_batch_sensitivity")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.quick:
        args.batch_sizes = [25, 100]
        args.n_repeats = 1
    run_eps_batch_sensitivity(
        n_repeats=args.n_repeats,
        feature_dim=args.feature_dim,
        total_samples=args.total_samples,
        ref_samples=args.ref_samples,
        batch_sizes=args.batch_sizes,
        output_prefix=args.output_prefix,
    )
