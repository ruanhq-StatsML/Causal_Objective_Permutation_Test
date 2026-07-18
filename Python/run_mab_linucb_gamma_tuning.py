"""LinUCB gamma tuning + UCB baseline on nonlinear_messy DGP (d=30)."""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_MAB import CONFIG, GAMMA_GRID
from run_mab_benchmark import _cfg_to_id, _policy_uses_context
from mab_benchmark_core import (
    context_dim,
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)

MESSY_DGP = {
    "name": "messy_d30_gamma_tune",
    "dgp": "nonlinear_messy",
    "shift_magnitude": 0.5,
    "shift_point_index": 50_000,
}


def build_gamma_policy_jobs(
    gamma_grid: List[float],
    lambda_reg: float = 100.0,
) -> List[Tuple[str, Dict[str, Any]]]:
    jobs: List[Tuple[str, Dict[str, Any]]] = [
        (
            "LinUCB_Vanilla",
            {
                "alpha": 0.1,
                "lambda_reg": float(lambda_reg),
                "use_momentum": False,
            },
        ),
        ("UCB", {}),
    ]
    for gamma in gamma_grid:
        g = float(gamma)
        jobs.append(
            (
                "LinUCB_Momentum",
                {
                    "alpha": 0.1,
                    "lambda_reg": float(lambda_reg),
                    "base_gamma": g,
                    "gamma_decay": g,
                    "window": 30,
                    "threshold": 1.5,
                    "use_momentum": True,
                    "alpha_growth": 1.02,
                },
            )
        )
    return jobs


def build_gamma_summary(metrics_df: pd.DataFrame, output_prefix: str) -> pd.DataFrame:
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
    )
    summary = summary.merge(
        metrics_df.groupby(["policy", "config_id"]).first().reset_index()[
            ["policy", "config_id", "hp_base_gamma", "hp_lambda_reg"]
        ],
        on=["policy", "config_id"],
        how="left",
    )
    summary.to_csv(f"{output_prefix}_summary.csv", index=False)

    mom = metrics_df[metrics_df.policy == "LinUCB_Momentum"].copy()
    if not mom.empty:
        mom_summary = (
            mom.groupby("hp_base_gamma")["final_regret"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "final_regret_mean", "std": "final_regret_std"})
            .sort_values("hp_base_gamma")
        )
        mom_summary.to_csv(f"{output_prefix}_gamma_curve.csv", index=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(
            mom_summary["hp_base_gamma"],
            mom_summary["final_regret_mean"],
            yerr=mom_summary["final_regret_std"],
            fmt="o-",
            capsize=4,
            color="#2CA02C",
            label="LinUCB_Momentum",
        )
        ucb_mean = metrics_df.loc[metrics_df.policy == "UCB", "final_regret"].mean()
        van_mean = metrics_df.loc[metrics_df.policy == "LinUCB_Vanilla", "final_regret"].mean()
        if not np.isnan(ucb_mean):
            ax.axhline(ucb_mean, color="#D62728", linestyle="--", label=f"UCB ({ucb_mean:.1f})")
        if not np.isnan(van_mean):
            ax.axhline(van_mean, color="#1F77B4", linestyle=":", label=f"LinUCB_Vanilla ({van_mean:.1f})")
        ax.set_xlabel("γ (base_gamma = gamma_decay)")
        ax.set_ylabel("Final regret (mean ± std)")
        ax.set_title("LinUCB_Momentum γ tuning — messy DGP d=30")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{output_prefix}_gamma_tuning.png", dpi=180)
        plt.close(fig)

    best = summary.sort_values("final_regret_mean").groupby("policy", as_index=False).first()
    best.to_csv(f"{output_prefix}_best_per_policy.csv", index=False)
    return best


def run_linucb_gamma_tuning(
    n_repeats: int = 5,
    feature_dim: int = 30,
    total_samples: int = 100_000,
    ref_samples: int = 10_000,
    batch_size: int = 500,
    gamma_grid: List[float] | None = None,
    lambda_reg: float = 100.0,
    output_prefix: str = "mab_linucb_gamma_d30",
) -> pd.DataFrame:
    gamma_grid = [float(g) for g in (gamma_grid or GAMMA_GRID)]
    policy_jobs = build_gamma_policy_jobs(gamma_grid, lambda_reg=lambda_reg)
    print(
        f"[gamma] {len(policy_jobs)} configs "
        f"(LinUCB_Momentum γ×{len(gamma_grid)} + Vanilla + UCB) × {n_repeats} repeats",
        flush=True,
    )

    run_config = copy.deepcopy(CONFIG)
    data_cfg = dict(run_config["data"])
    data_cfg.update(MESSY_DGP)
    data_cfg["feature_dim"] = feature_dim
    data_cfg["total_samples"] = total_samples
    data_cfg["ref_samples"] = ref_samples
    data_cfg["batch_size"] = batch_size

    n_arms = run_config["model_pool"]["n_arms"]
    d_ctx = context_dim(run_config["context"]["n_base_features"])
    rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    for rep in range(n_repeats):
        data_cfg_run = dict(data_cfg)
        data_cfg_run["random_seed"] = int(data_cfg["random_seed"]) + rep * 17
        t0 = time.time()
        print(
            f"[gamma] rep={rep + 1}/{n_repeats} generating DGP "
            f"(d={feature_dim}, n={total_samples})",
            flush=True,
        )
        data = generate_dgp_from_config(data_cfg_run, n_arms=n_arms)
        shift_batch = int(data["shift_batch_index"])
        cache = precompute_batch_cache(data, run_config)
        print(
            f"[gamma] DGP ready in {time.time() - t0:.1f}s, "
            f"n_batches={data['n_batches']}, shift_batch={shift_batch}",
            flush=True,
        )

        for policy_name, policy_cfg in policy_jobs:
            cfg_id = _cfg_to_id(policy_cfg)
            t1 = time.time()
            policy = make_policy(policy_name, n_arms, d_ctx, dict(policy_cfg))
            result = run_experiment(
                policy,
                data,
                run_config,
                use_context=_policy_uses_context(policy_name),
                batch_cache=cache,
            )
            rows.append(
                {
                    "dgp_name": MESSY_DGP["name"],
                    "rep": rep,
                    "n_batches": result["n_batches"],
                    "shift_batch_index": shift_batch,
                    "feature_dim": feature_dim,
                    "total_samples": total_samples,
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
                        "policy": policy_name,
                        "config_id": cfg_id,
                        "batch_index": batch_idx,
                        "cum_regret": float(cum_reg),
                    }
                )
            print(
                f"[gamma] rep={rep} {policy_name} {cfg_id} "
                f"regret={result['final_regret']:.2f} ({time.time()-t1:.1f}s)",
                flush=True,
            )

    metrics_df = pd.DataFrame(rows)
    curves_df = pd.DataFrame(curve_rows)
    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    curves_df.to_csv(f"{output_prefix}_cum_regret.csv", index=False)
    best = build_gamma_summary(metrics_df, output_prefix)

    print("\n=== BEST PER POLICY ===", flush=True)
    print(best.to_string(index=False), flush=True)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LinUCB γ tuning + UCB on messy DGP")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--feature-dim", type=int, default=30)
    parser.add_argument("--total-samples", type=int, default=100_000)
    parser.add_argument("--ref-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--lambda-reg", type=float, default=100.0)
    parser.add_argument("--output-prefix", default="mab_linucb_gamma_d30")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    gamma_grid = list(GAMMA_GRID)
    if args.quick:
        args.n_repeats = 1
        args.total_samples = 20_000
        args.ref_samples = 5_000
        args.batch_size = 100
        gamma_grid = gamma_grid[::3]
    run_linucb_gamma_tuning(
        n_repeats=args.n_repeats,
        feature_dim=args.feature_dim,
        total_samples=args.total_samples,
        ref_samples=args.ref_samples,
        batch_size=args.batch_size,
        gamma_grid=gamma_grid,
        lambda_reg=args.lambda_reg,
        output_prefix=args.output_prefix,
    )
