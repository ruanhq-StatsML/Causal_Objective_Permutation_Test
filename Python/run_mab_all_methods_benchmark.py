"""All MAB methods @ fixed ε=5%, γ=0.8 across batch-size grid on messy DGP d=30."""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_MAB import CONFIG
from run_mab_benchmark import _cfg_to_id
from mab_benchmark_core import (
    context_dim,
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)

MESSY_DGP = {
    "name": "messy_d30_all_methods",
    "dgp": "nonlinear_messy",
    "shift_magnitude": 0.5,
    "shift_point_index": 10_000,
}

FIXED_EPSILON = 0.05
FIXED_GAMMA = 0.8
DEFAULT_BATCH_SIZES = [10, 25, 50, 75, 100, 200, 250, 500]

ALL_METHODS: List[Tuple[str, Dict[str, Any]]] = [
    ("Epsilon_Greedy", {"epsilon": FIXED_EPSILON}),
    ("Epsilon_Momentum", {"epsilon": FIXED_EPSILON, "gamma": FIXED_GAMMA}),
    (
        "AdaptiveEpsilonGreedyEWMA",
        {
            "base_epsilon": FIXED_EPSILON,
            "max_epsilon": 0.5,
            "gamma": FIXED_GAMMA,
            "anomaly_sensitivity": 0.5,
        },
    ),
    (
        "AdaptiveEpsilonGreedy",
        {
            "base_epsilon": FIXED_EPSILON,
            "max_epsilon": 0.5,
            "gamma": FIXED_GAMMA,
            "anomaly_sensitivity": 0.5,
        },
    ),
    ("LinUCB_Vanilla", {"alpha": 0.1, "lambda_reg": 100.0, "use_momentum": False}),
    (
        "LinUCB_Momentum",
        {
            "alpha": 0.1,
            "lambda_reg": 100.0,
            "base_gamma": FIXED_GAMMA,
            "gamma_decay": FIXED_GAMMA,
            "window": 30,
            "threshold": 1.5,
            "use_momentum": True,
            "alpha_growth": 1.02,
        },
    ),
    ("UCB", {}),
    ("Thompson_Sampling", {"prior_lambda": 0.1}),
    ("Gaussian_Sampling", {"learning_rate": 0.1, "init_sigma": 5.0}),
    ("Random", {}),
]

POLICY_COLORS = {
    "Epsilon_Greedy": "#E08214",
    "Epsilon_Momentum": "#FF7F0E",
    "AdaptiveEpsilonGreedyEWMA": "#BCBD22",
    "AdaptiveEpsilonGreedy": "#9467BD",
    "LinUCB_Vanilla": "#1F77B4",
    "LinUCB_Momentum": "#2CA02C",
    "UCB": "#D62728",
    "Thompson_Sampling": "#8C564B",
    "Gaussian_Sampling": "#17BECF",
    "Random": "#7F7F7F",
}


def _policy_uses_context(policy_name: str) -> bool:
    return policy_name.startswith("LinUCB") or policy_name == "AdaptiveEpsilonGreedy"


def _build_policy_jobs(epsilon: float, gamma: float) -> List[Tuple[str, Dict[str, Any]]]:
    jobs = copy.deepcopy(ALL_METHODS)
    for name, cfg in jobs:
        if "epsilon" in cfg:
            cfg["epsilon"] = float(epsilon)
        if name == "Epsilon_Momentum":
            cfg["gamma"] = float(gamma)
        if name == "LinUCB_Momentum":
            cfg["base_gamma"] = float(gamma)
            cfg["gamma_decay"] = float(gamma)
        if name in ("AdaptiveEpsilonGreedy", "AdaptiveEpsilonGreedyEWMA"):
            cfg["base_epsilon"] = float(epsilon)
            cfg["gamma"] = float(gamma)
    return jobs


def build_statistical_summaries(metrics_df: pd.DataFrame, output_prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        metrics_df.groupby("policy")["final_regret"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "final_regret_mean",
                "std": "final_regret_std",
                "min": "final_regret_min",
                "max": "final_regret_max",
                "count": "n_obs",
            }
        )
    )
    overall["final_regret_se"] = overall["final_regret_std"] / np.sqrt(overall["n_obs"].clip(lower=1))
    overall = overall.sort_values("final_regret_mean")
    overall.to_csv(f"{output_prefix}_summary_overall.csv", index=False)

    by_batch = (
        metrics_df.groupby(["policy", "batch_size"])["final_regret"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "final_regret_mean", "std": "final_regret_std", "count": "n_repeats"})
    )
    by_batch["final_regret_se"] = by_batch["final_regret_std"] / np.sqrt(by_batch["n_repeats"].clip(lower=1))
    by_batch.to_csv(f"{output_prefix}_summary_by_batch.csv", index=False)

    pivot = by_batch.pivot(index="policy", columns="batch_size", values="final_regret_mean")
    pivot.to_csv(f"{output_prefix}_regret_pivot.csv")

    fig, ax = plt.subplots(figsize=(13, 7))
    for policy in sorted(metrics_df["policy"].unique()):
        sub = by_batch[by_batch.policy == policy].sort_values("batch_size")
        ax.errorbar(
            sub["batch_size"],
            sub["final_regret_mean"],
            yerr=sub["final_regret_std"],
            fmt="o-",
            capsize=3,
            label=policy,
            color=POLICY_COLORS.get(policy, "gray"),
            alpha=0.9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Final regret (mean ± std)")
    ax.set_title(f"All methods @ ε={FIXED_EPSILON:.0%}, γ={FIXED_GAMMA} — messy DGP d=30")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_regret_vs_batch.png", dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(overall))
    ax2.barh(
        y_pos,
        overall["final_regret_mean"],
        xerr=overall["final_regret_std"],
        color=[POLICY_COLORS.get(p, "gray") for p in overall["policy"]],
        capsize=4,
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(overall["policy"], fontsize=9)
    ax2.set_xlabel("Final regret (mean ± std, pooled over batch sizes)")
    ax2.invert_yaxis()
    fig2.tight_layout()
    fig2.savefig(f"{output_prefix}_overall_barplot.png", dpi=180)
    plt.close(fig2)

    eps_methods = [
        "Epsilon_Greedy",
        "Epsilon_Momentum",
        "AdaptiveEpsilonGreedy",
        "AdaptiveEpsilonGreedyEWMA",
    ]
    eps_sub = by_batch[by_batch.policy.isin(eps_methods)]
    if not eps_sub.empty:
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for policy in eps_methods:
            sub = eps_sub[eps_sub.policy == policy].sort_values("batch_size")
            if sub.empty:
                continue
            ax3.plot(
                sub["batch_size"],
                sub["final_regret_mean"],
                "o-",
                label=policy,
                color=POLICY_COLORS.get(policy, "gray"),
                linewidth=2,
            )
        ax3.set_xscale("log")
        ax3.set_xlabel("Batch size")
        ax3.set_ylabel("Final regret (mean)")
        ax3.set_title("ε-family comparison vs batch size")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(f"{output_prefix}_epsilon_family.png", dpi=180)
        plt.close(fig3)

    return overall, by_batch


def run_all_methods_benchmark(
    n_repeats: int = 3,
    feature_dim: int = 30,
    total_samples: int = 20_000,
    ref_samples: int = 5_000,
    batch_sizes: List[int] | None = None,
    epsilon: float = FIXED_EPSILON,
    gamma: float = FIXED_GAMMA,
    output_prefix: str = "mab_all_methods_d30",
) -> pd.DataFrame:
    batch_sizes = batch_sizes or list(DEFAULT_BATCH_SIZES)
    policy_jobs = _build_policy_jobs(epsilon, gamma)
    n_jobs = len(batch_sizes) * len(policy_jobs) * n_repeats
    print(
        f"[all-methods] {len(policy_jobs)} policies × {len(batch_sizes)} batch sizes "
        f"× {n_repeats} repeats = {n_jobs} runs @ ε={epsilon:.0%}, γ={gamma}",
        flush=True,
    )

    run_config = copy.deepcopy(CONFIG)
    n_arms = run_config["model_pool"]["n_arms"]
    d_ctx = context_dim(run_config["context"]["n_base_features"])
    rows: List[Dict[str, Any]] = []

    for batch_size in batch_sizes:
        data_cfg = dict(run_config["data"])
        data_cfg.update(MESSY_DGP)
        data_cfg["feature_dim"] = feature_dim
        data_cfg["total_samples"] = total_samples
        data_cfg["ref_samples"] = ref_samples
        data_cfg["batch_size"] = int(batch_size)

        for rep in range(n_repeats):
            data_cfg_run = dict(data_cfg)
            data_cfg_run["random_seed"] = int(data_cfg["random_seed"]) + rep * 19 + batch_size
            t0 = time.time()
            print(
                f"[all-methods] batch_size={batch_size} rep={rep + 1}/{n_repeats} "
                f"(d={feature_dim}, n={total_samples})",
                flush=True,
            )
            data = generate_dgp_from_config(data_cfg_run, n_arms=n_arms)
            cache = precompute_batch_cache(data, run_config)

            for policy_name, policy_cfg in policy_jobs:
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
                        "batch_size": batch_size,
                        "n_batches": result["n_batches"],
                        "shift_batch_index": data["shift_batch_index"],
                        "feature_dim": feature_dim,
                        "total_samples": total_samples,
                        "ref_samples": ref_samples,
                        "fixed_epsilon": epsilon,
                        "fixed_gamma": gamma,
                        "policy": policy_name,
                        "config_id": _cfg_to_id(policy_cfg),
                        "final_regret": result["final_regret"],
                        "mean_reward": result["mean_reward"],
                        **{f"hp_{k}": v for k, v in policy_cfg.items()},
                    }
                )
                print(
                    f"[all-methods] bs={batch_size} rep={rep} {policy_name} "
                    f"regret={result['final_regret']:.2f} ({time.time()-t1:.1f}s)",
                    flush=True,
                )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    overall, _ = build_statistical_summaries(metrics_df, output_prefix)

    print("\n=== OVERALL RANKING (pooled) ===", flush=True)
    print(overall.to_string(index=False), flush=True)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="All MAB methods @ fixed ε, γ, batch-size grid")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=30)
    parser.add_argument("--total-samples", type=int, default=20_000)
    parser.add_argument("--ref-samples", type=int, default=5_000)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--epsilon", type=float, default=FIXED_EPSILON)
    parser.add_argument("--gamma", type=float, default=FIXED_GAMMA)
    parser.add_argument("--output-prefix", default="mab_all_methods_d30")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.quick:
        args.n_repeats = 1
        args.batch_sizes = [50, 100]
    run_all_methods_benchmark(
        n_repeats=args.n_repeats,
        feature_dim=args.feature_dim,
        total_samples=args.total_samples,
        ref_samples=args.ref_samples,
        batch_sizes=args.batch_sizes,
        epsilon=args.epsilon,
        gamma=args.gamma,
        output_prefix=args.output_prefix,
    )
