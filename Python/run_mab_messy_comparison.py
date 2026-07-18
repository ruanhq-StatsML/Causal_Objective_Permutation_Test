"""Messy DGP (d=30, 100k): Epsilon-Greedy exploration sweep vs LinUCB vs UCB vs Adaptive."""
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

SCALE_DGP = {
    "name": "messy_d30_n100k",
    "dgp": "nonlinear_messy",
    "shift_magnitude": 0.5,
    "shift_point_index": 50000,
}

DEFAULT_EPS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

MESSY_COMPARISON_POLICIES = {
    "Epsilon_Greedy": {
        "epsilon": DEFAULT_EPS,
    },
    "LinUCB_Vanilla": {
        "alpha": 0.1,
        "lambda_reg": [10, 50, 100, 500],
        "use_momentum": False,
    },
    "LinUCB_Momentum": {
        "alpha": 0.1,
        "lambda_reg": [50, 100, 500],
        "base_gamma": [0.85, 0.90, 0.95],
        "window": 30,
        "threshold": 1.5,
        "use_momentum": True,
        "gamma_decay": [0.90, 0.95],
        "alpha_growth": 1.02,
    },
    "AdaptiveEpsilonGreedy": {
        "base_epsilon": [0.025, 0.05, 0.10],
        "max_epsilon": [0.5],
        "gamma": [0.9],
        "anomaly_sensitivity": [0.3, 0.5, 0.7],
    },
    "UCB": {},
}


def build_variance_explore_analysis(
    metrics_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    output_prefix: str,
) -> pd.DataFrame:
    eps_df = metrics_df[metrics_df.policy == "Epsilon_Greedy"].copy()
    if eps_df.empty:
        return pd.DataFrame()
    n_batches = int(eps_df["n_batches"].iloc[0])
    rows: List[Dict[str, Any]] = []
    for cfg_id in sorted(eps_df["config_id"].unique()):
        msub = eps_df[eps_df.config_id == cfg_id]
        eps = float(msub["hp_epsilon"].iloc[0])
        final_mean = float(msub["final_regret"].mean())
        final_std = float(msub["final_regret"].std(ddof=0)) if len(msub) > 1 else 0.0
        final_cv = final_std / final_mean if final_mean > 1e-9 else np.nan
        csub = curves_df[curves_df.config_id == cfg_id]
        batch_stds: List[float] = []
        for batch_idx in sorted(csub["batch_index"].unique()):
            b = csub[csub.batch_index == batch_idx]["cum_regret"].to_numpy()
            if len(b) > 1:
                batch_stds.append(float(np.std(b, ddof=0)))
        rows.append(
            {
                "config_id": cfg_id,
                "epsilon": eps,
                "explore_traffic_ratio": eps,
                "expected_explore_batches": eps * n_batches,
                "n_batches": n_batches,
                "final_regret_mean": final_mean,
                "final_regret_std": final_std,
                "final_regret_cv": final_cv,
                "path_cumregret_std_mean": float(np.mean(batch_stds)) if batch_stds else 0.0,
                "n_repeats": int(len(msub)),
            }
        )
    out = pd.DataFrame(rows).sort_values("epsilon")
    out.to_csv(f"{output_prefix}_epsilon_variance_vs_explore.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = out["explore_traffic_ratio"].to_numpy() * 100
    axes[0].errorbar(x, out["final_regret_mean"], yerr=out["final_regret_std"], fmt="o-", capsize=4, color="#E08214")
    axes[0].set_xlabel("Exploration traffic ε (%)")
    axes[0].set_ylabel("Final regret (mean ± std)")
    axes[0].set_title("Regret vs ε (messy DGP)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x, out["final_regret_std"], "s-", color="#1F77B4")
    axes[1].set_xlabel("Exploration traffic ε (%)")
    axes[1].set_ylabel("Std across repeats")
    axes[1].set_title("Repeat variance vs ε")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(x, out["final_regret_cv"], "^-", color="#2CA02C", label="CV")
    axes[2].plot(x, out["path_cumregret_std_mean"], "v--", color="#9467BD", label="Path std")
    axes[2].set_xlabel("Exploration traffic ε (%)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_epsilon_variance_vs_explore.png", dpi=180)
    plt.close(fig)
    return out


def build_policy_comparison_summary(metrics_df: pd.DataFrame, output_prefix: str) -> pd.DataFrame:
    summary = (
        metrics_df.groupby(["policy", "config_id"])["final_regret"]
        .agg(["mean", "std", "min", "count"])
        .reset_index()
        .rename(columns={"mean": "final_regret_mean", "std": "final_regret_std", "min": "final_regret_best", "count": "n_repeats"})
    )
    best_per_policy = (
        summary.sort_values("final_regret_mean")
        .groupby("policy", as_index=False)
        .first()
        .sort_values("final_regret_mean")
    )
    best_per_policy.to_csv(f"{output_prefix}_best_per_policy.csv", index=False)
    summary.sort_values("final_regret_mean").to_csv(f"{output_prefix}_all_configs_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {
        "Epsilon_Greedy": "#E08214",
        "LinUCB_Vanilla": "#1F77B4",
        "LinUCB_Momentum": "#2CA02C",
        "AdaptiveEpsilonGreedy": "#9467BD",
        "UCB": "#D62728",
    }
    y_pos = np.arange(len(best_per_policy))
    ax.barh(
        y_pos,
        best_per_policy["final_regret_mean"],
        xerr=best_per_policy["final_regret_std"],
        color=[colors.get(p, "gray") for p in best_per_policy["policy"]],
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r.policy}\n{r.config_id[:40]}" for r in best_per_policy.itertuples()],
        fontsize=8,
    )
    ax.set_xlabel("Final regret (mean ± std)")
    ax.set_title("Best config per policy family — messy DGP 500d/100k")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_best_per_policy.png", dpi=180)
    plt.close(fig)
    return best_per_policy


def plot_regret_curves(curves_df: pd.DataFrame, metrics_df: pd.DataFrame, output_prefix: str, shift_batch: int) -> None:
    best = (
        metrics_df.groupby(["policy", "config_id"])["final_regret"]
        .mean()
        .reset_index()
        .sort_values("final_regret")
        .groupby("policy")
        .first()
        .reset_index()[["policy", "config_id"]]
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = {
        "Epsilon_Greedy": "#E08214",
        "LinUCB_Vanilla": "#1F77B4",
        "LinUCB_Momentum": "#2CA02C",
        "AdaptiveEpsilonGreedy": "#9467BD",
        "UCB": "#D62728",
    }
    for _, row in best.iterrows():
        sub = curves_df[
            (curves_df.policy == row["policy"]) & (curves_df.config_id == row["config_id"])
        ]
        if sub.empty:
            continue
        stats = sub.groupby("batch_index")["cum_regret"].agg(["mean", "std"]).reset_index()
        label = f"{row['policy']} ({row['config_id'][:30]})"
        ax.plot(stats["batch_index"], stats["mean"], label=label, color=colors.get(row["policy"], "gray"), linewidth=2)
        ax.fill_between(
            stats["batch_index"],
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=colors.get(row["policy"], "gray"),
            alpha=0.15,
        )
    ax.axvline(x=shift_batch, color="red", linestyle="--", alpha=0.5, label=f"shift@{shift_batch}")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Best config per policy — cumulative regret bands")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_regret_curves_best.png", dpi=180)
    plt.close(fig)


def run_messy_comparison(
    n_repeats: int = 5,
    feature_dim: int = 500,
    total_samples: int = 100_000,
    ref_samples: int = 10_000,
    batch_size: int = 500,
    epsilon_grid: List[float] | None = None,
    output_prefix: str = "mab_messy_comparison",
    policy_names: List[str] | None = None,
) -> pd.DataFrame:
    policies_cfg = copy.deepcopy(MESSY_COMPARISON_POLICIES)
    if epsilon_grid is not None:
        policies_cfg["Epsilon_Greedy"]["epsilon"] = [float(x) for x in epsilon_grid]
    names = policy_names or list(policies_cfg.keys())
    policy_jobs = _expand_policy_grid(policies_cfg, policy_names=names)
    print(f"[messy] {len(policy_jobs)} policy configs across {names}", flush=True)

    run_config = copy.deepcopy(CONFIG)
    data_cfg = dict(run_config["data"])
    data_cfg.update(SCALE_DGP)
    data_cfg["feature_dim"] = feature_dim
    data_cfg["total_samples"] = total_samples
    data_cfg["ref_samples"] = ref_samples
    data_cfg["batch_size"] = batch_size

    n_arms = run_config["model_pool"]["n_arms"]
    rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    shift_batch = 0

    for rep in range(n_repeats):
        data_cfg_run = dict(data_cfg)
        data_cfg_run["random_seed"] = int(data_cfg["random_seed"]) + rep * 11
        t0 = time.time()
        print(
            f"[messy] rep={rep + 1}/{n_repeats} generating DGP "
            f"(d={feature_dim}, n={total_samples})",
            flush=True,
        )
        data = generate_dgp_from_config(data_cfg_run, n_arms=n_arms)
        shift_batch = int(data["shift_batch_index"])
        print(
            f"[messy] DGP ready in {time.time() - t0:.1f}s, "
            f"n_batches={data['n_batches']}, shift_batch={shift_batch}",
            flush=True,
        )
        cache = precompute_batch_cache(data, run_config)
        print(f"[messy] cache ready in {time.time() - t0:.1f}s total", flush=True)

        for policy_name, policy_cfg in policy_jobs:
            cfg_id = _cfg_to_id(policy_cfg)
            t1 = time.time()
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
                    "dgp_name": SCALE_DGP["name"],
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
                f"[messy] rep={rep} {policy_name} {cfg_id} "
                f"regret={result['final_regret']:.2f} ({time.time()-t1:.1f}s)",
                flush=True,
            )

    metrics_df = pd.DataFrame(rows)
    curves_df = pd.DataFrame(curve_rows)
    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    curves_df.to_csv(f"{output_prefix}_cum_regret.csv", index=False)

    build_variance_explore_analysis(metrics_df, curves_df, output_prefix)
    best = build_policy_comparison_summary(metrics_df, output_prefix)
    plot_regret_curves(curves_df, metrics_df, output_prefix, shift_batch)

    print("\n=== BEST PER POLICY FAMILY ===", flush=True)
    print(best.to_string(index=False), flush=True)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Messy DGP: ε-greedy vs LinUCB comparison")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--feature-dim", type=int, default=30)
    parser.add_argument("--total-samples", type=int, default=100_000)
    parser.add_argument("--ref-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--output-prefix", default="mab_messy_comparison")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help="subset of: Epsilon_Greedy LinUCB_Vanilla LinUCB_Momentum AdaptiveEpsilonGreedy UCB",
    )
    parser.add_argument("--quick", action="store_true", help="smaller scale smoke test")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.quick:
        args.feature_dim = 100
        args.total_samples = 20_000
        args.ref_samples = 5_000
        args.batch_size = 100
        args.n_repeats = 1
    run_messy_comparison(
        n_repeats=args.n_repeats,
        feature_dim=args.feature_dim,
        total_samples=args.total_samples,
        ref_samples=args.ref_samples,
        batch_size=args.batch_size,
        output_prefix=args.output_prefix,
        policy_names=args.policies,
    )
