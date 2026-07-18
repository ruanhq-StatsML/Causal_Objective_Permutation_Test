"""Large-scale epsilon sweep on nonlinear_messy DGP (high-dim, long stream)."""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_MAB import CONFIG, PROTOTYPE_EPSILON_SWEEP_POLICIES
from run_mab_benchmark import _cfg_to_id, _expand_policy_grid, _policy_uses_context
from run_mab_prototype_benchmark import plot_prototype_confidence_bands
from mab_benchmark_core import (
    context_dim,
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)

SCALE_DGP = {
    "name": "messy_d500_n100k",
    "dgp": "nonlinear_messy",
    "shift_magnitude": 0.5,
    "shift_point_index": 50000,
}

DEFAULT_EPS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]


def build_variance_explore_analysis(
    metrics_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    output_prefix: str,
) -> pd.DataFrame:
    """Summarize repeat variance vs epsilon exploration traffic ratio; save CSV + plots."""
    n_batches = int(metrics_df["n_batches"].iloc[0]) if len(metrics_df) else 0
    rows: List[Dict[str, Any]] = []

    for cfg_id in sorted(metrics_df["config_id"].unique()):
        msub = metrics_df[metrics_df.config_id == cfg_id]
        eps = float(msub["hp_epsilon"].iloc[0])
        explore_ratio = eps
        expected_explore_batches = eps * n_batches

        final_mean = float(msub["final_regret"].mean())
        final_std = float(msub["final_regret"].std(ddof=0)) if len(msub) > 1 else 0.0
        final_cv = final_std / final_mean if final_mean > 1e-9 else np.nan

        csub = curves_df[curves_df.config_id == cfg_id]
        batch_stds: List[float] = []
        inc_vars: List[float] = []
        for batch_idx in sorted(csub["batch_index"].unique()):
            b = csub[csub.batch_index == batch_idx]["cum_regret"].to_numpy()
            if len(b) > 1:
                batch_stds.append(float(np.std(b, ddof=0)))
            if batch_idx > 0:
                prev = csub[csub.batch_index == batch_idx - 1].set_index("rep")["cum_regret"]
                curr = csub[csub.batch_index == batch_idx].set_index("rep")["cum_regret"]
                aligned = curr - prev
                if len(aligned) > 1:
                    inc_vars.append(float(np.var(aligned, ddof=0)))

        rows.append(
            {
                "config_id": cfg_id,
                "epsilon": eps,
                "explore_traffic_ratio": explore_ratio,
                "expected_explore_batches": expected_explore_batches,
                "n_batches": n_batches,
                "final_regret_mean": final_mean,
                "final_regret_std": final_std,
                "final_regret_cv": final_cv,
                "path_cumregret_std_mean": float(np.mean(batch_stds)) if batch_stds else 0.0,
                "incremental_regret_var_mean": float(np.mean(inc_vars)) if inc_vars else 0.0,
                "n_repeats": int(len(msub)),
            }
        )

    out = pd.DataFrame(rows).sort_values("epsilon")
    out_path = f"{output_prefix}_variance_vs_explore.csv"
    out.to_csv(out_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = out["explore_traffic_ratio"].to_numpy() * 100

    axes[0].errorbar(
        x, out["final_regret_mean"], yerr=out["final_regret_std"],
        fmt="o-", capsize=4, color="#E08214", linewidth=2,
    )
    axes[0].set_xlabel("Exploration traffic ε (%)")
    axes[0].set_ylabel("Final regret (mean ± std)")
    axes[0].set_title("Regret vs ε")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, out["final_regret_std"], "s-", color="#1F77B4", linewidth=2)
    axes[1].set_xlabel("Exploration traffic ε (%)")
    axes[1].set_ylabel("Std across repeats")
    axes[1].set_title("Repeat variance vs ε")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, out["final_regret_cv"], "^-", color="#2CA02C", linewidth=2, label="CV (std/mean)")
    axes[2].plot(
        x, out["path_cumregret_std_mean"], "v--", color="#9467BD", linewidth=1.8, label="Path std (mean)",
    )
    axes[2].set_xlabel("Exploration traffic ε (%)")
    axes[2].set_ylabel("Normalized / path volatility")
    axes[2].set_title("Variance trend vs explore rate")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{output_prefix}_variance_vs_explore.png", dpi=180)
    plt.close(fig)
    print(f"[scale/eps] saved {out_path} ({len(out)} rows)", flush=True)
    print(f"[scale/eps] saved {output_prefix}_variance_vs_explore.png", flush=True)
    return out


def run_scale_eps_benchmark(
    n_repeats: int = 3,
    feature_dim: int = 500,
    total_samples: int = 100_000,
    ref_samples: int = 10_000,
    batch_size: int = 500,
    epsilon_grid: List[float] | None = None,
    output_prefix: str = "mab_scale_messy_eps",
    plot: bool = True,
) -> pd.DataFrame:
    if epsilon_grid is None:
        epsilon_grid = DEFAULT_EPS
    policies_cfg = copy.deepcopy(PROTOTYPE_EPSILON_SWEEP_POLICIES)
    policies_cfg["Epsilon_Greedy"]["epsilon"] = [float(x) for x in epsilon_grid]
    policy_jobs = _expand_policy_grid(policies_cfg)

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
    shift_batches: List[int] = []

    for rep in range(n_repeats):
        data_cfg_run = dict(data_cfg)
        data_cfg_run["random_seed"] = int(data_cfg["random_seed"]) + rep * 11
        t0 = time.time()
        print(
            f"[scale/eps] rep={rep + 1}/{n_repeats} generating DGP "
            f"(d={feature_dim}, n={total_samples}, ref={ref_samples}, batch={batch_size})",
            flush=True,
        )
        data = generate_dgp_from_config(data_cfg_run, n_arms=n_arms)
        shift_batches = [data["shift_batch_index"]]
        print(
            f"[scale/eps] DGP ready in {time.time() - t0:.1f}s, "
            f"n_batches={data['n_batches']}, shift_batch={data['shift_batch_index']}",
            flush=True,
        )

        cache = precompute_batch_cache(data, run_config)
        tc = time.time() - t0
        print(f"[scale/eps] batch cache in {tc:.1f}s total", flush=True)

        for policy_name, policy_cfg in policy_jobs:
            cfg_id = _cfg_to_id(policy_cfg)
            print(f"[scale/eps] rep={rep} cfg={cfg_id}", flush=True)
            policy = make_policy(policy_name, n_arms, context_dim(run_config["context"]["n_base_features"]), dict(policy_cfg))
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
                    "cooling_preset": "scale_default",
                    "rep": rep,
                    "n_batches": result["n_batches"],
                    "shift_batch_index": data["shift_batch_index"],
                    "feature_dim": feature_dim,
                    "total_samples": total_samples,
                    "ref_samples": ref_samples,
                    "batch_size": batch_size,
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
                        "cooling_preset": "scale_default",
                        "rep": rep,
                        "policy": policy_name,
                        "config_id": cfg_id,
                        "batch_index": batch_idx,
                        "cum_regret": float(cum_reg),
                    }
                )

    metrics_df = pd.DataFrame(rows)
    curves_df = pd.DataFrame(curve_rows)
    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
    curves_df.to_csv(f"{output_prefix}_cum_regret.csv", index=False)
    summary = (
        metrics_df.groupby(["policy", "config_id"])["final_regret"]
        .agg(["mean", "std", "min", "count"])
        .reset_index()
        .rename(columns={"mean": "final_regret_mean", "std": "final_regret_std", "min": "final_regret_best", "count": "n_repeats"})
        .sort_values("final_regret_mean")
    )
    summary.to_csv(f"{output_prefix}_summary.csv", index=False)
    print(f"[scale/eps] saved {output_prefix}_summary.csv", flush=True)
    build_variance_explore_analysis(metrics_df, curves_df, output_prefix)
    if plot and len(curves_df):
        metrics_df["cooling_preset"] = "scale_default"
        curves_df["cooling_preset"] = "scale_default"
        plot_prototype_confidence_bands(curves_df, metrics_df, output_prefix, shift_batch_indices=shift_batches)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale epsilon sweep on nonlinear_messy DGP")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=500)
    parser.add_argument("--total-samples", type=int, default=100_000)
    parser.add_argument("--ref-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--epsilon-grid", nargs="+", type=float, default=None)
    parser.add_argument("--output-prefix", default="mab_scale_messy_eps")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="build variance-vs-explore table/plot from existing CSVs",
    )
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.analyze_only:
        metrics_df = pd.read_csv(f"{args.output_prefix}_metrics.csv")
        curves_df = pd.read_csv(f"{args.output_prefix}_cum_regret.csv")
        build_variance_explore_analysis(metrics_df, curves_df, args.output_prefix)
    elif args.plot_only:
        curves_df = pd.read_csv(f"{args.output_prefix}_cum_regret.csv")
        metrics_df = pd.read_csv(f"{args.output_prefix}_metrics.csv")
        plot_prototype_confidence_bands(curves_df, metrics_df, args.output_prefix)
    else:
        run_scale_eps_benchmark(
            n_repeats=args.n_repeats,
            feature_dim=args.feature_dim,
            total_samples=args.total_samples,
            ref_samples=args.ref_samples,
            batch_size=args.batch_size,
            epsilon_grid=args.epsilon_grid,
            output_prefix=args.output_prefix,
            plot=not args.no_plot,
        )
