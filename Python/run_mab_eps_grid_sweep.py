"""ε-Greedy full grid sweep: feature_dim × batch_size × epsilon on nonlinear_messy DGP."""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Set, Tuple

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
    "dgp": "nonlinear_messy",
    "shift_magnitude": 0.5,
}

DEFAULT_FEATURE_DIMS = [10, 25, 50, 75, 100]
DEFAULT_BATCH_SIZES = [10, 25, 50, 75, 100, 200, 250, 500]
DEFAULT_EPS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

EPS_POLICY_GRID = {"Epsilon_Greedy": {"epsilon": DEFAULT_EPS}}


def _job_key(row: Dict[str, Any]) -> Tuple:
    return (
        int(row["feature_dim"]),
        int(row["batch_size"]),
        int(row["rep"]),
        str(row["config_id"]),
    )


def _load_checkpoint(prefix: str) -> pd.DataFrame:
    path = f"{prefix}_metrics.csv"
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def build_grid_summaries(metrics_df: pd.DataFrame, output_prefix: str) -> None:
    by_cell = (
        metrics_df.groupby(["feature_dim", "batch_size", "hp_epsilon"])["final_regret"]
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
    by_cell["final_regret_se"] = by_cell["final_regret_std"] / np.sqrt(by_cell["n_repeats"].clip(lower=1))
    by_cell.to_csv(f"{output_prefix}_summary_by_cell.csv", index=False)

    best_eps = (
        by_cell.sort_values("final_regret_mean")
        .groupby(["feature_dim", "batch_size"], as_index=False)
        .first()
        .sort_values(["feature_dim", "batch_size"])
    )
    best_eps.to_csv(f"{output_prefix}_best_epsilon_per_cell.csv", index=False)

    overall_eps = (
        metrics_df.groupby("hp_epsilon")["final_regret"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "final_regret_mean", "std": "final_regret_std", "count": "n_obs"})
        .sort_values("hp_epsilon")
    )
    overall_eps.to_csv(f"{output_prefix}_summary_by_epsilon.csv", index=False)

    for dim in sorted(metrics_df["feature_dim"].unique()):
        sub = by_cell[by_cell.feature_dim == dim]
        if sub.empty:
            continue
        pivot = sub.pivot(index="batch_size", columns="hp_epsilon", values="final_regret_mean")
        pivot.to_csv(f"{output_prefix}_heatmap_d{int(dim)}.csv")

        fig, ax = plt.subplots(figsize=(12, 6))
        for eps in sorted(sub["hp_epsilon"].unique()):
            s = sub[sub.hp_epsilon == eps].sort_values("batch_size")
            ax.plot(s["batch_size"], s["final_regret_mean"], "o-", label=f"ε={eps:g}", alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Final regret (mean)")
        ax.set_title(f"ε-Greedy regret vs batch size (d={int(dim)})")
        ax.legend(fontsize=6, ncol=3, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{output_prefix}_regret_vs_batch_d{int(dim)}.png", dpi=180)
        plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.errorbar(
        overall_eps["hp_epsilon"] * 100,
        overall_eps["final_regret_mean"],
        yerr=overall_eps["final_regret_std"],
        fmt="o-",
        capsize=4,
        color="#E08214",
    )
    ax2.set_xlabel("ε (%)")
    ax2.set_ylabel("Final regret (mean ± std, pooled)")
    ax2.set_title("ε-Greedy: pooled regret vs exploration rate")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{output_prefix}_epsilon_pooled.png", dpi=180)
    plt.close(fig2)


def run_eps_grid_sweep(
    n_repeats: int = 5,
    feature_dims: List[int] | None = None,
    batch_sizes: List[int] | None = None,
    epsilon_grid: List[float] | None = None,
    total_samples: int = 20_000,
    ref_samples: int = 5_000,
    output_prefix: str = "mab_eps_grid_sweep",
    resume: bool = True,
) -> pd.DataFrame:
    feature_dims = feature_dims or list(DEFAULT_FEATURE_DIMS)
    batch_sizes = batch_sizes or list(DEFAULT_BATCH_SIZES)
    epsilon_grid = epsilon_grid or list(DEFAULT_EPS)

    policies_cfg = copy.deepcopy(EPS_POLICY_GRID)
    policies_cfg["Epsilon_Greedy"]["epsilon"] = [float(x) for x in epsilon_grid]
    policy_jobs = _expand_policy_grid(policies_cfg)

    n_jobs = len(feature_dims) * len(batch_sizes) * len(policy_jobs) * n_repeats
    print(
        f"[eps-grid] d={feature_dims} × batch={batch_sizes} × ε×{len(epsilon_grid)} "
        f"× {n_repeats} repeats = {n_jobs} runs",
        flush=True,
    )

    run_config = copy.deepcopy(CONFIG)
    n_arms = run_config["model_pool"]["n_arms"]
    metrics_df = _load_checkpoint(output_prefix) if resume else pd.DataFrame()
    done: Set[Tuple] = set()
    if len(metrics_df):
        for _, r in metrics_df.iterrows():
            done.add(_job_key(r.to_dict()))

    shift_point = total_samples // 2

    for feature_dim in feature_dims:
        for batch_size in batch_sizes:
            data_cfg = dict(run_config["data"])
            data_cfg.update(MESSY_DGP)
            data_cfg["feature_dim"] = int(feature_dim)
            data_cfg["total_samples"] = total_samples
            data_cfg["ref_samples"] = ref_samples
            data_cfg["batch_size"] = int(batch_size)
            data_cfg["shift_point_index"] = shift_point

            for rep in range(n_repeats):
                pending = [
                    (pn, pc)
                    for pn, pc in policy_jobs
                    if _job_key(
                        {
                            "feature_dim": feature_dim,
                            "batch_size": batch_size,
                            "rep": rep,
                            "config_id": _cfg_to_id(pc),
                        }
                    )
                    not in done
                ]
                if not pending:
                    continue

                data_cfg_run = dict(data_cfg)
                data_cfg_run["random_seed"] = (
                    int(data_cfg["random_seed"]) + rep * 23 + feature_dim * 7 + batch_size
                )
                t0 = time.time()
                print(
                    f"[eps-grid] d={feature_dim} batch={batch_size} rep={rep + 1}/{n_repeats} "
                    f"({len(pending)} ε configs, n={total_samples})",
                    flush=True,
                )
                data = generate_dgp_from_config(data_cfg_run, n_arms=n_arms)
                cache = precompute_batch_cache(data, run_config)
                new_rows: List[Dict[str, Any]] = []

                for policy_name, policy_cfg in pending:
                    t1 = time.time()
                    policy = make_policy(policy_name, n_arms, 53, dict(policy_cfg))
                    result = run_experiment(
                        policy,
                        data,
                        run_config,
                        use_context=False,
                        batch_cache=cache,
                    )
                    row = {
                        "dgp_name": f"messy_d{feature_dim}",
                        "rep": rep,
                        "feature_dim": feature_dim,
                        "batch_size": batch_size,
                        "n_batches": result["n_batches"],
                        "shift_batch_index": data["shift_batch_index"],
                        "total_samples": total_samples,
                        "ref_samples": ref_samples,
                        "policy": policy_name,
                        "config_id": _cfg_to_id(policy_cfg),
                        "final_regret": result["final_regret"],
                        "mean_reward": result["mean_reward"],
                        **{f"hp_{k}": v for k, v in policy_cfg.items()},
                    }
                    new_rows.append(row)
                    done.add(_job_key(row))
                    print(
                        f"[eps-grid] d={feature_dim} bs={batch_size} rep={rep} "
                        f"ε={policy_cfg['epsilon']} regret={result['final_regret']:.2f} "
                        f"({time.time()-t1:.1f}s)",
                        flush=True,
                    )

                if new_rows:
                    metrics_df = pd.concat([metrics_df, pd.DataFrame(new_rows)], ignore_index=True)
                    metrics_df.to_csv(f"{output_prefix}_metrics.csv", index=False)
                    print(
                        f"[eps-grid] checkpoint saved ({len(metrics_df)} rows, "
                        f"DGP gen {time.time()-t0:.1f}s)",
                        flush=True,
                    )

    build_grid_summaries(metrics_df, output_prefix)
    print(f"\n[eps-grid] done — {len(metrics_df)} total rows", flush=True)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ε-Greedy dim × batch × ε grid sweep")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--feature-dims", nargs="+", type=int, default=DEFAULT_FEATURE_DIMS)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--epsilon", nargs="+", type=float, default=DEFAULT_EPS)
    parser.add_argument("--total-samples", type=int, default=20_000)
    parser.add_argument("--ref-samples", type=int, default=5_000)
    parser.add_argument("--output-prefix", default="mab_eps_grid_sweep")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if args.quick:
        args.feature_dims = [10, 30]
        args.batch_sizes = [50, 100]
        args.epsilon = [0.02, 0.05, 0.1]
        args.n_repeats = 1
    run_eps_grid_sweep(
        n_repeats=args.n_repeats,
        feature_dims=args.feature_dims,
        batch_sizes=args.batch_sizes,
        epsilon_grid=args.epsilon,
        total_samples=args.total_samples,
        ref_samples=args.ref_samples,
        output_prefix=args.output_prefix,
        resume=not args.no_resume,
    )
