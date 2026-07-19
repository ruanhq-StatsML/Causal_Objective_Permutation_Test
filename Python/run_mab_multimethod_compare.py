"""Multi-method MAB comparison: stationary vs shift (ultra-messy)."""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Optional

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

# Modest grids: enough to pick a best config per method without exploding runtime.
MULTI_POLICIES = {
    "Epsilon_Greedy": {
        "epsilon": [0.02, 0.05, 0.10],
    },
    "AdaptiveEpsilonGreedy": {
        "base_epsilon": [0.05, 0.10],
        "max_epsilon": [0.5],
        "gamma": [0.9],
        "anomaly_sensitivity": [0.3, 0.5],
    },
    "UCB": {},
    "Random": {},
    "LinUCB_Vanilla": {
        "alpha": [0.1],
        "lambda_reg": [50, 100],
        "use_momentum": [False],
    },
    "Thompson_Sampling": {
        "prior_lambda": [0.1, 0.15],
    },
}


def build_method_summary(metrics_df: pd.DataFrame, output_prefix: str) -> pd.DataFrame:
    """Per-config summary + best-config-per-method table + bar plot."""
    cfg_summary = (
        metrics_df.groupby(["policy", "config_id"], as_index=False)
        .agg(
            final_regret_mean=("final_regret", "mean"),
            final_regret_std=("final_regret", "std"),
            final_regret_cv=(
                "final_regret",
                lambda s: float(s.std(ddof=0) / s.mean()) if s.mean() > 1e-9 else np.nan,
            ),
            n_repeats=("final_regret", "count"),
        )
        .sort_values(["policy", "final_regret_mean"])
    )
    cfg_summary.to_csv(f"{output_prefix}_by_config.csv", index=False)

    idx = cfg_summary.groupby("policy")["final_regret_mean"].idxmin()
    best = cfg_summary.loc[idx].sort_values("final_regret_mean").reset_index(drop=True)
    best.to_csv(f"{output_prefix}_best_by_method.csv", index=False)
    print(f"[multi] saved {output_prefix}_by_config.csv ({len(cfg_summary)} configs)", flush=True)
    print(f"[multi] saved {output_prefix}_best_by_method.csv", flush=True)
    print(best.to_string(index=False), flush=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(best))
    ax.bar(x, best["final_regret_mean"], yerr=best["final_regret_std"].fillna(0.0),
           capsize=4, color="#4C72B0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n{c}" for p, c in zip(best["policy"], best["config_id"])],
        fontsize=7, rotation=20, ha="right",
    )
    ax.set_ylabel("Final regret (mean ± std)")
    ax.set_title(f"Best config per method — {output_prefix}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_best_by_method.png", dpi=180)
    plt.close(fig)
    print(f"[multi] saved {output_prefix}_best_by_method.png", flush=True)
    return best


def run_multimethod(
    setting: str,
    feature_dim: int = 30,
    n_repeats: int = 3,
    output_prefix: Optional[str] = None,
) -> pd.DataFrame:
    setting = setting.lower()
    if setting not in {"stationary", "shift", "ultra"}:
        raise ValueError("setting must be stationary|shift|ultra")

    if setting == "stationary":
        dgp_name = "stationary"
        dgp = "stationary"
        total_samples, ref_samples, batch_size = 20000, 5000, 25
        shift_magnitude = 0.0
        extra: Dict[str, Any] = {}
    else:
        # shift / ultra → messiest nonstationary DGP
        dgp_name = "nonlinear_messy_ultra_multimethod"
        dgp = "nonlinear_messy_ultra"
        total_samples, ref_samples, batch_size = 15000, 5000, 100
        shift_magnitude = 0.5
        extra = {
            "shift_interval": 2000,
            "mini_shift_interval": 400,
            "covariate_drift_strength": 0.45,
            "shift_point_index": total_samples // 2,
        }

    if output_prefix is None:
        output_prefix = f"mab_multi_{setting}_d{feature_dim}"

    policy_jobs = _expand_policy_grid(MULTI_POLICIES)
    run_config = copy.deepcopy(CONFIG)
    n_arms = run_config["model_pool"]["n_arms"]
    rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []

    print(
        f"[multi] setting={setting} dgp={dgp} d={feature_dim} "
        f"n={total_samples} ref={ref_samples} batch={batch_size} "
        f"policies_jobs={len(policy_jobs)} repeats={n_repeats}",
        flush=True,
    )

    for rep in range(n_repeats):
        data_cfg = dict(run_config["data"])
        data_cfg.update(
            {
                "name": dgp_name,
                "dgp": dgp,
                "feature_dim": feature_dim,
                "total_samples": total_samples,
                "ref_samples": ref_samples,
                "batch_size": batch_size,
                "shift_magnitude": shift_magnitude,
                "noise_scale": float(run_config["data"].get("noise_scale", 2.5)),
                "random_seed": int(run_config["data"]["random_seed"]) + rep * 41,
            }
        )
        data_cfg.update(extra)

        t0 = time.time()
        print(f"[multi] rep={rep + 1}/{n_repeats} generating DGP...", flush=True)
        data = generate_dgp_from_config(data_cfg, n_arms=n_arms)
        cache = precompute_batch_cache(data, run_config)
        print(
            f"[multi] DGP ready in {time.time() - t0:.1f}s "
            f"n_batches={data['n_batches']} shift_batch={data['shift_batch_index']}",
            flush=True,
        )

        for policy_name, policy_cfg in policy_jobs:
            cfg_id = _cfg_to_id(policy_cfg)
            print(f"[multi] rep={rep} policy={policy_name} cfg={cfg_id}", flush=True)
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
                "dgp_name": dgp_name,
                "dgp": dgp,
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
            for key, val in policy_cfg.items():
                row[f"hp_{key}"] = val
            rows.append(row)
            for batch_idx, cum_reg in enumerate(result["cum_regret"]):
                curve_rows.append(
                    {
                        "setting": setting,
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
    build_method_summary(metrics_df, output_prefix)
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-method MAB comparison")
    parser.add_argument("--setting", choices=["stationary", "shift", "ultra"], required=True)
    parser.add_argument("--feature-dim", type=int, default=30)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_multimethod(
        setting=args.setting,
        feature_dim=args.feature_dim,
        n_repeats=args.n_repeats,
        output_prefix=args.output_prefix,
    )
