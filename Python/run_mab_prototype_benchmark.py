"""MAB prototype: complex DGPs, dual-drift + adaptive cooling, epsilon vs adaptive."""
from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_MAB import (
    CONFIG,
    PROTOTYPE_COOLING_PRESETS,
    PROTOTYPE_DGP_GRID,
    PROTOTYPE_EPSILON_CALIB_POLICIES,
    PROTOTYPE_EPSILON_SWEEP_POLICIES,
    PROTOTYPE_EXTENDED_POLICIES,
    PROTOTYPE_LARGE_POLICIES,
    PROTOTYPE_POLICIES,
)
from run_mab_benchmark import (
    _cfg_to_id,
    _expand_policy_grid,
    _policy_uses_context,
    _resolve_dgp_grid,
)
from mab_benchmark_core import (
    context_dim,
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)


def _best_config_ids(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Best hyperparam config per (cooling, policy) by mean final_regret."""
    agg = (
        metrics_df.groupby(["cooling_preset", "policy", "config_id"], as_index=False)["final_regret"]
        .mean()
        .rename(columns={"final_regret": "final_regret_mean"})
    )
    idx = agg.groupby(["cooling_preset", "policy"])["final_regret_mean"].idxmin()
    return agg.loc[idx].reset_index(drop=True)


def build_confidence_band_table(
    curves_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config_ids: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if config_ids is None:
        config_ids = _best_config_ids(metrics_df)
    rows: List[Dict[str, Any]] = []
    for _, row in config_ids.iterrows():
        sub = curves_df[
            (curves_df.cooling_preset == row["cooling_preset"])
            & (curves_df.policy == row["policy"])
            & (curves_df.config_id == row["config_id"])
        ]
        if sub.empty:
            continue
        stats = (
            sub.groupby("batch_index")["cum_regret"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        stats["std"] = stats["std"].fillna(0.0)
        for _, s in stats.iterrows():
            rows.append(
                {
                    "cooling_preset": row["cooling_preset"],
                    "policy": row["policy"],
                    "config_id": row["config_id"],
                    "batch_index": int(s["batch_index"]),
                    "cum_regret_mean": float(s["mean"]),
                    "cum_regret_std": float(s["std"]),
                    "cum_regret_lower": float(s["mean"] - s["std"]),
                    "cum_regret_upper": float(s["mean"] + s["std"]),
                    "n_repeats": int(s["count"]),
                }
            )
    return pd.DataFrame(rows)


def plot_prototype_confidence_bands(
    curves_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_prefix: str,
    shift_batch_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Plot mean ± 1 std cumulative regret bands; save band table CSV."""
    band_df = build_confidence_band_table(curves_df, metrics_df)
    band_path = f"{output_prefix}_confidence_bands.csv"
    band_df.to_csv(band_path, index=False)
    all_cfg_ids = metrics_df[["cooling_preset", "policy", "config_id"]].drop_duplicates()
    all_band_df = build_confidence_band_table(curves_df, metrics_df, config_ids=all_cfg_ids)
    all_band_path = f"{output_prefix}_confidence_bands_all_configs.csv"
    all_band_df.to_csv(all_band_path, index=False)

    policy_colors = {
        "Epsilon_Greedy": "#E08214",
        "AdaptiveEpsilonGreedy": "#1F77B4",
        "UCB": "#2CA02C",
        "Random": "#7F7F7F",
        "LinUCB_Vanilla": "#D62728",
        "LinUCB_Momentum": "#8C564B",
        "Thompson_Sampling": "#17BECF",
        "Gaussian_Sampling": "#BCBD22",
    }
    # Distinct linestyle cycle so multiple configs of the same policy stay readable.
    style_cycle = ["-", "--", "-.", ":"]
    eps_colors = {
        0.01: "#FDBB84",
        0.02: "#FDB863",
        0.03: "#FDAE6B",
        0.05: "#E08214",
        0.08: "#D62728",
        0.10: "#9467BD",
        0.15: "#8C6BB1",
        0.20: "#4A0080",
        0.30: "#1A0030",
    }

    for cooling in sorted(all_band_df["cooling_preset"].unique()):
        # --- Figure 1: ALL methods / ALL configs (primary) ---
        sub_all = all_band_df[all_band_df.cooling_preset == cooling]
        fig, ax = plt.subplots(figsize=(14, 8))
        # Stable order: policy then config_id
        keys = (
            sub_all[["policy", "config_id"]]
            .drop_duplicates()
            .sort_values(["policy", "config_id"])
        )
        for _, key in keys.iterrows():
            policy = key["policy"]
            cfg_id = key["config_id"]
            psub = sub_all[
                (sub_all.policy == policy) & (sub_all.config_id == cfg_id)
            ].sort_values("batch_index")
            if psub.empty:
                continue
            # Index among this policy's configs → linestyle
            cfg_list = keys.loc[keys.policy == policy, "config_id"].tolist()
            ls = style_cycle[cfg_list.index(cfg_id) % len(style_cycle)]
            x = psub["batch_index"].to_numpy()
            mean = psub["cum_regret_mean"].to_numpy()
            std = psub["cum_regret_std"].to_numpy()
            color = policy_colors.get(policy, None)
            label = f"{policy} | {cfg_id}"
            ax.plot(x, mean, label=label, color=color, linewidth=1.8, linestyle=ls, alpha=0.95)
            ax.fill_between(x, mean - std, mean + std, color=color or "gray", alpha=0.10)
        if shift_batch_indices:
            for sb in shift_batch_indices:
                ax.axvline(x=sb, color="red", linestyle=":", alpha=0.35, linewidth=1.2)
        ax.set_xlabel("Batch index")
        ax.set_ylabel("Cumulative regret")
        ax.set_title(f"All methods / all configs — {cooling} (mean ± 1 std)")
        ax.legend(fontsize=6.5, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(f"{output_prefix}_{cooling}_all_methods_bands.png", dpi=180)
        plt.close(fig)

        # Keep a best-config overlay for quick glance (secondary).
        sub_best = band_df[band_df.cooling_preset == cooling]
        if len(sub_best):
            fig, ax = plt.subplots(figsize=(12, 7))
            for policy in sorted(sub_best["policy"].unique()):
                psub = sub_best[sub_best.policy == policy].sort_values("batch_index")
                x = psub["batch_index"].to_numpy()
                mean = psub["cum_regret_mean"].to_numpy()
                std = psub["cum_regret_std"].to_numpy()
                label = f"{policy} ({psub['config_id'].iloc[0]})"
                ax.plot(x, mean, label=label, color=policy_colors.get(policy, None), linewidth=2.2)
                ax.fill_between(
                    x, mean - std, mean + std, color=policy_colors.get(policy, "gray"), alpha=0.18
                )
            if shift_batch_indices:
                for sb in shift_batch_indices:
                    ax.axvline(x=sb, color="red", linestyle=":", alpha=0.35, linewidth=1.2)
            ax.set_xlabel("Batch index")
            ax.set_ylabel("Cumulative regret")
            ax.set_title(f"Best config per method — {cooling} (mean ± 1 std)")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(f"{output_prefix}_{cooling}_best_methods_bands.png", dpi=180)
            plt.close(fig)

        # --- Per-policy panels: every config of that method ---
        for policy in sorted(metrics_df["policy"].unique()):
            pol_cfg = metrics_df.loc[
                (metrics_df.policy == policy) & (metrics_df.cooling_preset == cooling),
                ["cooling_preset", "policy", "config_id"],
            ].drop_duplicates()
            if pol_cfg.empty:
                # cooling_preset may be absent on older metrics; fall back
                pol_cfg = metrics_df.loc[
                    metrics_df.policy == policy,
                    ["cooling_preset", "policy", "config_id"],
                ].drop_duplicates()
            if pol_cfg.empty:
                continue
            pol_band = all_band_df[
                (all_band_df.cooling_preset == cooling) & (all_band_df.policy == policy)
            ]
            if pol_band.empty:
                continue
            fig, ax = plt.subplots(figsize=(13, 7))
            for i, cfg_id in enumerate(sorted(pol_band["config_id"].unique())):
                csub = pol_band[pol_band.config_id == cfg_id].sort_values("batch_index")
                x = csub["batch_index"].to_numpy()
                mean = csub["cum_regret_mean"].to_numpy()
                std = csub["cum_regret_std"].to_numpy()
                if policy == "Epsilon_Greedy" and "epsilon=" in cfg_id:
                    eps_val = float(cfg_id.split("=")[-1])
                    color = eps_colors.get(round(eps_val, 2), policy_colors.get(policy, "gray"))
                    label = f"ε={eps_val:g}"
                else:
                    color = policy_colors.get(policy, None)
                    label = str(cfg_id).replace("|", " ")
                ax.plot(
                    x, mean, label=label, color=color,
                    linewidth=2.0, linestyle=style_cycle[i % len(style_cycle)], alpha=0.95,
                )
                ax.fill_between(x, mean - std, mean + std, color=color or "gray", alpha=0.12)
            if shift_batch_indices:
                for j, sb in enumerate(shift_batch_indices):
                    ax.axvline(
                        x=sb, color="red", linestyle="--", alpha=0.45, linewidth=1.5,
                        label="Drift" if j == 0 else None,
                    )
            ax.set_xlabel("Batch index")
            ax.set_ylabel("Cumulative regret")
            ax.set_title(f"{policy} — all configs — {cooling} (mean ± 1 std)")
            ax.legend(fontsize=8, loc="upper left", ncol=2)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            safe_name = policy.replace("/", "_")
            fig.savefig(f"{output_prefix}_{cooling}_{safe_name}_bands.png", dpi=180)
            plt.close(fig)

    print(
        f"[mab/prototype] saved {band_path} ({len(band_df)} rows, best config per method)",
        flush=True,
    )
    print(
        f"[mab/prototype] saved {all_band_path} ({len(all_band_df)} rows, all configs)",
        flush=True,
    )
    print(
        f"[mab/prototype] also wrote all-methods and per-policy band plots under {output_prefix}_*",
        flush=True,
    )
    return band_df


def _resolve_policies_cfg(
    preset: str | None = None,
    epsilon_grid: Iterable[float] | None = None,
) -> Dict[str, Any]:
    if preset == "epsilon_sweep":
        policies_cfg = copy.deepcopy(PROTOTYPE_EPSILON_SWEEP_POLICIES)
    elif preset == "extended":
        policies_cfg = copy.deepcopy(PROTOTYPE_EXTENDED_POLICIES)
    elif preset == "large":
        policies_cfg = copy.deepcopy(PROTOTYPE_LARGE_POLICIES)
    else:
        policies_cfg = copy.deepcopy(PROTOTYPE_POLICIES)
    if epsilon_grid is not None:
        if "Epsilon_Greedy" not in policies_cfg:
            policies_cfg["Epsilon_Greedy"] = {}
        policies_cfg["Epsilon_Greedy"]["epsilon"] = [float(x) for x in epsilon_grid]
    return policies_cfg


def run_prototype_benchmark(
    config: Dict[str, Any],
    n_repeats: int = 5,
    dgp_names: Iterable[str] | None = None,
    policy_names: Iterable[str] | None = None,
    cooling_names: Iterable[str] | None = None,
    output_prefix: str = "mab_prototype",
    plot: bool = True,
    policies_cfg: Dict[str, Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    run_config = copy.deepcopy(config)
    run_config["policies"] = policies_cfg or copy.deepcopy(PROTOTYPE_POLICIES)
    dgp_grid = _resolve_dgp_grid({"dgp_grid": PROTOTYPE_DGP_GRID}, dgp_names)
    policy_jobs = _expand_policy_grid(
        run_config["policies"], policy_names=policy_names
    )
    if cooling_names is None:
        cooling_items = list(PROTOTYPE_COOLING_PRESETS.items())
    else:
        names = set(cooling_names)
        cooling_items = [
            (k, v) for k, v in PROTOTYPE_COOLING_PRESETS.items() if k in names
        ]
        missing = names - {k for k, _ in cooling_items}
        if missing:
            raise ValueError(f"unknown cooling presets: {sorted(missing)}")
    d_ctx = context_dim(run_config["context"]["n_base_features"])
    n_arms = run_config["model_pool"]["n_arms"]
    base_data = run_config["data"]

    rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    last_shift_batches: List[int] = []

    for dgp_spec in dgp_grid:
        dgp_name = dgp_spec["name"]
        data_cfg = dict(base_data)
        for key, value in dgp_spec.items():
            if key != "name":
                data_cfg[key] = value

        for rep in range(n_repeats):
            data_cfg_run = dict(data_cfg)
            data_cfg_run["random_seed"] = int(base_data["random_seed"]) + rep * 7
            print(
                f"[mab/prototype] DGP={dgp_name} rep={rep + 1}/{n_repeats}",
                flush=True,
            )
            data = generate_dgp_from_config(data_cfg_run, n_arms=n_arms)
            last_shift_batches = list(data.get("shift_batch_indices", [data["shift_batch_index"]]))

            for cooling_name, cooling_overrides in cooling_items:
                cfg_run = copy.deepcopy(run_config)
                cfg_run["context"].update(cooling_overrides)
                cache = precompute_batch_cache(data, cfg_run)

                for policy_name, policy_cfg in policy_jobs:
                    cfg_id = _cfg_to_id(policy_cfg)
                    print(
                        f"[mab/prototype] dgp={dgp_name} cooling={cooling_name} "
                        f"rep={rep} policy={policy_name} cfg={cfg_id}",
                        flush=True,
                    )
                    policy = make_policy(policy_name, n_arms, d_ctx, dict(policy_cfg))
                    result = run_experiment(
                        policy,
                        data,
                        cfg_run,
                        use_context=_policy_uses_context(policy_name),
                        batch_cache=cache,
                    )
                    rows.append(
                        {
                            "dgp_name": dgp_name,
                            "cooling_preset": cooling_name,
                            "dgp_type": data_cfg.get("dgp"),
                            "rep": rep,
                            "shift_interval": data_cfg.get("shift_interval"),
                            "shift_magnitude": data_cfg.get("shift_magnitude"),
                            "total_samples": data_cfg_run["total_samples"],
                            "n_batches": result["n_batches"],
                            "shift_batch_index": data["shift_batch_index"],
                            "policy": policy_name,
                            "config_id": cfg_id,
                            "final_regret": result["final_regret"],
                            "mean_reward": result["mean_reward"],
                            **{f"hp_{k}": v for k, v in policy_cfg.items()},
                            **{f"ctx_{k}": v for k, v in cooling_overrides.items()},
                        }
                    )
                    for batch_idx, cum_reg in enumerate(result["cum_regret"]):
                        curve_rows.append(
                            {
                                "dgp_name": dgp_name,
                                "cooling_preset": cooling_name,
                                "rep": rep,
                                "policy": policy_name,
                                "config_id": cfg_id,
                                "batch_index": batch_idx,
                                "cum_regret": float(cum_reg),
                                "drift_delta": float(result["drift_deltas"][batch_idx]),
                            }
                        )

    metrics_df = pd.DataFrame(rows)
    curves_df = pd.DataFrame(curve_rows)
    metrics_path = f"{output_prefix}_metrics.csv"
    curves_path = f"{output_prefix}_cum_regret.csv"
    summary_path = f"{output_prefix}_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    summary = (
        metrics_df.groupby(["dgp_name", "cooling_preset", "policy", "config_id"])["final_regret"]
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
        .sort_values(["dgp_name", "final_regret_mean"])
    )
    summary.to_csv(summary_path, index=False)
    print(f"[mab/prototype] saved {metrics_path} ({len(metrics_df)} rows)", flush=True)
    print(f"[mab/prototype] saved {curves_path} ({len(curves_df)} rows)", flush=True)
    print(f"[mab/prototype] saved {summary_path}", flush=True)
    if plot and len(curves_df):
        plot_prototype_confidence_bands(
            curves_df,
            metrics_df,
            output_prefix,
            shift_batch_indices=last_shift_batches,
        )
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAB prototype benchmark (complex DGPs)")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--dgp", nargs="+", default=None)
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument(
        "--cooling",
        nargs="+",
        default=None,
        help="cooling presets: cool_default cool_slow cool_fast",
    )
    parser.add_argument("--output-prefix", default="mab_prototype")
    parser.add_argument(
        "--policies-preset",
        choices=["default", "epsilon_sweep", "extended", "large"],
        default="default",
        help="policy grid: epsilon_sweep, extended, large (11 eps + 144 adaptive + UCB/Random)",
    )
    parser.add_argument(
        "--epsilon-grid",
        nargs="+",
        type=float,
        default=None,
        help="override epsilon values, e.g. 0.01 0.02 0.05 ...",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="plot confidence bands from existing metrics/cum_regret CSVs",
    )
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    prefix = args.output_prefix
    if args.plot_only:
        curves_df = pd.read_csv(f"{prefix}_cum_regret.csv")
        metrics_df = pd.read_csv(f"{prefix}_metrics.csv")
        plot_prototype_confidence_bands(curves_df, metrics_df, prefix)
    else:
        policies_cfg = _resolve_policies_cfg(args.policies_preset, args.epsilon_grid)
        df = run_prototype_benchmark(
            CONFIG,
            n_repeats=args.n_repeats,
            dgp_names=args.dgp,
            policy_names=args.policies,
            cooling_names=args.cooling,
            output_prefix=prefix,
            plot=not args.no_plot,
            policies_cfg=policies_cfg,
        )
        print(
            df.groupby(["dgp_name", "policy"])["final_regret"]
            .mean()
            .sort_values()
            .to_string()
        )
