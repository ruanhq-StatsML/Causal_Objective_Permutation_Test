"""Shared helpers and parallel benchmark runner for MAB experiments."""
from __future__ import annotations

import copy
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from mab_benchmark_core import (
    context_dim,
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)


def _cfg_to_id(policy_cfg: Dict[str, Any]) -> str:
    if not policy_cfg:
        return "default"
    parts = [f"{k}={policy_cfg[k]}" for k in sorted(policy_cfg)]
    return "|".join(parts)


def _expand_policy_grid(
    policies_cfg: Dict[str, Dict[str, Any]],
    policy_names: Optional[Iterable[str]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    names = policy_names or list(policies_cfg.keys())
    jobs: List[Tuple[str, Dict[str, Any]]] = []
    for name in names:
        if name not in policies_cfg:
            raise ValueError(f"unknown policy: {name}")
        spec = policies_cfg[name]
        list_keys = [k for k, v in spec.items() if isinstance(v, list)]
        if not list_keys:
            jobs.append((name, dict(spec)))
            continue
        value_lists = [spec[k] if isinstance(spec[k], list) else [spec[k]] for k in list_keys]
        for combo in itertools.product(*value_lists):
            cfg = {k: v for k, v in spec.items() if k not in list_keys}
            for key, val in zip(list_keys, combo):
                cfg[key] = val
            jobs.append((name, cfg))
    return jobs


def _policy_uses_context(policy_name: str) -> bool:
    return policy_name.startswith("LinUCB")


def _resolve_dgp_grid(config: Dict[str, Any], dgp_names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    grid = config.get("dgp_grid", [])
    if dgp_names is None:
        return list(grid)
    wanted = set(dgp_names)
    return [g for g in grid if g["name"] in wanted]


def _load_checkpoint(prefix: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = f"{prefix}_metrics.csv"
    curves_path = f"{prefix}_cum_regret.csv"
    metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
    curves = pd.read_csv(curves_path) if os.path.exists(curves_path) else pd.DataFrame()
    return metrics, curves


def _job_key(row: Dict[str, Any]) -> Tuple:
    return (
        row.get("dgp_name"),
        row.get("rep"),
        row.get("policy"),
        row.get("config_id"),
    )


def run_single_job(
    data_cfg: Dict[str, Any],
    run_config: Dict[str, Any],
    dgp_name: str,
    rep: int,
    policy_name: str,
    policy_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cfg_run = copy.deepcopy(run_config)
    data_cfg_run = dict(data_cfg)
    data_cfg_run["random_seed"] = int(data_cfg["random_seed"]) + rep * 7
    n_arms = cfg_run["model_pool"]["n_arms"]
    d_ctx = context_dim(cfg_run["context"]["n_base_features"])
    data = generate_dgp_from_config(data_cfg_run, n_arms=n_arms)
    cache = precompute_batch_cache(data, cfg_run)
    cfg_id = _cfg_to_id(policy_cfg)
    policy = make_policy(policy_name, n_arms, d_ctx, dict(policy_cfg))
    result = run_experiment(
        policy,
        data,
        cfg_run,
        use_context=_policy_uses_context(policy_name),
        batch_cache=cache,
    )
    metric_row = {
        "dgp_name": dgp_name,
        "rep": rep,
        "policy": policy_name,
        "config_id": cfg_id,
        "final_regret": result["final_regret"],
        "mean_reward": result["mean_reward"],
        "n_batches": result["n_batches"],
        "shift_batch_index": data["shift_batch_index"],
        **{f"hp_{k}": v for k, v in policy_cfg.items()},
    }
    curve_rows = []
    for batch_idx, cum_reg in enumerate(result["cum_regret"]):
        curve_rows.append(
            {
                "dgp_name": dgp_name,
                "rep": rep,
                "policy": policy_name,
                "config_id": cfg_id,
                "batch_index": batch_idx,
                "cum_regret": float(cum_reg),
            }
        )
    return [metric_row], curve_rows


def run_benchmark(
    config: Dict[str, Any],
    quick: bool = False,
    dgp_names: Optional[Iterable[str]] = None,
    policy_names: Optional[Iterable[str]] = None,
    output_prefix: str = "mab",
    plot: bool = False,
    workers: int = 1,
    resume: bool = False,
    n_repeats: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    del plot, quick
    run_config = copy.deepcopy(config)
    policies_cfg = run_config["policies"]
    dgp_grid = _resolve_dgp_grid(run_config, dgp_names)
    policy_jobs = _expand_policy_grid(policies_cfg, policy_names=policy_names)
    base_data = run_config["data"]

    metrics_df, curves_df = _load_checkpoint(output_prefix) if resume else (pd.DataFrame(), pd.DataFrame())
    done = set()
    if len(metrics_df):
        for _, r in metrics_df.iterrows():
            done.add(_job_key(r.to_dict()))

    pending = []
    for dgp_spec in dgp_grid:
        dgp_name = dgp_spec["name"]
        data_cfg = dict(base_data)
        for key, value in dgp_spec.items():
            if key != "name":
                data_cfg[key] = value
        for rep in range(n_repeats):
            for policy_name, policy_cfg in policy_jobs:
                cfg_id = _cfg_to_id(policy_cfg)
                key = (dgp_name, rep, policy_name, cfg_id)
                if key in done:
                    continue
                pending.append((data_cfg, run_config, dgp_name, rep, policy_name, policy_cfg))

    print(f"[benchmark] {len(pending)} jobs pending, workers={workers}", flush=True)
    new_metrics: List[Dict[str, Any]] = []
    new_curves: List[Dict[str, Any]] = []

    if workers <= 1:
        for args in pending:
            mrows, crows = run_single_job(*args)
            new_metrics.extend(mrows)
            new_curves.extend(crows)
            print(f"[benchmark] done {args[3]} {args[4]} {_cfg_to_id(args[5])}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_single_job, *args): args for args in pending}
            for fut in as_completed(futures):
                mrows, crows = fut.result()
                new_metrics.extend(mrows)
                new_curves.extend(crows)
                args = futures[fut]
                print(f"[benchmark] done {args[3]} {args[4]} {_cfg_to_id(args[5])}", flush=True)

    if new_metrics:
        metrics_df = pd.concat([metrics_df, pd.DataFrame(new_metrics)], ignore_index=True)
    if new_curves:
        curves_df = pd.concat([curves_df, pd.DataFrame(new_curves)], ignore_index=True)

    metrics_path = f"{output_prefix}_metrics.csv"
    curves_path = f"{output_prefix}_cum_regret.csv"
    summary_path = f"{output_prefix}_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    summary = (
        metrics_df.groupby(["dgp_name", "policy", "config_id"])["final_regret"]
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
    summary.to_csv(summary_path, index=False)
    print(f"[benchmark] saved {metrics_path} ({len(metrics_df)} rows)", flush=True)
    return metrics_df, summary
