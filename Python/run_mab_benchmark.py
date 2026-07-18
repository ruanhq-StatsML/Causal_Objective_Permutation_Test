"""Generic MAB benchmark scheduler: policy-grid expansion, DGP filtering, parallel runs."""
from __future__ import annotations

import copy
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config_MAB import COMPLEXITY_SCORES, DGP_GRID
from mab_benchmark_core import (
    context_dim,
    generate_dgp_from_config,
    make_policy,
    precompute_batch_cache,
    run_experiment,
)

# Policies that require the context vector in select_arm / update.
_CONTEXT_POLICIES = {
    "LinUCB_Vanilla",
    "LinUCB_Momentum",
}


def _to_py(value: Any) -> Any:
    """Convert numpy scalars to plain Python for stable IDs / pickling."""
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _format_id_value(value: Any) -> str:
    value = _to_py(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _cfg_to_id(policy_cfg: Dict[str, Any]) -> str:
    """Stable config id, e.g. ``epsilon=0.1`` or ``base_epsilon=0.05|gamma=0.9``."""
    if not policy_cfg:
        return "default"
    parts = [
        f"{key}={_format_id_value(policy_cfg[key])}"
        for key in sorted(policy_cfg.keys())
    ]
    return "|".join(parts)


def _expand_policy_grid(
    policies_cfg: Dict[str, Dict[str, Any]],
    policy_names: Optional[Iterable[str]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Expand list-valued hyperparameters into concrete (policy, cfg) jobs."""
    if policy_names is None:
        selected = list(policies_cfg.keys())
    else:
        selected = list(policy_names)
        missing = [name for name in selected if name not in policies_cfg]
        if missing:
            raise ValueError(f"unknown policies: {missing}")

    jobs: List[Tuple[str, Dict[str, Any]]] = []
    for policy_name in selected:
        raw_cfg = dict(policies_cfg.get(policy_name) or {})
        if not raw_cfg:
            jobs.append((policy_name, {}))
            continue

        keys = list(raw_cfg.keys())
        value_lists: List[List[Any]] = []
        for key in keys:
            val = raw_cfg[key]
            if isinstance(val, (list, tuple)):
                value_lists.append([_to_py(v) for v in val])
            else:
                value_lists.append([_to_py(val)])

        for combo in itertools.product(*value_lists):
            concrete = {key: combo[i] for i, key in enumerate(keys)}
            jobs.append((policy_name, concrete))
    return jobs


def _policy_uses_context(policy_name: str) -> bool:
    """Return True when the policy needs the context feature vector."""
    if policy_name in _CONTEXT_POLICIES:
        return True
    return policy_name.startswith("LinUCB")


def _resolve_dgp_grid(
    config_or_dict: Dict[str, Any],
    dgp_names: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Return DGP specs from ``dgp_grid``, optionally filtered by name."""
    grid = list(config_or_dict.get("dgp_grid") or [])
    if not grid:
        grid = copy.deepcopy(DGP_GRID)
    if dgp_names is None:
        return grid
    wanted = list(dgp_names)
    by_name = {spec["name"]: spec for spec in grid}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"unknown dgp names: {missing}")
    return [copy.deepcopy(by_name[name]) for name in wanted]


def _shrink_policies_quick(
    policies_cfg: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Keep only the first value of each list-valued hyperparameter."""
    out: Dict[str, Dict[str, Any]] = {}
    for policy_name, cfg in policies_cfg.items():
        shrunk: Dict[str, Any] = {}
        for key, val in cfg.items():
            if isinstance(val, (list, tuple)) and len(val) > 0:
                shrunk[key] = [val[0]]
            else:
                shrunk[key] = val
        out[policy_name] = shrunk
    return out


def _job_key(dgp_name: str, policy: str, config_id: str, rep: int) -> Tuple[str, str, str, int]:
    return (dgp_name, policy, config_id, int(rep))


def _load_resume_metrics(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame(
            columns=[
                "policy",
                "config_id",
                "final_regret_mean",
                "final_regret_std",
                "final_regret_best",
                "n_repeats",
            ]
        )
    group_cols = ["policy", "config_id"]
    if "dgp_name" in metrics_df.columns and metrics_df["dgp_name"].nunique() > 1:
        group_cols = ["dgp_name", "policy", "config_id"]
    summary = (
        metrics_df.groupby(group_cols)["final_regret"]
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
    return summary


def _run_single_policy(
    policy_name: str,
    policy_cfg: Dict[str, Any],
    data: Dict[str, Any],
    run_config: Dict[str, Any],
    batch_cache: Dict[str, Any],
    dgp_name: str,
    dgp_spec: Dict[str, Any],
    data_cfg_run: Dict[str, Any],
    rep: int,
) -> Dict[str, Any]:
    n_arms = run_config["model_pool"]["n_arms"]
    d_ctx = context_dim(run_config["context"]["n_base_features"])
    cfg_id = _cfg_to_id(policy_cfg)
    policy = make_policy(policy_name, n_arms, d_ctx, dict(policy_cfg))
    result = run_experiment(
        policy,
        data,
        run_config,
        use_context=_policy_uses_context(policy_name),
        batch_cache=batch_cache,
    )
    row: Dict[str, Any] = {
        "dgp_name": dgp_name,
        "dgp_type": dgp_spec.get("dgp", data_cfg_run.get("dgp")),
        "rep": rep,
        "policy": policy_name,
        "config_id": cfg_id,
        "final_regret": result["final_regret"],
        "mean_reward": result["mean_reward"],
        "n_batches": result["n_batches"],
        "shift_batch_index": data.get("shift_batch_index"),
        "feature_dim": data_cfg_run.get("feature_dim"),
        "total_samples": data_cfg_run.get("total_samples"),
        "ref_samples": data_cfg_run.get("ref_samples"),
        "batch_size": data_cfg_run.get("batch_size"),
        "shift_magnitude": data_cfg_run.get("shift_magnitude"),
    }
    for key, val in policy_cfg.items():
        row[f"hp_{key}"] = val
    return row


# Process-pool worker state (set via initializer).
_POOL_STATE: Dict[str, Any] = {}


def _pool_initializer(
    data: Dict[str, Any],
    batch_cache: Dict[str, Any],
    run_config: Dict[str, Any],
    dgp_name: str,
    dgp_spec: Dict[str, Any],
    data_cfg_run: Dict[str, Any],
    rep: int,
) -> None:
    _POOL_STATE["data"] = data
    _POOL_STATE["batch_cache"] = batch_cache
    _POOL_STATE["run_config"] = run_config
    _POOL_STATE["dgp_name"] = dgp_name
    _POOL_STATE["dgp_spec"] = dgp_spec
    _POOL_STATE["data_cfg_run"] = data_cfg_run
    _POOL_STATE["rep"] = rep


def _pool_worker(job: Tuple[str, Dict[str, Any]]) -> Dict[str, Any]:
    policy_name, policy_cfg = job
    return _run_single_policy(
        policy_name,
        policy_cfg,
        _POOL_STATE["data"],
        _POOL_STATE["run_config"],
        _POOL_STATE["batch_cache"],
        _POOL_STATE["dgp_name"],
        _POOL_STATE["dgp_spec"],
        _POOL_STATE["data_cfg_run"],
        _POOL_STATE["rep"],
    )


def run_benchmark(
    config: Dict[str, Any],
    quick: bool = False,
    dgp_names: Optional[Iterable[str]] = None,
    policy_names: Optional[Iterable[str]] = None,
    output_prefix: str = "mab",
    plot: bool = False,
    workers: int = 1,
    resume: bool = False,
    n_repeats: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full policy × DGP grid.

    Returns
    -------
    metrics_df, summary_df
    """
    run_config = copy.deepcopy(config)
    run_config.setdefault("complexity_scores", COMPLEXITY_SCORES)
    if "policies" not in run_config:
        raise ValueError("config must contain 'policies'")
    if quick:
        run_config["policies"] = _shrink_policies_quick(run_config["policies"])

    dgp_grid = _resolve_dgp_grid(run_config, dgp_names=dgp_names)
    policy_jobs = _expand_policy_grid(run_config["policies"], policy_names=policy_names)
    repeats = int(n_repeats if n_repeats is not None else run_config.get("n_repeats", 1))
    repeats = max(1, repeats)
    workers = max(1, int(workers))

    metrics_path = f"{output_prefix}_metrics.csv"
    summary_path = f"{output_prefix}_summary.csv"

    rows: List[Dict[str, Any]] = []
    done_keys = set()
    if resume:
        existing = _load_resume_metrics(metrics_path)
        if len(existing):
            rows.extend(existing.to_dict(orient="records"))
            for _, r in existing.iterrows():
                done_keys.add(
                    _job_key(
                        str(r.get("dgp_name", "")),
                        str(r.get("policy", "")),
                        str(r.get("config_id", "")),
                        int(r.get("rep", 0)),
                    )
                )
            print(
                f"[mab/benchmark] resume: loaded {len(existing)} rows "
                f"({len(done_keys)} completed jobs)",
                flush=True,
            )

    base_data = run_config["data"]
    total_jobs = len(dgp_grid) * repeats * len(policy_jobs)
    print(
        f"[mab/benchmark] dgps={len(dgp_grid)} policies_jobs={len(policy_jobs)} "
        f"repeats={repeats} total={total_jobs} workers={workers} quick={quick}",
        flush=True,
    )

    for dgp_spec in dgp_grid:
        dgp_name = dgp_spec["name"]
        data_cfg = dict(base_data)
        for key, value in dgp_spec.items():
            if key != "name":
                data_cfg[key] = value

        for rep in range(repeats):
            pending = [
                (policy_name, policy_cfg)
                for policy_name, policy_cfg in policy_jobs
                if _job_key(dgp_name, policy_name, _cfg_to_id(policy_cfg), rep)
                not in done_keys
            ]
            if not pending:
                print(
                    f"[mab/benchmark] skip dgp={dgp_name} rep={rep} (all done)",
                    flush=True,
                )
                continue

            data_cfg_run = dict(data_cfg)
            data_cfg_run["random_seed"] = int(base_data.get("random_seed", 2026)) + rep * 17
            print(
                f"[mab/benchmark] DGP={dgp_name} rep={rep + 1}/{repeats} "
                f"pending={len(pending)}/{len(policy_jobs)}",
                flush=True,
            )
            data = generate_dgp_from_config(
                data_cfg_run, n_arms=run_config["model_pool"]["n_arms"]
            )
            batch_cache = precompute_batch_cache(data, run_config)

            new_rows: List[Dict[str, Any]] = []
            if workers == 1 or len(pending) == 1:
                for policy_name, policy_cfg in pending:
                    cfg_id = _cfg_to_id(policy_cfg)
                    print(
                        f"[mab/benchmark] dgp={dgp_name} rep={rep} "
                        f"policy={policy_name} cfg={cfg_id}",
                        flush=True,
                    )
                    new_rows.append(
                        _run_single_policy(
                            policy_name,
                            policy_cfg,
                            data,
                            run_config,
                            batch_cache,
                            dgp_name,
                            dgp_spec,
                            data_cfg_run,
                            rep,
                        )
                    )
            else:
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(pending)),
                    initializer=_pool_initializer,
                    initargs=(
                        data,
                        batch_cache,
                        run_config,
                        dgp_name,
                        dgp_spec,
                        data_cfg_run,
                        rep,
                    ),
                ) as pool:
                    futures = {pool.submit(_pool_worker, job): job for job in pending}
                    for fut in as_completed(futures):
                        policy_name, policy_cfg = futures[fut]
                        cfg_id = _cfg_to_id(policy_cfg)
                        try:
                            row = fut.result()
                        except Exception as exc:
                            print(
                                f"[mab/benchmark] FAILED dgp={dgp_name} rep={rep} "
                                f"policy={policy_name} cfg={cfg_id}: {exc}",
                                flush=True,
                            )
                            raise
                        print(
                            f"[mab/benchmark] done dgp={dgp_name} rep={rep} "
                            f"policy={policy_name} cfg={cfg_id} "
                            f"regret={row['final_regret']:.4f}",
                            flush=True,
                        )
                        new_rows.append(row)

            rows.extend(new_rows)
            for row in new_rows:
                done_keys.add(
                    _job_key(row["dgp_name"], row["policy"], row["config_id"], row["rep"])
                )
            # Checkpoint after each (dgp, rep) block.
            pd.DataFrame(rows).to_csv(metrics_path, index=False)

    metrics_df = pd.DataFrame(rows)
    if len(metrics_df):
        metrics_df.to_csv(metrics_path, index=False)
    summary = _summarize_metrics(metrics_df)
    summary.to_csv(summary_path, index=False)
    print(f"[mab/benchmark] saved {metrics_path} ({len(metrics_df)} rows)", flush=True)
    print(f"[mab/benchmark] saved {summary_path} ({len(summary)} rows)", flush=True)

    if plot and len(metrics_df):
        try:
            from run_mab_prototype_benchmark import plot_prototype_confidence_bands

            # Optional: only works if cum_regret was collected elsewhere.
            cum_path = f"{output_prefix}_cum_regret.csv"
            if os.path.exists(cum_path):
                curves_df = pd.read_csv(cum_path)
                plot_prototype_confidence_bands(curves_df, metrics_df, output_prefix)
            else:
                print(
                    "[mab/benchmark] plot=True but cum_regret CSV missing; skip plots",
                    flush=True,
                )
        except Exception as exc:
            print(f"[mab/benchmark] plot skipped: {exc}", flush=True)

    return metrics_df, summary


if __name__ == "__main__":
    from config_MAB import CONFIG

    cfg = copy.deepcopy(CONFIG)
    cfg["dgp_grid"] = copy.deepcopy(DGP_GRID[:1])
    metrics, summary = run_benchmark(
        cfg,
        quick=True,
        policy_names=["Epsilon_Greedy", "UCB"],
        output_prefix="mab_smoke",
        plot=False,
        workers=1,
        n_repeats=1,
    )
    print(summary.to_string(index=False))
