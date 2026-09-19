import gc
import glob
import json
import os
import re

os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")

from benchmark_root import chdir_to_script_dir
chdir_to_script_dir()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from adversarial_pert_DS import adv_pert
from realdata_vimp_benchmark import (
    append_and_save,
    benchmark_rows_from_result,
    benchmark_whole_feature_selection,
    load_completed_ids,
    save_json_result,
)
from synthetic_injection_linear import synthetic_injection_linear

DATASET_LIST = [
    "df_adult.csv",
    "df_house16H.csv",
    "df_elevators.csv",
    "df_pol.csv",
    "df_cpu.csv",
    "df_kin.csv",
    "df_Ail.csv",
    "df_bank32nh.csv",
]

SCORES_CSV = "realdata_vimp_scores.csv"
METRICS_CSV = "realdata_vimp_metrics.csv"
JSON_PATH = "realdata_vimp_results.json"
REAL_DATA_DIR = "real_data"
PAIR_PATTERN = re.compile(
    r"^(?P<s1>[A-Z]{2})_(?P<s2>[A-Z]{2})_subsample_(?P<n>\d+)_pairdf\.csv$"
)
SUBSAMPLE_N = 2000


def _load_tabular_dataset(path, dataset_idx):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)
    if "Y" in df.columns:
        y_col = "Y"
    else:
        y_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != y_col]
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "string":
            codes, _ = pd.factorize(X[col], sort=False)
            X[col] = codes
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    Y = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0)
    data = pd.concat([X, Y.rename("Y")], axis=1)
    df_part1, df_part2 = train_test_split(
        data, test_size=0.5, random_state=2000 + dataset_idx
    )
    df1_X = np.asarray(df_part1.drop(columns=["Y"]), dtype=float)
    df2_X = np.asarray(df_part2.drop(columns=["Y"]), dtype=float)
    df1_Y = np.asarray(df_part1["Y"], dtype=float).ravel()
    df2_Y = np.asarray(df_part2["Y"], dtype=float).ravel()
    return df1_X, df1_Y, df2_X, df2_Y


def _subsample_pair(df1, df2, seed):
    rng = np.random.default_rng(seed)
    if df1.shape[0] > SUBSAMPLE_N:
        idx1 = rng.choice(df1.shape[0], SUBSAMPLE_N, replace=False)
        df1 = df1[idx1]
    if df2.shape[0] > SUBSAMPLE_N:
        idx2 = rng.choice(df2.shape[0], SUBSAMPLE_N, replace=False)
        df2 = df2[idx2]
    return df1, df2


def _subsample_xy(df1_X, df1_Y, df2_X, df2_Y, seed):
    df1 = np.hstack([df1_X, np.asarray(df1_Y, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([df2_X, np.asarray(df2_Y, dtype=float).reshape(-1, 1)])
    df1, df2 = _subsample_pair(df1, df2, seed=seed)
    return (
        df1[:, :-1],
        df1[:, -1].ravel(),
        df2[:, :-1],
        df2[:, -1].ravel(),
    )


def _list_pair_files():
    files = []
    for path in sorted(glob.glob(os.path.join(REAL_DATA_DIR, "*_pairdf.csv"))):
        name = os.path.basename(path)
        m = PAIR_PATTERN.match(name)
        if not m:
            continue
        dataset_id = f"{m.group('s1')}_{m.group('s2')}"
        files.append((dataset_id, path))
    return files


def _load_pairdf(path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)
    y_col = "Y" if "Y" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != y_col]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    Y = pd.to_numeric(df[y_col], errors="coerce").fillna(0.0)
    arr = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float).ravel()
    n = len(y)
    half = n // 2
    return arr[:half], y[:half], arr[half:], y[half:]


def run_tabular_benchmark(shift_type="adv", completed=None):
    if completed is None:
        completed = load_completed_ids(SCORES_CSV)
    rows_scores = []
    rows_metrics = []

    for i, dataset_name in enumerate(DATASET_LIST):
        dataset_id = dataset_name.replace(".csv", "")
        key = ("tabular", dataset_id, shift_type, 0)
        if key in completed:
            print(f"[tabular {shift_type}] skip {dataset_id}", flush=True)
            continue

        print(f"[tabular {shift_type}] start {dataset_id}", flush=True)
        df1_X, df1_Y, df2_X, df2_Y = _load_tabular_dataset(dataset_name, i)
        df1_X, df1_Y, df2_X, df2_Y = _subsample_xy(
            df1_X, df1_Y, df2_X, df2_Y, seed=4000 + i
        )
        print(
            f"[tabular {shift_type}] subsampled n1={len(df1_Y)} n2={len(df2_Y)}",
            flush=True,
        )

        if shift_type == "adv":
            X_adv, Y_adv, feature_ind, n1 = adv_pert(
                df1_X, df1_Y, df2_X, df2_Y, n_shift_prop=0.3, n_epoch=4
            )
            feature_ind = np.asarray(feature_ind, dtype=int).ravel()
            df1 = np.hstack([X_adv[:n1, :], Y_adv[:n1].reshape(-1, 1)])
            df2 = np.hstack([X_adv[n1:, :], Y_adv[n1:].reshape(-1, 1)])
        elif shift_type == "linear":
            df1, df2, feature_ind = synthetic_injection_linear(
                df1_X, df1_Y, df2_X, df2_Y, n_prop=0.3, seed=3000 + i
            )
        else:
            raise ValueError(f"unknown shift_type: {shift_type}")

        result = benchmark_whole_feature_selection(
            df1, df2, feature_ind=feature_ind, seed=5000 + i
        )
        s_rows, m_rows = benchmark_rows_from_result(
            "tabular", dataset_id, shift_type, result, rep=0
        )
        rows_scores.extend(s_rows)
        rows_metrics.extend(m_rows)
        append_and_save(rows_scores, rows_metrics, SCORES_CSV, METRICS_CSV)
        save_json_result(JSON_PATH, f"tabular::{dataset_id}::{shift_type}", result)
        rows_scores.clear()
        rows_metrics.clear()
        completed.add(key)
        gc.collect()
        print(f"[tabular {shift_type}] done {dataset_id}", flush=True)


def run_whyshift_linear_benchmark(completed=None, pair_files=None):
    if completed is None:
        completed = load_completed_ids(SCORES_CSV)
    if pair_files is None:
        pair_files = _list_pair_files()
    rows_scores = []
    rows_metrics = []
    seed_counter = 0

    for pair_idx, (dataset_id, path) in enumerate(pair_files):
        key = ("whyshift", dataset_id, "linear", 0)
        if key in completed:
            print(f"[whyshift linear] skip {dataset_id}", flush=True)
            continue

        print(f"[whyshift linear] start {dataset_id} ({path})", flush=True)
        try:
            df_X1, df_Y1, df_X2, df_Y2 = _load_pairdf(path)
        except Exception as exc:
            print(
                f"[whyshift linear] skip {dataset_id} (load failed): {exc}",
                flush=True,
            )
            continue

        beta1 = np.zeros(df_X1.shape[1])
        beta1[:12] = np.linspace(1.5, 0.4, 12)
        df_Y2 = df_Y2 + df_X2 @ beta1
        df_1 = np.hstack([df_X1, df_Y1.reshape(-1, 1)])
        df_2 = np.hstack([df_X2, df_Y2.reshape(-1, 1)])

        df1, df2 = _subsample_pair(df_1, df_2, seed=7000 + pair_idx)
        print(
            f"[whyshift linear] subsampled n1={len(df1)} n2={len(df2)}",
            flush=True,
        )

        feature_ind = np.arange(12)
        try:
            result = benchmark_whole_feature_selection(
                df1, df2, feature_ind=feature_ind, seed=6000 + pair_idx
            )
        except Exception as exc:
            print(
                f"[whyshift linear] skip {dataset_id} (benchmark failed): {exc}",
                flush=True,
            )
            continue
        s_rows, m_rows = benchmark_rows_from_result(
            "whyshift", dataset_id, "linear", result, rep=0
        )
        rows_scores.extend(s_rows)
        rows_metrics.extend(m_rows)
        append_and_save(rows_scores, rows_metrics, SCORES_CSV, METRICS_CSV)
        save_json_result(JSON_PATH, f"whyshift::{dataset_id}::linear", result)
        rows_scores.clear()
        rows_metrics.clear()
        completed.add(key)
        gc.collect()
        print(f"[whyshift linear] done {dataset_id}", flush=True)


def run_whyshift_adv_benchmark(completed=None, pair_files=None):
    if completed is None:
        completed = load_completed_ids(SCORES_CSV)
    if pair_files is None:
        pair_files = _list_pair_files()
    rows_scores = []
    rows_metrics = []

    for pair_idx, (dataset_id, path) in enumerate(pair_files):
        key = ("whyshift", dataset_id, "adv", 0)
        if key in completed:
            print(f"[whyshift adv] skip {dataset_id}", flush=True)
            continue

        print(f"[whyshift adv] start {dataset_id} ({path})", flush=True)
        try:
            df1_X, df1_Y, df2_X, df2_Y = _load_pairdf(path)
        except Exception as exc:
            print(
                f"[whyshift adv] skip {dataset_id} (load failed): {exc}",
                flush=True,
            )
            continue

        df1_X, df1_Y, df2_X, df2_Y = _subsample_xy(
            df1_X, df1_Y, df2_X, df2_Y, seed=8000 + pair_idx
        )
        print(
            f"[whyshift adv] subsampled n1={len(df1_Y)} n2={len(df2_Y)}",
            flush=True,
        )

        try:
            X_adv, Y_adv, feature_ind, n1 = adv_pert(
                df1_X,
                df1_Y,
                df2_X,
                df2_Y,
                n_shift_prop=0.3,
                n_epoch=4,
                random_state=9000 + pair_idx,
            )
            feature_ind = np.asarray(feature_ind, dtype=int).ravel()
            df1 = np.hstack([X_adv[:n1, :], Y_adv[:n1].reshape(-1, 1)])
            df2 = np.hstack([X_adv[n1:, :], Y_adv[n1:].reshape(-1, 1)])
            result = benchmark_whole_feature_selection(
                df1, df2, feature_ind=feature_ind, seed=10000 + pair_idx
            )
        except Exception as exc:
            print(
                f"[whyshift adv] skip {dataset_id} (benchmark failed): {exc}",
                flush=True,
            )
            continue
        s_rows, m_rows = benchmark_rows_from_result(
            "whyshift", dataset_id, "adv", result, rep=0
        )
        rows_scores.extend(s_rows)
        rows_metrics.extend(m_rows)
        append_and_save(rows_scores, rows_metrics, SCORES_CSV, METRICS_CSV)
        save_json_result(JSON_PATH, f"whyshift::{dataset_id}::adv", result)
        rows_scores.clear()
        rows_metrics.clear()
        completed.add(key)
        gc.collect()
        print(f"[whyshift adv] done {dataset_id}", flush=True)


if __name__ == "__main__":
    import sys

    whyshift_only = "--whyshift-only" in sys.argv
    whyshift_adv_only = "--whyshift-adv-only" in sys.argv
    completed = load_completed_ids(SCORES_CSV)
    pair_files = _list_pair_files()
    print(
        f"[realdata] resume keys: {len(completed)}, pairdf files: {len(pair_files)}",
        flush=True,
    )
    if not whyshift_only and not whyshift_adv_only:
        run_tabular_benchmark("adv", completed=completed)
        run_tabular_benchmark("linear", completed=completed)
    if not whyshift_adv_only:
        run_whyshift_linear_benchmark(completed=completed, pair_files=pair_files)
    run_whyshift_adv_benchmark(completed=completed, pair_files=pair_files)
    print(f"[realdata] finished. scores -> {SCORES_CSV}, metrics -> {METRICS_CSV}", flush=True)
