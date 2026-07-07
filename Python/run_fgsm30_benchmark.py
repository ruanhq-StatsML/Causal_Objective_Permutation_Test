"""FGSM adversarial shift benchmark: 30% feature shift on 8 tabular datasets."""
import gc
import json
import os

os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")

from benchmark_root import chdir_to_script_dir
chdir_to_script_dir()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from FGSM_adversarial import adv_pert_reg
from realdata_vimp_benchmark import (
    append_and_save,
    benchmark_rows_from_result,
    benchmark_whole_feature_selection,
    load_completed_ids,
)

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
SUBSAMPLE_N = 1000
SCORES_CSV = "fgsm30_vimp_scores.csv"
METRICS_CSV = "fgsm30_vimp_metrics.csv"
JSON_PATH = "fgsm30_benchmark_results.json"
SHIFT_TYPE = "fgsm30"
N_SHIFT_PROP = 0.3
N_EPOCH = 4
EPSILON = 0.15


def _load_tabular_dataset(path, dataset_idx):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)
    y_col = "Y" if "Y" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != y_col]
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "string":
            X[col], _ = pd.factorize(X[col], sort=False)
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
        df1 = df1[rng.choice(df1.shape[0], SUBSAMPLE_N, replace=False)]
    if df2.shape[0] > SUBSAMPLE_N:
        df2 = df2[rng.choice(df2.shape[0], SUBSAMPLE_N, replace=False)]
    return df1, df2


def save_json_result(json_path, key, result):
    data = {}
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[str(key)] = result["metrics"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_fgsm30_benchmark(completed=None):
    if completed is None:
        completed = load_completed_ids(SCORES_CSV)

    for i, dataset_name in enumerate(DATASET_LIST):
        dataset_id = dataset_name.replace(".csv", "")
        key = ("tabular", dataset_id, SHIFT_TYPE, 0)
        if key in completed:
            print(f"[fgsm30] skip {dataset_id}", flush=True)
            continue

        print(f"[fgsm30] start {dataset_id}", flush=True)
        df1_X, df1_Y, df2_X, df2_Y = _load_tabular_dataset(dataset_name, i)

        X_adv, Y_adv, feature_ind, n1 = adv_pert_reg(
            df1_X,
            df1_Y,
            df2_X,
            df2_Y,
            n_shift_prop=N_SHIFT_PROP,
            n_epoch=N_EPOCH,
            epsilon=EPSILON,
            random_state=7000 + i,
        )
        feature_ind = np.asarray(feature_ind, dtype=int).ravel()
        df1 = np.hstack([X_adv[:n1, :], Y_adv[:n1].reshape(-1, 1)])
        df2 = np.hstack([X_adv[n1:, :], Y_adv[n1:].reshape(-1, 1)])
        df1, df2 = _subsample_pair(df1, df2, seed=8000 + i)

        result = benchmark_whole_feature_selection(
            df1, df2, feature_ind=feature_ind, seed=9000 + i
        )
        s_rows, m_rows = benchmark_rows_from_result(
            "tabular", dataset_id, SHIFT_TYPE, result, rep=0
        )
        append_and_save(s_rows, m_rows, SCORES_CSV, METRICS_CSV)
        save_json_result(JSON_PATH, f"tabular::{dataset_id}::{SHIFT_TYPE}", result)
        completed.add(key)
        gc.collect()
        print(f"[fgsm30] done {dataset_id} ({len(completed)}/8)", flush=True)


if __name__ == "__main__":
    done = load_completed_ids(SCORES_CSV)
    print(f"[fgsm30] resume: {len(done)}/8 datasets already done", flush=True)
    run_fgsm30_benchmark(completed=done)
    print(f"[fgsm30] finished -> {JSON_PATH}", flush=True)
