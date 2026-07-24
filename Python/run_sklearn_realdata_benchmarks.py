"""Run sklearn + feature_engine feature-selection benchmarks on real datasets."""
import argparse
import gc
import glob
import itertools
import os
import re

os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")

from benchmark_root import chdir_to_script_dir

chdir_to_script_dir()

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from adversarial_pert_DS import adv_pert
from adversarial_perturbation_distribution_shift_whole import impose_adv_shift
from benchmark_io import load_completed_ids, save_benchmark_result
from sklearn_feature_selection import sklearn_feature_selection
from synthetic_injection_linear import synthetic_injection_linear

REAL_DATA_DIR = "real_data"
PAIR_PATTERN = re.compile(
    r"^(?P<s1>[A-Z]{2})_(?P<s2>[A-Z]{2})_subsample_(?P<n>\d+)_pairdf\.csv$"
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

METHOD_ADV = ["miFGSM", "sini_FGSM", "vmi_FGSM", "rFGSM", "jitter"]
FMOW_PAIRWISE = [
    "space_facility",
    "airport",
    "debris_or_rubble",
    "border_checkpoint",
    "port",
]
EMB_DIR = os.path.join(REAL_DATA_DIR, "yearly_extracted_batch")
EMB_SUFFIX = "sentence_chronoberg_processed_Emb_gemma_embedding2.npy"
START_YEAR_LIST = np.round(np.linspace(1750, 1970, 23)).astype(int)

SUBSAMPLE = 2000
PCA_N = 30
RANDOM_STATE = 2000


def _subsample_rows(X, n, seed):
    rng = np.random.default_rng(seed)
    if X.shape[0] <= n:
        return X
    return X[rng.choice(X.shape[0], n, replace=False)]


def _subsample_pair(df1, df2, seed):
    rng = np.random.default_rng(seed)
    if df1.shape[0] > SUBSAMPLE:
        df1 = df1[rng.choice(df1.shape[0], SUBSAMPLE, replace=False)]
    if df2.shape[0] > SUBSAMPLE:
        df2 = df2[rng.choice(df2.shape[0], SUBSAMPLE, replace=False)]
    return df1, df2


def _subsample_xy(df1_X, df1_Y, df2_X, df2_Y, seed):
    df1 = np.hstack([df1_X, np.asarray(df1_Y, dtype=float).reshape(-1, 1)])
    df2 = np.hstack([df2_X, np.asarray(df2_Y, dtype=float).reshape(-1, 1)])
    return _split_xy(*_subsample_pair(df1, df2, seed))


def _split_xy(df1, df2):
    return (
        df1[:, :-1],
        df1[:, -1].ravel(),
        df2[:, :-1],
        df2[:, -1].ravel(),
    )


def _load_tabular_dataset(path, dataset_idx):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(["Unnamed: 0"], axis=1)
    y_col = "Y" if "Y" in df.columns else df.columns[-1]
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
    return (
        np.asarray(df_part1.drop(columns=["Y"]), dtype=float),
        np.asarray(df_part1["Y"], dtype=float).ravel(),
        np.asarray(df_part2.drop(columns=["Y"]), dtype=float),
        np.asarray(df_part2["Y"], dtype=float).ravel(),
    )


def _list_pair_files():
    files = []
    for path in sorted(glob.glob(os.path.join(REAL_DATA_DIR, "*_pairdf.csv"))):
        name = os.path.basename(path)
        match = PAIR_PATTERN.match(name)
        if not match:
            continue
        files.append((f"{match.group('s1')}_{match.group('s2')}", path))
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
    half = len(y) // 2
    return arr[:half], y[:half], arr[half:], y[half:]


def pca_pair(arr1, arr2, n_components=PCA_N, random_state=RANDOM_STATE, subsample=SUBSAMPLE, seed=0):
    X1 = _subsample_rows(np.asarray(arr1, dtype=float), subsample, seed=seed)
    X2 = _subsample_rows(np.asarray(arr2, dtype=float), subsample, seed=seed + 1)
    n_comp = min(
        n_components,
        X1.shape[0] - 1,
        X1.shape[1],
        X2.shape[0] - 1,
        X2.shape[1],
    )
    n_comp = max(1, int(n_comp))
    pca = PCA(n_components=n_comp, random_state=random_state)
    X1_p = pca.fit_transform(X1)
    X2_p = pca.transform(X2)
    out1 = np.hstack([X1_p, np.arange(X1_p.shape[0]).reshape(-1, 1)])
    out2 = np.hstack([X2_p, np.arange(X2_p.shape[0]).reshape(-1, 1)])
    return out1, out2


def _emb_path(year):
    return os.path.join(EMB_DIR, f"year_{int(year)}_{EMB_SUFFIX}")


def _run_and_save(scenario, dataset_id, shift_type, df1, df2, feature_ind, result_seed, outputs, rep=0):
    result = sklearn_feature_selection(
        df1, df2, feature_ind, regression=True, seed=result_seed
    )
    save_benchmark_result(
        scenario,
        dataset_id,
        shift_type,
        result,
        outputs["scores_csv"],
        outputs["metrics_csv"],
        json_path=outputs.get("json_path"),
        rep=rep,
    )
    return result


def run_tabular_benchmark(shift_type="adv", completed=None):
    outputs = {
        "scores_csv": "sklearn_tabular_benchmark_scores.csv",
        "metrics_csv": "sklearn_tabular_benchmark_metrics.csv",
        "json_path": "sklearn_tabular_benchmark_results.json",
    }
    if completed is None:
        completed = load_completed_ids(outputs["scores_csv"])

    for i, dataset_name in enumerate(DATASET_LIST):
        dataset_id = dataset_name.replace(".csv", "")
        key = ("tabular", dataset_id, shift_type, 0)
        if key in completed:
            print(f"[tabular/{shift_type}] skip {dataset_id}", flush=True)
            continue

        path = os.path.join(REAL_DATA_DIR, dataset_name)
        if not os.path.exists(path):
            print(f"[tabular/{shift_type}] skip {dataset_id} (missing {path})", flush=True)
            continue

        print(f"[tabular/{shift_type}] start {dataset_id}", flush=True)
        df1_X, df1_Y, df2_X, df2_Y = _load_tabular_dataset(path, i)
        df1_X, df1_Y, df2_X, df2_Y = _subsample_xy(df1_X, df1_Y, df2_X, df2_Y, seed=4000 + i)

        if shift_type == "adv":
            X_adv, Y_adv, feature_ind, n1 = adv_pert(
                df1_X, df1_Y, df2_X, df2_Y, n_shift_prop=0.3, n_epoch=4, random_state=5000 + i
            )
            feature_ind = np.asarray(feature_ind, dtype=int).ravel()
            df1 = np.hstack([X_adv[:n1], Y_adv[:n1].reshape(-1, 1)])
            df2 = np.hstack([X_adv[n1:], Y_adv[n1:].reshape(-1, 1)])
        elif shift_type == "linear":
            df1, df2, feature_ind = synthetic_injection_linear(
                df1_X, df1_Y, df2_X, df2_Y, n_prop=0.3, seed=3000 + i
            )
        else:
            raise ValueError(f"unknown shift_type: {shift_type}")

        _run_and_save("tabular", dataset_id, shift_type, df1, df2, feature_ind, 6000 + i, outputs)
        completed.add(key)
        gc.collect()
        print(f"[tabular/{shift_type}] done {dataset_id}", flush=True)


def run_whyshift_benchmark(shift_type="linear", completed=None, pair_files=None):
    outputs = {
        "scores_csv": "whyshift_benchmark_scores_sklearn.csv",
        "metrics_csv": "whyshift_benchmark_metrics_sklearn.csv",
        "json_path": "whyshift_benchmark_results_sklearn.json",
    }
    if completed is None:
        completed = load_completed_ids(outputs["scores_csv"])
    if pair_files is None:
        pair_files = _list_pair_files()

    for pair_idx, (dataset_id, path) in enumerate(pair_files):
        key = ("whyshift", dataset_id, shift_type, 0)
        if key in completed:
            print(f"[whyshift/{shift_type}] skip {dataset_id}", flush=True)
            continue

        print(f"[whyshift/{shift_type}] start {dataset_id}", flush=True)
        try:
            df_X1, df_Y1, df_X2, df_Y2 = _load_pairdf(path)
        except Exception as exc:
            print(f"[whyshift/{shift_type}] skip {dataset_id} (load failed): {exc}", flush=True)
            continue

        df1_X, df1_Y, df2_X, df2_Y = _subsample_xy(df_X1, df_Y1, df_X2, df_Y2, seed=7000 + pair_idx)

        if shift_type == "linear":
            beta1 = np.zeros(df1_X.shape[1])
            beta1[:11] = np.linspace(1.5, 0.5, 11)
            df2_Y = df2_Y + df2_X @ beta1
            df1 = np.hstack([df1_X, df1_Y.reshape(-1, 1)])
            df2 = np.hstack([df2_X, df2_Y.reshape(-1, 1)])
            feature_ind = np.arange(11)
        elif shift_type == "adv":
            X_adv, Y_adv, feature_ind, n1 = adv_pert(
                df1_X, df1_Y, df2_X, df2_Y, n_shift_prop=0.3, n_epoch=4, random_state=8000 + pair_idx
            )
            feature_ind = np.asarray(feature_ind, dtype=int).ravel()
            df1 = np.hstack([X_adv[:n1], Y_adv[:n1].reshape(-1, 1)])
            df2 = np.hstack([X_adv[n1:], Y_adv[n1:].reshape(-1, 1)])
        else:
            raise ValueError(f"unknown shift_type: {shift_type}")

        _run_and_save("whyshift", dataset_id, shift_type, df1, df2, feature_ind, 9000 + pair_idx, outputs)
        completed.add(key)
        gc.collect()
        print(f"[whyshift/{shift_type}] done {dataset_id}", flush=True)


def run_chronoberg_benchmark(completed=None):
    outputs = {
        "scores_csv": "ChronoBerg_benchmark_scores_sklearn.csv",
        "metrics_csv": "ChronoBerg_benchmark_metrics_sklearn.csv",
        "json_path": "ChronoBerg_benchmark_results_sklearn.json",
    }
    if completed is None:
        completed = load_completed_ids(outputs["scores_csv"])

    for pair_idx, start_year in enumerate(START_YEAR_LIST):
        end_year = int(start_year) + 10
        dataset_id = f"{start_year}_{end_year}"
        path1, path2 = _emb_path(start_year), _emb_path(end_year)
        if not os.path.exists(path1) or not os.path.exists(path2):
            print(f"[chronoberg] skip {dataset_id} (missing npy)", flush=True)
            continue

        arr1, arr2 = pca_pair(np.load(path1), np.load(path2), seed=3000 + pair_idx)
        df1_X, df1_Y, df2_X, df2_Y = arr1[:, :-1], arr1[:, -1], arr2[:, :-1], arr2[:, -1]

        for i, method in enumerate(METHOD_ADV):
            key = ("chronoberg", dataset_id, method, 0)
            if key in completed:
                print(f"[chronoberg] skip {dataset_id} {method}", flush=True)
                continue

            print(f"[chronoberg] start {dataset_id} {method}", flush=True)
            X_adv_t, Y_t, feature_ind, n1 = impose_adv_shift(
                df1_X, df1_Y, df2_X, df2_Y, method=method, task="reg"
            )
            df1 = np.hstack([X_adv_t[:n1], Y_t[:n1].reshape(-1, 1)])
            df2 = np.hstack([X_adv_t[n1:], Y_t[n1:].reshape(-1, 1)])
            _run_and_save(
                "chronoberg",
                dataset_id,
                method,
                df1,
                df2,
                feature_ind,
                2000 + pair_idx * 10 + i,
                outputs,
            )
            completed.add(key)
            gc.collect()
            print(f"[chronoberg] done {dataset_id} {method}", flush=True)


def run_fmow_benchmark(completed=None):
    outputs = {
        "scores_csv": "FMoW_benchmark_scores_sklearn.csv",
        "metrics_csv": "FMoW_benchmark_metrics_sklearn.csv",
        "json_path": "FMoW_benchmark_results_sklearn.json",
    }
    if completed is None:
        completed = load_completed_ids(outputs["scores_csv"])

    for grp_idx, (grp1, grp2) in enumerate(itertools.product(FMOW_PAIRWISE, FMOW_PAIRWISE)):
        if grp1 == grp2:
            continue
        dataset_id = f"{grp1}_{grp2}"
        path1 = os.path.join(REAL_DATA_DIR, f"df_{grp1}.csv")
        path2 = os.path.join(REAL_DATA_DIR, f"df_{grp2}.csv")
        if not os.path.exists(path1) or not os.path.exists(path2):
            print(f"[fmow] skip {dataset_id} (missing csv)", flush=True)
            continue

        raw1 = pd.read_csv(path1)
        raw2 = pd.read_csv(path2)
        if "Unnamed: 0" in raw1.columns:
            raw1 = raw1.drop(["Unnamed: 0"], axis=1)
        if "Unnamed: 0" in raw2.columns:
            raw2 = raw2.drop(["Unnamed: 0"], axis=1)
        arr1, arr2 = pca_pair(
            np.asarray(raw1, dtype=float),
            np.asarray(raw2, dtype=float),
            seed=4000 + grp_idx,
        )
        df1_X, df1_Y, df2_X, df2_Y = arr1[:, :-1], arr1[:, -1], arr2[:, :-1], arr2[:, -1]

        for i, method in enumerate(METHOD_ADV):
            key = ("fmow", dataset_id, method, 0)
            if key in completed:
                print(f"[fmow] skip {dataset_id} {method}", flush=True)
                continue

            print(f"[fmow] start {dataset_id} {method}", flush=True)
            X_adv_t, Y_t, feature_ind, n1 = impose_adv_shift(
                df1_X, df1_Y, df2_X, df2_Y, method=method, task="reg"
            )
            df1 = np.hstack([X_adv_t[:n1], Y_t[:n1].reshape(-1, 1)])
            df2 = np.hstack([X_adv_t[n1:], Y_t[n1:].reshape(-1, 1)])
            _run_and_save(
                "fmow",
                dataset_id,
                method,
                df1,
                df2,
                feature_ind,
                5000 + grp_idx * 10 + i,
                outputs,
            )
            completed.add(key)
            gc.collect()
            print(f"[fmow] done {dataset_id} {method}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Sklearn/feature_engine real-data benchmarks")
    parser.add_argument(
        "--benchmark",
        choices=("tabular", "whyshift", "chronoberg", "fmow", "all"),
        default="all",
    )
    parser.add_argument(
        "--shift-type",
        choices=("adv", "linear", "both"),
        default="both",
        help="Used for tabular / whyshift benchmarks",
    )
    args = parser.parse_args()

    if args.benchmark in ("tabular", "all"):
        shift_types = ["adv", "linear"] if args.shift_type == "both" else [args.shift_type]
        for shift_type in shift_types:
            run_tabular_benchmark(shift_type=shift_type)

    if args.benchmark in ("whyshift", "all"):
        shift_types = ["adv", "linear"] if args.shift_type == "both" else [args.shift_type]
        pair_files = _list_pair_files()
        for shift_type in shift_types:
            run_whyshift_benchmark(shift_type=shift_type, pair_files=pair_files)

    if args.benchmark in ("chronoberg", "all"):
        run_chronoberg_benchmark()

    if args.benchmark in ("fmow", "all"):
        run_fmow_benchmark()


if __name__ == "__main__":
    main()
