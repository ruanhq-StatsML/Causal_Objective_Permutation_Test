#ChronoBerg benchmark here:
import os

from benchmark_root import chdir_to_script_dir
chdir_to_script_dir()

import numpy as np
from sklearn.decomposition import PCA

from adversarial_perturbation_distribution_shift_whole import impose_adv_shift
from realdata_vimp_benchmark import (
    append_and_save,
    benchmark_rows_from_result,
    benchmark_whole_feature_selection,
    load_completed_ids,
    save_json_result,
)

EMB_DIR = "real_data/yearly_extracted_batch"
EMB_SUFFIX = "sentence_chronoberg_processed_Emb_gemma_embedding2.npy"
START_YEAR_LIST = np.round(np.linspace(1750, 1970, 23)).astype(int)

METHOD_ADV = [
    "miFGSM",
    "sini_FGSM",
    "vmi_FGSM",
    "rFGSM",
    "jitter",
]

SCORES_CSV = "ChronoBerg_benchmark_scores.csv"
METRICS_CSV = "ChronoBerg_benchmark_metrics.csv"
JSON_PATH = "ChronoBerg_benchmark_results.json"
SCENARIO = "chronoberg"
SUBSAMPLE = 2000
PCA_N = 30
RANDOM_STATE = 2000


def _subsample_rows(X, n, seed):
    rng = np.random.default_rng(seed)
    if X.shape[0] <= n:
        return X
    return X[rng.choice(X.shape[0], n, replace=False)]


def pca_pair(arr1, arr2, n_components=PCA_N, random_state=RANDOM_STATE, subsample=SUBSAMPLE, seed=0):
    X1 = np.asarray(arr1, dtype=float)
    X2 = np.asarray(arr2, dtype=float)
    X1 = _subsample_rows(X1, subsample, seed=seed)
    X2 = _subsample_rows(X2, subsample, seed=seed + 1)
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


completed = load_completed_ids(SCORES_CSV)
total = len(START_YEAR_LIST) * len(METHOD_ADV)
print(f"[chronoberg] resume keys: {len(completed)}, jobs: {total}", flush=True)

for pair_idx, start_year in enumerate(START_YEAR_LIST):
    end_year = int(start_year) + 10
    dataset_id = f"{start_year}_{end_year}"
    path1, path2 = _emb_path(start_year), _emb_path(end_year)
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"[chronoberg] skip {dataset_id} (missing npy)", flush=True)
        continue

    print(f"[chronoberg] load pair {dataset_id}", flush=True)
    arr1, arr2 = pca_pair(
        np.load(path1),
        np.load(path2),
        seed=3000 + pair_idx,
    )
    df1_X = arr1[:, :-1]
    df2_X = arr2[:, :-1]
    df1_Y = arr1[:, -1].ravel()
    df2_Y = arr2[:, -1].ravel()

    for i, method in enumerate(METHOD_ADV):
        key = (SCENARIO, dataset_id, method, 0)
        if key in completed:
            print(f"[chronoberg adv] skip {dataset_id} {method}", flush=True)
            continue
        print(f"[chronoberg adv] start {dataset_id} {method}", flush=True)
        X_adv_t, Y_t, feature_ind, n1 = impose_adv_shift(
            df1_X, df1_Y, df2_X, df2_Y, method=method, task="reg"
        )
        df1 = np.hstack([X_adv_t[:n1], Y_t[:n1].reshape(-1, 1)])
        df2 = np.hstack([X_adv_t[n1:], Y_t[n1:].reshape(-1, 1)])
        results = benchmark_whole_feature_selection(
            df1, df2, feature_ind, seed=2000 + pair_idx * 10 + i
        )
        s_rows, m_rows = benchmark_rows_from_result(
            SCENARIO, dataset_id, method, results, rep=0
        )
        append_and_save(s_rows, m_rows, SCORES_CSV, METRICS_CSV)
        save_json_result(JSON_PATH, f"{SCENARIO}::{dataset_id}::{method}", results)
        completed.add(key)
        print(f"[chronoberg adv] done {dataset_id} {method}", flush=True)

print(f"[chronoberg] finished. scores -> {SCORES_CSV}, metrics -> {METRICS_CSV}", flush=True)
