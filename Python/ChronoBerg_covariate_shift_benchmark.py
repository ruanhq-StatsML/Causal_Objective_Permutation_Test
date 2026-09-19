import os

from benchmark_root import chdir_to_script_dir
chdir_to_script_dir()

import numpy as np
from sklearn.decomposition import PCA

from realdata_vimp_benchmark import (
    append_and_save,
    benchmark_rows_from_result,
    benchmark_whole_feature_selection,
    load_completed_ids,
    save_json_result,
)

EMB_DIR = "real_data/yearly_extracted_batch"
EMB_SUFFIX = "sentence_chronoberg_processed_Emb_gemma_embedding2.npy"
SCORES_CSV = "ChronoBerg_cs30_benchmark_scores.csv"
METRICS_CSV = "ChronoBerg_cs30_benchmark_metrics.csv"
JSON_PATH = "ChronoBerg_cs30_benchmark_results.json"
SCENARIO = "chronoberg"
SHIFT_TYPE = "covariate_shift"
SUBSAMPLE = 2000
PCA_N = 30
RANDOM_STATE = 2000
YEAR_GAP = 30


def _available_years():
    years = []
    for name in os.listdir(EMB_DIR):
        if not name.endswith(EMB_SUFFIX) or not name.startswith("year_"):
            continue
        years.append(int(name.split("_")[1]))
    return sorted(set(years))


def _year_pairs(gap=YEAR_GAP, n_pairs=15):
    avail = set(_available_years())
    pairs = [(y, y + gap) for y in sorted(avail) if (y + gap) in avail]
    if not pairs:
        return []
    idx = np.round(np.linspace(0, len(pairs) - 1, min(n_pairs, len(pairs)))).astype(int)
    return [pairs[i] for i in idx]


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


year_pairs = _year_pairs()
completed = load_completed_ids(SCORES_CSV)
print(
    f"[chronoberg cs30] pairs: {len(year_pairs)}, resume keys: {len(completed)}",
    flush=True,
)

for pair_idx, (start_year, end_year) in enumerate(year_pairs):
    dataset_id = f"{start_year}_{end_year}"
    path1, path2 = _emb_path(start_year), _emb_path(end_year)
    key = (SCENARIO, dataset_id, SHIFT_TYPE, 0)
    if key in completed:
        print(f"[chronoberg cs30] skip {dataset_id}", flush=True)
        continue

    print(f"[chronoberg cs30] start {dataset_id}", flush=True)
    df1, df2 = pca_pair(np.load(path1), np.load(path2), seed=4000 + pair_idx)
    p = df1.shape[1] - 1
    feature_ind = np.arange(p)
    results = benchmark_whole_feature_selection(
        df1, df2, feature_ind, seed=5000 + pair_idx
    )
    s_rows, m_rows = benchmark_rows_from_result(
        SCENARIO, dataset_id, SHIFT_TYPE, results, rep=0
    )
    append_and_save(s_rows, m_rows, SCORES_CSV, METRICS_CSV)
    save_json_result(JSON_PATH, f"{SCENARIO}::{dataset_id}::{SHIFT_TYPE}", results)
    completed.add(key)
    print(f"[chronoberg cs30] done {dataset_id}", flush=True)

print(
    f"[chronoberg cs30] finished. scores -> {SCORES_CSV}, metrics -> {METRICS_CSV}",
    flush=True,
)
