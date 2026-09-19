import os

from benchmark_root import chdir_to_script_dir
chdir_to_script_dir()

import numpy as np
from itertools import combinations

from realdata_vimp_benchmark import (
    append_and_save,
    benchmark_rows_from_result,
    benchmark_whole_feature_selection,
    load_completed_ids,
    save_json_result,
)

VIDEO_IDS = [1, 2, 3, 4]
SCORES_CSV = "video1234_benchmark_scores.csv"
METRICS_CSV = "video1234_benchmark_metrics.csv"
JSON_PATH = "video1234_benchmark_results.json"
SCENARIO = "video"
SHIFT_TYPE = "covariate_shift"
SUBSAMPLE = 2000
N_REPS = 5


def _video_path(vid):
    return os.path.join("real_data", f"video{vid}_pca30.npy")


def _subsample_rows(arr, n, seed):
    rng = np.random.default_rng(seed)
    if arr.shape[0] <= n:
        return arr
    idx = rng.choice(arr.shape[0], n, replace=False)
    return arr[idx]


def _load_batch(vid, seed):
    arr = np.asarray(np.load(_video_path(vid)), dtype=float)
    X = arr[:, :-1]
    Y = np.arange(arr.shape[0], dtype=float)
    stacked = np.hstack([X, Y.reshape(-1, 1)])
    return _subsample_rows(stacked, SUBSAMPLE, seed=seed)


completed = load_completed_ids(SCORES_CSV)
pairs = list(combinations(VIDEO_IDS, 2))
print(
    f"[video1234 cs] pairs: {len(pairs)}, reps: {N_REPS}, resume keys: {len(completed)}",
    flush=True,
)

for rep in range(N_REPS):
    for pair_idx, (v1, v2) in enumerate(pairs):
        dataset_id = f"video{v1}_video{v2}"
        key = (SCENARIO, dataset_id, SHIFT_TYPE, rep)
        if key in completed:
            print(f"[video1234 cs] skip {dataset_id} rep={rep}", flush=True)
            continue

        print(f"[video1234 cs] start {dataset_id} rep={rep}", flush=True)
        df1 = _load_batch(v1, seed=6000 + pair_idx * 2 + rep * 100)
        df2 = _load_batch(v2, seed=6000 + pair_idx * 2 + 1 + rep * 100)
        p = df1.shape[1] - 1
        feature_ind = np.arange(p)
        results = benchmark_whole_feature_selection(
            df1, df2, feature_ind, seed=7000 + pair_idx + rep * 1000
        )
        s_rows, m_rows = benchmark_rows_from_result(
            SCENARIO, dataset_id, SHIFT_TYPE, results, rep=rep
        )
        append_and_save(s_rows, m_rows, SCORES_CSV, METRICS_CSV)
        save_json_result(JSON_PATH, f"{SCENARIO}::{dataset_id}::{SHIFT_TYPE}::rep{rep}", results)
        completed.add(key)
        print(f"[video1234 cs] done {dataset_id} rep={rep}", flush=True)

print(
    f"[video1234 cs] finished. scores -> {SCORES_CSV}, metrics -> {METRICS_CSV}",
    flush=True,
)
