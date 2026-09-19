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

AUDIO_IDS = [1, 2, 3, 4]
SCORES_CSV = "audio1234_benchmark_scores.csv"
METRICS_CSV = "audio1234_benchmark_metrics.csv"
JSON_PATH = "audio1234_benchmark_results.json"
SCENARIO = "audio"
SHIFT_TYPE = "covariate_shift"
SUBSAMPLE = 2000
N_REPS = 5


def _audio_path(aid):
    return os.path.join("real_data", f"audio{aid}_pca30.npy")


def _subsample_rows(arr, n, seed):
    rng = np.random.default_rng(seed)
    if arr.shape[0] <= n:
        return arr
    idx = rng.choice(arr.shape[0], n, replace=False)
    return arr[idx]


def _load_batch(aid, seed):
    path = _audio_path(aid)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.asarray(np.load(path), dtype=float)
    X = arr[:, :-1]
    Y = np.arange(arr.shape[0], dtype=float)
    stacked = np.hstack([X, Y.reshape(-1, 1)])
    return _subsample_rows(stacked, SUBSAMPLE, seed=seed)


completed = load_completed_ids(SCORES_CSV)
pairs = list(combinations(AUDIO_IDS, 2))
print(
    f"[audio1234 cs] pairs: {len(pairs)}, reps: {N_REPS}, resume keys: {len(completed)}",
    flush=True,
)

for rep in range(N_REPS):
    for pair_idx, (a1, a2) in enumerate(pairs):
        dataset_id = f"audio{a1}_audio{a2}"
        key = (SCENARIO, dataset_id, SHIFT_TYPE, rep)
        if key in completed:
            print(f"[audio1234 cs] skip {dataset_id} rep={rep}", flush=True)
            continue
        if not os.path.exists(_audio_path(a1)) or not os.path.exists(_audio_path(a2)):
            print(f"[audio1234 cs] skip {dataset_id} rep={rep} (missing npy)", flush=True)
            continue

        print(f"[audio1234 cs] start {dataset_id} rep={rep}", flush=True)
        df1 = _load_batch(a1, seed=8000 + pair_idx * 2 + rep * 100)
        df2 = _load_batch(a2, seed=8000 + pair_idx * 2 + 1 + rep * 100)
        p = df1.shape[1] - 1
        feature_ind = np.arange(p)
        results = benchmark_whole_feature_selection(
            df1, df2, feature_ind, seed=9000 + pair_idx + rep * 1000
        )
        s_rows, m_rows = benchmark_rows_from_result(
            SCENARIO, dataset_id, SHIFT_TYPE, results, rep=rep
        )
        append_and_save(s_rows, m_rows, SCORES_CSV, METRICS_CSV)
        save_json_result(JSON_PATH, f"{SCENARIO}::{dataset_id}::{SHIFT_TYPE}::rep{rep}", results)
        completed.add(key)
        print(f"[audio1234 cs] done {dataset_id} rep={rep}", flush=True)

print(
    f"[audio1234 cs] finished. scores -> {SCORES_CSV}, metrics -> {METRICS_CSV}",
    flush=True,
)
