#Wilds data benchmark:
import os
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from realdata_vimp_benchmark import *
from sklearn.model_selection import train_test_split
from adversarial_perturbation_distribution_shift_whole import impose_adv_shift

from benchmark_root import chdir_to_script_dir
chdir_to_script_dir()
df1 = pd.read_csv('real_data/wilds_image_saved_train.csv')
df2 = pd.read_csv('real_data/wilds_image_saved_test.csv')
################################################
df1 = df1.drop(['Unnamed: 0'], axis = 1)
df2 = df2.drop(['Unnamed: 0'], axis = 1)

RANDOM_STATE = 2000
SUBSAMPLE = 2000
df2_SUB = np.array(df2)[np.random.randint(0, df2.shape[0], SUBSAMPLE), :]

df1_X = PCA(n_components = 50, random_state = RANDOM_STATE).fit_transform(df1)
df2_X = PCA(n_components = 50, random_state = RANDOM_STATE).fit_transform(df2_SUB)
df1_Y = np.arange(df1_X.shape[0])
df2_Y = np.arange(df2_X.shape[0])
#Then conduct the PCA on two batches of data separately:

METHOD_ADV = [
'miFGSM', 'sini_FGSM', 
'vmi_FGSM', 'rFGSM', 'jitter'
]

SCORES_CSV = "wilds_vimp_scores.csv"
METRICS_CSV = "wilds_vimp_metrics.csv"
JSON_PATH = "wilds_vimp_results.json"
SCENARIO = "wilds"
DATASET_ID = "wilds_image"

completed = load_completed_ids(SCORES_CSV)
print(f"[wilds] resume keys: {len(completed)}", flush=True)

for i, method in enumerate(METHOD_ADV):
    key = (SCENARIO, DATASET_ID, method, 0)
    if key in completed:
        print(f"[wilds adv] skip {method}", flush=True)
        continue
    print(f"[wilds adv] start {method}", flush=True)
    X_adv_t, Y_t, feature_ind, n1 = impose_adv_shift(df1_X, 
        df1_Y, df2_X, df2_Y, method = method, task = 'reg')
    df1 = np.hstack([X_adv_t[:n1], Y_t[:n1].reshape(-1, 1)])
    df2 = np.hstack([X_adv_t[n1:], Y_t[n1:].reshape(-1, 1)])
    results = benchmark_whole_feature_selection(df1, 
        df2, feature_ind, seed = 2000 + i)
    s_rows, m_rows = benchmark_rows_from_result(
        SCENARIO, DATASET_ID, method, results, rep=0
    )
    append_and_save(s_rows, m_rows, SCORES_CSV, METRICS_CSV)
    save_json_result(JSON_PATH, f"{SCENARIO}::{DATASET_ID}::{method}", results)
    completed.add(key)
    print(f"[wilds adv] done {method} -> {METRICS_CSV}", flush=True)

print(f"[wilds] finished. scores -> {SCORES_CSV}, metrics -> {METRICS_CSV}", flush=True)

    

































