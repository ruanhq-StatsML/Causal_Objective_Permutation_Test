#Time profiling for DRPerm:
import os
os.chdir('/Users/heqiaoruan/Documents/GitHub/Causal_Objective_Permutation_Test/Python')
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import inspect 
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional, Union 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from utils import *
from model_registry_class import *
import time
sample_size_seq = [100, 200, 300, 500]

def make_folds(n, n_folds = 5, seed = 2026):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    return folds

time_comparison = pd.DataFrame({
'sample_size': sample_size_seq, 
'time_mean': [0] * len(sample_size_seq),
'time_sd': [0] * len(sample_size_seq)
})
model_registry = default_model_registry(
  ntree = 150,
  ridge_alpha = 0.25,
  nthread = 1, maxit = 200, max_depth = 5,
  gamma = 0.25, eta = 0.15, mlp_hidden_size = 4,
  mlp_decay = 1e-5, mlp_max_iter = 500, mlp_trace = False,
  mlp_max_coef_reg = 10000, mlp_max_coef_clf = 10000,
  warn_xgb_labels = True, positive_class = 1
)
for i, sample_size in enumerate(sample_size_seq):
    time_profile = []
    for k in range(10):
        rng = np.random.default_rng(2023+k)
        n = sample_size
        p = 20
        X = rng.normal(size = (n, p))
        Y = 10.0 * X[:, 0] + 1.25 * X[:, 1] + 0.25 * X[:, 2] + rng.normal(scale = 1.0, size = n)
        W = rng.binomial(1, 0.5, size = n)
        start = time.perf_counter()
        output = DRPerm(X, Y, W, n_perm = 150, model_e = 'logistic_classifier', model_registry = model_registry)    
        end = time.perf_counter()
        time_profile.append((end - start))#in seconds here!
    time_comparison['time_mean'][i] = np.mean(time_profile)
    time_comparison['time_sd'][i] = np.std(time_profile)
    time_comparison.to_csv('drperm_time_profile_sample_size.csv')























