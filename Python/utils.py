import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import kl_div
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler




"""
#R-learner weighted regression:
"""
def _fit_tau_rlearner_weighted(X, Y_tilde, W_tilde, seed: int = 0, clip_wtilde: float = 1e-3):
    X = np.asarray(X)
    Y_tilde = np.asarray(Y_tilde).reshape(-1)
    W_tilde = np.asarray(W_tilde).reshape(-1)
    mask = (np.abs(W_tilde) > clip_wtilde)
    z = Y_tilde[mask]/W_tilde[mask]
    weights = (W_tilde[mask] ** 2)
    tau_model = Pipeline(
      steps = [
      ('scaler', StandardScaler(with_mean = True, with_std = True)),
      ('ridge', Ridge(alpha = 1.0, random_state = seed))])
    tau_model.fit(X[mask], z, ridge__sample_weight = weights)
    return tau_model

'''
Making the folds here via the array_split:
'''
def make_folds(n, n_folds = 5, seed = 2026):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    return folds


'''
Cross-fitting for the outcome model and the propensity score model:
'''
def cross_nuisance_fit(X, Y, W, n_folds = 5,
  n_estimators = 100, binary_outcome = True,
  model_registry = None,
  model_m = 'rf_regressor', model_e = 'rf_classifier',
  clip_e = 1e-3, seed = 2026):
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    folds = make_folds(n, n_folds = n_folds, seed = seed)
    mu_hat = np.zeros(n, dtype = float)
    e_hat = np.zeros(n, dtype = float)
    if model_registry is None:
        from model_registry import ModelRegistry as default_registry
        model_registry = default_registry
    for k, test_idx in enumerate(make_folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]
        W_train = W[train_idx]
        model_propensity_score = model_registry[model_e]
        model_outcome = model_registry[model_m]
        #Fit the outcome model and propensity score models:
        fit_mu = model_outcome['fit'](
          X_train, Y_train, seed = seed + k
          )
        mu_hat[test_idx] = model_outcome['predict'](
          fit_obj = fit_mu, X_new = X_test
          )
        fit_e = model_propensity_score['fit'](
          X_train, W_train, seed = seed + k * 3
          )
        e_hat[test_idx] = model_outcome['predict'](
          fit_obj = fit_e, X_new = X_test
          )
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    return mu_hat, e_hat



    