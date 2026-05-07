#DRPerm packages:


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import KFold, StratifiedKFold
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional



@dataclass
class ModelSpec:
    name: str
    fit: Callable
    predict: Callable


def default_model_config():
    return{
      'global':{
        'positive_class': 1,
        'n_jobs': -1
      },
      'rf_regression':{
        'n_estimators': 200,
        'max_features': 'sqrt',
        'min_samples_leaf': 10
      },
      'rf_classification':{
        "n_estimators": 200,
        "max_features": 'sqrt',
        'min_samples_leaf': 1
      },
      'lm_regression': {},
      'logistic_classification': {
        'max_iter': 1000,
        'penalty': 'l2',
        'C': 1.0
      },
      'ridge_regression':{
        'alpha': 0.01
      },
      'mlp_regression':{
        'hidden_layer_sizes': (5, ),
        'alpha': 1e-5,
        'max_iter': 250
      },
      'mlp_classification':{
        'hidden_layer_sizes': (5, ),
        'alpha': 1e-5,
        'max_iter': 250
      },
      'xgb_regression':{
        'n_estimators': 150,
        'max_depth': 4,
        'gamma': 0.5,
        'learning_rate': 0.1,
        'n_jobs': 1
      },
      'xgb_classification':{
        'n_estimators': 150,
        'max_depth': 4,
        'gamma': 0.5,
        'leaerning_rate': 0.1,
        'n_jobs': 1,
        'eval_metrics': 'logloss'
      },
    }


def make_rf_regression(params, global_param):
    def fit(X, y, seed = None):
        model = RandomForestRegressor(
            **params,
            random_state = seed,
            n_jobs = global_param.get('n_jobs', -1)
        )
        model.fit(as_array(X), as_vector(y))


























