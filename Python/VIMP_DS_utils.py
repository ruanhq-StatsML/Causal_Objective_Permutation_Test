import numpy as np
import pandas as pd 
import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

def _rng_seed(seed):
    return None if seed is None else int(seed)

def _as_1d(y):
    return np.asarray(y, dtype = float).ravel()

def _as_2d(X):
    Xa = np.asarray(X, dtype = float)
    if Xa.ndim == 1:
        return Xa.reshape(-1, 1) #(n, ) -> (n, 1)
    return Xa

def _as_1d_float(y):
    return np.asarray(y, dtype = float).ravel()

def _as_2d_float(X):
    Xa = np.asarray(X, dtype = float)
    if Xa.ndim == 1:
        return Xa.reshape(-1, 1) #(n, ) -> (n, 1)
    return Xa

def _standardize_cols(X):
    cen = np.mean(X, axis = 0)
    sc = np.std(X, axis = 0, ddof = 0)
    sc = np.where(sc <= 1e-12, 1.0, sc)
    return (X - cen)/sc, cen, sc

def _safe_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = float(np.sum(v))
    return v / s if np.isfinite(s) and s > 0 else np.zeros_like(v)

def make_folds(n, n_folds = 5, seed = 1):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, n_folds)

def L2(X, Y):
    return ((np.array(Y) - np.array(X)) ** 2)

def multiva_L2(X, Y):
    return ((np.array(Y) - np.array(X)) ** 2).sum(axis = 1)

def bidir(df_exist, df_new, method = 'stouffer'):
    p_value_for = RFPerm(df_exist, df_new, method = method)
    p_value_bac = RFPerm(df_exist, df_new, method = method)
    p_value_bidir = combine_pvalues([p_value_for, p_value_bac], method = method)
    return p_value_bidir

def infer_response_type(Y):
    """
    Infer the types of response in Y:
    Y should either be Numeric or Object
    Return one of the "binary", "continuous" for the downstream modeling procedure.
    """
    if Y is None:
        raise ValueError("Y must not be None")
    Y_ = np.array(Y).reshape(-1)
    if Y_.size == 0:
        raise ValueError("Y must contain at least one observation")
    if pd.isna(Y_).any():
        raise ValueError("Y contains missing values. Please remove or impute them first.")
    dtype = Y_.dtype
    #Checking whether it's object or numeric, otherwise, ValueError
    is_numeric = np.issubdtype(dtype, np.number)
    is_object_like = dtype == object or np.issubdtype(dtype, np.str_)
    if not (is_numeric or is_object_like):
        raise TypeError(
            'Y must be numeric or categorical/object-like.'
            f"Got dtype={dtype}."
        )
    len_class = np.unique(Y_).size
    if len_class == 2:
        return 'binary'
    elif len_class > 6:
        return 'continuous'
    else:
        return 'multinomial'
    