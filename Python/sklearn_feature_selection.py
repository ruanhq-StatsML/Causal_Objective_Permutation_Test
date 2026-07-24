"""Sklearn + feature_engine feature-selection benchmark helpers."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFwe,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso

from benchmark_io import metrics_from_scores
from feature_engine.selection import MRMR


def _split_batches(df1, df2):
    p = df1.shape[1] - 1
    df1_X = np.asarray(df1[:, :p], dtype=float)
    df2_X = np.asarray(df2[:, :p], dtype=float)
    df1_Y = np.asarray(df1[:, p], dtype=float).ravel()
    df2_Y = np.asarray(df2[:, p], dtype=float).ravel()
    return df1_X, df1_Y, df2_X, df2_Y, p


def _to_score_vector(values, value_type, p):
    if value_type == "indices":
        score_vec = np.zeros(p, dtype=float)
        score_vec[np.asarray(values, dtype=int)] = 1.0
        return score_vec
    if value_type in ("scores", "ranking", "model"):
        arr = np.asarray(values, dtype=float).ravel()
        if value_type == "ranking":
            return -arr
        return arr
    raise ValueError(f"Unknown value_type: {value_type}")


def sklearn_feature_selection(df1, df2, feature_ind, regression=False, seed=0):
    """
    Benchmark classical sklearn / feature_engine selectors on pooled data.

    Returns the same dict shape as benchmark_whole_feature_selection:
    {"scores", "metrics", "feature_ind", "p", "timing"}.
    """
    df1_X, df1_Y, df2_X, df2_Y, p = _split_batches(df1, df2)
    X = np.vstack([df1_X, df2_X])
    Y = np.concatenate([df1_Y, df2_Y])
    feature_ind = np.asarray(feature_ind, dtype=int).ravel()
    k = max(1, min(len(feature_ind), p))

    if regression:
        score_func = f_regression
        mi_score_func = mutual_info_regression
        base_model = RandomForestRegressor(
            max_features="sqrt",
            n_estimators=150,
            min_samples_leaf=int(np.round(np.sqrt(X.shape[0]) // 2)),
            random_state=seed,
        )
        lasso = Lasso(alpha=0.1, random_state=seed)
    else:
        score_func = f_classif
        mi_score_func = mutual_info_classif
        base_model = RandomForestClassifier(
            max_features="sqrt",
            n_estimators=150,
            min_samples_leaf=int(np.round(np.sqrt(X.shape[0]) // 2)),
            random_state=seed,
        )
        lasso = Lasso(alpha=0.1, random_state=seed)

    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    Y_series = pd.Series(Y.ravel())

    selectors = {}
    timing = {}

    start = time.perf_counter()
    kbest_f = SelectKBest(score_func=score_func, k=k).fit(X, Y)
    selectors["SelectKBest_F"] = (kbest_f.scores_, "scores")
    timing["SelectKBest_F"] = time.perf_counter() - start

    start = time.perf_counter()
    kbest_mi = SelectKBest(score_func=mi_score_func, k=k).fit(X, Y)
    selectors["SelectKBest_MI"] = (kbest_mi.scores_, "scores")
    timing["SelectKBest_MI"] = time.perf_counter() - start

    start = time.perf_counter()
    percentile_sel = SelectPercentile(score_func=score_func, percentile=20).fit(X, Y)
    selectors["SelectPercentile"] = (percentile_sel.scores_, "scores")
    timing["SelectPercentile"] = time.perf_counter() - start

    start = time.perf_counter()
    fwe_sel = SelectFwe(score_func=score_func, alpha=0.05).fit(X, Y)
    selectors["SelectFwe"] = (fwe_sel.scores_, "scores")
    timing["SelectFwe"] = time.perf_counter() - start

    start = time.perf_counter()
    rfe = RFE(estimator=base_model, n_features_to_select=k).fit(X, Y)
    selectors["RFE_RF"] = (rfe.ranking_, "ranking")
    timing["RFE_RF"] = time.perf_counter() - start

    start = time.perf_counter()
    lasso_sel = SelectFromModel(estimator=lasso, max_features=k, threshold=-np.inf).fit(X, Y)
    selectors["SelectFromModel_L1"] = (np.abs(lasso_sel.estimator_.coef_), "scores")
    timing["SelectFromModel_L1"] = time.perf_counter() - start

    start = time.perf_counter()
    rf_sel = SelectFromModel(estimator=base_model, max_features=k, threshold=-np.inf).fit(X, Y)
    selectors["SelectFromModel_RF"] = (rf_sel.estimator_.feature_importances_, "model")
    timing["SelectFromModel_RF"] = time.perf_counter() - start

    start = time.perf_counter()
    mrmr_fcq = MRMR(method="FCQ", regression=regression).fit(X_df, Y_series)
    selectors["MRMR_FCQ"] = (mrmr_fcq.relevance_, "scores")
    timing["MRMR_FCQ"] = time.perf_counter() - start

    start = time.perf_counter()
    mrmr_miq = MRMR(method="MIQ", regression=regression).fit(X_df, Y_series)
    selectors["MRMR_MIQ"] = (mrmr_miq.relevance_, "scores")
    timing["MRMR_MIQ"] = time.perf_counter() - start

    scores = {}
    metrics_result = {}
    for name, (values, value_type) in selectors.items():
        score_vec = _to_score_vector(values, value_type, p)
        scores[name] = score_vec
        metrics_result[name] = metrics_from_scores(score_vec, feature_ind, p)

    return {
        "scores": scores,
        "metrics": metrics_result,
        "feature_ind": feature_ind.tolist(),
        "p": int(p),
        "timing": timing,
    }
