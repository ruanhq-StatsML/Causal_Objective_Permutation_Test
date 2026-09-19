import os
from benchmark_root import chdir_to_script_dir
chdir_to_script_dir()

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from VIMP_mmd_benchmark import MMD  # noqa: E402


def _rank_positions(vimp):
    order = np.argsort(-np.asarray(vimp, dtype=float))
    rank_pos = np.empty_like(order)
    rank_pos[order] = np.arange(len(vimp))
    return rank_pos
from grf_vimp_causalForest import cf_variable_importance
from LOCO_vimp_r_risk import vimp_loco_r_risk
from permuCATE_vimp import permuCATE_vimp

MODEL_M = "rf_regressor"
MODEL_E = "logistic_classifier"
MODEL_TAU = "rf_regressor"
MODEL_NU = "rf_regressor"
N_PERM_CATE = 20
N_ESTIMATORS = 120


def metrics(selected, shift_index, p, scores=None):
    sel = set(int(i) for i in selected)
    true = set(int(i) for i in shift_index)
    tp = len(sel & true)
    fp = len(sel - true)
    fn = len(true - sel)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    out = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
        "FDR": round(fdr, 4),
    }
    if scores is not None:
        membership = np.zeros(p, dtype=int)
        membership[list(true)] = 1
        sc = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if 0 < membership.sum() < p:
            out["selection_AUC"] = round(float(roc_auc_score(membership, sc)), 4)
    return out


def metrics_from_scores(scores, feature_ind, p):
    k = len(feature_ind)
    selected = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return metrics(selected, feature_ind, p, scores=scores)


def _split_batches(df1, df2):
    p = df1.shape[1] - 1
    df1_X = np.asarray(df1[:, :p], dtype=float)
    df2_X = np.asarray(df2[:, :p], dtype=float)
    df1_Y = np.asarray(df1[:, p], dtype=float).ravel()
    df2_Y = np.asarray(df2[:, p], dtype=float).ravel()
    return df1_X, df1_Y, df2_X, df2_Y, p


def _subsample_batch(X, max_n, rng):
    if X.shape[0] <= max_n:
        return X
    idx = rng.choice(X.shape[0], size=max_n, replace=False)
    return X[idx]


def compute_delta_shap(
    df1_X,
    df1_Y,
    df2_X,
    df2_Y,
    test_size=0.3,
    seed=2047,
):
    rng = np.random.default_rng(seed)
    p = df1_X.shape[1]
    n1 = df1_X.shape[0]
    n2 = df2_X.shape[0]
    rf_kwargs = dict(
        max_depth=max(2, round(p // 3)),
        n_estimators=100,
        min_samples_leaf=max(1, round(np.sqrt(n1 + n2) // 2)),
        n_jobs=1,
        random_state=int(rng.integers(0, 10000)),
    )
    model1 = RandomForestRegressor(**rf_kwargs)
    model2 = RandomForestRegressor(**{**rf_kwargs, "random_state": int(rng.integers(0, 10000))})

    X_train1, X_test1, Y_train1, _ = train_test_split(
        df1_X, df1_Y, test_size=test_size, random_state=seed + 10
    )
    X_train2, X_test2, Y_train2, _ = train_test_split(
        df2_X, df2_Y, test_size=test_size, random_state=seed + 10
    )
    model1.fit(X_train1, Y_train1)
    model2.fit(X_train2, Y_train2)

    explainer1 = shap.Explainer(model1, X_train1)
    shap_value_b1 = explainer1(X_test1, check_additivity=False)
    explainer2 = shap.Explainer(model2, X_train2)
    shap_value_b2 = explainer2(X_test2, check_additivity=False)

    mean_shap_b1 = np.mean(shap_value_b1.values, axis=0)
    mean_shap_b2 = np.mean(shap_value_b2.values, axis=0)
    delta_shap = np.abs(mean_shap_b2 - mean_shap_b1)
    return delta_shap, _rank_positions(delta_shap)


def compute_loco_mmd_batches(df1_X, df2_X, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X_exist = _subsample_batch(np.asarray(df1_X, dtype=float), max_n, rng)
    X_new = _subsample_batch(np.asarray(df2_X, dtype=float), max_n, rng)
    mmd_test = MMD(compute_kernel="rbf")
    mmd_orig, _ = mmd_test(X_exist, X_new)
    p = X_exist.shape[1]
    mmd_dic = np.zeros(p, dtype=float)
    for j in range(p):
        mmd_perm, _ = mmd_test(np.delete(X_exist, j, axis=1), np.delete(X_new, j, axis=1))
        mmd_dic[j] = mmd_perm
    mmd_vimp = mmd_orig - mmd_dic
    return mmd_vimp, _rank_positions(mmd_vimp)


def compute_rf_domain_vimp(df1_X, df2_X, seed=0):
    X = np.vstack([df1_X, df2_X])
    W = np.concatenate([np.zeros(len(df1_X)), np.ones(len(df2_X))]).astype(int)
    p = X.shape[1]
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=max(2, int(np.round(np.sqrt(p)))),
        min_samples_leaf=max(1, int(np.round(np.sqrt(len(X)) // 2))),
        n_jobs=1,
        random_state=seed,
    )
    clf.fit(X, W)
    return clf.feature_importances_, _rank_positions(clf.feature_importances_)


def sgs_shift_benchmark(
    df1_X,
    df1_Y,
    df2_X,
    df2_Y,
    task="regression",
    model_type="random_forest_reg",
    selection_tol=1e-4,
    random_state=24,
    lambdas=None,
    v=3,
    B=20,
    pi_thr=0.5,
):
    import SGShift

    if lambdas is None:
        lambdas = list(np.logspace(-2.5, 0.3, 8))
    solver = SGShift.fit_model(task, model_type, random_state, df1_X, df1_Y)
    beta_mis, delta_mis, _ = SGShift.cross_validate_lambda(
        df2_X, df2_Y, solver, lambdas, misspecified=True, X_S=df1_X, y_S=df1_Y, task=task
    )
    delta_unmis, _ = SGShift.cross_validate_lambda(
        df2_X, df2_Y, solver, lambdas, misspecified=False, X_S=df1_X, y_S=df1_Y, task=task
    )
    scores_l1mis = np.abs(delta_mis)
    scores_l1unmis = np.abs(delta_unmis)
    selected_l1mis = np.where(scores_l1mis > selection_tol)[0]
    selected_l1unmis = np.where(scores_l1unmis > selection_tol)[0]
    pi = SGShift.derandom_knock(
        solver, df1_X, df1_Y, df2_X, df2_Y, B, lambdas, v=v, task=task
    )
    freq = np.max(np.vstack([pi[l] for l in lambdas]), axis=0)
    selected_knock = np.where(freq >= pi_thr)[0]
    return {
        "l1_score_misspec": scores_l1mis,
        "l1_score_spec": scores_l1unmis,
        "knock_score": freq,
        "l1_selected_misspec": selected_l1mis,
        "l1_selected_spec": selected_l1unmis,
        "selected_knock": selected_knock,
    }


def run_all_causal_vimp(X, Y, W, seed=0):
    perm_vimp = permuCATE_vimp(
        X,
        Y,
        W,
        model_m=MODEL_M,
        model_e=MODEL_E,
        model_tau=MODEL_TAU,
        model_nu=MODEL_NU,
        n_perm=N_PERM_CATE,
        seed=seed,
    )
    loco_vimp = vimp_loco_r_risk(X, Y, W, model_m=MODEL_M, model_e=MODEL_E, seed=seed)
    cf_vimp = cf_variable_importance(
        X, Y, W, model_psm="logistic", model_outcome="rf", n_estimators=N_ESTIMATORS, seed=seed
    )
    return {
        "permucate": perm_vimp,
        "loco_r_risk": loco_vimp,
        "grf": cf_vimp,
    }


def benchmark_whole_feature_selection(
    df1,
    df2,
    feature_ind,
    seed=0,
    max_mmd_n=1000,
):
    feature_ind = np.asarray(feature_ind, dtype=int).ravel()
    df1_X, df1_Y, df2_X, df2_Y, p = _split_batches(df1, df2)
    n1, n2 = len(df1_X), len(df2_X)

    scores = {}
    metrics_result = {}

    rf_scores, _ = compute_rf_domain_vimp(df1_X, df2_X, seed=seed)
    scores["rf_domain"] = rf_scores
    metrics_result["rf_domain"] = metrics_from_scores(rf_scores, feature_ind, p)

    delta_shap, _ = compute_delta_shap(df1_X, df1_Y, df2_X, df2_Y, seed=seed)
    scores["delta_shap"] = delta_shap
    metrics_result["delta_shap"] = metrics_from_scores(delta_shap, feature_ind, p)

    mmd_vimp, _ = compute_loco_mmd_batches(df1_X, df2_X, max_n=max_mmd_n, seed=seed)
    scores["loco_mmd"] = mmd_vimp
    metrics_result["loco_mmd"] = metrics_from_scores(mmd_vimp, feature_ind, p)

    X = np.vstack([df1_X, df2_X])
    Y = np.concatenate([df1_Y, df2_Y])
    W = np.concatenate([np.zeros(n1), np.ones(n2)])
    causal = run_all_causal_vimp(X, Y, W, seed=seed + 4)
    for name, vimp in causal.items():
        scores[name] = vimp
        metrics_result[name] = metrics_from_scores(vimp, feature_ind, p)

    try:
        sgs = sgs_shift_benchmark(df1_X, df1_Y, df2_X, df2_Y, random_state=seed + 8)
        scores["l1_misspec"] = sgs["l1_score_misspec"]
        scores["l1_spec"] = sgs["l1_score_spec"]
        scores["knockoff"] = sgs["knock_score"]
        metrics_result["l1_misspec"] = metrics_from_scores(sgs["l1_score_misspec"], feature_ind, p)
        metrics_result["l1_spec"] = metrics_from_scores(sgs["l1_score_spec"], feature_ind, p)
        metrics_result["knockoff"] = metrics_from_scores(sgs["knock_score"], feature_ind, p)
    except Exception as exc:
        print(f"[realdata] SGShift skipped: {exc}", flush=True)

    return {
        "scores": scores,
        "metrics": metrics_result,
        "feature_ind": feature_ind.tolist(),
        "p": int(p),
    }


def benchmark_rows_from_result(
    scenario,
    dataset_id,
    shift_type,
    result,
    rep=0,
):
    rows_scores = []
    rows_metrics = []
    feature_ind = np.asarray(result["feature_ind"], dtype=int)
    shift_set = set(feature_ind.tolist())

    for method, scores in result["scores"].items():
        scores = np.asarray(scores, dtype=float).ravel()
        ranks = _rank_positions(scores)
        for feat_idx, (score, rank) in enumerate(zip(scores, ranks)):
            rows_scores.append(
                {
                    "scenario": scenario,
                    "dataset_id": dataset_id,
                    "shift_type": shift_type,
                    "rep": rep,
                    "method": method,
                    "feature": int(feat_idx),
                    "score": float(score),
                    "rank": int(rank),
                    "is_shift_feature": int(feat_idx in shift_set),
                }
            )
        m = result["metrics"][method]
        row = {
            "scenario": scenario,
            "dataset_id": dataset_id,
            "shift_type": shift_type,
            "rep": rep,
            "method": method,
            "n_shift_features": int(len(feature_ind)),
        }
        row.update(m)
        rows_metrics.append(row)
    return rows_scores, rows_metrics


def load_completed_ids(scores_csv):
    df = _read_csv_if_nonempty(scores_csv)
    if df is None:
        return set()
    df = df[["scenario", "dataset_id", "shift_type", "rep"]]
    return set(map(tuple, df.drop_duplicates().to_numpy()))


def load_completed_method_ids(scores_csv, method):
    if not os.path.exists(scores_csv) or os.path.getsize(scores_csv) == 0:
        return set()
    df = pd.read_csv(scores_csv)
    if "method" not in df.columns:
        return set()
    df = df[df["method"] == method]
    return set(map(tuple, df[["scenario", "dataset_id", "shift_type", "rep"]].drop_duplicates().to_numpy()))


def _read_csv_if_nonempty(path, retries=3):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    for attempt in range(retries):
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return None
        except (TimeoutError, OSError):
            if attempt + 1 >= retries:
                raise
            import time
            time.sleep(2 * (attempt + 1))
    return None


def append_and_save(rows_scores, rows_metrics, scores_csv, metrics_csv):
    score_df = pd.DataFrame(rows_scores)
    metric_df = pd.DataFrame(rows_metrics)
    prev_scores = _read_csv_if_nonempty(scores_csv)
    if prev_scores is not None:
        score_df = pd.concat([prev_scores, score_df], ignore_index=True)
    prev_metrics = _read_csv_if_nonempty(metrics_csv)
    if prev_metrics is not None:
        metric_df = pd.concat([prev_metrics, metric_df], ignore_index=True)
    score_df.to_csv(scores_csv, index=False)
    metric_df.to_csv(metrics_csv, index=False)


def save_json_result(json_path, key, result):
    data = {}
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data[str(key)] = result["metrics"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)