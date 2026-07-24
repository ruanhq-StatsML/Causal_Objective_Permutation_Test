"""CSV/JSON persistence helpers for real-data feature-selection benchmarks."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _rank_positions(vimp):
    order = np.argsort(-np.asarray(vimp, dtype=float))
    rank_pos = np.empty_like(order)
    rank_pos[order] = np.arange(len(vimp))
    return rank_pos


def metrics(selected, shift_index, p, scores=None):
    sel = set(int(i) for i in selected)
    true = set(int(i) for i in shift_index)
    tp = len(sel & true)
    fp = len(sel - true)
    fn = len(true - sel)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
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
        sc = np.nan_to_num(
            np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        if 0 < membership.sum() < p:
            out["selection_AUC"] = round(float(roc_auc_score(membership, sc)), 4)
    return out


def metrics_from_scores(scores, feature_ind, p):
    k = len(feature_ind)
    selected = np.argsort(-np.asarray(scores, dtype=float))[:k]
    return metrics(selected, feature_ind, p, scores=scores)


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
            time.sleep(2 * (attempt + 1))
    return None


def benchmark_rows_from_result(scenario, dataset_id, shift_type, result, rep=0):
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
        row = {
            "scenario": scenario,
            "dataset_id": dataset_id,
            "shift_type": shift_type,
            "rep": rep,
            "method": method,
            "n_shift_features": int(len(feature_ind)),
        }
        row.update(result["metrics"][method])
        rows_metrics.append(row)
    return rows_scores, rows_metrics


def load_completed_ids(scores_csv):
    df = _read_csv_if_nonempty(scores_csv)
    if df is None:
        return set()
    return set(
        map(tuple, df[["scenario", "dataset_id", "shift_type", "rep"]].drop_duplicates().to_numpy())
    )


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


def save_benchmark_result(
    scenario,
    dataset_id,
    shift_type,
    result,
    scores_csv,
    metrics_csv,
    json_path=None,
    rep=0,
):
    s_rows, m_rows = benchmark_rows_from_result(
        scenario, dataset_id, shift_type, result, rep=rep
    )
    append_and_save(s_rows, m_rows, scores_csv, metrics_csv)
    if json_path is not None:
        save_json_result(json_path, f"{scenario}::{dataset_id}::{shift_type}", result)
    return s_rows, m_rows
