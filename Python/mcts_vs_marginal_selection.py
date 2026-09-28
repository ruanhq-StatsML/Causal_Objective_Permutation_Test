"""Selection AUC of MCTS attribution against one-shot marginals.

Oracle subset is the neighbor set with the highest true-class logit lift.
A neighbor is a selected feature when it belongs to that subset. Selection
AUC ranks neighbors by a score. The feature-selection effect is the logit
lift of the set each method keeps, next to the oracle lift.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import multistart_mcts_attribution as m

TARGETS = {"Cora": [1708, 1710, 1713], "CiteSeer": [2312, 2316]}


def oracle_subset(value_fn, neighbors: set):
    nodes = list(neighbors)
    best_state, best_value = set(), 0.0
    for size in range(1, len(nodes) + 1):
        for comb in combinations(nodes, size):
            state = set(comb)
            value = value_fn(state)
            if value > best_value:
                best_value = value
                best_state = state
    return best_state, float(best_value)


def solo_scores(value_fn, neighbors: set) -> dict:
    return {node: value_fn({node}) for node in neighbors}


def loco_scores(value_fn, neighbors: set) -> dict:
    full = value_fn(neighbors)
    return {node: full - value_fn(neighbors - {node}) for node in neighbors}


def selection_auc(labels: np.ndarray, scores: np.ndarray):
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def selection_effect(value_fn, chosen: set, oracle: set, oracle_value: float) -> dict:
    value = float(value_fn(chosen))
    if not oracle and not chosen:
        precision, recall = 1.0, 1.0
    elif not chosen:
        precision, recall = 0.0, 0.0
    else:
        hit = len(chosen & oracle)
        precision = hit / len(chosen)
        recall = hit / len(oracle) if oracle else 0.0
    return {
        "size": len(chosen),
        "value": value,
        "gap": value - oracle_value,
        "precision": precision,
        "recall": recall,
    }


def positive_set(scores: dict) -> set:
    return {node for node, score in scores.items() if score > 0}


def evaluate(name: str, data, model, target: int) -> dict:
    m.random.seed(1000 + target)
    np.random.seed(1000 + target)
    torch.manual_seed(1000 + target)
    neighbors = set(m.build_adj(data.edge_index, data.num_nodes)[target]) - {target}
    value_fn = m.ValueFunction(model, data, target)
    oracle, oracle_value = oracle_subset(value_fn, neighbors)
    best_state, best_value, _, scores, visits, _ = m.multi_start_mcts(
        value_fn, neighbors, n_starts=5, n_iterations=80,
    )
    solo = solo_scores(value_fn, neighbors)
    loco = loco_scores(value_fn, neighbors)
    labels = np.array([1 if node in oracle else 0 for node in neighbors])
    order = list(neighbors)
    labels = np.array([1 if node in oracle else 0 for node in order])
    auc = {
        "mcts": selection_auc(labels, np.array([scores[node] for node in order])),
        "solo": selection_auc(labels, np.array([solo[node] for node in order])),
        "loco": selection_auc(labels, np.array([loco[node] for node in order])),
    }
    effect = {
        "mcts": selection_effect(value_fn, best_state, oracle, oracle_value),
        "solo": selection_effect(value_fn, positive_set(solo), oracle, oracle_value),
        "loco": selection_effect(value_fn, positive_set(loco), oracle, oracle_value),
    }
    return {
        "dataset": name,
        "target": int(target),
        "degree": len(neighbors),
        "oracle_size": len(oracle),
        "oracle_value": oracle_value,
        "oracle": sorted(int(node) for node in oracle),
        "mcts_subset": sorted(int(node) for node in best_state),
        "mcts_value": float(best_value),
        "auc": auc,
        "effect": effect,
        "n_cache": len(value_fn.cache),
    }


def main() -> None:
    rows = []
    for name, nodes in TARGETS.items():
        data, in_dim, n_classes = m.load_planetoid(name)
        torch.manual_seed(0)
        model = m.train_gcn(data, in_dim, n_classes)
        for target in nodes:
            row = evaluate(name, data, model, target)
            rows.append(row)
            auc = row["auc"]
            print(
                f"{name} {target} deg {row['degree']} oracle {row['oracle_size']} "
                f"V* {row['oracle_value']:+.3f}  "
                f"AUC mcts {auc['mcts']} solo {auc['solo']} loco {auc['loco']}"
            )
            for key in ("mcts", "solo", "loco"):
                effect = row["effect"][key]
                print(
                    f"  {key:4s} size {effect['size']} value {effect['value']:+.3f} "
                    f"gap {effect['gap']:+.3f} prec {effect['precision']:.2f} "
                    f"recall {effect['recall']:.2f}"
                )
    out = Path("/opt/cursor/artifacts/mcts_selection_auc.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    repo = Path(__file__).resolve().parent / "results" / "mcts_selection_auc.json"
    repo.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("SELECTION_AUC_OK", out)


if __name__ == "__main__":
    main()
