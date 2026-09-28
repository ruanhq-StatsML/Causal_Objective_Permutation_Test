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


def best_visited(root, value_fn):
    best_state, best_value = set(root.state), float(value_fn(root.state))

    def walk(node):
        nonlocal best_state, best_value
        value = float(value_fn(node.state))
        if value > best_value:
            best_value = value
            best_state = set(node.state)
        for child in node.children:
            walk(child)

    walk(root)
    return best_state, best_value


def search_from(value_fn, neighbors: set, start: set, seed: int):
    m.random.seed(seed)
    np.random.seed(seed)
    best_state, best_value = set(start), float(value_fn(start))
    for trial in range(5):
        m.random.seed(seed + trial)
        root = m.mcts_search(value_fn, neighbors, n_iterations=80, start=start)
        state, value = best_visited(root, value_fn)
        if value > best_value:
            best_value = value
            best_state = state
    return best_state, best_value


def start_comparison(name: str, data, model, target: int, solo: dict) -> dict:
    m.random.seed(1000 + target)
    np.random.seed(1000 + target)
    neighbors = set(m.build_adj(data.edge_index, data.num_nodes)[target]) - {target}
    value_fn = m.ValueFunction(model, data, target)
    oracle, oracle_value = oracle_subset(value_fn, neighbors)
    starts = {
        "full": set(neighbors),
        "empty": set(),
        "solo_positive": {node for node, score in solo.items() if score > 0},
    }
    found = {}
    for label, start in starts.items():
        state, value = search_from(value_fn, neighbors, start, seed=3000 + target)
        found[label] = {
            "start_size": len(start),
            "size": len(state),
            "value": value,
            "gap": value - oracle_value,
            "covers_oracle": oracle <= start,
        }
    return {
        "dataset": name,
        "target": int(target),
        "oracle_size": len(oracle),
        "oracle_value": oracle_value,
        "starts": found,
    }


def main() -> None:
    rows = []
    starts = []
    for name, nodes in TARGETS.items():
        data, in_dim, n_classes = m.load_planetoid(name)
        torch.manual_seed(0)
        model = m.train_gcn(data, in_dim, n_classes)
        for target in nodes:
            row = evaluate(name, data, model, target)
            rows.append(row)
            neighbors = set(m.build_adj(data.edge_index, data.num_nodes)[target]) - {target}
            value_fn = m.ValueFunction(model, data, target)
            solo = solo_scores(value_fn, neighbors)
            start_row = start_comparison(name, data, model, target, solo)
            starts.append(start_row)
            print(f"START {name} {target} oracle {start_row['oracle_value']:+.3f}")
            for label, item in start_row["starts"].items():
                print(
                    f"  {label:14s} pool {item['start_size']:2d} "
                    f"value {item['value']:+.3f} gap {item['gap']:+.3f} "
                    f"covers {item['covers_oracle']}"
                )
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
    payload = {"selection": rows, "starts": starts}
    out = Path("/opt/cursor/artifacts/mcts_selection_auc.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    repo = Path(__file__).resolve().parent / "results" / "mcts_selection_auc.json"
    repo.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SELECTION_AUC_OK", out)


if __name__ == "__main__":
    main()
