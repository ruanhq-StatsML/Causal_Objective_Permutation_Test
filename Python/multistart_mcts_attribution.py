"""Multi-start MCTS with entropy-regularized UCB, then an evolutionary refine.

The kept set S is a subset of one node's neighbors. The value of S is the
target node's true-class logit with only the edges from the target to S,
minus the same logit with none of those edges. Attribution of a neighbor is
the change in that value when the neighbor is removed from the best set, or
added to it when it was left out.

Two Planetoid graphs, Cora and CiteSeer. Each half-hour picks the next test
node and appends one record, stopping at 05:30 Asia/Shanghai.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

TZ = ZoneInfo("Asia/Shanghai")
DATASETS = ("Cora", "CiteSeer")


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def build_adj(edge_index, num_nodes: int) -> dict:
    adj = {i: set() for i in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        adj[u].add(v)
        adj[v].add(u)
    return adj


def perturb_graph(data, target: int, kept_neighbors: set) -> torch.Tensor:
    """Keep only edges between ``target`` and ``kept_neighbors``."""
    edge_index = data.edge_index
    src, dst = edge_index[0], edge_index[1]
    kept = torch.zeros(data.num_nodes, dtype=torch.bool)
    if kept_neighbors:
        kept[list(kept_neighbors)] = True
    mask = ((src == target) & kept[dst]) | ((dst == target) & kept[src])
    return edge_index[:, mask]


class ValueFunction:
    """True-class logit lift over the empty-neighborhood baseline."""

    def __init__(self, model, data, target: int):
        self.model = model
        self.data = data
        self.target = target
        self.true_class = int(data.y[target].item())
        self.cache = {}
        self.baseline = self._logit(set())

    def _logit(self, kept: set) -> float:
        edge_index = perturb_graph(self.data, self.target, kept)
        with torch.no_grad():
            logits = self.model(self.data.x, edge_index)[self.target]
        return float(logits[self.true_class].item())

    def __call__(self, kept: set) -> float:
        key = frozenset(kept)
        if key not in self.cache:
            self.cache[key] = self._logit(kept) - self.baseline
        return self.cache[key]


class MCTSNode:
    def __init__(self, state: set, parent=None):
        self.state = set(state)
        self.parent = parent
        self.children = []
        self.Q = 0.0
        self.N = 0
        self.untried = set(state)

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def best_child_ucb(self, c: float = 1.414, beta: float = 0.1):
        """UCB plus an entropy term on the child's visit share."""
        best_score, best_child = -np.inf, None
        for child in self.children:
            if child.N == 0:
                score = np.inf
            else:
                ucb = child.Q + c * np.sqrt(np.log(max(self.N, 1)) / child.N)
                share = child.N / max(self.N, 1)
                entropy = -share * np.log(share + 1e-10)
                score = ucb + beta * entropy
            if score > best_score:
                best_score, best_child = score, child
        return best_child


def mcts_search(value_fn, neighbors: set, n_iterations: int = 80, c: float = 1.414, beta: float = 0.1):
    root = MCTSNode(state=set(neighbors))
    for _ in range(n_iterations):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child_ucb(c, beta)
        if node.untried:
            neighbor = random.choice(tuple(node.untried))
            node.untried.remove(neighbor)
            child = MCTSNode(state=node.state - {neighbor}, parent=node)
            node.children.append(child)
            node = child
        value = value_fn(node.state)
        while node is not None:
            node.N += 1
            node.Q += (value - node.Q) / node.N
            node = node.parent
    return root


def extract_best_subset(root: MCTSNode) -> set:
    best_state, best_q = set(root.state), root.Q

    def walk(node: MCTSNode) -> None:
        nonlocal best_state, best_q
        if node.Q > best_q:
            best_q = node.Q
            best_state = set(node.state)
        for child in node.children:
            walk(child)

    walk(root)
    return best_state


def multi_start_mcts(value_fn, neighbors: set, n_starts: int = 4, n_iterations: int = 80):
    best_state, best_value, roots = None, -np.inf, []
    for _ in range(n_starts):
        root = mcts_search(value_fn, neighbors, n_iterations)
        roots.append(root)
        state = extract_best_subset(root)
        value = value_fn(state)
        if value > best_value:
            best_value = value
            best_state = state
    return best_state, best_value, roots


def uniform_crossover(left: set, right: set) -> set:
    child = set()
    for node in left | right:
        if random.random() < 0.5:
            child.add(node)
    return child


def flip_mutation(state: set, neighbors: set, mutation_rate: float = 0.1) -> set:
    state = set(state)
    for node in neighbors:
        if random.random() < mutation_rate:
            if node in state:
                state.remove(node)
            else:
                state.add(node)
    return state


def tournament_selection(population, fitness, k: int = 3):
    picks = random.sample(range(len(population)), min(k, len(population)))
    return population[max(picks, key=lambda i: fitness[i])]


def _collect_states(root: MCTSNode) -> list:
    found = []

    def walk(node: MCTSNode) -> None:
        found.append(set(node.state))
        for child in node.children:
            walk(child)

    walk(root)
    return found


def hybrid_search(value_fn, neighbors: set, n_starts: int = 5, n_iterations: int = 200,
                  pop_size: int = 30, n_gen: int = 50):
    """Multi-start MCTS, then evolutionary search seeded by those states."""
    best_mcts, value_mcts, roots = multi_start_mcts(value_fn, neighbors, n_starts, n_iterations)
    population = []
    for root in roots:
        for state in _collect_states(root):
            if len(population) >= pop_size:
                break
            population.append(state)
        if len(population) >= pop_size:
            break
    neighbor_list = list(neighbors)
    while len(population) < pop_size:
        size = random.randint(1, len(neighbor_list))
        population.append(set(random.sample(neighbor_list, size)))

    best_evo, value_evo = None, -np.inf
    for _ in range(n_gen):
        fitness = [value_fn(state) for state in population]
        top = int(np.argmax(fitness))
        if fitness[top] > value_evo:
            value_evo = fitness[top]
            best_evo = set(population[top])
        order = np.argsort(fitness)[::-1]
        nxt = [set(population[int(i)]) for i in order[:5]]
        while len(nxt) < pop_size:
            left = tournament_selection(population, fitness)
            right = tournament_selection(population, fitness)
            child = flip_mutation(uniform_crossover(left, right), neighbors, 0.1)
            nxt.append(child)
        population = nxt

    if value_evo > value_mcts:
        return best_evo, float(value_evo), float(value_mcts)
    return best_mcts, float(value_mcts), float(value_mcts)


def extract_attribution(value_fn, neighbors: set, best_state: set) -> dict:
    """Marginal change from removing a kept neighbor, or adding a left-out one."""
    base = value_fn(best_state)
    scores = {}
    for node in neighbors:
        if node in best_state:
            scores[node] = base - value_fn(best_state - {node})
        else:
            scores[node] = value_fn(best_state | {node}) - base
    return scores


def train_gcn(data, in_dim: int, n_classes: int, epochs: int = 200) -> GCN:
    model = GCN(in_dim, 16, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def load_planetoid(name: str):
    dataset = Planetoid(root="/tmp/planetoid", name=name)
    return dataset[0], dataset.num_features, dataset.num_classes


def run_one(name: str, data, model, target: int, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    adj = build_adj(data.edge_index, data.num_nodes)
    neighbors = set(adj[target]) - {target}
    if len(neighbors) < 2:
        return {"dataset": name, "target": int(target), "skipped": True, "degree": len(neighbors)}
    value_fn = ValueFunction(model, data, target)
    best_state, best_value, mcts_value = hybrid_search(value_fn, neighbors)
    scores = extract_attribution(value_fn, neighbors, best_state)
    ranked = sorted(scores.items(), key=lambda item: -abs(item[1]))
    inside = [scores[node] for node in best_state]
    outside = [scores[node] for node in neighbors - best_state]
    return {
        "dataset": name,
        "target": int(target),
        "true_class": int(data.y[target].item()),
        "degree": len(neighbors),
        "subset_size": len(best_state),
        "best_value": best_value,
        "mcts_value": mcts_value,
        "mean_in": float(np.mean(inside)) if inside else 0.0,
        "mean_out": float(np.mean(outside)) if outside else 0.0,
        "top": [{"neighbor": int(node), "attr": float(score), "in_subset": node in best_state}
                for node, score in ranked[:8]],
        "n_cache": len(value_fn.cache),
    }


def deadline_530() -> datetime:
    now = datetime.now(TZ)
    stop = now.replace(hour=5, minute=30, second=0, microsecond=0)
    if now >= stop:
        stop += timedelta(days=1)
    return stop


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_progress(path: Path) -> tuple[int, dict]:
    """Next round id and how many targets each dataset already has."""
    seen = {name: 0 for name in DATASETS}
    round_id = 0
    if not path.exists():
        return 0, seen
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        name = record.get("dataset")
        if name in seen:
            seen[name] += 1
        round_id = max(round_id, int(record.get("round", 0)) + 1)
    return round_id, seen


def test_targets(data, limit: int) -> list:
    nodes = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    return [int(node) for node in nodes[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("--per-round", type=int, default=1)
    parser.add_argument("--log", default="/opt/cursor/artifacts/multistart_mcts_log.jsonl")
    args = parser.parse_args()
    log_path = Path(args.log)
    loaded = {name: load_planetoid(name) for name in DATASETS}
    models = {}
    for name, (data, in_dim, n_classes) in loaded.items():
        torch.manual_seed(0)
        models[name] = train_gcn(data, in_dim, n_classes)
        pred = models[name](data.x, data.edge_index).argmax(dim=1)
        acc = float((pred[data.test_mask] == data.y[data.test_mask]).float().mean())
        print(f"{name} test acc={acc:.3f} nodes={data.num_nodes}")

    queues = {name: test_targets(loaded[name][0], 48) for name in DATASETS}
    round_id, seen = load_progress(log_path)
    cursor = {name: min(seen[name], len(queues[name])) for name in DATASETS}
    stop = deadline_530()
    print(f"stop at {stop.isoformat()} resume round {round_id}")
    while True:
        stamp = datetime.now(TZ).isoformat(timespec="seconds")
        for name in DATASETS:
            data = loaded[name][0]
            for _ in range(args.per_round):
                if cursor[name] >= len(queues[name]):
                    break
                target = queues[name][cursor[name]]
                cursor[name] += 1
                started = time.time()
                record = run_one(name, data, models[name], target, seed=1000 + round_id + target)
                record["round"] = round_id
                record["time"] = stamp
                record["seconds"] = round(time.time() - started, 2)
                append_record(log_path, record)
                if record.get("skipped"):
                    print(f"{name} target {target} degree {record['degree']} skipped")
                else:
                    print(
                        f"{name} target {target} deg {record['degree']} "
                        f"subset {record['subset_size']} value {record['best_value']:+.3f} "
                        f"in {record['mean_in']:+.3f} out {record['mean_out']:+.3f} "
                        f"({record['seconds']}s)"
                    )
        round_id += 1
        if not args.loop or datetime.now(TZ) >= stop:
            break
        remaining = (stop - datetime.now(TZ)).total_seconds()
        time.sleep(max(1.0, min(args.interval, remaining)))
    write_summary(log_path, log_path.with_name("multistart_mcts_summary.txt"))
    print("MULTISTART_MCTS_OK")


def write_summary(log_path: Path, summary_path: Path) -> None:
    rows = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    lines = [
        f"多起点 MCTS + 熵正则 + 进化  截止 {datetime.now(TZ).isoformat(timespec='seconds')}",
        f"停止时间 {deadline_530().isoformat(timespec='seconds')}  记录 {len(rows)} 条",
        "",
    ]
    for row in rows:
        if row.get("skipped"):
            lines.append(f"{row.get('time')} {row['dataset']} 节点 {row['target']} 度 {row['degree']} 跳过")
            continue
        top = row.get("top", [])
        head = ", ".join(f"{item['neighbor']}:{item['attr']:+.3f}" for item in top[:3])
        lines.append(
            f"{row.get('time')} {row['dataset']} 节点 {row['target']} 度 {row['degree']} "
            f"子集 {row['subset_size']} 价值 {row['best_value']:+.3f} "
            f"子集内 {row['mean_in']:+.3f} 子集外 {row['mean_out']:+.3f} 前三 {head}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
