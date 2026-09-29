"""MCTS on a tree, then PO-risk attribution.

One row is one finished search on

    root
    / | \\
    A  B  C
    /| /| /|
    L R L R L R

The search records the root value, each child's value, and each link's visit
share. PO-risk is mean(tau^2) on the pseudo-outcome (Y - mu) * (W - e), the
statistic in ``po_statistic``. Nuisances are the linear outcome fit and the
logistic propensity in ``porisk_path``. LOCO is the full risk minus the risk
with that column removed, as in ``po_risk_loco``.

Y is the root node value, or the probability of link B. A drifted search
moves those columns together and the risk stays at the null. The risk names
a column when the same search table maps to Y through a different slope.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

ACTIONS = ("A", "B", "C")
# True leaf reward of the path (first action, second action). Fixed across batches.
TRUE = {
    ("A", "L"): 1.0, ("A", "R"): 0.7,
    ("B", "L"): 0.45, ("B", "R"): 0.35,
    ("C", "L"): 0.3, ("C", "R"): 0.25,
}
VALUE_COLS = ["q_A", "q_B", "q_C", "p_A", "p_B", "p_C"]
LINK_COLS = ["q_A", "q_B", "q_C"]


class _Node:
    def __init__(self):
        self.n = 0
        self.w = 0.0
        self.children = {}

    @property
    def q(self) -> float:
        return 0.0 if self.n == 0 else self.w / self.n


def _uct(parent: _Node, child: _Node, c: float) -> float:
    return child.q + c * math.sqrt(parent.n + 1) / (1 + child.n)


def _select(node: _Node, c: float, actions) -> str:
    if not node.children:
        for name in actions:
            node.children[name] = _Node()
    return max(node.children, key=lambda name: _uct(node, node.children[name], c))


def mcts_search(bias_b: float, c_uct: float, rng: np.random.Generator, n_sim: int = 24) -> dict:
    """One search. ``bias_b`` is added only inside the value backup at B."""
    root = _Node()
    for _ in range(n_sim):
        action = _select(root, c_uct, ACTIONS)
        child = root.children[action]
        second = _select(child, c_uct, ("L", "R"))
        leaf = child.children[second]
        backed = TRUE[(action, second)] + float(rng.normal(0.0, 0.15))
        if action == "B":
            backed += bias_b
        for node in (leaf, child, root):
            node.n += 1
            node.w += backed
    total = max(root.n, 1)
    return {
        "value": root.q,
        "q_A": root.children["A"].q,
        "q_B": root.children["B"].q,
        "q_C": root.children["C"].q,
        "p_A": root.children["A"].n / total,
        "p_B": root.children["B"].n / total,
        "p_C": root.children["C"].n / total,
    }


def _rows(n: int, bias_b: float, c_uct: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [mcts_search(bias_b, c_uct, rng, n_sim=24) for _ in range(n)]


def _matrix(rows, keys):
    return np.column_stack([np.array([r[k] for r in rows], dtype=float) for k in keys])


def po_risk(X, Y, W) -> float:
    """mean(tau^2) on (Y - mu) * (W - e).

    mu and e are the linear outcome fit and the logistic propensity from
    ``porisk_path``. tau is the ridge fit of that pseudo-outcome, and the
    returned number is ``po_statistic``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).ravel()
    W = np.asarray(W, dtype=int).ravel()
    mu = LinearRegression().fit(X, Y).predict(X)
    e = LogisticRegression(max_iter=500).fit(X, W).predict_proba(X)[:, 1]
    e = np.clip(e, 0.02, 0.98)
    pseudo = (Y - mu) * (W.astype(float) - e)
    tau = Ridge(alpha=1.0).fit(X, pseudo).predict(X)
    return float(np.mean(tau ** 2))


def porisk_loco(x, y, w, names: list) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    w = np.asarray(w, dtype=int).ravel()
    full = po_risk(x, y, w)
    drops = []
    for j, name in enumerate(names):
        drops.append((name, full - po_risk(np.delete(x, j, axis=1), y, w)))
    drops.sort(key=lambda item: -item[1])
    return {
        "porisk": full,
        "loco": drops,
        "y_mean_0": float(y[w == 0].mean()),
        "y_mean_1": float(y[w == 1].mean()),
    }


def _print_case(title: str, out: dict) -> None:
    print(title)
    print(
        f"  Y mean batch0={out['y_mean_0']:.3f}  batch1={out['y_mean_1']:.3f}  "
        f"PO-risk={out['porisk']:.6f}"
    )
    for name, drop in out["loco"]:
        print(f"  LOCO {name:6s} {drop:+.6f}")


def main() -> None:
    n = 200
    rows = _rows(n, bias_b=0.0, c_uct=1.3, seed=20)
    w = np.random.default_rng(200).integers(0, 2, size=n)
    value = _matrix(rows, ["value"])[:, 0]
    q_a = _matrix(rows, ["q_A"])[:, 0]
    q_b = _matrix(rows, ["q_B"])[:, 0]
    p_b = _matrix(rows, ["p_B"])[:, 0]
    x_value = _matrix(rows, VALUE_COLS)
    x_link = _matrix(rows, LINK_COLS)
    # Slope on a centered child. The raw mean of that child is only a level
    # shift, and the intercept of tau absorbs a level shift.
    y_value = value + w * 4.0 * (q_b - q_b.mean())
    y_link = p_b + w * 4.0 * (q_a - q_a.mean())
    y_level = value + 0.40 * w

    rows_ref = _rows(100, bias_b=0.0, c_uct=1.3, seed=0)
    rows_new = _rows(100, bias_b=1.6, c_uct=1.3, seed=1)
    x_cov = np.vstack([_matrix(rows_ref, VALUE_COLS), _matrix(rows_new, VALUE_COLS)])
    y_cov = np.concatenate([
        _matrix(rows_ref, ["value"])[:, 0],
        _matrix(rows_new, ["value"])[:, 0],
    ])
    w_cov = np.array([0] * 100 + [1] * 100)

    print("tree: root - A,B,C - each with L,R")
    print(
        "rewards  A=(1.00,0.70)  B=(0.45,0.35)  C=(0.30,0.25)  "
        "UCT c=1.3  sims=24"
    )
    print(
        f"shared search  n={n}  value={value.mean():.3f}  "
        f"q_A={q_a.mean():.3f}  q_B={q_b.mean():.3f}  p_B={p_b.mean():.3f}"
    )
    print(
        "feature std  "
        + "  ".join(f"{name}={x_value[:, j].std():.3f}" for j, name in enumerate(VALUE_COLS))
    )

    value_case = porisk_loco(x_value, y_value, w, VALUE_COLS)
    link_case = porisk_loco(x_link, y_link, w, LINK_COLS)
    null_value = porisk_loco(x_value, value, w, VALUE_COLS)
    null_link = porisk_loco(x_link, p_b, w, LINK_COLS)
    level_case = porisk_loco(x_value, y_level, w, VALUE_COLS)
    cov_case = porisk_loco(x_cov, y_cov, w_cov, VALUE_COLS)

    _print_case("Y = node value, new batch adds a slope on q_B", value_case)
    _print_case("Y = P(link B), new batch adds a slope on q_A", link_case)
    _print_case("null Y = node value", null_value)
    _print_case("null Y = P(link B)", null_link)
    _print_case("level shift of the node value, no extra slope", level_case)
    _print_case("search itself drifts: backup bias 1.6 at B, Y = node value", cov_case)

    assert value_case["loco"][0][0] == "q_B"
    assert link_case["loco"][0][0] == "q_A"
    assert value_case["porisk"] > 10.0 * null_value["porisk"]
    assert link_case["porisk"] > 10.0 * null_link["porisk"]
    assert cov_case["porisk"] < value_case["porisk"]
    assert level_case["loco"][0][1] < 0.05 * level_case["porisk"]
    print("MCTS_PORISK_OK")


if __name__ == "__main__":
    main()
