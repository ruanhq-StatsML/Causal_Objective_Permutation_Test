"""
Local MCTS over branch token budgets with success-rate rollouts.

Uses FSDS water-fill plan as prior; MCTS applies local ±delta moves on
high-shift branches and maximizes simulated task success rate.
Selection: UCT + entropy regularization on action visit distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .budget import BranchBudget, SoTPlan, allocate_branch_budgets
from .entropy_reg import allocation_entropy
from .incremental_value import simulate_episode_outcome


@dataclass
class MCTSConfig:
    n_simulations: int = 48
    c_puct: float = 1.4
    entropy_coef: float = 0.12
    delta_tokens: int = 48
    max_local_branches: int = 3
    rollout_repeats: int = 1
    seed: int = 2026


@dataclass
class MCTSStats:
    success_rate: float
    n_simulations: int
    best_L: List[int]
    root_entropy: float
    visits: Dict[str, int] = field(default_factory=dict)


def _plan_from_L(
    L: np.ndarray,
    shift_scores: np.ndarray,
    quality: np.ndarray,
    *,
    latency_cap: int,
    total_budget: int,
) -> SoTPlan:
    B = len(L)
    s = np.asarray(shift_scores, dtype=float)
    q = np.asarray(quality, dtype=float)
    risk = s / (s.sum() + 1e-9) * (1.0 - np.clip(q, 0.05, 1.0))
    check = np.clip(np.round(1 + 2 * risk), 1, 5).astype(int)
    tiers: List[str] = []
    for i in range(B):
        if L[i] >= latency_cap * 0.95:
            tiers.append("large")
        elif L[i] < latency_cap * 0.45:
            tiers.append("small")
        else:
            tiers.append("medium")
    branches = [
        BranchBudget(
            branch_id=i,
            expansion_tokens=int(L[i]),
            check_budget=int(check[i]),
            model_tier=tiers[i],  # type: ignore[arg-type]
            shift_score=float(s[i]),
            quality_score=float(q[i]),
        )
        for i in range(B)
    ]
    return SoTPlan(
        branch_budgets=branches,
        latency_cap_tokens=latency_cap,
        total_token_budget=total_budget,
        use_parallel=True,
        clusters=[],
    )


def _legal_actions(L: np.ndarray, focus: List[int], delta: int, min_t: int, cap: int) -> List[np.ndarray]:
    actions: List[np.ndarray] = []
    base = L.copy()
    actions.append(base)
    for b in focus:
        for sign in (-1, 1):
            L2 = L.copy()
            L2[b] = int(np.clip(L2[b] + sign * delta, min_t, cap))
            if not np.array_equal(L2, L):
                actions.append(L2)
    # dedupe
    uniq: List[np.ndarray] = []
    seen = set()
    for a in actions:
        key = tuple(a.tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    return uniq


def _renormalize_total(L: np.ndarray, total_budget: int, min_t: int) -> np.ndarray:
    L = L.astype(int).copy()
    if L.sum() <= total_budget:
        return L
    scale = total_budget / L.sum()
    L = np.maximum((L * scale).astype(int), min_t)
    while L.sum() > total_budget and (L > min_t).any():
        i = int(np.argmax(L))
        L[i] -= 1
    return L


class LocalMCTS:
    def __init__(self, config: Optional[MCTSConfig] = None):
        self.cfg = config or MCTSConfig()

    def optimize_plan(
        self,
        shift_scores: np.ndarray,
        quality: np.ndarray,
        need: np.ndarray,
        *,
        total_token_budget: int = 2800,
        latency_cap_tokens: int = 520,
        min_tokens: int = 64,
        entropy_coef_budget: float = 0.15,
        kappa: float = 1.0,
        need_weights: Optional[np.ndarray] = None,
        min_tokens_by_branch: Optional[np.ndarray] = None,
    ) -> Tuple[SoTPlan, MCTSStats]:
        rng = np.random.default_rng(self.cfg.seed)
        prior = allocate_branch_budgets(
            shift_scores,
            quality,
            total_token_budget=total_token_budget,
            latency_cap_tokens=latency_cap_tokens,
            min_tokens=min_tokens,
            need_weights=need_weights,
            min_tokens_by_branch=min_tokens_by_branch,
            kappa=kappa,
            entropy_coef=entropy_coef_budget,
        )
        L0 = np.array([b.expansion_tokens for b in prior.branch_budgets], dtype=int)
        order = np.argsort(-shift_scores)[: self.cfg.max_local_branches]
        focus = order.tolist()

        # MCTS tree: state key -> {N, W, children actions}
        class Node:
            __slots__ = ("L", "N", "W", "children", "action_keys")

            def __init__(self, L: np.ndarray):
                self.L = L
                self.N = 0
                self.W = 0.0
                self.children: Dict[str, Tuple[np.ndarray, Node]] = {}
                self.action_keys: List[str] = []

        root = Node(L0)
        visits: Dict[str, int] = {}

        def state_key(L: np.ndarray) -> str:
            return ",".join(map(str, L.tolist()))

        def rollout(L: np.ndarray) -> float:
            L = _renormalize_total(L, total_token_budget, min_tokens)
            plan = _plan_from_L(L, shift_scores, quality, latency_cap=latency_cap_tokens, total_budget=total_token_budget)
            succ = 0.0
            for _ in range(self.cfg.rollout_repeats):
                s, _, _, _, _ = simulate_episode_outcome(
                    plan, need, quality, uniform_L=latency_cap_tokens, rng=rng
                )
                succ += s
            return succ / self.cfg.rollout_repeats

        def select(node: Node) -> Node:
            path = [node]
            while node.children:
                total_N = sum(c.N for _, c in node.children.values()) + 1
                best_score = -1e18
                best: Optional[Node] = None
                best_a: Optional[np.ndarray] = None
                visit_probs = []
                child_list = list(node.children.values())
                for _, ch in child_list:
                    visit_probs.append(ch.N + 1e-9)
                vp = np.array(visit_probs)
                vp = vp / vp.sum()
                ent = allocation_entropy(vp)

                for (L_a, child) in node.children.values():
                    if child.N == 0:
                        score = 1e9
                    else:
                        q = child.W / child.N
                        u = self.cfg.c_puct * np.sqrt(np.log(total_N) / child.N)
                        score = q + u + self.cfg.entropy_coef * ent
                    if score > best_score:
                        best_score = score
                        best = child
                if best is None:
                    break
                node = best
                path.append(node)
            return path[-1]

        def expand(node: Node) -> None:
            if node.children:
                return
            actions = _legal_actions(node.L, focus, self.cfg.delta_tokens, min_tokens, latency_cap_tokens)
            for L_a in actions:
                k = state_key(L_a)
                node.children[k] = (L_a, Node(L_a))
                node.action_keys.append(k)

        for _ in range(self.cfg.n_simulations):
            leaf = select(root)
            if leaf.N == 0 or not leaf.children:
                expand(leaf)
            if leaf.children:
                # pick random unexplored child first
                unexplored = [c for _, c in leaf.children.values() if c.N == 0]
                sim_node = unexplored[0] if unexplored else select(leaf)
            else:
                sim_node = leaf
            r = rollout(sim_node.L)
            visits[state_key(sim_node.L)] = visits.get(state_key(sim_node.L), 0) + 1
            # backprop
            cur = sim_node
            while cur is not None:
                cur.N += 1
                cur.W += r
                # find parent (linear scan — small tree)
                cur = None
                for n in [root]:
                    stack = [n]
                    while stack:
                        nd = stack.pop()
                        for _, ch in nd.children.values():
                            if ch is sim_node:
                                cur = nd
                                break
                            stack.append(ch)

        # Best child of root by success rate
        best_L = root.L
        best_rate = -1.0
        if root.children:
            for L_a, ch in root.children.values():
                rate = ch.W / max(ch.N, 1)
                if rate > best_rate:
                    best_rate = rate
                    best_L = L_a
        else:
            best_rate = rollout(root.L)

        best_L = _renormalize_total(best_L, total_token_budget, min_tokens)
        plan = _plan_from_L(
            best_L, shift_scores, quality, latency_cap=latency_cap_tokens, total_budget=total_token_budget
        )
        vp = np.array([root.children[k][1].N for k in root.action_keys] if root.action_keys else [1.0])
        stats = MCTSStats(
            success_rate=float(best_rate),
            n_simulations=self.cfg.n_simulations,
            best_L=best_L.tolist(),
            root_entropy=float(allocation_entropy(vp / vp.sum())),
            visits=visits,
        )
        return plan, stats
