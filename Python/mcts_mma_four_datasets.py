"""MCTS token search on four image datasets, next to leave-one-token-out.

Each image is a 4 by 4 grid of patch tokens. The query batch adds a constant
to the top-left 2 by 2 block. The value of a token set is the MMD drop from
putting the reference mean back on those tokens, minus a charge for the set
size. MCTS starts from the full token pool and only deletes. Leave-one-token-out
MMD is the score ``FSDS_runner`` already uses.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_olivetti_faces, load_digits
from sklearn.metrics import roc_auc_score

import multistart_mcts_attribution as search
from mma_wrapper import (
    HSIC,
    MMD,
    _perturb_tokens,
    mmd2_unbiased,
    patch_of,
    window_porisk,
)

GRID = (1, 4, 4)
N_TOKENS = 16
TRUTH = [h * 4 + w for h in range(2) for w in range(2)]


def _idx_images(path: str, n: int, seed: int) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    rng = np.random.default_rng(seed)
    take = rng.choice(len(images), size=n, replace=False)
    return images[take].astype(float) / 255.0


def _patches(images: np.ndarray, patch: int) -> np.ndarray:
    n, height, width = images.shape
    tokens = []
    for h in range(0, height, patch):
        for w in range(0, width, patch):
            block = images[:, h:h + patch, w:w + patch]
            tokens.append(np.stack([block.mean(axis=(1, 2)), block.std(axis=(1, 2))], axis=1))
    return np.stack(tokens, axis=1)


def _plant(images: np.ndarray, patch: int, amount: float) -> np.ndarray:
    out = np.array(images, copy=True)
    out[:, : 2 * patch, : 2 * patch] += amount
    return out


def load_four(n_each: int = 24):
    digits = load_digits().images[: 2 * n_each]
    olivetti = fetch_olivetti_faces(shuffle=True, random_state=0).images[: 2 * n_each]
    mnist = _idx_images("/tmp/imgdata/mnist.gz", 2 * n_each, seed=1)
    fashion = _idx_images("/tmp/imgdata/fashion.gz", 2 * n_each, seed=2)
    specs = [
        ("digits", digits, 2, 1.5),
        ("mnist", mnist, 7, 0.6),
        ("fashion", fashion, 7, 0.6),
        ("olivetti", olivetti, 16, 0.45),
    ]
    packs = []
    for name, images, patch, amount in specs:
        ref = _patches(images[:n_each], patch)
        query = _patches(_plant(images[n_each:], patch, amount), patch)
        y_ref = images[:n_each, : 2 * patch, : 2 * patch].mean(axis=(1, 2))
        y_query = _plant(images[n_each:], patch, amount)[:, : 2 * patch, : 2 * patch].mean(axis=(1, 2))
        packs.append((name, ref, query, np.concatenate([y_ref, y_query])))
    return packs


def _drop_value(tokens, w, gamma, base):
    """MMD drop from perturbing ``state``, with gamma held at the original pair."""

    def value(state: set) -> float:
        if not state:
            return 0.0
        pert = _perturb_tokens(tokens, w, list(state))
        flat = pert.reshape(len(pert), -1)
        return float(base - mmd2_unbiased(flat[w == 0], flat[w == 1], gamma))

    return value


def _auc(scores: dict) -> float:
    order = list(range(N_TOKENS))
    labels = np.array([1 if token in TRUTH else 0 for token in order])
    return float(roc_auc_score(labels, np.array([scores.get(token, 0.0) for token in order])))


def _deltas(tokens, y, w, chosen: set, gamma: float, base_mmd: float) -> dict:
    hsic_fn = HSIC()
    base_hsic = hsic_fn(tokens, w)
    base_po = window_porisk(tokens, y, w)
    pert = _perturb_tokens(tokens, w, list(chosen))
    flat = pert.reshape(len(pert), -1)
    return {
        "MMD": float(base_mmd - mmd2_unbiased(flat[w == 0], flat[w == 1], gamma)),
        "HSIC": float(base_hsic - hsic_fn(pert, w)),
        "PORisk": float(base_po - window_porisk(pert, y, w)),
        "precision": len(set(chosen) & set(TRUTH)) / max(len(chosen), 1),
    }


def run_one(name: str, ref, query, y) -> dict:
    tokens = np.vstack([ref, query])
    w = np.array([0] * len(ref) + [1] * len(query), dtype=int)
    flat = tokens.reshape(len(tokens), -1)
    gamma = 1.0 / max(np.median(np.sum((flat[:, None] - flat[None, :]) ** 2, axis=-1)[np.triu_indices(len(flat), 1)]), 1e-8)
    base = mmd2_unbiased(flat[w == 0], flat[w == 1], gamma)
    value_fn = _drop_value(tokens, w, gamma, base)
    search.random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    roots = []
    for trial in range(4):
        search.random.seed(10 + trial)
        roots.append(search.mcts_search(value_fn, set(range(N_TOKENS)), n_iterations=80))
    attr, _visits = search.merge_attribution(roots)
    loco = {i: float(v) for i, v in enumerate(MMD().MMD_LOCO(tokens, w))}
    solo = {token: value_fn({token}) for token in range(N_TOKENS)}
    k = len(TRUTH)
    best_state = set(int(i) for i in np.argsort([-attr.get(t, 0.0) for t in range(N_TOKENS)])[:k])
    loco_set = set(int(i) for i in np.argsort([-loco[t] for t in range(N_TOKENS)])[:k])
    solo_set = set(int(i) for i in np.argsort([-solo[t] for t in range(N_TOKENS)])[:k])
    return {
        "dataset": name,
        "truth_patches": [patch_of(token, GRID) for token in TRUTH],
        "auc": {"mcts": _auc(attr), "loco": _auc(loco), "solo": _auc(solo)},
        "mcts": {**_deltas(tokens, y, w, best_state, gamma, base), "tokens": sorted(best_state), "size": len(best_state)},
        "loco": {**_deltas(tokens, y, w, loco_set, gamma, base), "tokens": sorted(loco_set)},
        "solo": {**_deltas(tokens, y, w, solo_set, gamma, base), "tokens": sorted(solo_set)},
    }


def plot_rows(rows, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["dataset"] for row in rows]
    x = np.arange(len(names))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, key, title in (
        (axes[0], "auc", "selection AUC"),
        (axes[1], None, "MMD drop of the chosen tokens"),
    ):
        for offset, method, color in ((-width, "mcts", "#1f77b4"), (0.0, "loco", "#ff7f0e"), (width, "solo", "#2ca02c")):
            if key == "auc":
                vals = [row["auc"][method] for row in rows]
            else:
                vals = [row[method]["MMD"] for row in rows]
            ax.bar(x + offset, vals, width, label=method, color=color)
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    rows = [run_one(*pack) for pack in load_four()]
    for row in rows:
        print(
            f"{row['dataset']:10s} AUC mcts {row['auc']['mcts']:.3f} "
            f"loco {row['auc']['loco']:.3f} solo {row['auc']['solo']:.3f}  "
            f"MMD mcts {row['mcts']['MMD']:+.4f} loco {row['loco']['MMD']:+.4f} "
            f"solo {row['solo']['MMD']:+.4f}  "
            f"prec mcts {row['mcts']['precision']:.2f}"
        )
    out = Path("/opt/cursor/artifacts/mcts_mma_four.png")
    plot_rows(rows, str(out))
    payload = json.dumps(rows, indent=2)
    Path("/opt/cursor/artifacts/mcts_mma_four.json").write_text(payload, encoding="utf-8")
    Path(__file__).resolve().parent.joinpath("results", "mcts_mma_four.json").write_text(payload, encoding="utf-8")
    print("MCTS_MMA_OK", out)


if __name__ == "__main__":
    main()
