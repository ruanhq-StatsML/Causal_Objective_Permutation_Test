"""Dataloader -> ViViT tokens -> MMA wrapper.

The wrapper stacks the two batches, runs FSDS, and returns patch indices
``[[h, w], ...]`` plus token indices ``L``. Post-hoc localization then
perturbs those tokens and reports the change in MMD, PO-risk, and HSIC.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fsds_nonuniqueness import generate_concept_drift_dgp  # noqa: E402
from mma_wrapper import (  # noqa: E402
    graph_neighborhood_delta,
    mma_wrapper,
    neighborhood_indices,
    online_bootstrap_ci,
    online_pfi,
    plot_online_porisk,
)


class ViViTEmbedding(nn.Module):
    """Tubelet ViViT. Returns one embedding per patch token."""

    def __init__(
        self,
        *,
        image_size: int = 32,
        num_frames: int = 8,
        patch_size: int = 8,
        tubelet_size: int = 2,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        if image_size % patch_size or num_frames % tubelet_size:
            raise ValueError("frame count and image size must divide the tubelet size")
        self.tubelet = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.grid = (
            num_frames // tubelet_size,
            image_size // patch_size,
            image_size // patch_size,
        )
        n_tokens = self.grid[0] * self.grid[1] * self.grid[2]
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Token embeddings, index ``t * (H W) + h * W + w`` (CLS dropped)."""
        tokens = self.tubelet(clips).flatten(2).transpose(1, 2)
        cls = self.cls.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos[:, : tokens.size(1) + 1]
        return self.norm(self.encoder(x)[:, 1:])


class ClipDataset(Dataset):
    def __init__(self, n: int, shift: float, seed: int):
        rng = np.random.default_rng(seed)
        clips = rng.normal(0.0, 0.3, size=(n, 3, 8, 32, 32)).astype(np.float32)
        if shift:
            clips[:, :, 4:, :8, :8] += shift
        self.clips = torch.from_numpy(clips)

    def __len__(self) -> int:
        return int(self.clips.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.clips[idx]


def _video_loader(n: int, shift: float, seed: int, batch_size: int = 8) -> DataLoader:
    return DataLoader(ClipDataset(n, shift, seed), batch_size=batch_size, shuffle=False)


def _vivit_batch(loader: DataLoader, model: ViViTEmbedding):
    tokens, outcomes = [], []
    with torch.no_grad():
        for clips in loader:
            tokens.append(model(clips).cpu().numpy())
            outcomes.append(clips[:, :, 4:, :8, :8].mean(dim=(1, 2, 3, 4)).numpy())
    return np.concatenate(tokens, axis=0), np.concatenate(outcomes, axis=0)


def _print_result(name: str, out: dict) -> None:
    print(f"{name} patch-indices: {out['patch_indices']}")
    print(f"{name} token indices L: {out['L']}")
    delta = out["delta"]
    print(
        f"{name} delta  MMD={delta['MMD']:+.5f}  "
        f"PORisk={delta['PORisk']:+.5f}  HSIC={delta['HSIC']:+.5f}"
    )
    loco = out["loco"]
    for i, token in enumerate(out["L"]):
        print(
            f"{name} LOCO token {token} patch {out['patch_indices'][i]}  "
            f"MMD={loco['MMD'][i]:+.5f}  PORisk={loco['PORisk'][i]:+.5f}  "
            f"HSIC={loco['HSIC'][i]:+.5f}"
        )


def _print_online_porisk(name: str, rows: list) -> None:
    for row in rows:
        print(
            f"{name} onlinePORisk m={row['m']} "
            f"risk={row['porisk']:.5f} "
            f"CI=[{row['ci_lo']:.5f}, {row['ci_hi']:.5f}]"
        )


def _concept_online_porisk(path: str) -> None:
    """Their concept-drift stream, then the online bootstrap band."""
    X, Y, _W = generate_concept_drift_dgp(n=80, p=6, delta_beta=0.8, rho=0.2, seed=3)
    X0, Y0, _ = generate_concept_drift_dgp(n=80, p=6, delta_beta=0.0, rho=0.2, seed=4)
    shifted = online_bootstrap_ci(X, Y, ref_n=16, recent_n=12, step=8, n_boot=30, seed=3)
    null = online_bootstrap_ci(X0, Y0, ref_n=16, recent_n=12, step=8, n_boot=30, seed=4)
    _print_online_porisk("concept", shifted)
    _print_online_porisk("concept_null", null)
    plot_online_porisk(
        [("concept drift", shifted), ("no concept drift", null)],
        path,
        enter_m=52,
    )
    assert shifted[-1]["ci_lo"] <= shifted[-1]["ci_hi"]
    assert abs(shifted[0]["porisk"] - shifted[-1]["porisk"]) > abs(null[0]["porisk"] - null[-1]["porisk"])


def _print_online_pfi(name: str, rows: list) -> None:
    for row in rows:
        print(
            f"{name} onlinePFI m={row['m']} "
            f"token {row['top_token']} patch {row['top_patch']} "
            f"pfi={row['top_pfi']:+.5f}"
        )


def _synthetic_online_pfi() -> None:
    """Reference stream, then a query stream that only shifts token 5."""
    rng = np.random.default_rng(1)
    n, n_tokens, dim = 16, 8, 4
    ref = rng.normal(0.0, 1.0, size=(n, n_tokens, dim))
    query = rng.normal(0.0, 1.0, size=(n, n_tokens, dim))
    query[:, 5, :] += 2.5
    rows = online_pfi(np.vstack([ref, query]), grid=(2, 2, 4), ref_n=8, recent_n=6, step=4)
    _print_online_pfi("synthetic", rows)
    assert rows[-1]["top_token"] == 5
    assert rows[-1]["top_patch"] == [1, 1]
    assert rows[-1]["top_pfi"] > 0.0


def _synthetic_token_check() -> None:
    """One token carries the batch shift. FSDS should return that index first."""
    rng = np.random.default_rng(0)
    n, n_tokens, dim = 16, 8, 4
    ref = rng.normal(0.0, 1.0, size=(n, n_tokens, dim))
    query = rng.normal(0.0, 1.0, size=(n, n_tokens, dim))
    query[:, 5, :] += 2.5
    y = np.concatenate([ref[:, 5, :].mean(axis=1), query[:, 5, :].mean(axis=1)])
    out = mma_wrapper(ref, query, y, grid=(2, 2, 4), top_k=4)
    _print_result("synthetic", out)
    assert out["L"][0] == 5
    assert out["patch_indices"][0] == [1, 1]
    assert out["delta"]["MMD"] > 0.0


def _quadrant_means(tokens: np.ndarray, grid) -> np.ndarray:
    """Mean embedding in each 2 by 2 spatial block, averaged over time."""
    n_t, n_h, n_w = grid
    means = tokens.mean(axis=2)
    cols = []
    for ih in (0, n_h // 2):
        for iw in (0, n_w // 2):
            idx = []
            for t in range(n_t):
                for h in range(ih, ih + n_h // 2):
                    for w in range(iw, iw + n_w // 2):
                        idx.append(t * n_h * n_w + h * n_w + w)
            cols.append(means[:, idx].mean(axis=1))
    return np.column_stack(cols)


def _graph_neighborhood_try() -> None:
    """Shift one block. Perturb the 1-hop neighborhood of the moved node."""
    import networkx as nx

    rng = np.random.default_rng(0)
    sizes = [12, 12, 12]
    G = nx.stochastic_block_model(sizes, [[0.45, 0.02, 0.02], [0.02, 0.45, 0.02], [0.02, 0.02, 0.45]], seed=0)
    A = nx.to_numpy_array(G)
    membership = np.concatenate([np.full(s, k) for k, s in enumerate(sizes)])
    x_ref = rng.normal(size=(len(G), 4))
    x_new = x_ref + rng.normal(scale=0.05, size=x_ref.shape)
    x_new[membership == 0] += 1.8
    y_ref = x_ref[:, 0]
    y_new = x_new[:, 0]
    seed = int(np.argmax(np.linalg.norm(x_new - x_ref, axis=1)))
    neigh = neighborhood_indices(A, [seed])
    far_pool = np.flatnonzero(membership == 2)
    far = far_pool[:len(neigh)].tolist()
    hit = graph_neighborhood_delta(x_ref, x_new, y_ref, y_new, neigh)
    miss = graph_neighborhood_delta(x_ref, x_new, y_ref, y_new, far)
    print(f"graph seed {seed} block {int(membership[seed])}")
    print(f"graph neighborhood indices: {hit['neighborhood']}")
    d, m = hit["delta"], miss["delta"]
    print(
        f"graph neighborhood delta  MMD={d['MMD']:+.5f}  "
        f"PORisk={d['PORisk']:+.5f}  HSIC={d['HSIC']:+.5f}"
    )
    print(
        f"graph other-block delta   MMD={m['MMD']:+.5f}  "
        f"PORisk={m['PORisk']:+.5f}  HSIC={m['HSIC']:+.5f}"
    )
    assert int(membership[seed]) == 0
    assert np.mean(membership[neigh] == 0) > 0.5
    assert hit["delta"]["MMD"] > miss["delta"]["MMD"]


def main() -> None:
    out_dir = Path("/opt/cursor/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    _graph_neighborhood_try()
    _synthetic_token_check()
    _synthetic_online_pfi()
    _concept_online_porisk(str(out_dir / "online_porisk_concept_ci.png"))

    torch.manual_seed(0)
    model = ViViTEmbedding().eval()
    z_ref, y_ref = _vivit_batch(_video_loader(16, shift=0.0, seed=0), model)
    z_query, y_query = _vivit_batch(_video_loader(16, shift=3.0, seed=1), model)
    z_null, y_null = _vivit_batch(_video_loader(16, shift=0.0, seed=2), model)
    print(f"ViViT tokens {z_ref.shape} grid {model.grid}")

    shifted = mma_wrapper(z_ref, z_query, np.concatenate([y_ref, y_query]), model.grid, top_k=4)
    null = mma_wrapper(z_ref, z_null, np.concatenate([y_ref, y_null]), model.grid, top_k=4)
    _print_result("video", shifted)
    _print_result("video_null", null)
    stream = np.vstack([z_ref, z_query])
    null_stream = np.vstack([z_ref, z_null])
    pfi = online_pfi(stream, model.grid, ref_n=8, recent_n=6, step=8)
    pfi_null = online_pfi(null_stream, model.grid, ref_n=8, recent_n=6, step=8)
    _print_online_pfi("video", pfi)
    _print_online_pfi("video_null", pfi_null)
    y_shift = np.concatenate([y_ref, y_query])
    y_quiet = np.concatenate([y_ref, y_null])
    porisk_shift = online_bootstrap_ci(
        _quadrant_means(stream, model.grid), y_shift,
        ref_n=8, recent_n=6, step=4, n_boot=30, seed=7,
    )
    porisk_null = online_bootstrap_ci(
        _quadrant_means(null_stream, model.grid), y_quiet,
        ref_n=8, recent_n=6, step=4, n_boot=30, seed=8,
    )
    _print_online_porisk("video", porisk_shift)
    _print_online_porisk("video_null", porisk_null)
    plot_online_porisk(
        [("video local shift", porisk_shift), ("video no shift", porisk_null)],
        str(out_dir / "online_porisk_video_ci.png"),
        enter_m=22,
    )

    assert shifted["base"]["MMD"] > null["base"]["MMD"]
    assert [0, 0] in shifted["patch_indices"]
    assert pfi[-1]["top_patch"] == [0, 0]
    assert pfi[-1]["top_pfi"] > pfi_null[-1]["top_pfi"]
    print("PIPELINE_OK")


if __name__ == "__main__":
    main()
