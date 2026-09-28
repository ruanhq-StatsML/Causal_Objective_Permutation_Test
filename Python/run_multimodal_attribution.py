"""Standard pipeline: dataloader -> VAE or ViViT embedding -> FSDS.

Image batches use ``run_vae_domain_shift`` (ConvVAE ``mu``).
Video batches use a ViViT embedding.
Both embeddings are handed to ``graph_fsds.fsds_core`` unchanged:
``covariate_shift_test`` and ``domain_vimp``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_fsds.fsds_core import covariate_shift_test, domain_vimp
from run_vae_domain_shift import (
    ConvVAE,
    ImageDataset,
    encode_dataset,
    make_synthetic_dataset,
    make_transform,
    train_vae,
)


class ViViTEmbedding(nn.Module):
    """Tubelet ViViT. Returns one embedding per clip."""

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
        n_tokens = (num_frames // tubelet_size) * (image_size // patch_size) ** 2
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
        tokens = self.tubelet(clips).flatten(2).transpose(1, 2)
        cls = self.cls.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos[:, : tokens.size(1) + 1]
        return self.norm(self.encoder(x)[:, 0])


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


def _image_loaders(root: Path, n: int = 16, batch_size: int = 8):
    train_dir = root / "synthetic_train"
    eval_dir = root / "synthetic_eval"
    make_synthetic_dataset(train_dir, n_images=n, seed=0, color_shift=0.0)
    make_synthetic_dataset(eval_dir, n_images=n, seed=1, color_shift=40.0)
    transform = make_transform(64)
    train_loader = DataLoader(
        ImageDataset(str(train_dir), transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    eval_loader = DataLoader(
        ImageDataset(str(eval_dir), transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, eval_loader


def _vae_embeddings(train_loader: DataLoader, eval_loader: DataLoader, out_dir: Path):
    device = torch.device("cpu")
    vae = ConvVAE(latent_dim=32, img_channels=3, img_size=64).to(device)
    model_path = out_dir / "vae_model.pth"
    train_vae(vae, train_loader, device, num_epochs=2, model_path=model_path)
    vae.eval()

    def encode(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = vae.encode(x.to(device))
        return mu

    z_ref = encode_dataset(train_loader, encode).numpy()
    z_query = encode_dataset(eval_loader, encode).numpy()
    return z_ref, z_query


def _video_loader(n: int, shift: float, seed: int, batch_size: int = 8) -> DataLoader:
    return DataLoader(ClipDataset(n, shift, seed), batch_size=batch_size, shuffle=False)


def _vivit_embeddings(loader: DataLoader, model: ViViTEmbedding) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for clips in loader:
            chunks.append(model(clips).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def fsds_downstream(z_ref: np.ndarray, z_query: np.ndarray, name: str, seed: int = 2026) -> dict:
    """Existing FSDS covariate test and domain VIMP on one embedding pair."""
    x = np.vstack([z_ref, z_query])
    w = np.concatenate([
        np.zeros(z_ref.shape[0], dtype=int),
        np.ones(z_query.shape[0], dtype=int),
    ])
    cov = covariate_shift_test(
        x, w, n_perm=19, n_folds=2, domain_model="logistic", alpha=0.05, seed=seed,
    )
    vimp = domain_vimp(
        x, w, domain_model="logistic", n_repeats=4, n_folds=2, seed=seed,
    )
    top = vimp["vimp_rank"][:5].tolist()
    print(
        f"{name:16} auc={cov['auc']:.3f} p={cov['p_value']:.3f} "
        f"reject={cov['reject']} top_dims={top}"
    )
    return {"name": name, **cov, "top_dims": top}


def main() -> None:
    out_dir = Path("/opt/cursor/artifacts/embedding_fsds")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, eval_loader = _image_loaders(out_dir, n=16, batch_size=8)
    z_img_ref, z_img_query = _vae_embeddings(train_loader, eval_loader, out_dir)
    print(f"VAE embeddings {z_img_ref.shape} {z_img_query.shape}")

    vivit = ViViTEmbedding().eval()
    z_vid_ref = _vivit_embeddings(_video_loader(24, shift=0.0, seed=0), vivit)
    z_vid_query = _vivit_embeddings(_video_loader(24, shift=1.5, seed=1), vivit)
    z_vid_null = _vivit_embeddings(_video_loader(24, shift=0.0, seed=2), vivit)
    print(f"ViViT embeddings {z_vid_ref.shape}")

    rows = [
        fsds_downstream(z_img_ref, z_img_query, "image_vae", seed=2026),
        fsds_downstream(z_vid_ref, z_vid_query, "video_vivit", seed=2027),
        fsds_downstream(z_vid_ref, z_vid_null, "video_null", seed=2028),
    ]
    shifted = next(row for row in rows if row["name"] == "video_vivit")
    null = next(row for row in rows if row["name"] == "video_null")
    assert shifted["auc"] > null["auc"]
    print("PIPELINE_OK")


if __name__ == "__main__":
    main()
