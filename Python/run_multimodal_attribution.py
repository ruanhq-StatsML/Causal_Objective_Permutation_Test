"""Upper wrapper around the existing VAE domain-shift VIMP pipeline.

Image attribution calls ``run_vae_domain_shift.run_pipeline`` unchanged.
Video attribution encodes clips with a ViViT and scores those embeddings
with the same ``rf_domain_classifier``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_vae_domain_shift import rf_domain_classifier, run_pipeline


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


def _video_batch(n: int, shift: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    clips = rng.normal(0.0, 0.3, size=(n, 3, 8, 32, 32)).astype(np.float32)
    if shift:
        clips[:, :, 4:, :8, :8] += shift
    return torch.from_numpy(clips)


def attribute_videos(seed: int = 42) -> None:
    device = torch.device("cpu")
    model = ViViTEmbedding().to(device).eval()
    ref = _video_batch(24, shift=0.0, seed=0).to(device)
    query = _video_batch(24, shift=1.5, seed=1).to(device)
    null = _video_batch(24, shift=0.0, seed=2).to(device)
    with torch.no_grad():
        z_ref = model(ref).cpu()
        z_query = model(query).cpu()
        z_null = model(null).cpu()
    importance, oob = rf_domain_classifier(z_ref, z_query, seed=seed)
    _, oob_null = rf_domain_classifier(z_ref, z_null, seed=seed)
    rank = np.argsort(-importance)
    print("ViViT embedding", tuple(z_ref.shape))
    print(f"ViViT shifted OOB {oob:.4f} | null OOB {oob_null:.4f}")
    print(f"ViViT top embedding dims {rank[:8].tolist()}")


def main() -> None:
    out_dir = Path("/opt/cursor/artifacts/vae_domain_shift_outputs")
    args = argparse.Namespace(
        train_dir="",
        eval_dir="",
        output_dir=str(out_dir),
        image_size=64,
        batch_size=8,
        num_epochs=2,
        latent_dim=32,
        num_workers=0,
        seed=42,
        top_k=[1, 5, 10],
        retrain=True,
        cpu=True,
        synthetic=True,
        synthetic_n=16,
    )
    run_pipeline(args)
    attribute_videos(seed=args.seed)
    print("MULTIMODAL_ATTRIBUTION_OK")


if __name__ == "__main__":
    main()
