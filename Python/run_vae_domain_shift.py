"""VAE-based domain-shift VIMP and region-selection pipeline (FrEIA alternative)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

IMAGE_SIZE = 64
LATENT_DIM = 128


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths: List[str] = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(root, file))
        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, img_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc_decode(z))

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_vae(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss


def make_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def make_synthetic_dataset(out_dir: Path, n_images: int = 64, seed: int = 0, color_shift: float = 0.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_images):
        base = rng.integers(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        if color_shift:
            base = np.clip(base.astype(float) + color_shift, 0, 255).astype(np.uint8)
        Image.fromarray(base).save(out_dir / f"img_{i:04d}.png")


def train_vae(
    vae: ConvVAE,
    loader_train: DataLoader,
    device: torch.device,
    num_epochs: int,
    model_path: Path,
) -> None:
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_imgs, _ in tqdm(loader_train, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch_imgs = batch_imgs.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = vae(batch_imgs)
            loss = loss_vae(recon, batch_imgs, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader_train):.6f}")
    torch.save(vae.state_dict(), model_path)
    print(f"VAE saved to {model_path}")


def encode_dataset(loader: DataLoader, encode_func: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    z_list = []
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc="Encoding"):
            z_list.append(encode_func(batch_imgs).cpu())
    return torch.cat(z_list, dim=0)


def rf_domain_classifier(z_train: torch.Tensor, z_eval: torch.Tensor, seed: int = 42):
    X = np.vstack([z_train.numpy(), z_eval.numpy()])
    y = np.concatenate([np.zeros(z_train.shape[0]), np.ones(z_eval.shape[0])])
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        max_features=round(np.sqrt(X.shape[1]) / X.shape[1], 3),
        min_samples_leaf=max(1, round(np.sqrt(X.shape[0]) / 2)),
        random_state=seed,
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(X, y)
    return rf.feature_importances_, float(rf.oob_score_)


def evaluate_region_selection(
    loader_eval: DataLoader,
    encode_func: Callable[[torch.Tensor], torch.Tensor],
    decode_func: Callable[[torch.Tensor], torch.Tensor],
    z_delta: torch.Tensor,
    importance: np.ndarray,
    device: torch.device,
    top_k_list: List[int] | None = None,
) -> Dict[int, Dict[str, float]]:
    top_k_list = top_k_list or [1, 5, 10, 20, 50]
    sorted_idx = np.argsort(-importance)
    results = {k: {"top10_ratios": [], "gini_coeffs": [], "active_areas": []} for k in top_k_list}
    z_delta = z_delta.to(device)

    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader_eval, desc="Evaluating regions"):
            batch_imgs = batch_imgs.to(device)
            z_ref = encode_func(batch_imgs)
            for k in top_k_list:
                top_idx = sorted_idx[:k]
                z_pert = z_ref.clone()
                z_pert[:, top_idx] += z_delta[top_idx]
                img_pert = decode_func(z_pert)
                heatmap = (batch_imgs - img_pert).abs().mean(dim=1)
                for b in range(batch_imgs.size(0)):
                    h = heatmap[b].cpu().numpy()
                    flat = h.flatten()
                    flat_sorted = np.sort(flat)[::-1]
                    top10_sum = flat_sorted[: max(1, int(0.1 * len(flat)))].sum()
                    total_sum = flat.sum() + 1e-8
                    top10_ratio = top10_sum / total_sum
                    n_pixels = len(flat)
                    gini = (
                        2 * np.sum(np.arange(1, n_pixels + 1) * flat_sorted)
                        / (n_pixels * np.sum(flat_sorted))
                        - (n_pixels + 1) / n_pixels
                    )
                    gini = float(np.clip(gini, 0, 1))
                    active_area = float((h > h.mean() + h.std()).sum() / h.size)
                    results[k]["top10_ratios"].append(top10_ratio)
                    results[k]["gini_coeffs"].append(gini)
                    results[k]["active_areas"].append(active_area)

    summary: Dict[int, Dict[str, float]] = {}
    for k in top_k_list:
        summary[k] = {
            "top10_ratio": float(np.mean(results[k]["top10_ratios"])),
            "gini_coeff": float(np.mean(results[k]["gini_coeffs"])),
            "active_area": float(np.mean(results[k]["active_areas"])),
        }
        print(
            f"K={k:2d} | Top10%: {summary[k]['top10_ratio']:.4f} | "
            f"Gini: {summary[k]['gini_coeff']:.4f} | Active: {summary[k]['active_area']:.4f}"
        )
    return summary


def save_visualization(img: torch.Tensor, img_pert: torch.Tensor, heatmap: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def to_np(t: torch.Tensor) -> np.ndarray:
        return (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)

    axes[0].imshow(to_np(img))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(to_np(img_pert))
    axes[1].set_title("Perturbed")
    axes[1].axis("off")
    axes[2].imshow(to_np(img))
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title("Heatmap")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    train_dir = Path(args.train_dir)
    eval_dir = Path(args.eval_dir)
    if args.synthetic:
        train_dir = out_dir / "synthetic_train"
        eval_dir = out_dir / "synthetic_eval"
        make_synthetic_dataset(train_dir, n_images=args.synthetic_n, seed=0, color_shift=0.0)
        make_synthetic_dataset(eval_dir, n_images=args.synthetic_n, seed=1, color_shift=40.0)

    transform = make_transform(args.image_size)
    loader_train = DataLoader(
        ImageDataset(str(train_dir), transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loader_eval = DataLoader(
        ImageDataset(str(eval_dir), transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    vae = ConvVAE(latent_dim=args.latent_dim, img_channels=3, img_size=args.image_size).to(device)
    model_path = out_dir / "vae_model.pth"

    if model_path.exists() and not args.retrain:
        print(f"Loading pre-trained VAE from {model_path}")
        vae.load_state_dict(torch.load(model_path, map_location=device))
    else:
        train_vae(vae, loader_train, device, args.num_epochs, model_path)

    vae.eval()

    def vae_encode(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = vae.encode(x.to(device))
        return mu

    def vae_decode(z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return vae.decode(z.to(device))

    z_train = encode_dataset(loader_train, lambda x: vae_encode(x))
    z_eval = encode_dataset(loader_eval, lambda x: vae_encode(x))
    print(f"z_train: {z_train.shape}, z_eval: {z_eval.shape}")

    importance, oob_auc = rf_domain_classifier(z_train, z_eval, seed=args.seed)
    vimp_rank = np.argsort(-importance)
    print(f"OOB score: {oob_auc:.4f}")
    print(f"Top 10 latent dims: {vimp_rank[:10].tolist()}")

    z_delta = z_eval.mean(dim=0) - z_train.mean(dim=0)
    summary = evaluate_region_selection(
        loader_eval,
        lambda x: vae_encode(x),
        lambda z: vae_decode(z),
        z_delta,
        importance,
        device,
        top_k_list=args.top_k,
    )

    sample_img, _ = next(iter(loader_eval))
    sample_img = sample_img[:1].to(device)
    z_sample = vae_encode(sample_img)
    k_vis = min(10, args.top_k[-1] if args.top_k else 10)
    top_idx = vimp_rank[:k_vis]
    z_pert = z_sample.clone()
    z_pert[0, top_idx] += z_delta[top_idx].to(device)
    img_pert = vae_decode(z_pert)
    heatmap = (sample_img - img_pert).abs().mean(dim=1).squeeze(0).cpu().numpy()
    save_visualization(sample_img, img_pert, heatmap, out_dir / "sample_heatmap.png")

    torch.save(z_train, out_dir / "z_train_vae.pt")
    torch.save(z_eval, out_dir / "z_eval_vae.pt")
    torch.save(z_delta.cpu(), out_dir / "z_delta_vae.pt")
    np.save(out_dir / "importance_vae.npy", importance)
    np.save(out_dir / "vimp_rank_vae.npy", vimp_rank)
    with open(out_dir / "summary_vae.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    pd.DataFrame(
        [{"latent_dim": i, "importance": importance[i], "rank": int(np.where(vimp_rank == i)[0][0]) + 1}
         for i in range(len(importance))]
    ).to_csv(out_dir / "vimp_latent_dims.csv", index=False)

    print(f"\n===== VAE pipeline complete — outputs in {out_dir} =====")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAE domain-shift VIMP pipeline")
    parser.add_argument(
        "--train-dir",
        default="/workspace/data/fmow_subset/train",
        help="training/source domain images",
    )
    parser.add_argument(
        "--eval-dir",
        default="/workspace/data/fmow_subset/eval",
        help="evaluation/target domain images",
    )
    parser.add_argument("--output-dir", default="vae_domain_shift_outputs")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="smoke test with random images")
    parser.add_argument("--synthetic-n", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
