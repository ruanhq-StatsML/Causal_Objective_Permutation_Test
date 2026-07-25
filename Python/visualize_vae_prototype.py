"""Quick cross-domain VAE prototype: debris -> lake, latent perturb -> pixel heatmap."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from run_vae_domain_shift import (
    ConvVAE,
    ImageDataset,
    encode_dataset,
    make_transform,
    rf_domain_classifier,
    train_vae,
)


def subsample_dataset(image_dir: str, transform, max_images: int, seed: int) -> ImageDataset:
    ds = ImageDataset(image_dir, transform=transform)
    if len(ds.image_paths) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds.image_paths), max_images, replace=False)
        ds.image_paths = [ds.image_paths[i] for i in sorted(idx)]
        print(f"Subsampled to {len(ds.image_paths)} images")
    return ds


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)


def save_prototype_figure(
    out_path: Path,
    img_d1: torch.Tensor,
    img_d2: torch.Tensor,
    img_pert: torch.Tensor,
    heatmap: np.ndarray,
    z_train: torch.Tensor,
    z_eval: torch.Tensor,
    importance: np.ndarray,
    vimp_rank: np.ndarray,
    oob: float,
    top_k: int,
) -> None:
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(tensor_to_np(img_d1))
    ax0.set_title("Domain1 (debris)\nsource / train")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(tensor_to_np(img_d2))
    ax1.set_title("Domain2 (lake)\noriginal")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(tensor_to_np(img_pert))
    ax2.set_title(f"Domain2 perturbed\nTop-{top_k} latent + z_delta")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(tensor_to_np(img_d2))
    ax3.imshow(heatmap, cmap="jet", alpha=0.55)
    ax3.set_title("Pixel-space change\n|original - perturbed|")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, :2])
    z_all = np.vstack([z_train.numpy(), z_eval.numpy()])
    labels = np.array([0] * z_train.shape[0] + [1] * z_eval.shape[0])
    pcs = PCA(n_components=2).fit_transform(z_all)
    ax4.scatter(pcs[labels == 0, 0], pcs[labels == 0, 1], s=8, alpha=0.5, label="domain1")
    ax4.scatter(pcs[labels == 1, 0], pcs[labels == 1, 1], s=8, alpha=0.5, label="domain2")
    ax4.set_title(f"Latent PCA (OOB={oob:.3f})")
    ax4.legend()
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2:])
    top_n = 15
    top_idx = vimp_rank[:top_n]
    ax5.barh(top_idx.astype(str), importance[top_idx], color="#4C72B0")
    ax5.invert_yaxis()
    ax5.set_xlabel("VIMP importance")
    ax5.set_title(f"Top-{top_n} latent dims (domain separator)")

    fig.suptitle("VAE domain-shift prototype: latent perturbation -> pixel heatmap", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        default="/workspace/data/fmow_subset/raw/1lhFuyukjnZYdWFYfDRTur78lmSB7fzVr/debris_or_rubble_processed",
    )
    parser.add_argument(
        "--eval-dir",
        default="/workspace/data/fmow_subset/raw/1MI0BVcWIo8st4NVXfA2evwuOebZu_3RQ/lake_or_pond_processed",
    )
    parser.add_argument("--output-dir", default="vae_prototype_outputs")
    parser.add_argument("--max-train-images", type=int, default=256)
    parser.add_argument("--max-eval-images", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    transform = make_transform()

    loader_train = DataLoader(
        subsample_dataset(args.train_dir, transform, args.max_train_images, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    loader_eval = DataLoader(
        subsample_dataset(args.eval_dir, transform, args.max_eval_images, args.seed + 1),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    vae = ConvVAE().to(device)
    model_path = out_dir / "vae_model.pth"
    train_vae(vae, loader_train, device, args.num_epochs, model_path)
    vae.eval()

    def encode(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _ = vae.encode(x.to(device))
        return mu

    def decode(z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return vae.decode(z.to(device))

    z_train = encode_dataset(loader_train, encode)
    z_eval = encode_dataset(loader_eval, encode)
    importance, oob = rf_domain_classifier(z_train, z_eval, seed=args.seed)
    vimp_rank = np.argsort(-importance)
    z_delta = z_eval.mean(dim=0) - z_train.mean(dim=0)

    img_d1, _ = next(iter(loader_train))
    img_d2, _ = next(iter(loader_eval))
    img_d1 = img_d1[:1].to(device)
    img_d2 = img_d2[:1].to(device)

    z_ref = encode(img_d2)
    top_idx = vimp_rank[: args.top_k]
    z_pert = z_ref.clone()
    z_pert[:, top_idx] += z_delta[top_idx].to(device)
    img_pert = decode(z_pert)
    heatmap = (img_d2 - img_pert).abs().mean(dim=1).squeeze(0).cpu().numpy()

    fig_path = out_dir / "prototype_cross_domain.png"
    save_prototype_figure(
        fig_path,
        img_d1,
        img_d2,
        img_pert,
        heatmap,
        z_train,
        z_eval,
        importance,
        vimp_rank,
        oob,
        args.top_k,
    )
    print(f"Prototype saved to {fig_path}")


if __name__ == "__main__":
    main()
