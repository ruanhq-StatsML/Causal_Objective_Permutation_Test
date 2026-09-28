"""Run the multimodal attribution wrapper on a small synthetic shift."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multimodal_attribution import MultimodalAttribution


def _feature_shift(n: int, p: int, shifted: list[int], delta: float, seed: int):
    rng = np.random.default_rng(seed)
    X_ref = rng.normal(size=(n, p))
    X_query = rng.normal(size=(n, p))
    X_query[:, shifted] += delta
    return X_ref, X_query


def _images(n: int, shifted_patches: list[int], delta: float, seed: int, patch: int = 8):
    rng = np.random.default_rng(seed)
    height = width = 32
    grid = height // patch
    images_ref = rng.normal(0.45, 0.08, size=(n, 3, height, width))
    images_query = rng.normal(0.45, 0.08, size=(n, 3, height, width))
    yy, xx = np.mgrid[0:height, 0:width]
    texture = 0.08 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    images_ref += texture
    images_query += texture
    for patch_id in shifted_patches:
        row, col = divmod(patch_id, grid)
        r0, c0 = row * patch, col * patch
        images_query[:, :, r0:r0 + patch, c0:c0 + patch] += delta
    return np.clip(images_ref, 0, 1), np.clip(images_query, 0, 1)


def _global_images(n: int, delta: float, seed: int):
    rng = np.random.default_rng(seed)
    images_ref = rng.normal(0.45, 0.08, size=(n, 3, 32, 32))
    images_query = images_ref + delta + rng.normal(0, 0.02, size=images_ref.shape)
    return np.clip(images_ref, 0, 1), np.clip(images_query, 0, 1)


def main() -> None:
    tabular_truth = [0, 1, 2]
    token_truth = [5, 9]
    image_truth = [0, 1]
    X_ref, X_query = _feature_shift(240, 12, tabular_truth, delta=1.8, seed=1)
    X_null_ref, X_null_query = _feature_shift(240, 12, [], delta=0.0, seed=2)
    T_ref, T_query = _feature_shift(240, 16, token_truth, delta=1.6, seed=3)
    img_ref, img_query = _images(64, image_truth, delta=0.55, seed=4)
    glob_ref, glob_query = _global_images(64, delta=0.25, seed=5)

    wrapper = MultimodalAttribution(n_estimators=60, seed=2026, top_k=(1, 2, 4, 8))
    wrapper.add_features("tabular", X_ref, X_query, truth_idx=tabular_truth)
    wrapper.add_features("tabular_null", X_null_ref, X_null_query, truth_idx=[0, 1, 2])
    wrapper.add_features("tokens", T_ref, T_query, truth_idx=token_truth)
    wrapper.add_images("image_local", img_ref, img_query, patch=8, truth_idx=image_truth)
    wrapper.add_images("image_global", glob_ref, glob_query, patch=8, truth_idx=image_truth)

    result = wrapper.run()
    summary = result.summary.copy()
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    print(summary.round(4).to_string(index=False))

    out_dir = Path("/opt/cursor/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "multimodal_attribution_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    tab_imp = result.importance["tabular"]
    colors = ["#c4512c" if i in tabular_truth else "#8aa0b4" for i in range(tab_imp.size)]
    axes[0].bar(np.arange(tab_imp.size), tab_imp, color=colors)
    axes[0].set_title("Tabular importance (shifted features in red)")
    axes[0].set_xlabel("feature")
    axes[0].set_ylabel("importance")
    heat = result.example_heatmap["image_local"]
    axes[1].imshow(heat, cmap="magma")
    axes[1].set_title("Image perturbation heatmap")
    axes[1].axis("off")
    fig.tight_layout()
    fig_path = out_dir / "multimodal_attribution_demo.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"saved {fig_path}")

    by_name = summary.set_index("modality")
    assert by_name.loc["tabular", "recall_at_truth"] == 1.0
    assert by_name.loc["tokens", "recall_at_truth"] == 1.0
    assert by_name.loc["tabular", "oob_score"] > by_name.loc["tabular_null", "oob_score"] + 0.15
    assert by_name.loc["image_local", "recall_at_truth"] == 1.0
    assert by_name.loc["image_local", "importance_entropy"] < by_name.loc["image_global", "importance_entropy"]
    assert by_name.loc["image_local", "mass_on_truth"] > by_name.loc["image_global", "mass_on_truth"]
    print("DEMO_OK")


if __name__ == "__main__":
    main()
