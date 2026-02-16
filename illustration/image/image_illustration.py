"""
Image illustration for GXI-Shapley (single-feature game).

This script:
- loads a trained CIFAR-10 CNN,
- selects the dog test image (index = 16),
- prints prediction for sanity check,
- constructs one GXIShapleyGame for a fixed superpixel,
- approximates Shapley values using PermutationSamplingSV,
- visualizes coalition-conditioned GXI signals.

The goal is purely illustrative and not experimental.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from shapiq.approximator import PermutationSamplingSV

from src.data.illustration_image_cifar10 import get_cifar10_sample
from src.models.illustration_image_cnn import CIFAR10CNN
from src.gxi.game import GXIShapleyGame
from src.masking.image import ImageMasker


def main() -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results/illustration/image")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "cnn_cifar10.pt"

    # Superpixel configuration
    H, W, C = 32, 32, 3
    sp_h, sp_w = 4, 4
    n_sp_h = H // sp_h
    n_sp_w = W // sp_w
    n_players = n_sp_h * n_sp_w  # 64

    # Target feature (superpixel index)
    feature_index = 18

    # Horse image index
    TARGET_INDEX = 7886

    # Shapley approximation settings
    budget = 10000
    batch_size = 10
    random_state = 0


    # Load model
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load horse test image
    x_single, y = get_cifar10_sample(
        index=TARGET_INDEX,
        split="test",
        device=device,
    )

    print(f"\nSelected test index: {TARGET_INDEX}")

    # Prediction sanity check
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    with torch.no_grad():
        logits = model(x_single.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)

        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()

    print("Ground truth :", CIFAR10_CLASSES[y])
    print("Prediction   :", CIFAR10_CLASSES[pred_class])
    print(f"Confidence   : {confidence:.4f}")

    # Define masker and baseline
    baseline = torch.zeros((C, H, W), device=device)

    masker = ImageMasker(
        image_height=H,
        image_width=W,
        channels=C,
        sp_height=sp_h,
        sp_width=sp_w,
        baseline=baseline,
    )

    # Construct GXI-Shapley game
    game = GXIShapleyGame(
        n_players=n_players,
        model=model,
        x=x_single,
        masker=masker,
        device=device,
        aggregation="signed",
        normalize=False,
    )

    game.feature_index = feature_index
    game.target_index = None

    # Approximate Shapley values
    approximator = PermutationSamplingSV(
        n=n_players,
        random_state=random_state,
    )

    ivs = approximator.approximate(
        game=game,
        budget=budget,
        batch_size=batch_size,
    )

    sv = ivs.values[1:]  # remove null player
    sv_grid = sv.reshape(n_sp_h, n_sp_w)

    # Shared color scale (RAW values)
    vmin = sv_grid.min()
    vmax = sv_grid.max()

    # Plot 1 — Superpixel GXI response grid
    fig1, ax1 = plt.subplots(figsize=(6, 6))

    im = ax1.imshow(
        sv_grid,
        cmap="hot",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig1.colorbar(im, ax=ax1)
    cbar.set_label(
        "Signed GXI–Shapley Value"
    )

    ax1.set_title(
        f"Heat Map of GXI-Shapley Values\n"
        f"for Selected Superpixel {feature_index}",
        fontsize=13
    )

    # Highlight target superpixel
    i = feature_index // n_sp_w
    j = feature_index % n_sp_w
    ax1.scatter(j, i, s=200, c="cyan", marker="x", linewidths=2)

    fig1.tight_layout()
    fig1.savefig(
        results_dir / "superpixel_gxi_response_grid.pdf",
        bbox_inches="tight",
    )
    plt.close(fig1)

    # Upsample grid to image resolution
    heat = torch.tensor(
        sv_grid,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    heat_up = F.interpolate(
        heat,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    # Normalize image for display only
    img = x_single.permute(1, 2, 0).cpu().numpy()
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Plot 2 — Overlay + original
    fig2, axs = plt.subplots(
        1, 2,
        figsize=(12, 5),
        constrained_layout=True,
    )

    fig2.suptitle(
        f"Heat Map Overlay of Signed GXI–Shapley Values\n"
        f"(for Selected Superpixel {feature_index})",
        fontsize=14,
    )

    # Overlay
    axs[0].imshow(img_norm, alpha=0.6)

    heatmap_im = axs[0].imshow(
        heat_up,
        cmap="magma",
        alpha=0.7,
        vmin=vmin,
        vmax=vmax,
    )

    x_center = j * sp_w + sp_w / 2
    y_center = i * sp_h + sp_h / 2

    axs[0].scatter(
        x_center,
        y_center,
        s=450,
        facecolors="none",
        edgecolors="cyan",
        linewidths=2.5,
    )

    axs[0].scatter(
        x_center,
        y_center,
        s=300,
        c="cyan",
        marker="x",
        linewidths=3,
    )

    axs[0].set_title("GXI Overlay", fontsize=11)
    axs[0].axis("off")

    cbar = fig2.colorbar(
        heatmap_im,
        ax=axs[0],
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "Signed GXI–Shapley Value"
    )

    # Original image
    axs[1].imshow(img_norm)
    axs[1].set_title("Original Image", fontsize=11)
    axs[1].axis("off")

    fig2.savefig(
        results_dir / "gxi_response_overlay.pdf",
        bbox_inches="tight",
    )
    plt.close(fig2)

    print("\nFigures saved to:")
    print(f"  {results_dir / 'superpixel_gxi_response_grid.pdf'}")
    print(f"  {results_dir / 'gxi_response_overlay.pdf'}")


if __name__ == "__main__":
    main()
