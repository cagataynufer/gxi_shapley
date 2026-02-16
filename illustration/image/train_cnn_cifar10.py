"""
Train a CNN on CIFAR-10 for image illustration purposes.

This script trains a lightweight convolutional neural network on the
CIFAR-10 dataset and saves the trained weights to disk. The resulting
model is used solely to obtain a non-degenerate image model for the
illustration of GXI Shapley attributions.

This script performs no attribution or experimental evaluation beyond
reporting final training loss and test accuracy.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.illustration_image_cifar10 import load_cifar10
from src.models.illustration_image_cnn import CIFAR10CNN
from src.utils.illustration_image_cnn_training import train_cnn


def main() -> None:
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results/illustration/image")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "cnn_cifar10.pt"

    # Training hyperparameters (kept modest on purpose)
    epochs = 30
    batch_size = 128
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    step_size = 50
    gamma = 0.1
    seed = 0


    # Load data
    train_dataset = load_cifar10(split="train")
    test_dataset = load_cifar10(split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )


    # Initialize model
    model = CIFAR10CNN(num_classes=10)


    # Train model
    model, train_loss, test_acc = train_cnn(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
        device=device,
        seed=seed,
        save_path=model_path,
    )


    # Report results
    print("\nTraining completed.")
    print(f"Final train loss: {train_loss:.6f}")
    print(f"Final test accuracy: {test_acc * 100:.2f}%")
    print(f"Model saved to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
